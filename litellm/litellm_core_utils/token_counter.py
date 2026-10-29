# What is this?
## Helper utilities for token counting
import base64
import io
import math
import re
import struct
from bisect import bisect_right
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from itertools import accumulate
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, cast

import anyio
import anyio.lowlevel
import httpx
import tiktoken
from tiktoken.model import MODEL_TO_ENCODING
from typing_extensions import ParamSpec, TypeVar

import litellm
from litellm import verbose_logger
from litellm._lazy_imports import get_default_encoding
from litellm.constants import (
    ANTHROPIC_IMAGE_MAX_LONG_EDGE_PX,
    ANTHROPIC_IMAGE_MAX_PIXELS,
    ANTHROPIC_IMAGE_PIXELS_PER_TOKEN,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_TOKEN_COUNT,
    DEFAULT_IMAGE_WIDTH,
    MAX_IMAGE_URL_DOWNLOAD_SIZE_MB,
    MAX_LONG_SIDE_FOR_IMAGE_HIGH_RES,
    MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES,
    MAX_TILE_HEIGHT,
    MAX_TILE_WIDTH,
    PDF_DATA_URL_PREFIX,
    TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS,
    TOKEN_COUNTER_MAX_CONCURRENT_COUNTS,
    TOKEN_COUNTER_MAX_EXACT_CHARS,
)
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.prompt_templates.chat_template_tools import (
    ChatTemplateToolSection,
    chat_template_tool_section,
)
from litellm.litellm_core_utils.tokenizer import HuggingFaceTokenizer, OpenAIEncoding

if TYPE_CHECKING:
    from litellm.litellm_core_utils.tokenizer import Encoding, HuggingFace
from litellm.litellm_core_utils.url_utils import safe_get
from litellm.llms.custom_httpx.http_handler import get_httpx_client, header_value
from litellm.router_utils.deployment_pricing import deployment_pricing_models
from litellm.rust_bridge.tokenizer import get_encoding
from litellm.types.llms.anthropic import (
    AnthropicContentParamSource,
    AnthropicContentParamSourceFileId,
    AnthropicContentParamSourceUrl,
    AnthropicMessagesDocumentParam,
    AnthropicMessagesImageParam,
    AnthropicMessagesTextParam,
    AnthropicMessagesToolResultParam,
    AnthropicMessagesToolUseParam,
    AnthropicSearchResultParam,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionDocumentObject,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
    ChatCompletionToolParam,
    OpenAIMessageContentListBlock,
)
from litellm.types.utils import Message, SelectTokenizerResponse


class UnsupportedContentTypeError(ValueError):
    pass


def get_modified_max_tokens(
    model: str,
    base_model: str,
    messages: list[AllMessageValues] | None,
    user_max_tokens: int | None,
    buffer_perc: float | None,
    buffer_num: float | None,
) -> int | None:
    """
    Params:

    Returns the user's max output tokens, adjusted for:
    - the size of input - for models where input + output can't exceed X
    - model max output tokens - for models where there is a separate output token limit
    """
    try:
        if user_max_tokens is None:
            return None

        ## MODEL INFO
        _model_info: Final = litellm.get_model_info(model=model)

        max_output_tokens: Final = litellm.get_max_tokens(model=base_model)  # assume min context window is 4k tokens

        ## UNKNOWN MAX OUTPUT TOKENS - return user defined amount
        if max_output_tokens is None:
            return user_max_tokens

        input_tokens = litellm.token_counter(model=base_model, messages=messages)

        # token buffer
        if buffer_perc is None:
            buffer_perc = 0.1
        if buffer_num is None:
            buffer_num = 10
        token_buffer: Final = max(
            buffer_perc * input_tokens, buffer_num
        )  # give at least a 10 token buffer. token counting can be imprecise.

        input_tokens += int(token_buffer)
        verbose_logger.debug("max_output_tokens: %s, user_max_tokens: %s", max_output_tokens, user_max_tokens)
        ## CASE 1: model input + output can't exceed X - happens when max input = max output, e.g. gpt-3.5-turbo
        if _model_info["max_input_tokens"] == max_output_tokens:
            verbose_logger.debug("input_tokens: %s, max_output_tokens: %s", input_tokens, max_output_tokens)
            if input_tokens > max_output_tokens:
                pass  # allow call to fail normally - don't set max_tokens to negative.
            elif (
                user_max_tokens + input_tokens > max_output_tokens
            ):  # we can still modify to keep it positive but below the limit
                verbose_logger.debug(
                    "MODIFYING MAX TOKENS - user_max_tokens=%s, input_tokens=%s, max_output_tokens=%s",
                    user_max_tokens,
                    input_tokens,
                    max_output_tokens,
                )
                user_max_tokens = int(max_output_tokens - input_tokens)
        ## CASE 2: user_max_tokens> model max output tokens
        elif user_max_tokens > max_output_tokens:
            user_max_tokens = max_output_tokens

        verbose_logger.debug(
            "litellm.litellm_core_utils.token_counter.py::get_modified_max_tokens() - user_max_tokens: %s",
            user_max_tokens,
        )

        return user_max_tokens
    except Exception as e:
        verbose_logger.debug(
            "litellm.litellm_core_utils.token_counter.py::get_modified_max_tokens() - Error while checking max token limit: %s\nmodel=%s, base_model=%s",
            e,
            model,
            base_model,
        )
        return user_max_tokens


def resize_image_high_res(
    width: int,
    height: int,
) -> tuple[int, int]:
    # Maximum dimensions for high res mode
    max_short_side: Final = MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES
    max_long_side: Final = MAX_LONG_SIDE_FOR_IMAGE_HIGH_RES

    # Return early if no resizing is needed
    if width <= MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES and height <= MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES:
        return width, height

    # Determine the longer and shorter sides
    longer_side: Final = max(width, height)
    shorter_side: Final = min(width, height)

    # Calculate the aspect ratio
    aspect_ratio: Final = longer_side / shorter_side

    # Resize based on the short side being 768px
    if width <= height:  # Portrait or square
        resized_width = max_short_side
        resized_height = int(resized_width * aspect_ratio)
        # if the long side exceeds the limit after resizing, adjust both sides accordingly
        if resized_height > max_long_side:
            resized_height = max_long_side
            resized_width = int(resized_height / aspect_ratio)
    else:  # Landscape
        resized_height = max_short_side
        resized_width = int(resized_height * aspect_ratio)
        # if the long side exceeds the limit after resizing, adjust both sides accordingly
        if resized_width > max_long_side:
            resized_width = max_long_side
            resized_height = int(resized_width / aspect_ratio)

    return resized_width, resized_height


# Test the function with the given example
def calculate_tiles_needed(
    resized_width: int,
    resized_height: int,
    tile_width: int = MAX_TILE_WIDTH,
    tile_height: int = MAX_TILE_HEIGHT,
) -> int:
    tiles_across: Final = (resized_width + tile_width - 1) // tile_width
    tiles_down: Final = (resized_height + tile_height - 1) // tile_height
    total_tiles: Final = tiles_across * tiles_down
    return total_tiles


def high_detail_image_token_upper_bound(base_tokens: int = 85, tile_tokens: int | None = None) -> int:
    largest_tile_count: Final = calculate_tiles_needed(
        MAX_LONG_SIDE_FOR_IMAGE_HIGH_RES, MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES
    )
    per_tile: Final = base_tokens * 2 if tile_tokens is None else tile_tokens
    return base_tokens + per_tile * largest_tile_count


def _unpack_ints(fmt: str, buffer: bytes) -> tuple[int, ...]:
    return struct.unpack(fmt, buffer)


def get_image_type(image_data: bytes) -> str | None:
    """take an image (really only the first ~100 bytes max are needed)
    and return 'png' 'gif' 'jpeg' 'webp' 'heic' or None. method added to
    allow deprecation of imghdr in 3.13"""

    if image_data[0:8] == b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a":
        return "png"

    if image_data[0:4] == b"GIF8" and image_data[5:6] == b"a":
        return "gif"

    if image_data[0:3] == b"\xff\xd8\xff":
        return "jpeg"

    if image_data[4:8] == b"ftyp":
        return "heic"

    if image_data[0:4] == b"RIFF" and image_data[8:12] == b"WEBP":
        return "webp"

    return None


def get_image_dimensions(
    data: str,
) -> tuple[int, int]:
    """
    Async Function to get the dimensions of an image from a URL or base64 encoded string.

    Args:
        data (str): The URL or base64 encoded string of the image.

    Returns:
        Tuple[int, int]: The width and height of the image.
    """
    img_data = None
    if data.startswith(("http://", "https://")):
        try:
            client: Final = get_httpx_client()
            response: Final[httpx.Response] = safe_get(client, data)
            max_bytes: Final = int(MAX_IMAGE_URL_DOWNLOAD_SIZE_MB * 1024 * 1024)
            content_length: Final[str | None] = header_value(response.headers, "Content-Length")
            if content_length is not None and int(content_length) > max_bytes:
                pass  # skip download; img_data stays None
            else:
                body: Final = response.read()
                if len(body) <= max_bytes:
                    img_data = body
        except Exception:
            pass
    if img_data is None:
        # Not a URL or fetch failed — assume base64
        _header, encoded = data.split(",", 1)
        img_data = base64.b64decode(encoded)

    return _image_dimensions_from_bytes(img_data)


def _image_dimensions_from_bytes(img_data: bytes) -> tuple[int, int]:
    img_type: Final = get_image_type(img_data)

    if img_type == "png":
        w, h = _unpack_ints(">LL", img_data[16:24])
        return w, h
    elif img_type == "gif":
        w, h = _unpack_ints("<HH", img_data[6:10])
        return w, h
    elif img_type == "jpeg":
        with io.BytesIO(img_data) as fhandle:
            fhandle.seek(0)
            size = 2
            ftype = 0
            while not 0xC0 <= ftype <= 0xCF or ftype in (0xC4, 0xC8, 0xCC):
                fhandle.seek(size, 1)
                byte = fhandle.read(1)
                while ord(byte) == 0xFF:
                    byte = fhandle.read(1)
                ftype = ord(byte)
                size = _unpack_ints(">H", fhandle.read(2))[0] - 2
            fhandle.seek(1, 1)
            h, w = _unpack_ints(">HH", fhandle.read(4))
        return w, h
    elif img_type == "webp":
        # For WebP, the dimensions are stored at different offsets depending on the format
        # Check for VP8X (extended format)
        if img_data[12:16] == b"VP8X":
            w = _unpack_ints("<I", img_data[24:27] + b"\x00")[0] + 1
            h = _unpack_ints("<I", img_data[27:30] + b"\x00")[0] + 1
            return w, h
        # Check for VP8 (lossy format)
        elif img_data[12:16] == b"VP8 ":
            w = _unpack_ints("<H", img_data[26:28])[0] & 0x3FFF
            h = _unpack_ints("<H", img_data[28:30])[0] & 0x3FFF
            return w, h
        # Check for VP8L (lossless format)
        elif img_data[12:16] == b"VP8L":
            bits: Final = _unpack_ints("<I", img_data[21:25])[0]
            w = (bits & 0x3FFF) + 1
            h = ((bits >> 14) & 0x3FFF) + 1
            return w, h

    # return sensible default image dimensions if unable to get dimensions
    return DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT


def _auto_image_dimensions(data: str) -> tuple[int, int] | None:
    """Read bounded inline image data without fetching caller-supplied URLs."""
    if not data.startswith("data:image/"):
        return None
    max_bytes: Final = int(MAX_IMAGE_URL_DOWNLOAD_SIZE_MB * 1024 * 1024)
    max_encoded_length: Final = 4 * ((max_bytes + 2) // 3)
    # Bound allocations before splitting or decoding the data URL, including its header.
    if max_bytes <= 0 or len(data) > max_encoded_length + 128:
        return None
    header, separator, encoded = data.partition(",")
    if not separator or len(header) > 128 or not header.endswith(";base64") or len(encoded) > max_encoded_length:
        return None
    try:
        image_data: Final = base64.b64decode(encoded, validate=True)
        if len(image_data) > max_bytes or get_image_type(image_data) not in (
            "png",
            "gif",
            "jpeg",
            "webp",
        ):
            return None
        width, height = _image_dimensions_from_bytes(image_data)
    except (ValueError, TypeError, struct.error):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def calculate_img_tokens(
    data: str,
    mode: Literal["low", "high", "auto"] = "auto",
    base_tokens: int = 85,  # openai default - https://openai.com/pricing
    use_default_image_token_count: bool = False,
):
    """
    Calculate the number of tokens for an image.

    Args:
        data (str): The URL or base64 encoded string of the image.
        mode (Literal["low", "high", "auto"]): Auto uses high-detail tiles for readable inline images;
            URLs and unreadable images keep the base estimate without network I/O.
        base_tokens (int): The base number of tokens for an image.
        use_default_image_token_count (bool): When True, will NOT make a GET request to the image URL and instead return the default image dimensions.

    Returns:
        int: The number of tokens for the image.
    """
    if use_default_image_token_count:
        verbose_logger.debug("Using default image token count: %s", DEFAULT_IMAGE_TOKEN_COUNT)
        return DEFAULT_IMAGE_TOKEN_COUNT
    if mode == "low":
        return base_tokens
    if mode not in ("auto", "high"):
        return None
    dimensions: Final = _auto_image_dimensions(data) if mode == "auto" else get_image_dimensions(data=data)
    if dimensions is None:
        return base_tokens
    width, height = dimensions
    resized_width, resized_height = resize_image_high_res(width=width, height=height)
    tiles_needed_high_res = calculate_tiles_needed(resized_width=resized_width, resized_height=resized_height)
    tile_tokens: Final = (base_tokens * 2) * tiles_needed_high_res
    total_tokens: Final = base_tokens + tile_tokens
    return total_tokens


_IMAGE_HEADER_BASE64_CHARS: Final = 65_536
_MAX_IMAGE_PATCHES: Final = 30_000
_ANTHROPIC_IMAGE_LIMITS: Final = MappingProxyType(
    {"claude-standard": (ANTHROPIC_IMAGE_MAX_LONG_EDGE_PX, 1568), "claude-high": (2576, 4784)}
)
_TILE_IMAGE_TOKEN_COSTS: Final = MappingProxyType(
    {
        "gpt-4o": (85, 170),
        "gpt-4.1": (85, 170),
        "gpt-4o-mini": (2833, 5667),
        "gpt-5": (70, 140),
        "gpt-5.1": (70, 140),
        "o1": (75, 150),
        "o1-pro": (75, 150),
        "o3": (75, 150),
    }
)
_PATCH_IMAGE_MULTIPLIERS: Final = MappingProxyType(
    {
        "gpt-6-astra": 1.2,
        "gpt-5.6-sol": 1.2,
        "gpt-5.6-terra": 1.2,
        "gpt-5.6-luna": 1.2,
        "gpt-5.5": 1.2,
        "gpt-5.4": 1.2,
        "gpt-5.4-mini": 1.2,
        "gpt-5.4-nano": 1.2,
        "gpt-5.2": 1.2,
        "gpt-4.1-mini": 1.62,
    }
)


def _inline_image_dimensions(data: str) -> tuple[int, int] | None:
    separator: Final = data.find(",", 0, 256)
    if not data.startswith("data:") or separator < 0 or not data[:separator].endswith(";base64"):
        return None
    encoded_header: Final = data[separator + 1 : separator + 1 + _IMAGE_HEADER_BASE64_CHARS]
    try:
        header: Final = base64.b64decode(encoded_header, validate=True)
        image_type: Final = get_image_type(header)
        if image_type not in ("png", "gif", "jpeg", "webp"):
            return None
        if image_type == "webp" and header[12:16] not in (b"VP8X", b"VP8 ", b"VP8L"):
            return None
        dimensions: Final = _image_dimensions_from_bytes(header)
    except (ValueError, TypeError, struct.error):
        return None
    width, height = dimensions
    return (width, height) if width > 0 and height > 0 else None


def _patch_image_limits(model: str, detail: str) -> tuple[int, int | None] | None:
    if model in ("gpt-5.2", "gpt-4.1-mini"):
        return None if detail == "original" else (2048, 6144)
    original_default: Final = model in ("gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5")
    effective_detail: Final = ("original" if original_default else "high") if detail == "auto" else detail
    if effective_detail == "low":
        return (2048, 6144) if model in ("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano") else (512, None)
    if effective_detail == "high":
        return (65_535 if model == "gpt-6-astra" else 2048, 2500)
    return (
        (65_535, None) if model in ("gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna") else (6000, 10_000)
    )


def _image_patch_count(width: int, height: int, max_dimension: int, budget: int | None) -> int:
    dimension_scale: Final = min(1.0, max_dimension / max(width, height))
    fitted_width: Final = max(1, round(width * dimension_scale))
    fitted_height: Final = max(1, round(height * dimension_scale))
    patches: Final = math.ceil(fitted_width / 32) * math.ceil(fitted_height / 32)
    if budget is None or patches <= budget:
        return patches
    shrink: Final = math.sqrt(32**2 * budget / (fitted_width * fitted_height))
    width_patches: Final = fitted_width * shrink / 32
    height_patches: Final = fitted_height * shrink / 32
    aligned_shrink: Final = shrink * min(
        math.floor(width_patches) / width_patches, math.floor(height_patches) / height_patches
    )
    resized_width: Final = max(1, math.floor(fitted_width * aligned_shrink))
    resized_height: Final = max(1, math.floor(fitted_height * aligned_shrink))
    return math.ceil(resized_width / 32) * math.ceil(resized_height / 32)


def _image_policy_model_name(model: str) -> str | None:
    provider, separator, provider_model = model.partition("/")
    if not separator:
        return model
    if provider == "openai":
        return provider_model
    if provider == "openrouter":
        return provider_model.removeprefix("openai/") if provider_model.startswith("openai/") else None
    if provider not in ("azure", "azure_ai"):
        return None
    catalog_entry: Final = as_str_mapping(
        cast(  # cast-ok: read-only view of the legacy string-keyed registry; as_str_mapping checks the entry
            Mapping[str, object], litellm.model_cost
        ).get(model)
    )
    if catalog_entry is not None and catalog_entry.get("litellm_provider") in ("azure", "azure_ai"):
        return provider_model.rsplit("/", 1)[-1]
    return provider_model.removeprefix("responses/").removeprefix("o_series/").removeprefix("openai/")


def _anthropic_image_family(model: str) -> str | None:
    from litellm.utils import get_potential_model_names

    provider, separator, _ = model.partition("/")
    if separator and provider not in (
        "anthropic",
        "bedrock",
        "bedrock_converse",
        "bedrock_mantle",
        "vertex_ai",
        "vertex_ai_beta",
        "azure_ai",
        "openrouter",
    ):
        return None
    candidates: Final = get_potential_model_names(model, provider if separator else "anthropic")
    catalog: Final = cast(Mapping[str, object], litellm.model_cost)  # cast-ok: read-only view of the legacy registry
    for candidate in dict.fromkeys(candidates.values()):
        if not isinstance(candidate, str):
            continue
        if (entry := as_str_mapping(catalog.get(candidate))) is None or entry.get("supports_vision") is not True:
            continue
        if (
            version := re.search(
                r"(?:^|[./])claude-(?:(?:haiku|sonnet|opus)-)?(\d+)(?:[-.](\d{1,2})(?!\d))?", candidate
            )
        ) is None:
            continue
        major, minor = int(version[1]), int(version[2] or 0)
        return "claude-high" if (major, minor) >= (4, 7) else "claude-standard"
    return None


def _anthropic_image_tokens(dimensions: tuple[int, int] | None, limits: tuple[int, int]) -> int:
    max_edge, token_budget = limits
    if dimensions is None:
        return token_budget
    long_edge, short_edge = max(dimensions), min(dimensions)
    aspect_ratio: Final = long_edge / short_edge

    def tokens_at_edge(edge: int) -> int:
        return math.ceil(edge / 28) * math.ceil(max(round(edge / aspect_ratio), 1) / 28)

    fitted_edge: Final = bisect_right(range(1, min(long_edge, max_edge) + 1), token_budget, key=tokens_at_edge)
    return tokens_at_edge(max(fitted_edge, 1))


def _image_model_family(model: str) -> str | None:
    provider_model: Final = _image_policy_model_name(model)
    if provider_model is None:
        return _anthropic_image_family(model)
    family: Final = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", provider_model)
    canonical: Final = litellm.azure_llms.get(f"azure/{family}", family) if model.startswith("azure/") else family
    return (
        canonical
        if canonical in _TILE_IMAGE_TOKEN_COSTS or canonical in _PATCH_IMAGE_MULTIPLIERS
        else _anthropic_image_family(model)
    )


def _image_policy_tokens(family: str | None, dimensions: tuple[int, int] | None, detail: str) -> int:
    if family is None:
        return DEFAULT_IMAGE_TOKEN_COUNT
    anthropic_limits: Final = _ANTHROPIC_IMAGE_LIMITS.get(family)
    if anthropic_limits is not None:
        return _anthropic_image_tokens(dimensions, anthropic_limits)
    tile_costs: Final = _TILE_IMAGE_TOKEN_COSTS.get(family)
    multiplier: Final = _PATCH_IMAGE_MULTIPLIERS.get(family)
    if tile_costs is not None:
        base_tokens, tile_tokens = tile_costs
        if detail == "original":
            return DEFAULT_IMAGE_TOKEN_COUNT
        if detail == "low":
            return base_tokens
        if dimensions is None:
            return high_detail_image_token_upper_bound(base_tokens, tile_tokens)
        width, height = dimensions
        scale: Final = min(1.0, 2048 / max(width, height), 768 / min(width, height))
        tiles: Final = calculate_tiles_needed(max(1, math.floor(width * scale)), max(1, math.floor(height * scale)))
        return base_tokens + tiles * tile_tokens
    limits: Final = _patch_image_limits(family, detail)
    if limits is None or multiplier is None:
        return DEFAULT_IMAGE_TOKEN_COUNT
    max_dimension, patch_budget = limits
    patches: Final = (
        min(math.ceil(max_dimension / 32) ** 2, patch_budget if patch_budget is not None else _MAX_IMAGE_PATCHES)
        if dimensions is None
        else _image_patch_count(*dimensions, max_dimension, patch_budget)
    )
    return math.ceil(patches * multiplier)


def estimate_image_tokens(
    model: str | Sequence[str], data: str, detail: str = "auto", *, fallback_model: str | None = None
) -> int:
    """Estimate image policies from at most 48 KiB of inline headers, without fetching media."""
    if detail not in ("low", "high", "auto", "original"):
        return DEFAULT_IMAGE_TOKEN_COUNT
    models: Final = (model,) if isinstance(model, str) else model
    fallback_family: Final = _image_model_family(fallback_model) if fallback_model is not None else None
    families: Final = tuple(
        dict.fromkeys(_image_model_family(candidate) or fallback_family for candidate in models or ("",))
    )
    if all(family is None for family in families):
        return DEFAULT_IMAGE_TOKEN_COUNT
    dimensions: Final = _inline_image_dimensions(data)
    return max(_image_policy_tokens(family, dimensions, detail) for family in families)


TokenCounterFunction = Callable[[str], int]
"""
Type for a function that counts tokens in a string.
"""

EXTRAPOLATION_SAMPLES: Final = 16
T_ParamSpec: Final = ParamSpec("T_ParamSpec")
T_Retval = TypeVar("T_Retval")
_COUNT_OFFLOAD_LIMITER: Final = anyio.lowlevel.RunVar[anyio.CapacityLimiter]("litellm_count_offload_limiter")
_ANTHROPIC_TOOL_KEYS_MAPPED_BY_BRIDGE: Final = frozenset(
    ("name", "input_schema", "description", "cache_control", "strict", "type", "eager_input_streaming", "custom")
)


def _count_offload_limiter_for_this_loop() -> anyio.CapacityLimiter:
    existing: Final = _COUNT_OFFLOAD_LIMITER.get(None)
    if existing is not None:
        return existing
    created: Final = anyio.CapacityLimiter(TOKEN_COUNTER_MAX_CONCURRENT_COUNTS)
    _COUNT_OFFLOAD_LIMITER.set(created)
    return created


def offload_token_count(
    function: Callable[T_ParamSpec, T_Retval],
) -> Callable[T_ParamSpec, Awaitable[T_Retval]]:
    async def offloaded(
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,  # kwargs-ok: ParamSpec keeps the wrapped function's own keyword contract
    ) -> T_Retval:
        return await asyncify(function, limiter=_count_offload_limiter_for_this_loop())(*args, **kwargs)

    return offloaded


def _get_tiktoken_count_function(
    encode_length: Callable[[str], int],
    chunk_size: int = TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS,
) -> TokenCounterFunction:
    def count_tokens(text: str) -> int:
        return sum(encode_length(text[start : start + chunk_size]) for start in range(0, len(text), chunk_size))

    return count_tokens


class _MessageCountParams:
    """
    A class to hold the parameters for counting tokens in messages.
    """

    def __init__(
        self,
        model: str,
        custom_tokenizer: dict | SelectTokenizerResponse | None,
    ):
        from litellm.utils import print_verbose

        actual_model: Final = _fix_model_name(model)
        if uses_legacy_message_accounting(model):
            self.tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            self.tokens_per_name = -1  # if there's a name, the role is omitted
        elif actual_model in litellm.open_ai_chat_completion_models or actual_model in litellm.azure_llms:
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        else:
            print_verbose(f"Warning: unknown model {model}. Using default token params.")
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        self.count_function = _get_count_function(model, custom_tokenizer)
        self.tool_section = chat_template_tool_section(model)


def token_counter(
    model="",
    custom_tokenizer: dict | SelectTokenizerResponse | None = None,
    text: str | list[str] | None = None,
    messages: Sequence[AllMessageValues | Message] | None = None,
    count_response_tokens: bool | None = False,
    tools: list[ChatCompletionToolParam] | None = None,
    tool_choice: str | Mapping[str, object] | None = None,
    use_default_image_token_count: bool | None = False,
    default_token_count: int | None = None,
) -> int:
    """
    Count the number of tokens in a given text using a specified model.

    Args:
    model (str): The name of the model to use for tokenization. Default is an empty string.
    custom_tokenizer (Optional[dict]): A custom tokenizer created with the `create_pretrained_tokenizer` or `create_tokenizer` method. Must be a dictionary with a string value for `type` and Tokenizer for `tokenizer`. Default is None.
    text (str): The raw text string to be passed to the model. Default is None.
    messages (Optional[List[AllMessageValues]]): Alternative to passing in text. A list of dictionaries representing messages with "role" and "content" keys. Default is None.
    count_response_tokens (Optional[bool]): set to True to indicate we are processing a stream response.
    tools (Optional[List[ChatCompletionToolParam]]): The available tools. Default is None.
    tool_choice (Optional[ChatCompletionNamedToolChoiceParam]): The tool choice. Default is None.
    use_default_image_token_count (Optional[bool]): When True, will NOT make a GET request to the image URL and instead return the default image dimensions. Default is False.
    default_token_count (Optional[int]): The default number of tokens to return for a message block, if an error occurs. Default is None.

    Returns:
    int: The number of tokens in the text.
    """
    from litellm.utils import convert_list_message_to_dict

    #########################################################
    # Flag to disable token counter
    # We've gotten reports of this consuming CPU cycles,
    # exposing this flag to allow users to disable
    # it to confirm if this is indeed the issue
    #########################################################
    if litellm.disable_token_counter is True:
        return 0

    verbose_logger.debug("messages in token_counter: %s, text in token_counter: %s", messages, text)
    if text is not None and messages is not None:
        raise ValueError("text and messages cannot both be set")
    if use_default_image_token_count is None:
        use_default_image_token_count = False

    if text is not None:
        if tools or tool_choice:
            raise ValueError("tools or tool_choice cannot be set if using text")
        if isinstance(text, list):
            text_to_count = "".join(t for t in text if isinstance(t, str))
        elif isinstance(text, str):
            text_to_count = text
        count_function: Final = _get_count_function(model, custom_tokenizer)
        num_tokens = count_function(text_to_count)

    elif messages is not None:
        new_messages: Final = cast(list[AllMessageValues], convert_list_message_to_dict(messages))
        params: Final = _MessageCountParams(model, custom_tokenizer)
        num_tokens = _count_messages(params, new_messages, use_default_image_token_count, default_token_count)
        if count_response_tokens is False:
            includes_system_message: Final = any(message.get("role", None) == "system" for message in new_messages)
            num_tokens += _count_extra(
                params.count_function, tools, tool_choice, includes_system_message, params.tool_section
            )

    else:
        raise ValueError("Either text or messages must be provided")

    return num_tokens


def messages_reach_token_count(
    model: str,
    messages: Sequence[AllMessageValues | Message],
    threshold: int,
    tools: list[ChatCompletionToolParam] | None = None,
    use_default_image_token_count: bool = False,
) -> bool:
    """Whether ``messages`` plus ``tools`` hold at least ``threshold`` prompt tokens for ``model``.

    Same arithmetic as ``token_counter(messages=..., tools=...) >= threshold``, counted one message
    at a time and stopped at the first message that crosses the threshold, so a prompt far above it
    costs the tokenizer a few messages rather than the whole conversation.
    """
    from litellm.utils import convert_list_message_to_dict

    if litellm.disable_token_counter is True:
        return threshold <= 0
    new_messages: Final = cast(  # cast-ok: convert_list_message_to_dict is untyped, same as token_counter
        list[AllMessageValues], convert_list_message_to_dict(messages)
    )
    params: Final = _MessageCountParams(model, None)
    includes_system_message: Final = any(message.get("role", None) == "system" for message in new_messages)
    per_message_counts: Final = (
        _count_messages(params, [message], use_default_image_token_count, None) for message in new_messages
    )
    running_totals: Final = accumulate(
        per_message_counts,
        initial=_count_extra(params.count_function, tools, None, includes_system_message, params.tool_section),
    )
    return any(total >= threshold for total in running_totals)


def _count_function_call_tokens(
    key: str,
    value: object,
    message: Mapping[str, object],
    count_function: TokenCounterFunction,
) -> int:
    """
    Count tokens contributed by an assistant message's tool/function call payload.

    Handles both the modern `tool_calls` list and the legacy OpenAI
    `function_call` dict. Only the `arguments` string is counted (matching the
    existing tool_calls behavior); names are accounted for elsewhere via the
    tool/function definitions and `tool_choice`.
    """
    if key == "tool_calls":
        if not isinstance(value, list):
            raise ValueError(f"Unsupported type {type(value)} for key tool_calls in message {message}")
        total = 0
        for tool_call in value:
            if "function" not in tool_call:
                raise ValueError(f"Unsupported tool call {tool_call} must contain a function key")
            function_arguments = tool_call["function"].get("arguments", "")
            total += count_function(str(function_arguments))
        return total
    if key == "function_call":
        if not isinstance(value, Mapping):
            raise ValueError(f"Unsupported type {type(value)} for key function_call in message {message}")
        return count_function(str(value.get("arguments", "")))
    raise ValueError(f"Unexpected key {key!r}; expected 'tool_calls' or 'function_call'")


def _count_messages(
    params: _MessageCountParams,
    messages: list[AllMessageValues],
    use_default_image_token_count: bool,
    default_token_count: int | None,
) -> int:
    """
    Count the number of tokens in a list of messages.

    Args:
        params (_MessageCountParams): The parameters for counting tokens.
        messages (List[AllMessageValues]): The list of messages to count tokens in.
        use_default_image_token_count (bool): When True, will NOT make a GET request to the image URL and instead return the default image dimensions.
        default_token_count (Optional[int]): The default number of tokens to return for a message block, if an error occurs.
    """
    num_tokens = 0
    if len(messages) == 0:
        return num_tokens
    for message in messages:
        num_tokens += params.tokens_per_message
        for key, value in message.items():
            if value is None:
                pass
            elif key in ("tool_calls", "function_call"):
                num_tokens += _count_function_call_tokens(key, value, message, params.count_function)
            elif isinstance(value, str):
                num_tokens += params.count_function(value)
                if key == "name":
                    num_tokens += params.tokens_per_name
            elif key == "content" and isinstance(value, list):
                num_tokens += _count_content_list(
                    params.count_function,
                    value,
                    use_default_image_token_count,
                    default_token_count,
                )
            elif key == "search_results" and isinstance(value, list):
                from litellm.litellm_core_utils.prompt_templates.common_utils import (
                    extract_search_results_text,
                )

                search_results_text = extract_search_results_text(value)
                if search_results_text:
                    num_tokens += params.count_function(search_results_text)
            else:
                # Skip unsupported keys instead of raising an error
                continue
    return num_tokens


def _count_extra(
    count_function: TokenCounterFunction,
    tools: list[ChatCompletionToolParam] | None,
    tool_choice: str | Mapping[str, object] | None,
    includes_system_message: bool,
    tool_section: ChatTemplateToolSection | None,
) -> int:
    """Count extra tokens for function definitions and tool choices.
    Args:
        count_function (TokenCounterFunction): The function to count tokens.
        tools (Optional[List[ChatCompletionToolParam]]): The available tools.
        tool_choice (Optional[ChatCompletionNamedToolChoiceParam]): The tool choice.
        includes_system_message (bool): Whether the messages include a system message.
        tool_section (Optional[ChatTemplateToolSection]): How the model's chat template renders tools, if known.
    """
    reply_primer_tokens: Final = 3
    tool_choice_tokens: Final = 0 if tool_section is not None else _count_tool_choice(count_function, tool_choice)
    return (
        reply_primer_tokens
        + _count_tool_definitions(count_function, tools, includes_system_message, tool_section)
        + tool_choice_tokens
    )


def _count_tool_definitions(
    count_function: TokenCounterFunction,
    tools: Sequence[object] | None,
    includes_system_message: bool,
    tool_section: ChatTemplateToolSection | None,
) -> int:
    if not tools:
        return 0
    if tool_section is not None:
        return _count_chat_template_tools(count_function, tools, includes_system_message, tool_section)
    system_message_discount: Final = 4 if includes_system_message else 0
    return count_function(_format_function_definitions(tools)) + 9 - system_message_discount


def _count_tool_choice(
    count_function: TokenCounterFunction,
    tool_choice: str | Mapping[str, object] | None,
) -> int:
    match tool_choice:
        case "none":
            return 1
        case {"function": {"name": str() as name}}:
            return 7 + count_function(name)
        case {"type": "none"}:
            return 1
        case {"name": str() as name}:
            return 7 + count_function(name)
        case _:
            return 0


def _count_chat_template_tools(
    count_function: TokenCounterFunction,
    tools: Sequence[object],
    includes_system_message: bool,
    tool_section: ChatTemplateToolSection,
) -> int:
    functions: Final = tuple(
        function for function in _function_definitions(_tools_sent_as_functions(tools)) if function.get("name")
    )
    if not functions:
        return 0
    system_turn_tokens: Final = 0 if includes_system_message else tool_section.implicit_system_turn_tokens
    schemas: Final = tool_section.render_tool_schemas(functions)
    return tool_section.preamble_tokens + system_turn_tokens + count_function(schemas)


def _get_extrapolating_count_function(
    count_exactly: TokenCounterFunction,
    max_exact_chars: int = TOKEN_COUNTER_MAX_EXACT_CHARS,
) -> TokenCounterFunction:
    def count_tokens(text: str) -> int:
        if len(text) <= max_exact_chars:
            return count_exactly(text)
        samples: Final = _evenly_spaced_samples(text, max_exact_chars)
        sampled_chars: Final = sum(len(sample) for sample in samples)
        return round(sum(count_exactly(sample) for sample in samples) * len(text) / sampled_chars)

    return count_tokens


def _evenly_spaced_samples(text: str, total_chars: int) -> tuple[str, ...]:
    sample_count: Final = min(EXTRAPOLATION_SAMPLES, total_chars)
    sample_chars: Final = total_chars // sample_count
    last_start: Final = len(text) - sample_chars
    return tuple(
        text[start : start + sample_chars]
        for start in (last_start * index // max(sample_count - 1, 1) for index in range(sample_count))
    )


def _get_count_function(
    model: str | None,
    custom_tokenizer: dict | SelectTokenizerResponse | None = None,
) -> TokenCounterFunction:
    return _get_extrapolating_count_function(_get_exact_count_function(model, custom_tokenizer))


def _get_exact_count_function(
    model: str | None,
    custom_tokenizer: dict | SelectTokenizerResponse | None = None,
) -> TokenCounterFunction:
    """
    Get the function to count tokens based on the model and custom tokenizer."""
    from litellm.utils import select_tokenizer

    if model is not None or custom_tokenizer is not None:
        tokenizer_json: Final = custom_tokenizer or select_tokenizer(model)
    else:
        default_encoding: Final = get_default_encoding()

        def encode_length(text: str) -> int:
            return _encoding_count(default_encoding, text)

        return _get_tiktoken_count_function(encode_length)
    if tokenizer_json["type"] == "huggingface_tokenizer":
        tokenizer: Final[HuggingFace] = tokenizer_json["tokenizer"]

        def count_tokens(text: str) -> int:
            if isinstance(tokenizer, HuggingFaceTokenizer):
                return tokenizer.count(text)
            return len(tokenizer.encode_batch_fast([text])[0])

        return count_tokens
    if tokenizer_json["type"] == "openai_tokenizer":
        encoding: Final = openai_tokenizer_encoding(model)

        def encode_length(text: str) -> int:
            return _encoding_count(encoding, text)

        return _get_tiktoken_count_function(encode_length)
    raise ValueError("Unsupported tokenizer type")


def _encoding_count(encoding: "Encoding", text: str) -> int:
    if isinstance(encoding, OpenAIEncoding):
        return encoding.count(text)
    return len(encoding.encode(text, disallowed_special=()))


def openai_tokenizer_encoding(model: str) -> "Encoding":
    """The encoding `token_counter` uses for a model on the `openai_tokenizer` path."""
    return get_encoding(openai_tokenizer_encoding_name(model))


def openai_tokenizer_encoding_name(model: str) -> str:
    """The tiktoken encoding name for `model`, without loading the encoding.

    A listed OpenAI or Azure model that tiktoken does not know is either a suffixed variant of a model it does
    know (`text-embedding-ada-002-v2`) or a newer release than its table, which gets o200k_base, the encoding every
    OpenAI model since gpt-4o uses. Provider and region prefixes (`azure/eu/`, `azure_ai/`) are dropped, a name
    listed only under Azure (`gpt-41`, `openai/o1-mini`) counts as listed through its `azure/` cost-map key, and an
    unlisted name tiktoken counts with o200k_base (`openai/gpt-5.9`) or named like a gpt-5 or o-series release
    (`openai/gpt-6.2`) gets it."""
    model_to_use: Final = _openai_model_name(model) or _fix_model_name(model)
    if "gpt-4o" in model_to_use:
        return "o200k_base"
    try:
        return tiktoken.encoding_name_for_model(model_to_use)
    except KeyError:
        return _suffixed_variant_encoding_name(model_to_use) or "o200k_base"


def unresolved_tokenizer_model(model: str) -> str:
    """`model`, or `gpt-3.5-turbo` when resolving the OpenAI model `model` names moves it off cl100k_base, the encoding
    tiktoken gives the name as written, so a count with the result never drops below that one."""
    as_written: Final = _fix_model_name(model)
    as_written_encoding: Final = (
        "o200k_base" if "gpt-4o" in as_written else (_tiktoken_encoding_name(as_written) or "cl100k_base")
    )
    moved_off_cl100k: Final = (
        as_written_encoding == "cl100k_base" and openai_tokenizer_encoding_name(model) != "cl100k_base"
    )
    return "gpt-3.5-turbo" if moved_off_cl100k else model


def usage_tokenizer_model(model: str, base_model: str | None) -> str:
    """The model to estimate missing usage with: a deployment's `base_model` when it names an OpenAI model, since a
    custom deployment name says nothing about its tokenizer, else `model`."""
    return base_model if base_model is not None and is_openai_model(base_model) else model


def is_openai_model(model: str) -> bool:
    """Whether `model` names an OpenAI or Azure OpenAI model, listed in the cost map or counted with o200k_base by
    tiktoken, with or without its prefixes."""
    return _openai_model_name(model) is not None


def deployment_openai_model(deployment: object) -> str | None:
    """The OpenAI model a router deployment serves, read from its base model first, since a custom deployment name
    (`azure/my-gpt-5`) says nothing about its tokenizer, or None when it serves no OpenAI model."""
    return _as_openai_model(_served_model(deployment))


def served_openai_models(deployments: Iterable[object]) -> tuple[str | None, ...]:
    """The distinct OpenAI models `deployments` serve, with None standing for any deployment serving another model."""
    served: Final = dict.fromkeys(_served_model(deployment) for deployment in deployments)
    return tuple(dict.fromkeys(_as_openai_model(model) for model in served))


def _served_model(deployment: object) -> str | None:
    return next(iter(deployment_pricing_models(deployment)), None)


def _as_openai_model(model: str | None) -> str | None:
    return model if model is not None and is_openai_model(model) else None


def single_encoding_openai_model(served: Sequence[str | None]) -> str | None:
    """An OpenAI model to count a request with, when every `served_openai_models` entry is an OpenAI model on the same
    encoding, else None."""
    if not served or None in served:
        return None
    encodings: Final = frozenset(openai_tokenizer_encoding_name(model) for model in served if model is not None)
    return served[0] if len(encodings) == 1 else None


def _openai_model_name(model: str) -> str | None:
    return _listed_openai_model_name(model) or _tiktoken_o200k_model_name(model) or _versioned_openai_model_name(model)


_VERSIONED_OPENAI_MODEL: Final = re.compile(r"(?:gpt-(?:[5-9]|[1-9]\d+)|o[1-9]\d*)(?:[.-]|$)")


def _versioned_openai_model_name(model: str) -> str | None:
    name: Final = model.rsplit("/", 1)[-1]
    return name if _VERSIONED_OPENAI_MODEL.match(name) else None


def _tiktoken_o200k_model_name(model: str) -> str | None:
    name: Final = model.rsplit("/", 1)[-1]
    return name if _tiktoken_encoding_name(name) == "o200k_base" else None


def _listed_openai_model_name(model: str) -> str | None:
    segments: Final = model.split("/")
    suffixes: Final = ("/".join(segments[start:]) for start in range(len(segments)))
    listed: Final = any(
        _is_openai_cost_map_model(suffix) or _is_openai_cost_map_model(f"azure/{suffix}") for suffix in suffixes
    )
    return segments[-1] if listed else None


def _is_openai_cost_map_model(model: str) -> bool:
    return (
        model in litellm.azure_llms
        or model in litellm.open_ai_chat_completion_models
        or _is_azure_openai_text_model(model)
    )


_TEXT_MODEL_MODES: Final = frozenset({"chat", "completion", "responses", "realtime"})


def _is_azure_openai_text_model(model: str) -> bool:
    cost_info: Final[Mapping[str, object]] = litellm.model_cost.get(model) or {}
    return (
        cost_info.get("litellm_provider") == "azure"
        and cost_info.get("mode") in _TEXT_MODEL_MODES
        and _is_openai_model_name(model.rsplit("/", 1)[-1])
    )


def _is_openai_model_name(name: str) -> bool:
    return (
        name in litellm.open_ai_chat_completion_models
        or _tiktoken_encoding_name(name) is not None
        or any(
            name.startswith(f"{listed}-") or listed.startswith(f"{name}-")
            for listed in tuple(litellm.open_ai_chat_completion_models)
        )
    )


def _tiktoken_encoding_name(model: str) -> str | None:
    try:
        return tiktoken.encoding_name_for_model(model)
    except KeyError:
        return None


def _suffixed_variant_encoding_name(model: str) -> str | None:
    known: Final = tuple(name for name in MODEL_TO_ENCODING if model.startswith(f"{name}-"))
    return MODEL_TO_ENCODING[max(known, key=len)] if known else None


def uses_legacy_message_accounting(model: str) -> bool:
    """Whether `token_counter` prices messages with the `gpt-3.5-turbo-0301` constants (4 per message, -1 per name)."""
    return _fix_model_name(model) == "gpt-3.5-turbo-0301"


def _fix_model_name(model: str) -> str:
    """We normalize some model names to others"""
    if model in litellm.azure_llms:
        # azure llms use gpt-35-turbo instead of gpt-3.5-turbo 🙃
        return model.replace("-35", "-3.5")
    elif model in litellm.open_ai_chat_completion_models:
        return model
    else:
        return "gpt-3.5-turbo"


def _count_image_tokens(
    image_url: object,
    use_default_image_token_count: bool,
) -> int:
    """
    Count tokens for an image_url content block.

    Args:
        image_url: The image URL data - can be a string URL or dict with 'url' and 'detail'
        use_default_image_token_count: Whether to use default image token counts

    Returns:
        int: Number of tokens for the image

    Raises:
        ValueError: If image_url is invalid type or detail value is invalid
    """
    if isinstance(image_url, dict):
        detail: Final = image_url.get("detail", "auto")
        if detail not in ["low", "high", "auto"]:
            raise ValueError(f"Invalid detail value: {detail}. Expected 'low', 'high', or 'auto'.")
        url: Final = image_url.get("url")
        if not url:
            raise ValueError("Missing required key 'url' in image_url dict.")
        return calculate_img_tokens(
            data=url,
            mode=detail,
            use_default_image_token_count=use_default_image_token_count,
        )
    elif isinstance(image_url, str):
        if not image_url.strip():
            raise ValueError("Empty image_url string is not valid.")
        return calculate_img_tokens(
            data=image_url,
            mode="auto",
            use_default_image_token_count=use_default_image_token_count,
        )
    else:
        raise ValueError(f"Invalid image_url type: {type(image_url).__name__}. Expected str or dict with 'url' field.")


def _validate_anthropic_content(content: Mapping[str, object]) -> type:
    """
    Validate and determine which Anthropic TypedDict applies.

    Returns the corresponding TypedDict class if recognized, otherwise raises.
    """
    content_type: Final = content.get("type")
    if not content_type:
        raise ValueError("Anthropic content missing required field: 'type'")

    mapping: Final = {
        "tool_use": AnthropicMessagesToolUseParam,
        "tool_result": AnthropicMessagesToolResultParam,
        "search_result": AnthropicSearchResultParam,
    }

    expected_cls: Final = mapping.get(content_type) if isinstance(content_type, str) else None
    if expected_cls is None:
        raise ValueError(f"Unknown Anthropic content type: '{content_type}'")

    missing: Final = [k for k in getattr(expected_cls, "__required_keys__", set()) if k not in content]
    if missing:
        raise ValueError(f"Missing required fields in {content_type} block: {', '.join(missing)}")

    return expected_cls


def _anthropic_image_source_data(
    source: AnthropicContentParamSource | AnthropicContentParamSourceUrl | AnthropicContentParamSourceFileId,
) -> str:
    if source["type"] == "base64":
        data: Final = source.get("data")
        if not data:
            return ""
        media_type: Final = source.get("media_type") or "image/png"
        return f"data:{media_type};base64,{data}"
    if source["type"] == "url":
        return source.get("url") or ""
    return ""


def _anthropic_rendered_page_image_tokens(width: float, height: float) -> int:
    """A PDF page is rasterized within Anthropic's image limits, then billed by its pixel area."""
    if width <= 0 or height <= 0:
        return 0
    area_at_max_edge: Final = ANTHROPIC_IMAGE_MAX_LONG_EDGE_PX**2 * min(width, height) / max(width, height)
    return math.ceil(min(float(ANTHROPIC_IMAGE_MAX_PIXELS), area_at_max_edge) / ANTHROPIC_IMAGE_PIXELS_PER_TOKEN)


def _count_inline_pdf_tokens(data_url: str, count_function: TokenCounterFunction) -> int | None:
    if not data_url.startswith(PDF_DATA_URL_PREFIX):
        return None
    try:
        from pypdf import PdfReader
    except ImportError:
        verbose_logger.debug("pypdf is not installed, so the PDF document is priced like one image")
        return None
    try:
        reader: Final = PdfReader(io.BytesIO(base64.b64decode(data_url[len(PDF_DATA_URL_PREFIX) :])))
        return sum(
            count_function(page.extract_text() or "")
            + _anthropic_rendered_page_image_tokens(float(page.mediabox.width), float(page.mediabox.height))
            for page in reader.pages
        )
    except Exception as e:
        verbose_logger.debug("Could not read the PDF document's pages (%s), so it is priced like one image", e)
        return None


def _count_opaque_document_tokens(
    data_url: str,
    count_function: TokenCounterFunction,
    use_default_image_token_count: bool,
) -> int:
    pdf_tokens: Final = _count_inline_pdf_tokens(data_url, count_function)
    if pdf_tokens is not None:
        return pdf_tokens
    return calculate_img_tokens(
        data=data_url,
        mode="auto",
        use_default_image_token_count=use_default_image_token_count,
    )


def _count_document_tokens(
    document: ChatCompletionDocumentObject | AnthropicMessagesDocumentParam,
    count_function: TokenCounterFunction,
    use_default_image_token_count: bool,
    default_token_count: int | None,
) -> int:
    source: Final = document["source"]
    metadata_tokens: Final = sum(
        count_function(text) for text in (document.get("title"), document.get("context")) if text
    )
    if source["type"] == "text":
        return metadata_tokens + count_function(source["data"])
    if source["type"] == "content":
        content: Final = source["content"]
        if isinstance(content, str):
            return metadata_tokens + count_function(content)
        return metadata_tokens + _count_content_list(
            count_function, content, use_default_image_token_count, default_token_count
        )
    return metadata_tokens + _count_opaque_document_tokens(
        _anthropic_image_source_data(source), count_function, use_default_image_token_count
    )


def _count_file_tokens(
    file_value: object,
    count_function: TokenCounterFunction,
    use_default_image_token_count: bool,
) -> int:
    """An OpenAI `file` block is the chat-completions spelling of a document, so it prices like one."""
    if not isinstance(file_value, Mapping):
        return 0
    filename: Final = file_value.get("filename")
    file_data: Final = file_value.get("file_data")
    name_tokens: Final = count_function(filename) if isinstance(filename, str) and filename else 0
    if not isinstance(file_data, str) or not file_data:
        return name_tokens
    return name_tokens + _count_opaque_document_tokens(file_data, count_function, use_default_image_token_count)


def _count_anthropic_content(
    content: Mapping[str, object],
    count_function: TokenCounterFunction,
    use_default_image_token_count: bool,
    default_token_count: int | None,
) -> int:
    """
    Count tokens in Anthropic-specific content blocks (tool_use, tool_result, etc.).

    Uses TypedDict definitions from litellm.types.llms.anthropic to determine
    what fields to count and how to handle nested structures.

    Dynamically infers which fields to count based on the TypedDict definition,
    avoiding hardcoded field names.
    """
    typeddict_cls: Final = _validate_anthropic_content(content)
    type_hints: Final[Mapping[str, object]] = getattr(typeddict_cls, "__annotations__", {})
    tokens = 0

    # Fields to skip (metadata/identifiers that don't contribute to prompt tokens)
    skip_fields: Final = {"type", "id", "tool_use_id", "cache_control", "is_error"}

    # Iterate over all fields defined in the TypedDict
    for field_name, field_type in type_hints.items():
        if field_name in skip_fields:
            continue

        field_value = content.get(field_name)
        if field_value is None:
            continue
        try:
            if isinstance(field_value, str):
                tokens += count_function(field_value)
            elif isinstance(field_value, list):
                tokens += _count_content_list(
                    count_function,
                    field_value,
                    use_default_image_token_count,
                    default_token_count,
                )
            elif isinstance(field_value, dict):
                tokens += count_function(str(field_value))
        except Exception as e:
            if default_token_count is not None:
                return default_token_count
            raise (UnsupportedContentTypeError if isinstance(e, UnsupportedContentTypeError) else ValueError)(
                f"Error counting field '{field_name}': {e}"
            )
    return tokens


def _count_content_list(
    count_function: TokenCounterFunction,
    content_list: str
    | Iterable[
        OpenAIMessageContentListBlock
        | ChatCompletionThinkingBlock
        | ChatCompletionRedactedThinkingBlock
        | AnthropicMessagesTextParam
        | AnthropicMessagesImageParam
        | AnthropicMessagesDocumentParam
        | AnthropicSearchResultParam
    ],
    use_default_image_token_count: bool,
    default_token_count: int | None,
) -> int:
    """Recursively count tokens from a list of content blocks."""
    try:
        num_tokens = 0
        for c in content_list:
            if isinstance(c, str):
                num_tokens += count_function(c)
            elif c["type"] == "text":
                num_tokens += count_function(str(c.get("text", "")))
            elif c["type"] == "image_url":
                image_url = c.get("image_url")
                num_tokens += _count_image_tokens(image_url, use_default_image_token_count)
            elif c["type"] == "image":
                num_tokens += calculate_img_tokens(
                    data=_anthropic_image_source_data(c["source"]),
                    mode="auto",
                    use_default_image_token_count=use_default_image_token_count,
                )
            elif c["type"] == "document":
                num_tokens += _count_document_tokens(
                    c,
                    count_function,
                    use_default_image_token_count,
                    default_token_count,
                )
            elif c["type"] == "file":
                num_tokens += _count_file_tokens(
                    c.get("file"),
                    count_function,
                    use_default_image_token_count,
                )
            elif c["type"] in ("tool_use", "tool_result", "search_result"):
                num_tokens += _count_anthropic_content(
                    c,
                    count_function,
                    use_default_image_token_count,
                    default_token_count,
                )
            elif c["type"] in ("thinking", "redacted_thinking"):
                # Claude extended thinking content block
                # Count the thinking text and skip the opaque blobs (signature, redacted data)
                thinking_text = str(c.get("thinking", ""))
                if thinking_text:
                    num_tokens += count_function(thinking_text)
            elif c["type"] == "tool_reference":
                # Anthropic tool-search reference block: a lightweight pointer to
                # a deferred tool, e.g. {"type": "tool_reference", "tool_name": ...}.
                # The full tool definition is counted via the `tools` param, so we
                # only count the referenced name here. Without this branch,
                # token_counter raises on tool-search traffic; on the streaming
                # anthropic_messages path that nulls response_cost and causes the
                # proxy to drop the SpendLogs row entirely (silent cost undercount).
                tool_name = str(c.get("tool_name") or "")
                if tool_name:
                    num_tokens += count_function(tool_name)
            else:
                content_type = c.get("type", type(c).__name__) if isinstance(c, dict) else type(c).__name__
                raise UnsupportedContentTypeError(
                    f"Invalid content item type: {content_type}. "
                    f"Expected str or dict with 'type' field "
                    f"(text, image_url, image, document, file, tool_use, tool_result, thinking, redacted_thinking, "
                    f"tool_reference, search_result)."
                )
        return num_tokens
    except Exception as e:
        if default_token_count is not None:
            return default_token_count
        error_type: Final = UnsupportedContentTypeError if isinstance(e, UnsupportedContentTypeError) else ValueError
        raise error_type(
            f"Error getting number of tokens from content list: {e}, default_token_count={default_token_count}"
        )


def _format_function_definitions(tools: Sequence[object]) -> str:
    """Formats tool definitions in the format that OpenAI appears to use.
    Based on https://github.com/forestwanglin/openai-java/blob/main/jtokkit/src/main/java/xyz/felh/openai/jtokkit/utils/TikTokenUtils.java
    """
    lines: Final = []
    lines.append("namespace functions {")
    lines.append("")
    for function in _function_definitions(tools):
        lines.extend(_format_single_function_definition(function))
    lines.append("} // namespace functions")
    return "\n".join(lines)


def _tools_sent_as_functions(tools: Sequence[object]) -> tuple[Mapping[str, object], ...]:
    return tuple(tool for tool in _tool_mappings(tools) if not _is_anthropic_web_search_tool(tool))


def _is_anthropic_web_search_tool(tool: Mapping[str, object]) -> bool:
    return tool.get("name") == "web_search" and "input_schema" not in tool and tool.get("type") != "function"


def _tool_mappings(tools: Sequence[object]) -> Iterable[Mapping[str, object]]:
    for tool in tools:
        if isinstance(tool, Mapping):
            yield cast(Mapping[str, object], tool)


def _function_definitions(tools: Sequence[object]) -> Iterable[Mapping[str, object]]:
    for tool in _tool_mappings(tools):
        yield from _function_definitions_for_tool(tool)


def _function_definitions_for_tool(tool: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    function: Final = tool.get("function")
    if isinstance(function, Mapping):
        yield function
        return
    declarations: Final = tool.get("function_declarations") or tool.get("functionDeclarations")
    if isinstance(declarations, list):
        for declaration in declarations:
            if isinstance(declaration, Mapping):
                yield declaration
        return
    input_schema: Final = tool.get("input_schema")
    parameters: Final = input_schema or tool.get("parameters") or {}
    normalized_parameters: Final = parameters if isinstance(parameters, Mapping) else {}
    unmapped_anthropic_keys: Final = (
        {key: value for key, value in tool.items() if key not in _ANTHROPIC_TOOL_KEYS_MAPPED_BY_BRIDGE}
        if input_schema is not None
        else {}
    )
    description: Final = tool.get("description")
    strict: Final = tool.get("strict")
    yield {
        "name": tool.get("name"),
        "parameters": {**normalized_parameters, **unmapped_anthropic_keys},
        **({"description": description} if description is not None else {}),
        **({"strict": bool(strict)} if strict is not None else {}),
    }


def _format_single_function_definition(function: Mapping[str, object]) -> tuple[str, ...]:
    function_name: Final = function.get("name")
    if not function_name:
        return ()
    function_description: Final = function.get("description")
    parameters_value: Final = function.get("parameters") or {}
    parameters: Final = parameters_value if isinstance(parameters_value, Mapping) else {}
    properties: Final = parameters.get("properties")
    if isinstance(properties, Mapping) and properties:
        return (
            *((f"// {function_description}",) if function_description else ()),
            f"type {function_name} = (_: {{",
            _format_object_parameters(parameters, 0),
            "}) => any;",
            "",
        )
    return (
        *((f"// {function_description}",) if function_description else ()),
        f"type {function_name} = () => any;",
        "",
    )


def _format_object_parameters(parameters, indent):
    properties: Final = parameters.get("properties")
    if not properties:
        return ""
    required_params: Final = parameters.get("required", [])
    lines: Final = []
    for key, props in properties.items():
        description = props.get("description")
        if description:
            lines.append(f"// {description}")
        question = "?"
        if required_params and key in required_params:
            question = ""
        lines.append(f"{key}{question}: {_format_type(props, indent)},")
    return "\n".join([" " * max(0, indent) + line for line in lines])


def _format_type(props, indent):
    type: Final = props.get("type")
    if type == "string":
        if "enum" in props:
            return " | ".join([f'"{item}"' for item in props["enum"]])
        return "string"
    elif type == "array":
        # items is required, OpenAI throws an error if it's missing
        return f"{_format_type(props['items'], indent)}[]"
    elif type == "object":
        return f"{{\n{_format_object_parameters(props, indent + 2)}\n}}"
    elif type in ["integer", "number"]:
        if "enum" in props:
            return " | ".join([f'"{item}"' for item in props["enum"]])
        return "number"
    elif type == "boolean":
        return "boolean"
    elif type == "null":
        return "null"
    else:
        # This is a guess, as an empty string doesn't yield the expected token count
        return "any"
