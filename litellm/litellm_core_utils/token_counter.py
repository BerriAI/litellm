# What is this?
## Helper utilities for token counting
import base64
import importlib
import io
import math
import re
import struct
from bisect import bisect_right
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import accumulate, chain, groupby, islice, takewhile, tee
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias, cast, runtime_checkable

import anyio
import anyio.lowlevel
import httpx
import tiktoken
from pydantic import TypeAdapter
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
    OPENAI_CHAT_MODES,
    OPENAI_MODEL_NAME_MAX_DASHES,
    OPENAI_PDF_TEXT_PAGE_FRAMING_TOKENS,
    PDF_DATA_URL_PREFIX,
    PDF_MAX_FORM_XOBJECT_DEPTH,
    PDF_MAX_XOBJECT_DRAWS_PER_PAGE,
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
from litellm.litellm_core_utils.deployment_pricing import deployment_pricing_models
from litellm.litellm_core_utils.url_utils import safe_get
from litellm.llms.custom_httpx.http_handler import get_httpx_client, header_value
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
    CompactionBlock,
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


_MEDIA_FILE_CHARS_PER_TOKEN: Final = 4
_EmbeddingInputT: Final = TypeVar("_EmbeddingInputT")


def _max_input_tokens(model: str) -> int | None:
    try:
        limit: Final = litellm.get_model_info(model).get("max_input_tokens")
    except Exception:  # noqa: BLE001  # get_model_info raises a bare Exception for an unmapped model
        return None
    return limit if isinstance(limit, int) and limit > 0 else None


def _media_file_tokens(data: str, models: Sequence[str]) -> int:
    text_estimate: Final = len(data) // _MEDIA_FILE_CHARS_PER_TOKEN
    ceilings: Final = tuple(map(_max_input_tokens, models))
    known_ceilings: Final = tuple(ceiling for ceiling in ceilings if ceiling is not None)
    if not known_ceilings or len(known_ceilings) != len(ceilings):
        return text_estimate
    return min(text_estimate, max(known_ceilings))


def _block_str_field(block: Mapping[str, object], container: str, field: str) -> str | None:
    inner: Final = as_str_mapping(block.get(container))
    value: Final = inner.get(field) if inner is not None else None
    return value if isinstance(value, str) else None


def estimate_media_block_tokens(block: Mapping[str, object], models: Sequence[str], fallback_model: str | None) -> int:
    from litellm.litellm_core_utils.audio_utils.utils import estimate_audio_tokens

    match block.get("type"):
        case "input_audio":
            return estimate_audio_tokens(
                _block_str_field(block, "input_audio", "data"), models, fallback_model=fallback_model
            )
        case "file":
            return _media_file_tokens(_block_str_field(block, "file", "file_data") or "", models)
        case _:
            detail: Final = _block_str_field(block, "image_url", "detail")
            return estimate_image_tokens(
                models,
                _block_str_field(block, "image_url", "url") or "",
                "auto" if detail is None else detail,
                fallback_model=fallback_model,
            )


_PER_ITEM_DATA_URI_EMBEDDING_PROVIDERS: Final = frozenset({"vertex_ai", "gemini", "azure_ai"})

_DataUriMediaScope: TypeAlias = Literal["per_item", "whole_batch"]


def _is_pdf_block(block: Mapping[str, object]) -> bool:
    return (_block_str_field(block, "file", "file_data") or "").startswith("data:application/pdf")


def _bedrock_data_uri_media_scope(
    resolved_model: str, media: Sequence[Mapping[str, object]]
) -> _DataUriMediaScope | None:
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
    from litellm.llms.bedrock.common_utils import BedrockModelInfo
    from litellm.utils import supports_embedding_image_input

    base_model: Final = BedrockModelInfo.get_base_model(resolved_model.removeprefix("async_invoke/"))
    if not supports_embedding_image_input(f"bedrock/{base_model}"):
        return None
    match BaseAWSLLM.get_bedrock_embedding_provider(base_model):
        case "cohere":
            return "whole_batch"
        case "twelvelabs":
            return None
        case "nova" if any(map(_is_pdf_block, media)):
            return None
        case _:
            return "per_item"


def _data_uri_media_scope(model: str, media: Sequence[Mapping[str, object]]) -> _DataUriMediaScope | None:
    from litellm.utils import supports_embedding_image_input

    try:
        resolved_model, provider, _, _ = litellm.get_llm_provider(model=model)
    except litellm.exceptions.BadRequestError:
        return None
    if provider == "bedrock":
        return _bedrock_data_uri_media_scope(resolved_model, media)
    if not supports_embedding_image_input(model):
        return None
    if provider == "cohere":
        return "whole_batch"
    return "per_item" if provider in _PER_ITEM_DATA_URI_EMBEDDING_PROVIDERS else None


def split_embedding_input_tokens(
    value: _EmbeddingInputT, media_models: Callable[[], Sequence[str]], fallback_model: str | None = None
) -> tuple[_EmbeddingInputT, int]:
    from litellm.litellm_core_utils.prompt_templates.tpm_media_utils import (
        embedding_input_is_all_media,
        split_embedding_media_for_tpm,
    )

    text, media = split_embedding_media_for_tpm(value)
    if not media:
        return value, 0
    models: Final = tuple(media_models())
    scopes: Final = frozenset(_data_uri_media_scope(model, media) for model in models)
    if not models or None in scopes or ("whole_batch" in scopes and not embedding_input_is_all_media(value)):
        return value, 0
    media_tokens: Final = sum(estimate_media_block_tokens(block, models, fallback_model) for block in media)
    return cast(_EmbeddingInputT, text), media_tokens  # cast-ok: the projection keeps the input's container shape


def embedding_aware_token_count(
    messages: Sequence[Mapping[str, str]] | None,
    text: str | list[str] | None,  # mutable-ok: litellm.token_counter only takes text as list[str]
    media_models: Callable[[], Sequence[str]],
) -> int:
    embedding_text, media_tokens = split_embedding_input_tokens(text, media_models)
    return litellm.token_counter(messages=messages, text=embedding_text) + media_tokens


TokenCounterFunction = Callable[[str], int]
"""
Type for a function that counts tokens in a string.
"""

PdfPagePricing: TypeAlias = Literal["openai", "claude"]


class _PdfBox(Protocol):
    @property
    def width(self) -> float: ...

    @property
    def height(self) -> float: ...


class _PdfContent(Protocol):
    def get_data(self) -> bytes: ...


class _PdfPage(Protocol):
    @property
    def mediabox(self) -> _PdfBox: ...

    def extract_text(self) -> str: ...

    def get_contents(self) -> _PdfContent | None: ...

    def get(self, key: str) -> object | None: ...

    def items(self) -> Iterable[tuple[str, object]]: ...


class _PdfReader(Protocol):
    @property
    def pages(self) -> Sequence[_PdfPage]: ...


EXTRAPOLATION_SAMPLES: Final = 16
PDF_PAGE_MAX_EXACT_CONTENT_BYTES: Final = 64 * 1024
PDF_WHITESPACE_CLASS: Final = b"[\x00\t\n\x0c\r ]"
PDF_HEX_STRING_OPERAND: Final = rb"<[0-9A-Fa-f" + PDF_WHITESPACE_CLASS[1:-1] + rb"]*>"
PDF_LITERAL_STRING_OPERAND: Final = rb"\((?:\\[\x00-\xff]|[^\\()]|\((?:\\[\x00-\xff]|[^\\()])*\)?)*\)?"
PDF_STRING_OPERAND: Final = re.compile(PDF_LITERAL_STRING_OPERAND + rb"|" + PDF_HEX_STRING_OPERAND + rb"|%[^\r\n]*")
PDF_SIMPLE_LITERAL_STRING: Final = re.compile(rb"\([^()]*\)")
PDF_LITERAL_ESCAPE: Final = re.compile(rb"\\([nrtbf()\\]|[0-7]{1,3}|\r\n?|\n|[\x00-\xff])")
PDF_INLINE_IMAGE: Final = re.compile(
    PDF_LITERAL_STRING_OPERAND
    + rb"|"
    + PDF_HEX_STRING_OPERAND
    + rb"|%[^\r\n]*|(?P<inline_image>(?<![^"
    + PDF_WHITESPACE_CLASS[1:-1]
    + rb"\]>)])BI(?="
    + PDF_WHITESPACE_CLASS
    + rb"|[/\[\]()<>{}%])[\x00-\xff]*?(?:(?<="
    + PDF_WHITESPACE_CLASS
    + rb")ID"
    + PDF_WHITESPACE_CLASS
    + rb"[\x00-\xff]*?(?:(?P<image_end>"
    + PDF_WHITESPACE_CLASS
    + rb"EI(?![^"
    + PDF_WHITESPACE_CLASS[1:-1]
    + rb"\[/<(%]))|\Z)|\Z))"
)
PDF_TEXT_OBJECT: Final = re.compile(
    PDF_LITERAL_STRING_OPERAND
    + rb"|"
    + PDF_HEX_STRING_OPERAND
    + rb"|%[^\r\n]*|(?P<text_object>(?<![^"
    + PDF_WHITESPACE_CLASS[1:-1]
    + rb"\]>)])BT(?="
    + PDF_WHITESPACE_CLASS
    + rb"|[/\[\]()<>{}%])"
    + rb"(?:(?:"
    + PDF_LITERAL_STRING_OPERAND
    + rb"|"
    + PDF_HEX_STRING_OPERAND
    + rb"|%[^\r\n]*|[\x00-\xff])*?(?:(?P<text_object_end>(?<="
    + PDF_WHITESPACE_CLASS
    + rb")ET(?![^"
    + PDF_WHITESPACE_CLASS[1:-1]
    + rb"\[\]()<>{}/%]))|\Z)))"
)
PDF_XOBJECT_INVOCATION: Final = re.compile(
    PDF_LITERAL_STRING_OPERAND
    + rb"|%[^\r\n]*(?:[\r\n]|\Z)|/([^"
    + PDF_WHITESPACE_CLASS[1:-1]
    + rb"/\[\]()<>{}%]+)(?:"
    + PDF_WHITESPACE_CLASS
    + rb"|%[^\r\n]*(?:[\r\n]|\Z))+Do\b"
)
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
        elif _is_openai_chat_model(actual_model):
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        else:
            print_verbose(f"Warning: unknown model {model}. Using default token params.")
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        self.count_function = _get_count_function(model, custom_tokenizer)
        self.tool_section = chat_template_tool_section(model)
        self._model = model

    @cached_property
    def pdf_pricing(self) -> PdfPagePricing:
        return pdf_page_pricing(self._model)


def _is_openai_chat_model(model: str) -> bool:
    return model in litellm.open_ai_chat_completion_models or model in litellm.azure_llms


def _openai_deployment_name(model: str) -> str:
    name: Final = model.rpartition("/")[2]
    return name.split(":")[1] if name.startswith("ft:") else name


def _openai_model_family(model: str) -> str | None:
    words: Final = _openai_deployment_name(model).split("-", OPENAI_MODEL_NAME_MAX_DASHES)
    prefixes: Final = ("-".join(words[:end]) for end in range(len(words), 0, -1))
    listed: Final = next((prefix for prefix in prefixes if _is_openai_chat_model(prefix)), None)
    return None if listed is None or _is_listed_non_chat_model(listed) else listed


def _untyped_cost_map() -> object:
    return litellm.model_cost  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # loaded untyped


def _cost_map_entry(model: str) -> Mapping[str, object]:
    cost_map: Final = _pdf_dictionary(_untyped_cost_map()) or {}
    return _pdf_dictionary(cost_map.get(model)) or {}


def _is_azure_openai_chat_model(model: str) -> bool:
    name: Final = _openai_deployment_name(model)
    entry: Final = _cost_map_entry(f"azure/{name}")
    return (
        entry.get("litellm_provider") == "azure"
        and entry.get("mode") in OPENAI_CHAT_MODES
        and (entry.get("supports_pdf_input") is True or name.startswith("gpt-"))
    )


def _is_openai_or_azure_model(model: str) -> bool:
    if _is_listed_non_chat_model(_openai_deployment_name(model)):
        return False
    return _is_openai_chat_model(model) or _openai_model_family(model) is not None or _is_azure_openai_chat_model(model)


def pdf_page_pricing(model: str | None) -> PdfPagePricing:
    return "openai" if _is_openai_or_azure_model(model or "") else "claude"


def reads_pdf_pages(model: str) -> bool:
    segments: Final = model.split("/")
    names: Final = ("/".join(segments[start:]) for start in range(len(segments)))
    stated: Final = (_cost_map_entry(name).get("supports_pdf_input") for name in names)
    return next((support for support in stated if support is not None), False) is True


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
                    params.pdf_pricing,
                )
            elif key == "provider_specific_fields":
                num_tokens += sum(map(params.count_function, _compaction_summaries(value)))
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


_OBJECT_LIST: Final = TypeAdapter(list[object])


def _compaction_summaries(provider_specific_fields: object) -> tuple[str, ...]:
    fields: Final = as_str_mapping(provider_specific_fields)
    blocks: Final = fields.get("compaction_blocks") if fields is not None else None
    block_fields: Final = (
        tuple(map(as_str_mapping, _OBJECT_LIST.validate_python(blocks))) if isinstance(blocks, list) else ()
    )
    return tuple(
        content
        for block in block_fields
        if block is not None
        and block.get("type") == "compaction"
        and isinstance(content := block.get("content"), str)
        and content
    )


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
    tool_choice_tokens: Final = (
        _count_tool_choice(count_function, tool_choice)
        if tool_section is None or tool_section.counts_tool_choice
        else 0
    )
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


def _named_function_definitions(tools: Sequence[object]) -> Iterable[Mapping[str, object]]:
    return (function for function in _function_definitions(_tools_sent_as_functions(tools)) if function.get("name"))


def sends_function_tools(tools: Sequence[object] | None) -> bool:
    return tools is not None and any(_named_function_definitions(tools))


def _without_false_strict(function: Mapping[str, object]) -> Mapping[str, object]:
    return {key: value for key, value in function.items() if key != "strict" or value}


def _count_chat_template_tools(
    count_function: TokenCounterFunction,
    tools: Sequence[object],
    includes_system_message: bool,
    tool_section: ChatTemplateToolSection,
) -> int:
    named_functions: Final = tuple(_named_function_definitions(tools))
    functions: Final = (
        tuple(map(_without_false_strict, named_functions))
        if tool_section.renders_strict_only_when_true
        else named_functions
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
        encoding: Final = openai_tokenizer_encoding(model or "")

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


def _is_listed_non_chat_model(model: str) -> bool:
    modes: Final = (_cost_map_entry(name).get("mode") for name in (model, f"azure/{model}"))
    return any(mode is not None and mode not in OPENAI_CHAT_MODES for mode in modes)


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
        return _suffixed_variant_encoding_name(model_to_use) or (
            "cl100k_base" if _is_listed_non_chat_model(model_to_use) else "o200k_base"
        )


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
    return (
        _listed_openai_model_name(model)
        or _tiktoken_o200k_model_name(model)
        or _versioned_openai_model_name(model)
        or _azure_openai_chat_model_name(model)
    )


def _azure_openai_chat_model_name(model: str) -> str | None:
    return _openai_deployment_name(model) if _is_azure_openai_chat_model(model) else None


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


class _PdfContentStream(Protocol):
    def get_data(self) -> bytes: ...


@runtime_checkable
class _PdfResourceDictionary(Protocol):
    def get(self, key: str) -> object | None: ...

    def items(self) -> Iterable[tuple[str, object]]: ...


@runtime_checkable
class _PdfFormXObject(_PdfResourceDictionary, Protocol):
    def get_data(self) -> bytes: ...


@runtime_checkable
class _PdfIndirectReference(Protocol):
    def get_object(self) -> object: ...


class _PdfWritablePage(_PdfPage, Protocol):
    def __setitem__(self, key: str, value: object) -> None: ...


class _PdfPageFactory(Protocol):
    @staticmethod
    def create_blank_page(pdf: object | None, width: float, height: float) -> _PdfWritablePage: ...


class _PdfDecodedContentStream(_PdfContentStream, Protocol):
    def set_data(self, data: bytes) -> None: ...


class _PdfGenericModule(Protocol):
    DecodedStreamObject: Callable[[], _PdfDecodedContentStream]
    DictionaryObject: Callable[[Mapping[str, object]], _PdfResourceDictionary]
    NameObject: Callable[[str], str]


class _PdfModule(Protocol):
    PageObject: _PdfPageFactory


@dataclass(frozen=True, slots=True)
class _PdfStreamScan:
    content: bytes
    operand_bytes: int
    invoked_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PdfStreamMetrics:
    content: bytes
    content_bytes: int
    operand_bytes: int
    resources: _PdfResourceDictionary
    forms: tuple["_PdfStreamMetrics", ...]
    extraction_limited: bool = False


@dataclass(frozen=True, slots=True)
class _PdfTextObjectRecord:
    stream: _PdfStreamMetrics
    invocation_path: tuple[int, ...]
    start: int
    end: int
    first_in_stream: bool
    truncated: bool
    closed: bool


_PDF_EMPTY_RESOURCES: Final[_PdfResourceDictionary] = (
    cast(  # cast-ok: immutable empty map satisfies the resource protocol
        _PdfResourceDictionary, MappingProxyType({})
    )
)


def _pdf_resolve_object(value: object) -> object:
    return value.get_object() if isinstance(value, _PdfIndirectReference) else value


def _pdf_decode_name(name: bytes) -> str:
    def decode_escape(match: re.Match[bytes]) -> bytes:
        return bytes((int(match.group(1), 16),))

    return "/" + re.sub(rb"#([0-9A-Fa-f]{2})", decode_escape, name).decode("latin-1")


def _pdf_invocation_name(invocation: re.Match[bytes]) -> str | None:
    name: Final = invocation.group(1)
    return _pdf_decode_name(name) if name is not None else None


def _pdf_inline_image_replacement(match: re.Match[bytes]) -> bytes:
    inline_image: Final = match.group("inline_image")
    image_end: Final = match.group("image_end")
    return b" " if inline_image is not None and image_end is not None else match.group(0)


def _pdf_strip_inline_images(content: bytes) -> bytes:
    return PDF_INLINE_IMAGE.sub(_pdf_inline_image_replacement, content) if b"BI" in content else content


def _pdf_scan_stream(content: bytes) -> _PdfStreamScan:
    return _PdfStreamScan(
        content=content,
        operand_bytes=_pdf_string_operand_bytes(content),
        invoked_names=tuple(
            name
            for invocation in PDF_XOBJECT_INVOCATION.finditer(content)
            if (name := _pdf_invocation_name(invocation)) is not None
        ),
    )


def _pdf_page_content_metrics(page: _PdfPage, page_content: bytes) -> _PdfStreamMetrics:
    form_metrics_memo: Final[dict[tuple[int, int], _PdfStreamMetrics]] = {}  # mutable-ok: page-local walk cache
    stream_scans: Final[dict[int, _PdfStreamScan]] = {}  # mutable-ok: scan each stream identity once per page

    def stream_metrics(
        scan: _PdfStreamScan,
        resources: _PdfResourceDictionary,
        ancestors: frozenset[int],
    ) -> _PdfStreamMetrics:
        xobjects: Final = _pdf_resolve_object(resources.get("/XObject"))
        xobject_resources: Final = xobjects if isinstance(xobjects, _PdfResourceDictionary) else _PDF_EMPTY_RESOURCES
        form_metrics: Final = tuple(
            form_metrics_for_xobject(
                _pdf_resolve_object(xobject_resources.get(name)),
                resources,
                ancestors,
            )
            for name in scan.invoked_names
            if xobject_resources
        )
        return _PdfStreamMetrics(
            content=scan.content,
            content_bytes=len(scan.content) + sum(metrics.content_bytes for metrics in form_metrics),
            operand_bytes=scan.operand_bytes + sum(metrics.operand_bytes for metrics in form_metrics),
            resources=resources,
            forms=form_metrics,
            extraction_limited=any(metrics.extraction_limited for metrics in form_metrics),
        )

    def form_metrics_for_xobject(
        xobject: object,
        inherited_resources: _PdfResourceDictionary,
        ancestors: frozenset[int],
    ) -> _PdfStreamMetrics:
        resolved: Final = _pdf_resolve_object(xobject)
        identity: Final = id(resolved)
        if identity in ancestors or len(ancestors) >= PDF_MAX_FORM_XOBJECT_DEPTH:
            return _PdfStreamMetrics(b"", 0, 0, inherited_resources, (), extraction_limited=True)
        if not isinstance(resolved, _PdfFormXObject) or resolved.get("/Subtype") != "/Form":
            return _PdfStreamMetrics(b"", 0, 0, inherited_resources, ())
        form_resources: Final = _pdf_resolve_object(resolved.get("/Resources"))
        resources: Final = form_resources if isinstance(form_resources, _PdfResourceDictionary) else inherited_resources
        cache_key: Final = (identity, id(resources))
        cached: Final = form_metrics_memo.get(cache_key)
        if cached is not None:
            return cached
        if identity not in stream_scans:
            stream_scans[identity] = _pdf_scan_stream(_pdf_strip_inline_images(resolved.get_data()))
        metrics: Final = stream_metrics(stream_scans[identity], resources, ancestors | {identity})
        form_metrics_memo[cache_key] = metrics
        return metrics

    resources: Final = _pdf_resolve_object(page.get("/Resources"))
    page_resources: Final = resources if isinstance(resources, _PdfResourceDictionary) else _PDF_EMPTY_RESOURCES
    return stream_metrics(_pdf_scan_stream(page_content), page_resources, frozenset())


def _evenly_spaced_page_indices(page_count: int) -> tuple[int, ...]:
    if page_count <= EXTRAPOLATION_SAMPLES:
        return tuple(range(page_count))
    return tuple((page_count - 1) * index // (EXTRAPOLATION_SAMPLES - 1) for index in range(EXTRAPOLATION_SAMPLES))


def _pdf_string_operand_bytes(content: bytes) -> int:
    if b"\\" not in content and b"%" not in content and b"<" not in content and b"BI" not in content:
        content_without_literals, literal_count = PDF_SIMPLE_LITERAL_STRING.subn(b"", content)
        if literal_count == content.count(b"(") == content.count(b")"):
            return len(content) - len(content_without_literals) - 2 * literal_count
    return sum(len(operand) for operand in _pdf_decoded_string_operands(content))


def _pdf_decode_literal_escape(match: re.Match[bytes]) -> bytes:
    escape: Final = match.group(1)
    if escape is None:
        return b""
    if escape == b"n":
        return b"\n"
    if escape == b"r":
        return b"\r"
    if escape == b"t":
        return b"\t"
    if escape == b"b":
        return b"\b"
    if escape == b"f":
        return b"\f"
    if escape in (b"(", b")", b"\\"):
        return escape
    if escape in (b"\r", b"\n", b"\r\n"):
        return b""
    if escape[:1] in b"01234567":
        return bytes((int(escape, 8) & 0xFF,))
    return escape


def _pdf_decode_string_operand(operand: bytes) -> bytes:
    if operand.startswith(b"<"):
        digits: Final = re.sub(PDF_WHITESPACE_CLASS + rb"+", b"", operand[1:-1])
        padded_digits: Final = digits if len(digits) % 2 == 0 else digits + b"0"
        return bytes.fromhex(padded_digits.decode("ascii"))
    literal: Final = operand[1:-1] if operand.endswith(b")") else operand[1:]
    return PDF_LITERAL_ESCAPE.sub(_pdf_decode_literal_escape, literal) if b"\\" in literal else literal


def _pdf_decoded_string_operands(content: bytes) -> Iterable[bytes]:
    return (
        _pdf_decode_string_operand(match.group(0))
        for match in PDF_STRING_OPERAND.finditer(content)
        if not match.group(0).startswith(b"%")
    )


_PdfTextObjectSpans: TypeAlias = tuple[tuple[int, int, bool, bool], ...]
_PdfTextTreeWalkItem: TypeAlias = tuple[_PdfStreamMetrics, bool]
_PdfTextRecordWalkItem: TypeAlias = tuple[_PdfStreamMetrics, tuple[int, ...]]


def _pdf_text_object_records(  # noqa: C901  # coordinates memoized per-stream PDF traversal
    metrics: _PdfStreamMetrics,
    invocation_path: tuple[int, ...] = (),
) -> Iterable[_PdfTextObjectRecord]:
    text_object_span_memo: Final[dict[int, _PdfTextObjectSpans]] = {}  # mutable-ok: page-local regex cache
    subtree_text_memo: Final[dict[int, bool]] = {}  # mutable-ok: page-local subtree cache

    def partial_text_object_end(content: bytes, start: int) -> tuple[int, bool, bool]:
        end: Final = min(len(content), start + PDF_PAGE_MAX_EXACT_CONTENT_BYTES)
        bounded_content: Final = content[start:end]
        match: Final = next(
            (match for match in PDF_TEXT_OBJECT.finditer(bounded_content) if match.group("text_object") is not None),
            None,
        )
        if match is None:
            return start, False, False
        closed: Final = match.group("text_object_end") is not None
        truncated: Final = match.end("text_object") == len(bounded_content) and end < len(content) and not closed
        return start + match.end("text_object"), closed, truncated

    def text_object_spans_in_window(content: bytes, window_end: int) -> Iterable[tuple[int, int, bool, bool]]:
        window_content: Final = content[:window_end]
        for match in PDF_TEXT_OBJECT.finditer(window_content):
            if match.group("text_object") is None:
                continue
            if (
                match.end("text_object") == window_end
                and window_end < len(content)
                and match.group("text_object_end") is None
            ):
                partial_end, partial_closed, partial_truncated = partial_text_object_end(
                    content, match.start("text_object")
                )
                yield (
                    match.start("text_object"),
                    partial_end,
                    partial_truncated,
                    partial_closed,
                )
                return
            yield (
                match.start("text_object"),
                match.end("text_object"),
                False,
                match.group("text_object_end") is not None,
            )

    def scan_text_object_spans(
        content: bytes,
        window_end: int,
    ) -> tuple[_PdfTextObjectSpans, int]:
        spans: Final[list[tuple[int, int, bool, bool]]] = []  # mutable-ok: capped page-local span cache
        cumulative_length: int = 0  # rebind-ok: accumulates record lengths until the sample cap
        for start, end, truncated, closed in text_object_spans_in_window(content, window_end):
            spans.append((start, end, truncated, closed))
            cumulative_length += end - start + (0 if closed else len(b"\nET")) + (0 if len(spans) == 1 else len(b"\n"))
            if cumulative_length >= PDF_PAGE_MAX_EXACT_CONTENT_BYTES:
                break
        return tuple(spans), cumulative_length

    def text_object_spans_for(stream_metrics: _PdfStreamMetrics) -> _PdfTextObjectSpans:
        identity: Final = id(stream_metrics)
        cached: Final = text_object_span_memo.get(identity)
        if cached is not None:
            return cached
        first_window_end: Final = min(len(stream_metrics.content), PDF_PAGE_MAX_EXACT_CONTENT_BYTES)
        first_spans, first_length = scan_text_object_spans(stream_metrics.content, first_window_end)
        cached_spans: Final = (
            first_spans
            if first_length >= PDF_PAGE_MAX_EXACT_CONTENT_BYTES or first_window_end == len(stream_metrics.content)
            else scan_text_object_spans(stream_metrics.content, len(stream_metrics.content))[0]
        )
        text_object_span_memo[identity] = cached_spans
        return cached_spans

    def subtree_has_text(stream_metrics: _PdfStreamMetrics) -> bool:
        identity: Final = id(stream_metrics)
        if identity in subtree_text_memo:
            return subtree_text_memo[identity]
        pending: Final[list[_PdfTextTreeWalkItem]] = [(stream_metrics, False)]  # mutable-ok: DFS stack
        while pending:
            current_metrics, expanded = pending.pop()
            current_identity = id(current_metrics)
            if current_identity in subtree_text_memo:
                continue
            if expanded:
                subtree_text_memo[current_identity] = any(
                    subtree_text_memo.get(id(form_metrics), False) for form_metrics in current_metrics.forms
                )
                continue
            if text_object_spans_for(current_metrics):
                subtree_text_memo[current_identity] = True
                continue
            pending.append((current_metrics, True))
            pending.extend((form_metrics, False) for form_metrics in current_metrics.forms)
        return subtree_text_memo[identity]

    if not subtree_has_text(metrics):
        return
    pending_records: Final[list[_PdfTextRecordWalkItem]] = [(metrics, invocation_path)]  # mutable-ok: DFS stack
    while pending_records:
        current_metrics, current_path = pending_records.pop()
        for index, (start, end, truncated, closed) in enumerate(text_object_spans_for(current_metrics)):
            yield _PdfTextObjectRecord(current_metrics, current_path, start, end, index == 0, truncated, closed)
        pending_records.extend(
            (form_metrics, current_path + (index,))
            for index, form_metrics in reversed(tuple(enumerate(current_metrics.forms)))
            if subtree_has_text(form_metrics)
        )


def _pdf_text_object_record_length(record: _PdfTextObjectRecord) -> int:
    text_object_length: Final = record.end - record.start
    separator_bytes: Final = 0 if record.first_in_stream else len(b"\n")
    if record.closed:
        return text_object_length + separator_bytes
    text_object: Final = record.stream.content[record.start : record.end]
    prefix_length, closures = _pdf_truncated_text_sample_closures(
        text_object,
        include_text_show=True,
        maximum_length=PDF_PAGE_MAX_EXACT_CONTENT_BYTES - separator_bytes - len(b"\nET"),
    )
    return prefix_length + len(closures) + len(b"\nET") + separator_bytes


def _pdf_text_object_record_bytes(record: _PdfTextObjectRecord) -> bytes:
    text_object: Final = record.stream.content[record.start : record.end]
    separator: Final = b"" if record.first_in_stream else b"\n"
    if record.closed:
        return separator + text_object
    prefix_length, closures = _pdf_truncated_text_sample_closures(
        text_object,
        include_text_show=True,
        maximum_length=PDF_PAGE_MAX_EXACT_CONTENT_BYTES - len(separator) - len(b"\nET"),
    )
    return separator + text_object[:prefix_length] + closures + b"\nET"


def _pdf_truncated_text_sample_closures(  # noqa: C901  # single-pass PDF syntax-state scanner
    prefix: bytes, include_text_show: bool, maximum_length: int | None = None
) -> tuple[int, bytes]:
    comment_open: bool = False  # rebind-ok: scans one bounded PDF prefix
    escape_open: bool = False  # rebind-ok: scans one bounded PDF prefix
    literal_depth: int = 0  # rebind-ok: tracks balanced literal strings
    hex_open: bool = False  # rebind-ok: scans one bounded PDF prefix
    array_depth: int = 0  # rebind-ok: tracks arrays outside strings
    bare_string_complete: bool = False  # rebind-ok: tracks the final string operand
    best_prefix_length: int = 0  # rebind-ok: retains the latest prefix whose closures fit the sample cap
    sample_limit: Final = len(prefix) if maximum_length is None else maximum_length
    best_state: tuple[bool, bool, int, bool, int, bool] = (  # rebind-ok: scanner state
        False,
        False,
        0,
        False,
        0,
        False,
    )
    for index, byte in enumerate(prefix):
        if comment_open:
            if byte in (ord("\r"), ord("\n")):
                comment_open = False
        elif literal_depth:
            if escape_open:
                escape_open = False
            elif byte == ord("\\"):
                escape_open = True
            elif byte == ord("("):
                literal_depth += 1
            elif byte == ord(")"):
                literal_depth -= 1
                if literal_depth == 0:
                    bare_string_complete = True
        elif hex_open:
            if byte == ord(">"):
                hex_open = False
                bare_string_complete = True
        elif byte == ord("%"):
            comment_open = True
        elif byte == ord("("):
            literal_depth = 1
            bare_string_complete = False
        elif byte == ord("<") and (index + 1 == len(prefix) or prefix[index + 1] != ord("<")):
            hex_open = True
            bare_string_complete = False
        elif byte == ord("["):
            array_depth += 1
            bare_string_complete = False
        elif byte == ord("]"):
            array_depth = max(0, array_depth - 1)
            bare_string_complete = False
        elif byte not in PDF_WHITESPACE_CLASS[1:-1]:
            bare_string_complete = False
        if (
            index
            + 1
            + (1 if comment_open else 0)
            + (2 if escape_open else 0)
            + literal_depth
            + (1 if hex_open else 0)
            + (
                array_depth + len(b" TJ")
                if include_text_show and array_depth
                else len(b" Tj")
                if include_text_show and (literal_depth or hex_open or bare_string_complete)
                else 0
            )
            <= sample_limit
        ):
            best_prefix_length = index + 1
            best_state = (
                comment_open,
                escape_open,
                literal_depth,
                hex_open,
                array_depth,
                bare_string_complete,
            )
    (
        best_comment_open,
        best_escape_open,
        best_literal_depth,
        best_hex_open,
        best_array_depth,
        best_bare_string_complete,
    ) = best_state
    operand_closures: Final = (
        (b"\\" if best_escape_open else b"")
        + (b")" * best_literal_depth if best_literal_depth else b"")
        + (b">" if best_hex_open else b"")
    )
    comment_closure: Final = b"\n" if best_comment_open else b""
    if not include_text_show:
        return best_prefix_length, comment_closure + operand_closures
    if best_array_depth:
        return best_prefix_length, comment_closure + operand_closures + b"]" * best_array_depth + b" TJ"
    if best_literal_depth or best_hex_open or best_bare_string_complete:
        return best_prefix_length, comment_closure + operand_closures + b" Tj"
    return best_prefix_length, comment_closure + operand_closures


def _pdf_truncated_text_object_prefix(
    content: bytes,
    start: int,
    prefix_length: int,
) -> bytes:
    prefix: Final = content[start : start + prefix_length]
    closed_prefix_length, closures = _pdf_truncated_text_sample_closures(
        prefix, include_text_show=True, maximum_length=prefix_length
    )
    return prefix[:closed_prefix_length] + closures + b"\nET"


def _pdf_truncated_string_operand_prefix(operand: bytes, prefix_length: int) -> bytes:
    prefix: Final = operand[:prefix_length]
    closed_prefix_length, closures = _pdf_truncated_text_sample_closures(prefix, include_text_show=False)
    return prefix[:closed_prefix_length] + closures


def _pdf_truncated_text_object(record: _PdfTextObjectRecord) -> bytes:
    content: Final = record.stream.content
    if record.truncated:
        return _pdf_truncated_text_object_prefix(
            content,
            record.start,
            PDF_PAGE_MAX_EXACT_CONTENT_BYTES - len(b"\nET"),
        )
    string_operands: Final = (
        operand
        for operand in PDF_STRING_OPERAND.finditer(content, record.start, record.end)
        if not operand.group(0).startswith(b"%")
    )
    largest_operand: Final = max(string_operands, key=lambda operand: len(operand.group(0)), default=None)
    if largest_operand is not None and len(largest_operand.group(0)) > PDF_PAGE_MAX_EXACT_CONTENT_BYTES // 2:
        prefix: Final = content[record.start : largest_operand.start()]
        suffix: Final = content[largest_operand.end() : record.end]
        object_closure: Final = b"" if record.closed else b"\nET"
        operand_prefix_length: Final = (
            PDF_PAGE_MAX_EXACT_CONTENT_BYTES - len(prefix) - len(suffix) - len(object_closure)
        )
        if operand_prefix_length > 0:
            sample: Final = (
                prefix
                + _pdf_truncated_string_operand_prefix(largest_operand.group(0), operand_prefix_length)
                + suffix
                + object_closure
            )
            if len(sample) <= PDF_PAGE_MAX_EXACT_CONTENT_BYTES:
                return sample
    return _pdf_truncated_text_object_prefix(
        content,
        record.start,
        PDF_PAGE_MAX_EXACT_CONTENT_BYTES - len(b"\nET"),
    )


def _pdf_text_sample_for_scope(records: Iterable[_PdfTextObjectRecord]) -> tuple[_PdfResourceDictionary, bytes]:
    scoped_records: Final = tuple(records)
    return (
        scoped_records[0].stream.resources,
        b"".join(_pdf_text_object_record_bytes(record) for record in scoped_records),
    )


def _pdf_bounded_text_samples(metrics: _PdfStreamMetrics) -> tuple[tuple[_PdfResourceDictionary, bytes], ...]:
    records, sized_records = tee(_pdf_text_object_records(metrics))
    first_record: Final = next(records, None)
    if first_record is None:
        return ()
    first_record_length: Final = _pdf_text_object_record_length(first_record)
    if first_record.truncated or first_record_length > PDF_PAGE_MAX_EXACT_CONTENT_BYTES:
        return ((first_record.stream.resources, _pdf_truncated_text_object(first_record)),)
    record_sizes: Final = (_pdf_text_object_record_length(record) for record in islice(sized_records, 1, None))
    cumulative_sizes: Final = islice(accumulate(record_sizes, initial=first_record_length), 1, None)
    bounded_records: Final = tuple(
        record
        for record, _ in takewhile(
            lambda item: item[1] <= PDF_PAGE_MAX_EXACT_CONTENT_BYTES,
            zip(records, cumulative_sizes),
        )
    )
    sampled_records: Final = (first_record, *bounded_records)
    return tuple(
        _pdf_text_sample_for_scope(group)
        for _, group in groupby(sampled_records, key=lambda record: record.invocation_path)
    )


def _pdf_extract_sample_text(
    page: _PdfPage,
    resources: _PdfResourceDictionary,
    content: bytes,
) -> str:
    pypdf_module: Final = cast(  # cast-ok: optional pypdf module has no typed module API
        _PdfModule, importlib.import_module("pypdf")
    )
    generic_module: Final = cast(  # cast-ok: optional pypdf module has no typed module API
        _PdfGenericModule, importlib.import_module("pypdf.generic")
    )

    copied_resources: Final = generic_module.DictionaryObject(
        {generic_module.NameObject(str(name)): value for name, value in resources.items() if str(name) != "/XObject"}
    )
    synthetic_page: Final = pypdf_module.PageObject.create_blank_page(None, page.mediabox.width, page.mediabox.height)
    synthetic_page[generic_module.NameObject("/Resources")] = copied_resources
    content_stream: Final = generic_module.DecodedStreamObject()
    content_stream.set_data(content)
    synthetic_page[generic_module.NameObject("/Contents")] = content_stream
    return synthetic_page.extract_text() or ""


def _pdf_page_text_tokens(page: _PdfPage, count_function: TokenCounterFunction) -> int:
    """Oversized pages use bounded pypdf text-object samples scaled to decoded operand bytes."""
    contents: Final = page.get_contents()
    content: Final = _pdf_strip_inline_images(contents.get_data() if contents is not None else b"")
    metrics: Final = _pdf_page_content_metrics(page, content)
    if metrics.extraction_limited or metrics.content_bytes > PDF_PAGE_MAX_EXACT_CONTENT_BYTES:
        samples: Final = _pdf_bounded_text_samples(metrics)
        sampled_operand_bytes: Final = sum(_pdf_string_operand_bytes(sample) for _, sample in samples)
        if sampled_operand_bytes == 0:
            return 0
        sampled_text_tokens: Final = sum(
            count_function(_pdf_extract_sample_text(page, resources, sample)) for resources, sample in samples
        )
        return round(sampled_text_tokens * metrics.operand_bytes / sampled_operand_bytes)
    return count_function(page.extract_text() or "")


def _price_pdf_pages(
    pages: Sequence[_PdfPage], count_function: TokenCounterFunction, pdf_pricing: PdfPagePricing = "claude"
) -> int:
    if pdf_pricing == "openai":
        samples: Final = tuple(pages[index] for index in _evenly_spaced_page_indices(len(pages)))
        if not samples:
            return 0
        sampled_tokens: Final = sum(_pdf_page_tokens(page, count_function, pdf_pricing) for page in samples)
        return round(sampled_tokens * len(pages) / len(samples))
    rendering_tokens: Final = sum(
        _anthropic_rendered_page_image_tokens(page.mediabox.width, page.mediabox.height) for page in pages
    )
    sampled_pages: Final = tuple(pages[index] for index in _evenly_spaced_page_indices(len(pages)))
    if not sampled_pages:
        return rendering_tokens
    sampled_text_tokens: Final = sum(_pdf_page_text_tokens(page, count_function) for page in sampled_pages)
    return rendering_tokens + round(sampled_text_tokens * len(pages) / len(sampled_pages))


def _pdf_dictionary(node: object) -> Mapping[str, object] | None:
    return node if isinstance(node, Mapping) else None  # pyright: ignore[reportUnknownVariableType]  # read per key


def _pdf_value(container: Mapping[str, object], key: str) -> object:
    return container[key] if key in container else None  # noqa: SIM401  # pypdf resolves references only in []


_PdfXObjects: TypeAlias = Mapping[str, object] | None
_PdfXObjectUse: TypeAlias = tuple[Mapping[str, object], _PdfXObjects]


@dataclass(frozen=True, slots=True)
class _PdfContentSummary:
    drawn_names: tuple[str, ...]
    draws_inline_image: bool


_PdfDrawing: TypeAlias = tuple[_PdfContentSummary, _PdfXObjects]


@dataclass(frozen=True, slots=True)
class _PdfWalk:
    frontier: tuple[_PdfDrawing, ...]
    seen: frozenset[tuple[int, int]]
    summaries: Mapping[int, _PdfContentSummary | None]
    draws: int


def _pdf_xobjects(container: Mapping[str, object], inherited: _PdfXObjects) -> _PdfXObjects:
    resources: Final = _pdf_dictionary(_pdf_value(container, "/Resources"))
    if resources is None:
        return inherited
    return _pdf_dictionary(_pdf_value(resources, "/XObject"))


_PDF_WHITESPACE: Final = rb"\x00\t\n\x0c\r "
_PDF_TOKEN_END: Final = _PDF_WHITESPACE + rb"()<>\[\]{}/%"
_PDF_XOBJECT_DRAW: Final = re.compile(
    rb"/([^" + _PDF_TOKEN_END + rb"]*)(?:[" + _PDF_WHITESPACE + rb"]|%[^\r\n]*[\r\n])+Do(?![^" + _PDF_TOKEN_END + rb"])"
)
_PDF_INLINE_IMAGE: Final = re.compile(rb"(?<![^" + _PDF_WHITESPACE + rb"])BI(?![^" + _PDF_WHITESPACE + rb"/])")
_PDF_NAME_ESCAPE: Final = re.compile(rb"#([0-9A-Fa-f]{2})")
_PDF_NAME_CHARSETS: Final = ("utf-8", "gbk")


def _decoded(raw: bytes, charset: str) -> str | None:
    try:
        return raw.decode(charset)
    except UnicodeDecodeError:
        return None


def _pdf_name(escaped: bytes) -> str:
    raw: Final = _PDF_NAME_ESCAPE.sub(lambda escape: bytes.fromhex(escape[1].decode("ascii")), escaped)
    decoded: Final = (_decoded(raw, charset) for charset in _PDF_NAME_CHARSETS)
    return "/" + next((name for name in decoded if name is not None), raw.decode("latin-1"))


def _pdf_content_summary(content: _PdfContent) -> _PdfContentSummary:
    data: Final = content.get_data()
    draws: Final = islice(_PDF_XOBJECT_DRAW.finditer(data), PDF_MAX_XOBJECT_DRAWS_PER_PAGE + 1)
    return _PdfContentSummary(
        drawn_names=tuple(_pdf_name(draw[1]) for draw in draws),
        draws_inline_image=_PDF_INLINE_IMAGE.search(data) is not None,
    )


def _pdf_form_summary(form: Mapping[str, object]) -> _PdfContentSummary | None:
    from pypdf.generic import StreamObject

    if not isinstance(form, StreamObject):
        return None
    return _pdf_content_summary(cast(_PdfContent, form))  # cast-ok: pypdf is untyped


def _pdf_drawn_xobjects(drawing: _PdfDrawing) -> tuple[_PdfXObjectUse, ...]:
    summary, xobjects = drawing
    if xobjects is None:
        return ()
    resolved: Final = (_pdf_dictionary(_pdf_value(xobjects, name)) for name in summary.drawn_names)
    return tuple((xobject, xobjects) for xobject in resolved if xobject is not None)


def _pdf_drawing_key(use: _PdfXObjectUse) -> tuple[int, int]:
    xobject, inherited = use
    return id(xobject), id(_pdf_xobjects(xobject, inherited))


def _pdf_next_walk(walk: _PdfWalk) -> _PdfWalk | None:
    draws: Final = walk.draws + sum(len(summary.drawn_names) for summary, _ in walk.frontier)
    if draws > PDF_MAX_XOBJECT_DRAWS_PER_PAGE:
        return None
    drawn: Final = tuple(chain.from_iterable(map(_pdf_drawn_xobjects, walk.frontier)))
    if any(_pdf_value(xobject, "/Subtype") == "/Image" for xobject, _ in drawn):
        return None
    forms: Final = {_pdf_drawing_key(use): use for use in drawn if _pdf_value(use[0], "/Subtype") == "/Form"}
    unseen: Final = tuple(forms[key] for key in forms.keys() - walk.seen)
    unparsed: Final = {id(form): form for form, _ in unseen if id(form) not in walk.summaries}
    summaries: Final = MappingProxyType(
        {**walk.summaries, **{key: _pdf_form_summary(form) for key, form in unparsed.items()}}
    )
    drawings: Final = ((summaries[id(form)], _pdf_xobjects(form, inherited)) for form, inherited in unseen)
    return _PdfWalk(
        frontier=tuple((summary, xobjects) for summary, xobjects in drawings if summary is not None),
        seen=walk.seen | frozenset(forms),
        summaries=summaries,
        draws=draws,
    )


def _pdf_page_draws_raster_image(page: _PdfPage) -> bool:
    contents: Final = page.get_contents()
    page_dictionary: Final = _pdf_dictionary(page)
    if contents is None or page_dictionary is None:
        return False
    walk: _PdfWalk = _PdfWalk(  # rebind-ok: breadth-first walk, recursion is banned
        frontier=((_pdf_content_summary(contents), _pdf_xobjects(page_dictionary, None)),),
        seen=frozenset(),
        summaries=MappingProxyType({}),
        draws=0,
    )
    for _ in range(PDF_MAX_FORM_XOBJECT_DEPTH):
        if not walk.frontier:
            return False
        if any(summary.draws_inline_image for summary, _ in walk.frontier):
            return True
        if (next_walk := _pdf_next_walk(walk)) is None:
            return True
        walk = next_walk
    return bool(walk.frontier)


def _openai_pdf_page_tokens(page: _PdfPage, count_function: TokenCounterFunction) -> int:
    if _pdf_page_draws_raster_image(page):
        return _claude_pdf_page_tokens(page, count_function)
    text_tokens: Final = _pdf_page_text_tokens(page, lambda text: count_function(text) if text.strip() else 0)
    return text_tokens + OPENAI_PDF_TEXT_PAGE_FRAMING_TOKENS if text_tokens else 0


def _claude_pdf_page_tokens(page: _PdfPage, count_function: TokenCounterFunction) -> int:
    return _pdf_page_text_tokens(page, count_function) + _anthropic_rendered_page_image_tokens(
        float(page.mediabox.width), float(page.mediabox.height)
    )


def _pdf_page_tokens(page: _PdfPage, count_function: TokenCounterFunction, pdf_pricing: PdfPagePricing) -> int:
    match pdf_pricing:
        case "openai":
            return _openai_pdf_page_tokens(page, count_function)
        case "claude":
            return _claude_pdf_page_tokens(page, count_function)


def _count_inline_pdf_tokens(
    data_url: str,
    count_function: TokenCounterFunction,
    pdf_pricing: PdfPagePricing = "claude",
) -> int | None:
    if not data_url.startswith(PDF_DATA_URL_PREFIX):
        return None
    try:
        from pypdf import PdfReader
    except ImportError:
        verbose_logger.debug("pypdf is not installed, so the PDF document is priced like one image")
        return None
    try:
        reader: Final = cast(  # cast-ok: pypdf's PdfReader satisfies _PdfReader, typed only where pypdf is installed
            _PdfReader, PdfReader(io.BytesIO(base64.b64decode(data_url[len(PDF_DATA_URL_PREFIX) :])))
        )
        return _price_pdf_pages(reader.pages, count_function, pdf_pricing)
    except Exception as e:
        verbose_logger.debug("Could not read the PDF document's pages (%s), so it is priced like one image", e)
        return None


def _count_opaque_document_tokens(
    data_url: str,
    count_function: TokenCounterFunction,
    use_default_image_token_count: bool,
    pdf_pricing: PdfPagePricing = "claude",
) -> int:
    pdf_tokens: Final = _count_inline_pdf_tokens(data_url, count_function, pdf_pricing)
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
    pdf_pricing: PdfPagePricing = "claude",
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
            count_function, content, use_default_image_token_count, default_token_count, pdf_pricing
        )
    return metadata_tokens + _count_opaque_document_tokens(
        _anthropic_image_source_data(source), count_function, use_default_image_token_count, pdf_pricing
    )


def _count_file_tokens(
    file_value: object,
    count_function: TokenCounterFunction,
    use_default_image_token_count: bool,
    pdf_pricing: PdfPagePricing = "claude",
) -> int:
    """An OpenAI `file` block is the chat-completions spelling of a document, so it prices like one."""
    if not isinstance(file_value, Mapping):
        return 0
    filename: Final = file_value.get("filename")
    file_data: Final = file_value.get("file_data")
    name_tokens: Final = count_function(filename) if isinstance(filename, str) and filename else 0
    if not isinstance(file_data, str) or not file_data:
        return name_tokens
    return name_tokens + _count_opaque_document_tokens(
        file_data, count_function, use_default_image_token_count, pdf_pricing
    )


def _count_anthropic_content(
    content: Mapping[str, object],
    count_function: TokenCounterFunction,
    use_default_image_token_count: bool,
    default_token_count: int | None,
    pdf_pricing: PdfPagePricing = "claude",
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
                    pdf_pricing,
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


_TEXT_FIELD_BY_BLOCK_TYPE: Final = MappingProxyType(
    {"text": "text", "thinking": "thinking", "redacted_thinking": "thinking", "compaction": "content"}
)


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
        | CompactionBlock
    ],
    use_default_image_token_count: bool,
    default_token_count: int | None,
    pdf_pricing: PdfPagePricing = "claude",
) -> int:
    """Recursively count tokens from a list of content blocks."""
    try:
        num_tokens = 0
        for c in content_list:
            if isinstance(c, str):
                num_tokens += count_function(c)
            elif c["type"] in _TEXT_FIELD_BY_BLOCK_TYPE:
                block_text = str(c.get(_TEXT_FIELD_BY_BLOCK_TYPE[c["type"]]) or "")
                if block_text:
                    num_tokens += count_function(block_text)
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
                    pdf_pricing,
                )
            elif c["type"] == "file":
                num_tokens += _count_file_tokens(
                    c.get("file"),
                    count_function,
                    use_default_image_token_count,
                    pdf_pricing,
                )
            elif c["type"] in ("tool_use", "tool_result", "search_result"):
                num_tokens += _count_anthropic_content(
                    c,
                    count_function,
                    use_default_image_token_count,
                    default_token_count,
                    pdf_pricing,
                )
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
                    f"compaction, tool_reference, search_result)."
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
    if not isinstance(properties, Mapping) or not properties:
        return ""
    required: Final = parameters.get("required")
    required_params: Final = tuple(required) if isinstance(required, list) else ()
    lines: Final = []
    for key, props in properties.items():
        description = props.get("description") if isinstance(props, Mapping) else None
        if description:
            lines.append(f"// {description}")
        question = "?"
        if required_params and key in required_params:
            question = ""
        lines.append(f"{key}{question}: {_format_type(props, indent)},")
    return "\n".join([" " * max(0, indent) + line for line in lines])


def _format_type(props, indent):
    if not isinstance(props, Mapping):
        return "any"
    type: Final = props.get("type")
    enum: Final = props.get("enum")
    if type == "string":
        if isinstance(enum, list):
            return " | ".join([f'"{item}"' for item in enum])
        return "string"
    elif type == "array":
        return f"{_format_type(props.get('items'), indent)}[]"
    elif type == "object":
        return f"{{\n{_format_object_parameters(props, indent + 2)}\n}}"
    elif type in ["integer", "number"]:
        if isinstance(enum, list):
            return " | ".join([f'"{item}"' for item in enum])
        return "number"
    elif type == "boolean":
        return "boolean"
    elif type == "null":
        return "null"
    else:
        # This is a guess, as an empty string doesn't yield the expected token count
        return "any"
