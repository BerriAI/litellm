# What is this?
## Helper utilities for token counting
import base64
import importlib
import io
import math
import re
import struct
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, groupby, islice, takewhile, tee
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast, runtime_checkable

import anyio
import anyio.lowlevel
import httpx
import tiktoken
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
from litellm.litellm_core_utils.prompt_templates.chat_template_tools import (
    ChatTemplateToolSection,
    chat_template_tool_section,
)
from litellm.litellm_core_utils.tokenizer import HuggingFaceTokenizer, OpenAIEncoding

if TYPE_CHECKING:
    from litellm.litellm_core_utils.tokenizer import Encoding, HuggingFace
from litellm.litellm_core_utils.url_utils import safe_get
from litellm.llms.custom_httpx.http_handler import get_httpx_client
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
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionDocumentObject,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
    ChatCompletionToolParam,
    OpenAIMessageContentListBlock,
)
from litellm.types.utils import Message, SelectTokenizerResponse


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
    resized_width,
    resized_height,
    tile_width=MAX_TILE_WIDTH,
    tile_height=MAX_TILE_HEIGHT,
):
    tiles_across: Final = (resized_width + tile_width - 1) // tile_width
    tiles_down: Final = (resized_height + tile_height - 1) // tile_height
    total_tiles: Final = tiles_across * tiles_down
    return total_tiles


def high_detail_image_token_upper_bound(base_tokens: int = 85) -> int:
    largest_tile_count: Final = calculate_tiles_needed(
        MAX_LONG_SIDE_FOR_IMAGE_HIGH_RES, MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES
    )
    return base_tokens + (base_tokens * 2) * largest_tile_count


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
            content_length: Final[str | None] = response.headers.get("Content-Length")
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


def calculate_img_tokens(
    data,
    mode: Literal["low", "high", "auto"] = "auto",
    base_tokens: int = 85,  # openai default - https://openai.com/pricing
    use_default_image_token_count: bool = False,
):
    """
    Calculate the number of tokens for an image.

    Args:
        data (str): The URL or base64 encoded string of the image.
        mode (Literal["low", "high", "auto"]): The mode to use for calculating the number of tokens.
        base_tokens (int): The base number of tokens for an image.
        use_default_image_token_count (bool): When True, will NOT make a GET request to the image URL and instead return the default image dimensions.

    Returns:
        int: The number of tokens for the image.
    """
    if use_default_image_token_count:
        verbose_logger.debug("Using default image token count: %s", DEFAULT_IMAGE_TOKEN_COUNT)
        return DEFAULT_IMAGE_TOKEN_COUNT
    if mode == "low" or mode == "auto":
        return base_tokens
    elif mode == "high":
        # Run the async function using the helper
        width, height = get_image_dimensions(
            data=data,
        )
        resized_width, resized_height = resize_image_high_res(width=width, height=height)
        tiles_needed_high_res = calculate_tiles_needed(resized_width=resized_width, resized_height=resized_height)
        tile_tokens: Final = (base_tokens * 2) * tiles_needed_high_res
        total_tokens: Final = base_tokens + tile_tokens
        return total_tokens


TokenCounterFunction = Callable[[str], int]
"""
Type for a function that counts tokens in a string.
"""

EXTRAPOLATION_SAMPLES: Final = 16
PDF_PAGE_MAX_EXACT_CONTENT_BYTES: Final = 64 * 1024
PDF_LITERAL_STRING_OPERAND: Final = rb"\((?:\\[\s\S]|[^\\()]|\((?:\\[\s\S]|[^\\()])*\)?)*\)?"
PDF_STRING_OPERAND: Final = re.compile(PDF_LITERAL_STRING_OPERAND + rb"|<[0-9A-Fa-f\s]*>|%[^\r\n]*")
PDF_LITERAL_ESCAPE: Final = re.compile(rb"\\([nrtbf()\\]|[0-7]{1,3}|\r\n?|\n|[\s\S])")
PDF_INLINE_IMAGE: Final = re.compile(
    PDF_LITERAL_STRING_OPERAND
    + rb"|<[0-9A-Fa-f\s]*>|%[^\r\n]*|(?P<inline_image>(?<![^\s\]>)])BI(?=\s)"
    + rb"[\s\S]*?(?:(?<=\s)ID\s[\s\S]*?(?:(?P<image_end>\sEI(?![^\s\[/<(%]))|\Z)|\Z))"
)
PDF_TEXT_OBJECT: Final = re.compile(
    PDF_LITERAL_STRING_OPERAND
    + rb"|<[0-9A-Fa-f\s]*>|%[^\r\n]*|(?P<text_object>(?<![^\s\]>)])BT(?=\s)"
    + rb"(?:(?:"
    + PDF_LITERAL_STRING_OPERAND
    + rb"|<[0-9A-Fa-f\s]*>|%[^\r\n]*|[\s\S])*?"
    + rb"(?:(?<=\s)ET(?![^\s\[\]()<>{}/%])|\Z)))"
)
PDF_XOBJECT_INVOCATION: Final = re.compile(
    PDF_LITERAL_STRING_OPERAND + rb"|%[^\r\n]*(?:[\r\n]|\Z)|/([^\s/\[\]()<>{}%]+)(?:\s|%[^\r\n]*(?:[\r\n]|\Z))+Do\b"
)
T_ParamSpec: Final = ParamSpec("T_ParamSpec")
T_Retval = TypeVar("T_Retval")
_COUNT_OFFLOAD_LIMITER: Final = anyio.lowlevel.RunVar[anyio.CapacityLimiter]("litellm_count_offload_limiter")
_ANTHROPIC_TOOL_KEYS_MAPPED_BY_BRIDGE: Final = frozenset(
    ("name", "input_schema", "description", "cache_control", "strict", "type", "eager_input_streaming")
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
    tool_choice: ChatCompletionNamedToolChoiceParam | None = None,
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
    tool_choice: ChatCompletionNamedToolChoiceParam | None,
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
    tool_choice: ChatCompletionNamedToolChoiceParam | None,
) -> int:
    if tool_choice == "none":
        return 1
    if isinstance(tool_choice, dict):
        return 7 + count_function(str(tool_choice["function"]["name"]))
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
    """The tiktoken encoding name for `model`, without loading the encoding."""
    from litellm.utils import print_verbose

    model_to_use: Final = _fix_model_name(model)
    if "gpt-4o" in model_to_use:
        return "o200k_base"
    try:
        return tiktoken.encoding_name_for_model(model_to_use)
    except KeyError:
        print_verbose("Warning: model not found. Using cl100k_base encoding.")
        return "cl100k_base"


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


class _PdfMediaBox(Protocol):
    @property
    def width(self) -> float: ...

    @property
    def height(self) -> float: ...


class _PdfPage(_PdfResourceDictionary, Protocol):
    @property
    def mediabox(self) -> _PdfMediaBox: ...

    def get_contents(self) -> _PdfContentStream | None: ...

    def extract_text(self) -> str: ...


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
class _PdfStreamMetrics:
    content: bytes
    content_bytes: int
    operand_bytes: int
    resources: _PdfResourceDictionary
    forms: tuple["_PdfStreamMetrics", ...]


@dataclass(frozen=True, slots=True)
class _PdfTextObjectRecord:
    stream: _PdfStreamMetrics
    invocation_path: tuple[int, ...]
    start: int
    end: int
    first_in_stream: bool


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


def _pdf_page_content_metrics(page: _PdfPage, page_content: bytes) -> _PdfStreamMetrics:
    form_metrics_memo: Final[dict[int, _PdfStreamMetrics]] = {}  # mutable-ok: page-local cache avoids repeated walks

    def stream_metrics(
        content: bytes,
        resources: _PdfResourceDictionary,
        ancestors: frozenset[int],
    ) -> _PdfStreamMetrics:
        xobjects: Final = _pdf_resolve_object(resources.get("/XObject"))
        xobject_resources: Final = xobjects if isinstance(xobjects, _PdfResourceDictionary) else _PDF_EMPTY_RESOURCES
        invoked_names: Final = tuple(
            name
            for invocation in PDF_XOBJECT_INVOCATION.finditer(content)
            if (name := _pdf_invocation_name(invocation)) is not None
        )
        form_metrics: Final = tuple(
            form_metrics_for_xobject(
                _pdf_resolve_object(xobject_resources.get(name)),
                resources,
                ancestors,
            )
            for name in invoked_names
        )
        return _PdfStreamMetrics(
            content=content,
            content_bytes=len(content) + sum(metrics.content_bytes for metrics in form_metrics),
            operand_bytes=_pdf_string_operand_bytes(content) + sum(metrics.operand_bytes for metrics in form_metrics),
            resources=resources,
            forms=form_metrics,
        )

    def form_metrics_for_xobject(
        xobject: object,
        inherited_resources: _PdfResourceDictionary,
        ancestors: frozenset[int],
    ) -> _PdfStreamMetrics:
        resolved: Final = _pdf_resolve_object(xobject)
        identity: Final = id(resolved)
        if identity in ancestors:
            return _PdfStreamMetrics(b"", 0, 0, inherited_resources, ())
        if not isinstance(resolved, _PdfFormXObject) or resolved.get("/Subtype") != "/Form":
            return _PdfStreamMetrics(b"", 0, 0, inherited_resources, ())
        cached: Final = form_metrics_memo.get(identity)
        if cached is not None:
            return cached
        content: Final = PDF_INLINE_IMAGE.sub(_pdf_inline_image_replacement, resolved.get_data())
        form_resources: Final = _pdf_resolve_object(resolved.get("/Resources"))
        resources: Final = form_resources if isinstance(form_resources, _PdfResourceDictionary) else inherited_resources
        metrics: Final = stream_metrics(content, resources, ancestors | {identity})
        form_metrics_memo[identity] = metrics
        return metrics

    resources: Final = _pdf_resolve_object(page.get("/Resources"))
    page_resources: Final = resources if isinstance(resources, _PdfResourceDictionary) else _PDF_EMPTY_RESOURCES
    return stream_metrics(page_content, page_resources, frozenset())


def _evenly_spaced_page_indices(page_count: int) -> tuple[int, ...]:
    if page_count <= EXTRAPOLATION_SAMPLES:
        return tuple(range(page_count))
    return tuple((page_count - 1) * index // (EXTRAPOLATION_SAMPLES - 1) for index in range(EXTRAPOLATION_SAMPLES))


def _pdf_string_operand_bytes(content: bytes) -> int:
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
        digits: Final = re.sub(rb"\s+", b"", operand[1:-1])
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


def _pdf_text_object_records(
    metrics: _PdfStreamMetrics,
    invocation_path: tuple[int, ...] = (),
) -> Iterable[_PdfTextObjectRecord]:
    text_object_spans: Final = (
        (match.start("text_object"), match.end("text_object"))
        for match in PDF_TEXT_OBJECT.finditer(metrics.content)
        if match.group("text_object") is not None
    )
    for index, (start, end) in enumerate(text_object_spans):
        yield _PdfTextObjectRecord(metrics, invocation_path, start, end, index == 0)
    for index, form_metrics in enumerate(metrics.forms):
        yield from _pdf_text_object_records(form_metrics, invocation_path + (index,))


def _pdf_text_object_record_length(record: _PdfTextObjectRecord) -> int:
    text_object_length: Final = record.end - record.start
    is_closed: Final = record.stream.content[record.end - 2 : record.end] == b"ET"
    closure_bytes: Final = 0 if is_closed else len(b"\nET")
    separator_bytes: Final = 0 if record.first_in_stream else len(b"\n")
    return text_object_length + closure_bytes + separator_bytes


def _pdf_text_object_record_bytes(record: _PdfTextObjectRecord) -> bytes:
    text_object: Final = record.stream.content[record.start : record.end]
    separator: Final = b"" if record.first_in_stream else b"\n"
    closure: Final = b"" if text_object.endswith(b"ET") else b"\nET"
    return separator + text_object + closure


def _pdf_truncated_operand_closures(prefix: bytes) -> bytes:
    operands: Final = tuple(PDF_STRING_OPERAND.finditer(prefix))
    last_operand: Final = operands[-1] if operands else None
    if last_operand is not None and last_operand.end() == len(prefix):
        operand: Final = last_operand.group(0)
        if operand.startswith(b"%"):
            return b""
        if operand.startswith(b"("):
            unescaped_literal: Final = PDF_LITERAL_ESCAPE.sub(b"", operand)
            open_parentheses: Final = max(0, unescaped_literal.count(b"(") - unescaped_literal.count(b")"))
            trailing_backslashes: Final = re.search(rb"\\+$", operand)
            escaped_trailing_backslash: Final = (
                b"\\" if trailing_backslashes is not None and len(trailing_backslashes.group(0)) % 2 else b""
            )
            return escaped_trailing_backslash + b")" * open_parentheses
    if re.search(rb"<[0-9A-Fa-f\s]*$", prefix) is not None:
        return b">"
    return b""


def _pdf_truncated_text_object_prefix(
    content: bytes,
    start: int,
    prefix_length: int,
) -> bytes:
    prefix: Final = content[start : start + prefix_length]
    operand_closures: Final = _pdf_truncated_operand_closures(prefix)
    if len(prefix) + len(operand_closures) + len(b"\nET") <= PDF_PAGE_MAX_EXACT_CONTENT_BYTES:
        return prefix + operand_closures + b"\nET"
    reduced_prefix_length: Final = (
        prefix_length + PDF_PAGE_MAX_EXACT_CONTENT_BYTES - len(operand_closures) - len(b"\nET")
    ) // 2
    return _pdf_truncated_text_object_prefix(content, start, reduced_prefix_length)


def _pdf_truncated_string_operand_prefix(operand: bytes, prefix_length: int) -> bytes:
    def prefix_for_length(current_length: int) -> bytes:
        prefix: Final = operand[:current_length]
        operand_closures: Final = _pdf_truncated_operand_closures(prefix)
        if len(prefix) + len(operand_closures) <= prefix_length:
            return prefix + operand_closures
        reduced_length: Final = (current_length + prefix_length - len(operand_closures)) // 2
        return prefix_for_length(reduced_length)

    return prefix_for_length(prefix_length)


def _pdf_truncated_text_object(record: _PdfTextObjectRecord) -> bytes:
    content: Final = record.stream.content
    string_operands: Final = (
        operand
        for operand in PDF_STRING_OPERAND.finditer(content, record.start, record.end)
        if not operand.group(0).startswith(b"%")
    )
    largest_operand: Final = max(string_operands, key=lambda operand: len(operand.group(0)), default=None)
    if largest_operand is not None and len(largest_operand.group(0)) > PDF_PAGE_MAX_EXACT_CONTENT_BYTES // 2:
        prefix: Final = content[record.start : largest_operand.start()]
        suffix: Final = content[largest_operand.end() : record.end]
        object_closure: Final = b"" if suffix.endswith(b"ET") else b"\nET"
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
    if first_record_length > PDF_PAGE_MAX_EXACT_CONTENT_BYTES:
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
    content: Final = PDF_INLINE_IMAGE.sub(
        _pdf_inline_image_replacement,
        contents.get_data() if contents is not None else b"",
    )
    metrics: Final = _pdf_page_content_metrics(page, content)
    if metrics.content_bytes > PDF_PAGE_MAX_EXACT_CONTENT_BYTES:
        samples: Final = _pdf_bounded_text_samples(metrics)
        sampled_operand_bytes: Final = sum(_pdf_string_operand_bytes(sample) for _, sample in samples)
        if sampled_operand_bytes == 0:
            return 0
        sampled_text_tokens: Final = sum(
            count_function(_pdf_extract_sample_text(page, resources, sample)) for resources, sample in samples
        )
        return round(sampled_text_tokens * metrics.operand_bytes / sampled_operand_bytes)
    return count_function(page.extract_text() or "")


def _price_pdf_pages(pages: Sequence[_PdfPage], count_function: TokenCounterFunction) -> int:
    rendering_tokens: Final = sum(
        _anthropic_rendered_page_image_tokens(page.mediabox.width, page.mediabox.height) for page in pages
    )
    sampled_pages: Final = tuple(pages[index] for index in _evenly_spaced_page_indices(len(pages)))
    if not sampled_pages:
        return rendering_tokens
    sampled_text_tokens: Final = sum(_pdf_page_text_tokens(page, count_function) for page in sampled_pages)
    return rendering_tokens + round(sampled_text_tokens * len(pages) / len(sampled_pages))


def _count_inline_pdf_tokens(data_url: str, count_function: TokenCounterFunction) -> int | None:
    if not data_url.startswith(PDF_DATA_URL_PREFIX):
        return None
    try:
        from pypdf import PdfReader
    except ImportError:
        verbose_logger.debug("pypdf is not installed, so the PDF document is priced like one image")
        return None
    try:
        pages: Final = cast(  # cast-ok: pypdf is an optional extra the type-check env does not install
            Sequence[_PdfPage], PdfReader(io.BytesIO(base64.b64decode(data_url[len(PDF_DATA_URL_PREFIX) :]))).pages
        )
        return _price_pdf_pages(pages, count_function)
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
            raise ValueError(f"Error counting field '{field_name}': {e}")
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
            elif c["type"] in ("tool_use", "tool_result"):
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
                raise ValueError(
                    f"Invalid content item type: {content_type}. "
                    f"Expected str or dict with 'type' field "
                    f"(text, image_url, image, document, file, tool_use, tool_result, thinking, redacted_thinking, "
                    f"tool_reference)."
                )
        return num_tokens
    except Exception as e:
        if default_token_count is not None:
            return default_token_count
        raise ValueError(
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
