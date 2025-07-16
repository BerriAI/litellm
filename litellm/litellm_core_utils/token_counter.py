# What is this?
## Helper utilities for token counting
import base64
import io
import struct
from typing import Callable, List, Literal, Optional, Tuple, Union, cast

import tiktoken

import litellm
from litellm import verbose_logger
from litellm.constants import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_TOKEN_COUNT,
    DEFAULT_IMAGE_WIDTH,
    MAX_LONG_SIDE_FOR_IMAGE_HIGH_RES,
    MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES,
    MAX_TILE_HEIGHT,
    MAX_TILE_WIDTH,
)
from litellm.litellm_core_utils.default_encoding import encoding as default_encoding
from litellm.llms.custom_httpx.http_handler import _get_httpx_client
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolParam,
    OpenAIMessageContent,
)
from litellm.types.utils import Message, SelectTokenizerResponse


def get_modified_max_tokens(
    model: str,
    base_model: str,
    messages: Optional[List[AllMessageValues]],
    user_max_tokens: Optional[int],
    buffer_perc: Optional[float],
    buffer_num: Optional[float],
) -> Optional[int]:
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
        _model_info = litellm.get_model_info(model=model)

        max_output_tokens = litellm.get_max_tokens(
            model=base_model
        )  # assume min context window is 4k tokens

        ## UNKNOWN MAX OUTPUT TOKENS - return user defined amount
        if max_output_tokens is None:
            return user_max_tokens

        input_tokens = litellm.token_counter(model=base_model, messages=messages)

        # token buffer
        if buffer_perc is None:
            buffer_perc = 0.1
        if buffer_num is None:
            buffer_num = 10
        token_buffer = max(
            buffer_perc * input_tokens, buffer_num
        )  # give at least a 10 token buffer. token counting can be imprecise.

        input_tokens += int(token_buffer)
        verbose_logger.debug(
            f"max_output_tokens: {max_output_tokens}, user_max_tokens: {user_max_tokens}"
        )
        ## CASE 1: model input + output can't exceed X - happens when max input = max output, e.g. gpt-3.5-turbo
        if _model_info["max_input_tokens"] == max_output_tokens:
            verbose_logger.debug(
                f"input_tokens: {input_tokens}, max_output_tokens: {max_output_tokens}"
            )
            if input_tokens > max_output_tokens:
                pass  # allow call to fail normally - don't set max_tokens to negative.
            elif (
                user_max_tokens + input_tokens > max_output_tokens
            ):  # we can still modify to keep it positive but below the limit
                verbose_logger.debug(
                    f"MODIFYING MAX TOKENS - user_max_tokens={user_max_tokens}, input_tokens={input_tokens}, max_output_tokens={max_output_tokens}"
                )
                user_max_tokens = int(max_output_tokens - input_tokens)
        ## CASE 2: user_max_tokens> model max output tokens
        elif user_max_tokens > max_output_tokens:
            user_max_tokens = max_output_tokens

        verbose_logger.debug(
            f"litellm.litellm_core_utils.token_counter.py::get_modified_max_tokens() - user_max_tokens: {user_max_tokens}"
        )

        return user_max_tokens
    except Exception as e:
        verbose_logger.debug(
            "litellm.litellm_core_utils.token_counter.py::get_modified_max_tokens() - Error while checking max token limit: {}\nmodel={}, base_model={}".format(
                str(e), model, base_model
            )
        )
        return user_max_tokens


def resize_image_high_res(
    width: int,
    height: int,
) -> Tuple[int, int]:
    # Maximum dimensions for high res mode
    max_short_side = MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES
    max_long_side = MAX_LONG_SIDE_FOR_IMAGE_HIGH_RES

    # Return early if no resizing is needed
    if (
        width <= MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES
        and height <= MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES
    ):
        return width, height

    # Determine the longer and shorter sides
    longer_side = max(width, height)
    shorter_side = min(width, height)

    # Calculate the aspect ratio
    aspect_ratio = longer_side / shorter_side

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
    tiles_across = (resized_width + tile_width - 1) // tile_width
    tiles_down = (resized_height + tile_height - 1) // tile_height
    total_tiles = tiles_across * tiles_down
    return total_tiles


def get_image_type(image_data: bytes) -> Union[str, None]:
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
) -> Tuple[int, int]:
    """
    Async Function to get the dimensions of an image from a URL or base64 encoded string.

    Args:
        data (str): The URL or base64 encoded string of the image.

    Returns:
        Tuple[int, int]: The width and height of the image.
    """
    img_data = None
    try:
        # Try to open as URL
        client = _get_httpx_client()
        response = client.get(data)
        img_data = response.read()
    except Exception:
        # If not URL, assume it's base64
        _header, encoded = data.split(",", 1)
        img_data = base64.b64decode(encoded)

    img_type = get_image_type(img_data)

    if img_type == "png":
        w, h = struct.unpack(">LL", img_data[16:24])
        return w, h
    elif img_type == "gif":
        w, h = struct.unpack("<HH", img_data[6:10])
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
                size = struct.unpack(">H", fhandle.read(2))[0] - 2
            fhandle.seek(1, 1)
            h, w = struct.unpack(">HH", fhandle.read(4))
        return w, h
    elif img_type == "webp":
        # For WebP, the dimensions are stored at different offsets depending on the format
        # Check for VP8X (extended format)
        if img_data[12:16] == b"VP8X":
            w = struct.unpack("<I", img_data[24:27] + b"\x00")[0] + 1
            h = struct.unpack("<I", img_data[27:30] + b"\x00")[0] + 1
            return w, h
        # Check for VP8 (lossy format)
        elif img_data[12:16] == b"VP8 ":
            w = struct.unpack("<H", img_data[26:28])[0] & 0x3FFF
            h = struct.unpack("<H", img_data[28:30])[0] & 0x3FFF
            return w, h
        # Check for VP8L (lossless format)
        elif img_data[12:16] == b"VP8L":
            bits = struct.unpack("<I", img_data[21:25])[0]
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
        verbose_logger.debug(
            "Using default image token count: {}".format(DEFAULT_IMAGE_TOKEN_COUNT)
        )
        return DEFAULT_IMAGE_TOKEN_COUNT
    if mode == "low" or mode == "auto":
        return base_tokens
    elif mode == "high":
        # Run the async function using the helper
        width, height = get_image_dimensions(
            data=data,
        )
        resized_width, resized_height = resize_image_high_res(
            width=width, height=height
        )
        tiles_needed_high_res = calculate_tiles_needed(
            resized_width=resized_width, resized_height=resized_height
        )
        tile_tokens = (base_tokens * 2) * tiles_needed_high_res
        total_tokens = base_tokens + tile_tokens
        return total_tokens


TokenCounterFunction = Callable[[str], int]
"""
Type for a function that counts tokens in a string.
"""


class _MessageCountParams:
    """
    A class to hold the parameters for counting tokens in messages.
    """

    def __init__(
        self,
        model: str,
        custom_tokenizer: Optional[Union[dict, SelectTokenizerResponse]],
    ):
        from litellm.utils import print_verbose

        actual_model = _fix_model_name(model)
        if actual_model == "gpt-3.5-turbo-0301":
            self.tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            self.tokens_per_name = -1  # if there's a name, the role is omitted
        elif actual_model in litellm.open_ai_chat_completion_models:
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        elif actual_model in litellm.azure_llms:
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        else:
            print_verbose(
                f"Warning: unknown model {model}. Using default token params."
            )
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        self.count_function = _get_count_function(model, custom_tokenizer)


def token_counter(
    model="",
    custom_tokenizer: Optional[Union[dict, SelectTokenizerResponse]] = None,
    text: Optional[Union[str, List[str]]] = None,
    messages: Optional[List[Union[AllMessageValues, Message]]] = None,
    count_response_tokens: Optional[bool] = False,
    tools: Optional[List[ChatCompletionToolParam]] = None,
    tool_choice: Optional[ChatCompletionNamedToolChoiceParam] = None,
    use_default_image_token_count: Optional[bool] = False,
    default_token_count: Optional[int] = None,
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

    verbose_logger.debug(
        f"messages in token_counter: {messages}, text in token_counter: {text}"
    )
    if text is not None and messages is not None:
        raise ValueError("text and messages cannot both be set")
    if use_default_image_token_count is None:
        use_default_image_token_count = False

    if text is not None:
        if tools or tool_choice:
            raise ValueError("tools or tool_choice cannot be set if using text")
        if isinstance(text, List):
            text_to_count = "".join(t for t in text if isinstance(t, str))
        elif isinstance(text, str):
            text_to_count = text
        count_function = _get_count_function(model, custom_tokenizer)
        num_tokens = count_function(text_to_count)

    elif messages is not None:
        new_messages = cast(
            List[AllMessageValues], convert_list_message_to_dict(messages)
        )
        params = _MessageCountParams(model, custom_tokenizer)
        num_tokens = _count_messages(
            params, new_messages, use_default_image_token_count, default_token_count
        )
        if count_response_tokens is False:
            includes_system_message = any(
                [message.get("role", None) == "system" for message in new_messages]
            )
            num_tokens += _count_extra(
                params.count_function, tools, tool_choice, includes_system_message
            )

    else:
        raise ValueError("Either text or messages must be provided")

    return num_tokens


def _count_messages(
    params: _MessageCountParams,
    messages: List[AllMessageValues],
    use_default_image_token_count: bool,
    default_token_count: Optional[int],
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
            elif key == "tool_calls":
                if isinstance(value, List):
                    for tool_call in value:
                        if "function" in tool_call:
                            function_arguments = tool_call["function"].get(
                                "arguments", []
                            )
                            num_tokens += params.count_function(str(function_arguments))
                        else:
                            raise ValueError(
                                f"Unsupported tool call {tool_call} must contain a function key"
                            )
                else:
                    raise ValueError(
                        f"Unsupported type {type(value)} for key tool_calls in message {message}"
                    )
            elif isinstance(value, str):
                num_tokens += params.count_function(value)
                if key == "name":
                    num_tokens += params.tokens_per_name
            elif key == "content" and isinstance(value, List):
                num_tokens += _count_content_list(
                    params.count_function,
                    value,
                    use_default_image_token_count,
                    default_token_count,
                )
            else:
                raise ValueError(
                    f"Unsupported type {type(value)} for key {key} in message {message}"
                )
    return num_tokens


def _count_extra(
    count_function: TokenCounterFunction,
    tools: Optional[List[ChatCompletionToolParam]],
    tool_choice: Optional[ChatCompletionNamedToolChoiceParam],
    includes_system_message: bool,
) -> int:
    """Count extra tokens for function definitions and tool choices.
    Args:
        count_function (TokenCounterFunction): The function to count tokens.
        tools (Optional[List[ChatCompletionToolParam]]): The available tools.
        tool_choice (Optional[ChatCompletionNamedToolChoiceParam]): The tool choice.
        includes_system_message (bool): Whether the messages include a system message.
    """

    num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>

    if tools:
        num_tokens += count_function(_format_function_definitions(tools))
        num_tokens += 9  # Additional tokens for function definition of tools
    # If there's a system message and tools are present, subtract four tokens
    if tools and includes_system_message:
        num_tokens -= 4
    # If tool_choice is 'none', add one token.
    # If it's an object, add 4 + the number of tokens in the function name.
    # If it's undefined or 'auto', don't add anything.
    if tool_choice == "none":
        num_tokens += 1
    elif isinstance(tool_choice, dict):
        num_tokens += 7
        num_tokens += count_function(str(tool_choice["function"]["name"]))

    return num_tokens


def _get_count_function(
    model: Optional[str],
    custom_tokenizer: Optional[Union[dict, SelectTokenizerResponse]] = None,
) -> TokenCounterFunction:
    """
    Get the function to count tokens based on the model and custom tokenizer."""
    from litellm.utils import _select_tokenizer, print_verbose

    if model is not None or custom_tokenizer is not None:
        tokenizer_json = custom_tokenizer or _select_tokenizer(model)  # type: ignore
        if tokenizer_json["type"] == "huggingface_tokenizer":

            def count_tokens(text: str) -> int:
                enc = tokenizer_json["tokenizer"].encode(text)
                return len(enc.ids)

        elif tokenizer_json["type"] == "openai_tokenizer":
            model_to_use = _fix_model_name(model)  # type: ignore
            try:
                if "gpt-4o" in model_to_use:
                    encoding = tiktoken.get_encoding("o200k_base")
                else:
                    encoding = tiktoken.encoding_for_model(model_to_use)
            except KeyError:
                print_verbose("Warning: model not found. Using cl100k_base encoding.")
                encoding = tiktoken.get_encoding("cl100k_base")

            def count_tokens(text: str) -> int:
                return len(encoding.encode(text))

        else:
            raise ValueError("Unsupported tokenizer type")
    else:

        def count_tokens(text: str) -> int:
            return len(default_encoding.encode(text, disallowed_special=()))

    return count_tokens


def _fix_model_name(model: str) -> str:
    """We normalize some model names to others"""
    if model in litellm.azure_llms:
        # azure llms use gpt-35-turbo instead of gpt-3.5-turbo 🙃
        return model.replace("-35", "-3.5")
    elif model in litellm.open_ai_chat_completion_models:
        return model  # type: ignore
    else:
        return "gpt-3.5-turbo"


def _count_content_list(
    count_function: TokenCounterFunction,
    content_list: OpenAIMessageContent,
    use_default_image_token_count: bool,
    default_token_count: Optional[int],
) -> int:
    """
    Get the number of tokens from a list of content.
    """
    try:
        num_tokens = 0
        for c in content_list:
            if isinstance(c, str):
                num_tokens += count_function(c)
            elif c["type"] == "text":
                num_tokens += count_function(c["text"])
            elif c["type"] == "image_url":
                if isinstance(c["image_url"], dict):
                    image_url_dict = c["image_url"]
                    detail = image_url_dict.get("detail", "auto")
                    if detail not in ["low", "high", "auto"]:
                        raise ValueError(
                            f"Invalid detail value: {detail}. Expected 'low', 'high', or 'auto'."
                        )
                    url = image_url_dict.get("url")
                    num_tokens += calculate_img_tokens(
                        data=url,
                        mode=detail,  # type: ignore
                        use_default_image_token_count=use_default_image_token_count,
                    )
                elif isinstance(c["image_url"], str):
                    image_url_str = c["image_url"]
                    num_tokens += calculate_img_tokens(
                        data=image_url_str,
                        mode="auto",
                        use_default_image_token_count=use_default_image_token_count,
                    )
                else:
                    raise ValueError(
                        f"Invalid image_url type: {type(c['image_url'])}. Expected str or dict."
                    )
            else:
                raise ValueError(
                    f"Invalid content type: {type(c)}. Expected str or dict."
                )
        return num_tokens
    except Exception as e:
        if default_token_count is not None:
            return default_token_count
        raise ValueError(
            f"Error getting number of tokens from content list: {e}, default_token_count={default_token_count}"
        )


def _format_function_definitions(tools):
    """Formats tool definitions in the format that OpenAI appears to use.
    Based on https://github.com/forestwanglin/openai-java/blob/main/jtokkit/src/main/java/xyz/felh/openai/jtokkit/utils/TikTokenUtils.java
    """
    lines = []
    lines.append("namespace functions {")
    lines.append("")
    for tool in tools:
        function = tool.get("function")
        if function_description := function.get("description"):
            lines.append(f"// {function_description}")
        function_name = function.get("name")
        parameters = function.get("parameters", {})
        properties = parameters.get("properties")
        if properties and properties.keys():
            lines.append(f"type {function_name} = (_: {{")
            lines.append(_format_object_parameters(parameters, 0))
            lines.append("}) => any;")
        else:
            lines.append(f"type {function_name} = () => any;")
        lines.append("")
    lines.append("} // namespace functions")
    return "\n".join(lines)


def _format_object_parameters(parameters, indent):
    properties = parameters.get("properties")
    if not properties:
        return ""
    required_params = parameters.get("required", [])
    lines = []
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
    type = props.get("type")
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
