# What is this?
## Helper utilities for token counting
import base64
import io
import struct
from typing import Literal, Optional, Tuple, Union

import litellm
from litellm import verbose_logger
from litellm.constants import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_TOKEN_COUNT,
    DEFAULT_IMAGE_WIDTH,
)
from litellm.llms.custom_httpx.http_handler import _get_httpx_client


def get_modified_max_tokens(
    model: str,
    base_model: str,
    messages: Optional[list],
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
        verbose_logger.error(
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
    max_short_side = 768
    max_long_side = 2000

    # Return early if no resizing is needed
    if width <= 768 and height <= 768:
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
    resized_width, resized_height, tile_width=512, tile_height=512
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
