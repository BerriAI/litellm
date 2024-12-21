from typing import Optional, Union

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.chat.transformation import BaseLLMException


class OllamaError(BaseLLMException):
    def __init__(
        self, status_code: int, message: str, headers: Union[dict, httpx.Headers]
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


def _convert_image(image):
    """
    Convert image to base64 encoded image if not already in base64 format

    If image is already in base64 format AND is a jpeg/png, return it

    If image is not JPEG/PNG, convert it to JPEG base64 format
    """
    import base64
    import io

    try:
        from PIL import Image
    except Exception:
        raise Exception(
            "ollama image conversion failed please run `pip install Pillow`"
        )

    orig = image
    if image.startswith("data:"):
        image = image.split(",")[-1]
    try:
        image_data = Image.open(io.BytesIO(base64.b64decode(image)))
        if image_data.format in ["JPEG", "PNG"]:
            return image
    except Exception:
        return orig
    jpeg_image = io.BytesIO()
    image_data.convert("RGB").save(jpeg_image, "JPEG")
    jpeg_image.seek(0)
    return base64.b64encode(jpeg_image.getvalue()).decode("utf-8")


def process_response_format(response_format: dict) -> Optional[str]:
    """
    Process OpenAI-style response format specification into Ollama API format
    string

    Args:
        response_format (dict): OpenAI-style response format specification

    Returns:
        str: Format string for Ollama API
    """
    format_type = response_format.get("type")
    if format_type == "json_object":
        return "json"
    elif format_type == "json_schema":
        schema = response_format.get("json_schema", {}).get("schema")
        if not schema:
            raise ValueError("Invalid JSON schema format")
        return schema
    else:
        verbose_logger.warning(f"Unsupported response format type: {format_type}")
