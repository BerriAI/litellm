import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ImageObject

TOGETHER_AI_DEFAULT_API_BASE = "https://api.together.xyz/v1"


class TogetherAIException(BaseLLMException):
    pass


def resolve_together_ai_api_key(api_key: str | None) -> str | None:
    return (
        api_key
        or litellm.api_key
        or get_secret_str("TOGETHER_API_KEY")
        or get_secret_str("TOGETHER_AI_API_KEY")
        or get_secret_str("TOGETHERAI_API_KEY")
        or get_secret_str("TOGETHER_AI_TOKEN")
    )


def get_together_ai_images_generations_url(api_base: str | None) -> str:
    base = (api_base or get_secret_str("TOGETHER_AI_API_BASE") or TOGETHER_AI_DEFAULT_API_BASE).rstrip("/")
    if base.endswith("/images/generations"):
        return base
    return f"{base}/images/generations"


def parse_openai_size_to_width_height(size: str) -> tuple[int, int] | None:
    parts = size.lower().split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return None
    return int(parts[0]), int(parts[1])


def map_openai_image_param_to_together_ai(key: str, value: object) -> tuple[tuple[str, object], ...]:
    if key == "size" and isinstance(value, str):
        dimensions = parse_openai_size_to_width_height(value)
        if dimensions is None:
            return ()
        width, height = dimensions
        return (("width", width), ("height", height))
    if key == "response_format" and value == "b64_json":
        return (("response_format", "base64"),)
    return ((key, value),)


def together_ai_image_data_to_image_objects(response_json: dict) -> list[ImageObject]:
    data = response_json.get("data")
    if not isinstance(data, list):
        return []
    return [
        ImageObject(
            url=item.get("url"),
            b64_json=item.get("b64_json"),
            revised_prompt=None,
        )
        for item in data
        if isinstance(item, dict)
    ]
