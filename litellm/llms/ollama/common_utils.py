from typing import List, Optional, Union

import httpx

from litellm import verbose_logger
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


from litellm.llms.base_llm.base_utils import BaseLLMModelInfo


class OllamaModelInfo(BaseLLMModelInfo):
    """
    Dynamic model listing for Ollama server.
    Fetches /api/models and /api/tags, then for each tag also /api/models?tag=...
    Returns the union of all model names.
    """

    @staticmethod
    def get_api_key(api_key=None) -> None:
        return None  # Ollama does not use an API key by default

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        from litellm.secret_managers.main import get_secret_str

        # env var OLLAMA_API_BASE or default
        return api_base or get_secret_str("OLLAMA_API_BASE") or "http://localhost:11434"

    def get_models(self, api_key=None, api_base: Optional[str] = None) -> List[str]:
        """
        List all models available on the Ollama server via /api/tags endpoint.
        """

        base = self.get_api_base(api_base)
        names: set[str] = set()
        try:
            resp = httpx.get(f"{base}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            # Expecting a dict with a 'models' list
            models_list = []
            if (
                isinstance(data, dict)
                and "models" in data
                and isinstance(data["models"], list)
            ):
                models_list = data["models"]
            elif isinstance(data, list):
                models_list = data
            # Extract model names
            for entry in models_list:
                if not isinstance(entry, dict):
                    continue
                nm = entry.get("name") or entry.get("model")
                if isinstance(nm, str):
                    names.add(nm)
        except Exception as e:
            verbose_logger.warning(f"Error retrieving ollama tag endpoint: {e}")
            # If tags endpoint fails, fall back to static list
            try:
                from litellm import models_by_provider

                static = models_by_provider.get("ollama", []) or []
                return [f"ollama/{m}" for m in static]
            except Exception as e1:
                verbose_logger.warning(
                    f"Error retrieving static ollama models as fallback: {e1}"
                )
                return []
        # assemble full model names
        result = sorted(names)
        return result

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key=None,
        api_base=None,
    ) -> dict:
        """
        No-op environment validation for Ollama.
        """
        return {}

    @staticmethod
    def get_base_model(model: str) -> str:
        """
        Return the base model name for Ollama (no-op).
        """
        return model
