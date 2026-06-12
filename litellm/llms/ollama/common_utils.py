from typing import Any, List, Optional, Union

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
    def get_api_key(api_key=None) -> Optional[str]:
        """Get API key from environment variables or litellm configuration"""
        import os

        import litellm
        from litellm.secret_managers.main import get_secret_str

        return (
            api_key
            or os.environ.get("OLLAMA_API_KEY")
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OLLAMA_API_KEY")
        )

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        from litellm.secret_managers.main import get_secret_str

        # env var OLLAMA_API_BASE or default
        return api_base or get_secret_str("OLLAMA_API_BASE") or "http://localhost:11434"

    @classmethod
    def get_server_api_base(cls, api_base: Optional[str] = None) -> str:
        api_base = cls.get_api_base(api_base).rstrip("/")
        for suffix in (
            "/api/generate",
            "/api/chat",
            "/api/embed",
            "/api/embeddings",
            "/api/show",
            "/api/tags",
        ):
            if api_base.endswith(suffix):
                return api_base[: -len(suffix)]
        return api_base

    def get_models(self, api_key=None, api_base: Optional[str] = None) -> List[str]:
        """
        List all models available on the Ollama server via /api/tags endpoint.
        """

        passed_api_base = api_base
        base = self.get_server_api_base(api_base)
        api_key = (
            self.get_api_key(api_key) if passed_api_base is None or api_key else None
        )
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        names: set[str] = set()
        try:
            resp = httpx.get(f"{base}/api/tags", headers=headers)
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
                    names.add(nm if nm.startswith("ollama/") else f"ollama/{nm}")
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

    @staticmethod
    def _strip_ollama_model_prefix(model: str) -> str:
        if model.startswith("ollama/") or model.startswith("ollama_chat/"):
            return model.split("/", 1)[1]
        return model

    @staticmethod
    def _is_static_ollama_model(model: str) -> bool:
        from litellm import model_cost

        stripped_model = OllamaModelInfo._strip_ollama_model_prefix(model)
        potential_model_names = {
            model,
            stripped_model,
            "ollama/" + stripped_model,
            "ollama_chat/" + stripped_model,
        }
        model_cost_keys = {key.lower() for key in model_cost}
        return any(name.lower() in model_cost_keys for name in potential_model_names)

    @staticmethod
    def _supports_function_calling(ollama_model_info: dict) -> bool:
        _template: str = str(ollama_model_info.get("template", "") or "")
        return "tools" in _template.lower()

    @staticmethod
    def _get_max_tokens(ollama_model_info: dict) -> Optional[int]:
        _model_info: dict = ollama_model_info.get("model_info", {})

        for key, value in _model_info.items():
            if "context_length" in key:
                return value
        return None

    def get_runtime_model_info(
        self,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dict[str, Any]:
        from litellm import module_level_client

        model = self._strip_ollama_model_prefix(model)
        passed_api_base = api_base
        api_base = self.get_server_api_base(api_base)
        api_key = (
            self.get_api_key(api_key) if passed_api_base is None or api_key else None
        )
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        try:
            response = module_level_client.post(
                url=f"{api_base}/api/show",
                json={"name": model},
                headers=headers,
            )
            response.raise_for_status()
        except Exception:
            verbose_logger.debug("OllamaError: Could not get model info.")
            return {
                "key": model,
                "litellm_provider": "ollama",
                "mode": "chat",
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "max_tokens": None,
                "max_input_tokens": None,
                "max_output_tokens": None,
            }

        model_info = response.json()
        max_tokens = self._get_max_tokens(model_info)

        return {
            "key": model,
            "litellm_provider": "ollama",
            "mode": "chat",
            "supports_function_calling": self._supports_function_calling(model_info),
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "max_tokens": max_tokens,
            "max_input_tokens": max_tokens,
            "max_output_tokens": max_tokens,
        }

    def get_model_info(
        self,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        if self._is_static_ollama_model(model):
            return None
        return self.get_runtime_model_info(
            model=model, api_base=api_base, api_key=api_key
        )

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
