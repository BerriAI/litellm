from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ..base_llm.base_utils import BaseLLMModelInfo
from ..base_llm.chat.transformation import BaseLLMException


class TopazException(BaseLLMException):
    pass


class TopazModelInfo(BaseLLMModelInfo):
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        if api_key is None:
            raise ValueError("API key is required for Topaz image variations. Set via `TOPAZ_API_KEY` or `api_key=..`")
        return {
            # "Content-Type": "multipart/form-data",
            "Accept": "image/jpeg",
            "X-API-Key": api_key,
        }

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        return [
            "topaz/Standard V2",
            "topaz/Low Resolution V2",
            "topaz/CGI",
            "topaz/High Resolution V2",
            "topaz/Text Refine",
        ]

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("TOPAZ_API_KEY")

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return api_base or get_secret_str("TOPAZ_API_BASE") or "https://api.topazlabs.com"

    @staticmethod
    def get_base_model(model: str) -> str:
        return model
