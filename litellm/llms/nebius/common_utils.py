from typing import List, Optional, Union

import httpx

import litellm
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues


class NebiusError(BaseLLMException):
    pass


class NebiusModelInfo(BaseLLMModelInfo):
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """Nebius AI Studio sends api key in query params"""
        return headers

    @property
    def api_version(self) -> str:
        return "v1beta"

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return (
            api_base
            or get_secret_str("NEBIUS_API_BASE")
            or "https://api.studio.nebius.ai"
        )

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or (get_secret_str("NEBIUS_API_KEY"))

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        return model.replace("nebius/", "")

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        api_base = NebiusModelInfo.get_api_base(api_base)
        api_key = NebiusModelInfo.get_api_key(api_key)
        endpoint = f"/{self.api_version}/models"
        if api_base is None or api_key is None:
            raise ValueError(
                "NEBIUS_API_BASE or NEBIUS_API_KEY is not set. Please set the environment variable, to query Nebius's `/models` endpoint."
            )

        response = litellm.module_level_client.get(
            url=f"{api_base}{endpoint}?key={api_key}",
        )

        if response.status_code != 200:
            raise ValueError(
                f"Failed to fetch models from Nebius. Status code: {response.status_code}, Response: {response.json()}"
            )

        models = response.json()["models"]

        litellm_model_names = []
        for model in models:
            stripped_model_name = model["name"].strip("models/")
            litellm_model_name = "nebius/" + stripped_model_name
            litellm_model_names.append(litellm_model_name)
        return litellm_model_names

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return NebiusError(
            status_code=status_code, message=error_message, headers=headers
        )
