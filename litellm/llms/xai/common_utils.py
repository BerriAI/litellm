from typing import List, Optional

import httpx

import litellm
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues


class XAIModelInfo(BaseLLMModelInfo):
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
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        # Ensure Content-Type is set to application/json
        if "content-type" not in headers and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        return headers

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return api_base or get_secret_str("XAI_API_BASE") or "https://api.x.ai"

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("XAI_API_KEY")

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        return model.replace("xai/", "")

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        api_base = self.get_api_base(api_base)
        api_key = self.get_api_key(api_key)
        if api_base is None or api_key is None:
            raise ValueError(
                "XAI_API_BASE or XAI_API_KEY is not set. Please set the environment variable, to query XAI's `/models` endpoint."
            )
        response = litellm.module_level_client.get(
            url=f"{api_base}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise Exception(
                f"Failed to fetch models from XAI. Status code: {response.status_code}, Response: {response.text}"
            )

        models = response.json()["data"]

        litellm_model_names = []
        for model in models:
            stripped_model_name = model["id"]
            litellm_model_name = "xai/" + stripped_model_name
            litellm_model_names.append(litellm_model_name)
        return litellm_model_names
