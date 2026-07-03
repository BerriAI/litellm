from typing import Optional

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

WEB_SEARCH_TOOL_TYPES: tuple[str, ...] = ("web_search", "web_search_premium")

STR_OBJ_DICT: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])
OBJ_LIST: TypeAdapter[list[object]] = TypeAdapter(list[object])


def is_web_search_request(optional_params: dict) -> bool:
    """True when a Mistral request should route to the Conversations API for web search."""
    params = STR_OBJ_DICT.validate_python(optional_params)
    if params.get("web_search_options") is not None:
        return True
    tools = params.get("tools")
    if isinstance(tools, list):
        return any(
            isinstance(tool, dict) and STR_OBJ_DICT.validate_python(tool).get("type") in WEB_SEARCH_TOOL_TYPES
            for tool in OBJ_LIST.validate_python(tools)
        )
    return False


class MistralModelInfo(BaseLLMModelInfo):
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        if "content-type" not in headers and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return headers

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return api_base or get_secret_str("MISTRAL_API_BASE") or "https://api.mistral.ai"

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("MISTRAL_API_KEY")

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        return model.replace("mistral/", "")

    def get_models(self, api_key: Optional[str] = None, api_base: Optional[str] = None) -> list[str]:
        api_base = self.get_api_base(api_base)
        api_key = self.get_api_key(api_key)
        if api_base is None or api_key is None:
            raise ValueError(
                "MISTRAL_API_BASE or MISTRAL_API_KEY is not set. Set them in the environment or pass them in."
            )
        response = litellm.module_level_client.get(
            url=f"{api_base}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise Exception(
                f"Failed to fetch models from Mistral. Status code: {response.status_code}, Response: {response.text}"
            )
        return [f"mistral/{model['id']}" for model in response.json()["data"]]
