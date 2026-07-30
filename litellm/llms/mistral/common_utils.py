from typing import Mapping, Sequence

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

WEB_SEARCH_TOOL_TYPES: tuple[str, ...] = ("web_search", "web_search_premium")

STR_OBJ_DICT: TypeAdapter[Mapping[str, object]] = TypeAdapter(Mapping[str, object])
OBJ_LIST: TypeAdapter[Sequence[object]] = TypeAdapter(Sequence[object])


def is_web_search_request(optional_params: Mapping[str, object]) -> bool:
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
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: BaseLLMModelInfo contract returns the headers dict
        auth = {"Authorization": f"Bearer {api_key}"} if api_key is not None else {}
        content_type = (
            {} if "content-type" in headers or "Content-Type" in headers else {"Content-Type": "application/json"}
        )
        return {**headers, **auth, **content_type}

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return api_base or get_secret_str("MISTRAL_API_BASE") or "https://api.mistral.ai"

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("MISTRAL_API_KEY")

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model.replace("mistral/", "")

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: BaseLLMModelInfo contract returns List[str]
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
