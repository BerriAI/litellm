from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

import litellm
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)


class _ModelList(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_Model, ...]


class NanoGPTModelInfo(BaseLLMModelInfo):
    def __init__(self, client: HTTPHandler | None = None) -> None:
        self._client = client

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return api_base or get_secret_str("NANOGPT_API_BASE") or "https://nano-gpt.com/api/v1"

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("NANOGPT_API_KEY")

    @staticmethod
    def get_base_model(model: str) -> str:
        return model.removeprefix("nano-gpt/")

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: BaseLLMModelInfo requires a dict return
        resolved_key: Final = self.get_api_key(api_key)
        auth: Final = (("Authorization", f"Bearer {resolved_key}"),) if resolved_key else ()
        return dict((*headers.items(), *auth))  # mutable-ok: BaseLLMModelInfo requires a dict return

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: BaseLLMModelInfo requires a list return
        resolved_key: Final = self.get_api_key(api_key)
        client: Final = self._client or litellm.module_level_client
        response: Final = client.client.get(
            url=f"{self.get_api_base(api_base).rstrip('/')}/models",
            headers=MappingProxyType({"Authorization": f"Bearer {resolved_key}"}) if resolved_key else None,
            timeout=10.0,
            follow_redirects=False,
        )
        response.raise_for_status()
        catalog: Final = _ModelList.model_validate_json(response.content)
        unique_ids: Final = tuple(dict.fromkeys(f"nano-gpt/{model.id}" for model in catalog.data))
        return list(unique_ids)  # mutable-ok: BaseLLMModelInfo requires a list return
