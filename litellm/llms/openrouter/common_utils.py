from collections.abc import Mapping, Sequence
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict

import litellm
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

_DEFAULT_API_BASE: Final = "https://openrouter.ai/api/v1"
_CHAT_CATALOG_ROUTE: Final = "models/user"
_CHAT_COMPLETIONS_PATH: Final = "/chat/completions"
_BATCH_ONLY_SUFFIX: Final = ":batch"
_CATALOG_TIMEOUT: Final = httpx.Timeout(10.0)


class OpenRouterException(BaseLLMException):
    pass


class _CatalogModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str


class _Catalog(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_CatalogModel, ...]


def _chat_catalog_url(api_base: str) -> str:
    return join_api_base(api_base.rstrip("/").removesuffix(_CHAT_COMPLETIONS_PATH), _CHAT_CATALOG_ROUTE)


class OpenRouterModelInfo(BaseLLMModelInfo):
    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return (
            api_key
            or litellm.api_key
            or litellm.openrouter_key
            or get_secret_str("OPENROUTER_API_KEY")
            or get_secret_str("OR_API_KEY")
        )

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE") or _DEFAULT_API_BASE

    @staticmethod
    def get_base_model(model: str) -> str:
        return model.removeprefix("openrouter/")

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseLLMModelInfo.validate_environment returns a dict
        resolved_key: Final = self.get_api_key(api_key)
        if resolved_key is None:
            return {**headers}
        return {**headers, "Authorization": f"Bearer {resolved_key}"}

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: matches get_models' list[str] contract shared by every provider override
        resolved_key: Final = self.get_api_key(api_key)
        if resolved_key is None:
            raise ValueError("Set OPENROUTER_API_KEY or the deployment's api_key to list OpenRouter models")
        headers: Final = self.validate_environment(
            headers={}, model="", messages=(), optional_params={}, litellm_params={}, api_key=resolved_key
        )
        response: Final = litellm.module_level_client.get(  # pyright: ignore[reportUnknownMemberType]  # legacy HTTP wrapper has untyped query parameters
            url=_chat_catalog_url(self.get_api_base(api_base)), headers=headers, timeout=_CATALOG_TIMEOUT
        )
        response.raise_for_status()
        return [
            f"openrouter/{model.id}"
            for model in _Catalog.model_validate_json(response.content).data
            if not model.id.endswith(_BATCH_ONLY_SUFFIX)
        ]
