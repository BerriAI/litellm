import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Final, TypeAlias

import httpx
from pydantic import BeforeValidator, ConfigDict

import litellm
from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import add_path_to_api_base

MODEL_INFO_REFRESH_SECONDS: Final = 300
MODEL_INFO_REFRESH_CONCURRENCY: Final = 8
MODEL_INFO_DISCOVERY_PROVIDERS: Final = frozenset({"hosted_vllm", "openai", "text-completion-openai", "openai_like"})
_EMPTY_LIMITS: Final[Mapping[str, int]] = MappingProxyType({})


def _positive_limit(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


_TokenLimit: TypeAlias = Annotated[int | None, BeforeValidator(_positive_limit)]


class _ModelCard(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    max_model_len: _TokenLimit = None
    context_length: _TokenLimit = None
    max_input_tokens: _TokenLimit = None
    max_output_tokens: _TokenLimit = None

    def token_limits(self) -> Mapping[str, int]:
        context: Final = self.max_model_len or self.context_length
        input_limit: Final = self.max_input_tokens or context
        output_limit: Final = self.max_output_tokens or context
        return MappingProxyType(
            {
                key: value
                for key, value in (
                    ("max_tokens", context),
                    ("max_input_tokens", min(input_limit, context) if input_limit and context else input_limit),
                    ("max_output_tokens", min(output_limit, context) if output_limit and context else output_limit),
                )
                if value is not None
            }
        )


class _ModelList(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_ModelCard, ...] = ()


async def get_openai_compatible_model_info(
    *,
    model: str,
    api_base: str,
    headers: Mapping[str, str],
    client: AsyncHTTPHandler,
    cache: InMemoryCache,
) -> Mapping[str, int]:
    url: Final = add_path_to_api_base(api_base, "/v1/models")
    cache_key: Final = (
        "upstream_model_info:" + hashlib.sha256(json.dumps((url, sorted(headers.items()))).encode()).hexdigest()
    )
    cached: Final[object] = cache.get_cache(cache_key)
    if isinstance(cached, _ModelList):
        return next((card.token_limits() for card in cached.data if card.id == model), _EMPTY_LIMITS)

    try:
        response: Final = await client.get(
            url=url,
            headers=dict(headers),
            timeout=httpx.Timeout(5.0),
            follow_redirects=False,
            max_response_bytes=2 * 1024 * 1024,
        )
        response.raise_for_status()
        models: Final = _ModelList.model_validate_json(response.content)
    except Exception:  # noqa: BLE001  # optional upstream metadata must not interrupt proxy refresh
        verbose_logger.debug("Could not discover upstream model token limits")
        cache.set_cache(cache_key, _ModelList(), ttl=60)
        return _EMPTY_LIMITS

    cache.set_cache(cache_key, models, ttl=MODEL_INFO_REFRESH_SECONDS)
    return next((card.token_limits() for card in models.data if card.id == model), _EMPTY_LIMITS)


def list_openai_compatible_model_ids(
    *, models_url: str, api_key: str | None
) -> list[str]:  # mutable-ok: matches get_models' list[str] contract shared by every provider override
    """
    The model ids an OpenAI-compatible server lists at `models_url`.
    Raises on a non-2xx answer so wildcard discovery logs the failure instead of showing an empty list as success.
    """
    response: Final = litellm.module_level_client.get(
        url=models_url,
        headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
        follow_redirects=False,
    )
    response.raise_for_status()
    return [card.id for card in _ModelList.model_validate_json(response.content).data]


class CustomOpenAIModelInfo(BaseLLMModelInfo):
    """
    Lists the models behind a `custom_openai/*` wildcard by asking the server chat calls go to, with the key chat
    calls send, so `/v1/models` and inference can't disagree about where that server is or how to reach it.
    """

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        from litellm.responses.bridge_check import resolve_openai_api_base

        return resolve_openai_api_base(api_base)

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or litellm.api_key or litellm.openai_key or get_secret_str("OPENAI_API_KEY")

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: mirrors override contract
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],  # mutable-ok: mirrors override contract
        litellm_params: dict[str, object],  # mutable-ok: mirrors override contract
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: mirrors override contract
        return headers

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        resolved_api_base: Final = self.get_api_base(api_base)
        if resolved_api_base is None:
            raise ValueError("custom_openai needs an api_base to list models from")
        # Chat calls go to `{api_base}/chat/completions` (the OpenAI client's base_url contract), so the list lives at
        # `{api_base}/models`; adding `/v1` here would list models from a path chat never calls.
        return list_openai_compatible_model_ids(
            models_url=join_api_base(resolved_api_base, "models"), api_key=self.get_api_key(api_key)
        )
