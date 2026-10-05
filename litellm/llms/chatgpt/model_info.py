import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai_like.model_info import MODEL_INFO_REFRESH_SECONDS
from litellm.types.proxy.model_metadata import GatewayModelMetadata

from .authenticator import Authenticator
from .common_utils import get_chatgpt_default_headers

_CATALOG_CLIENT_VERSION: Final = "0.159.3"
_EMPTY_METADATA: Final[Mapping[str, object]] = MappingProxyType({})


class _ReasoningLevel(BaseModel):
    effort: str


class _ChatGPTModel(GatewayModelMetadata):
    slug: str
    max_context_window: int | None = None
    default_reasoning_level: str | None = None
    supported_reasoning_levels: tuple[_ReasoningLevel, ...] | None = None
    input_modalities: Sequence[str] | None = None
    supports_parallel_tool_calls: bool | None = None
    multi_agent_reasoning_effort: str | None = None

    @field_validator("max_context_window", mode="before")
    @classmethod
    def maximum_context(cls, value: object) -> int | None:
        return GatewayModelMetadata.model_validate({"context_window": value}).context_window

    def metadata(self) -> Mapping[str, object]:
        maximum: Final = GatewayModelMetadata(context_window=self.max_context_window).context_window
        native_efforts: Final = (
            self.reasoning_effort_levels
            if self.reasoning_effort_levels is not None
            else tuple(level.effort for level in self.supported_reasoning_levels)
            if self.supported_reasoning_levels is not None
            else None
        )
        # Codex Ultra is a multi-agent UI choice, not an API reasoning effort.
        api_efforts: Final = (
            tuple("disabled" if effort == "persistent" else effort for effort in native_efforts if effort != "ultra")
            if native_efforts is not None
            else None
        )
        native_default: Final = self.default_reasoning_effort or self.default_reasoning_level
        default_effort: Final = (
            self.multi_agent_reasoning_effort
            if native_default == "ultra" and self.multi_agent_reasoning_effort in (api_efforts or ())
            else "max"
            if native_default == "ultra" and "max" in (api_efforts or ())
            else api_efforts[-1]
            if native_default == "ultra" and api_efforts
            else "disabled"
            if native_default == "persistent"
            else native_default
        )
        normalized: Final = GatewayModelMetadata(
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            supports_function_calling=self.supports_function_calling,
            supports_parallel_function_calling=(
                self.supports_parallel_function_calling
                if self.supports_parallel_function_calling is not None
                else self.supports_parallel_tool_calls
            ),
            supports_reasoning=(
                self.supports_reasoning
                if self.supports_reasoning is not None
                else bool(self.supported_reasoning_levels)
                if self.supported_reasoning_levels is not None
                else None
            ),
            reasoning_effort_levels=api_efforts,
            default_reasoning_effort=default_effort,
            supported_endpoints=("/responses",),
            supported_modalities=(
                self.supported_modalities if self.supported_modalities is not None else self.input_modalities
            ),
            supported_output_modalities=self.supported_output_modalities,
        )
        native_windows: Final = {
            "codex_context_window": self.context_window,
            "codex_max_context_window": maximum,
        }
        return MappingProxyType(
            {
                **normalized.model_dump(exclude_none=True),
                **{key: value for key, value in native_windows.items() if value is not None},
            }
        )


class _ChatGPTCatalog(BaseModel):
    model_config = ConfigDict(frozen=True)

    models: tuple[_ChatGPTModel, ...] = ()


async def get_chatgpt_model_info(
    *,
    model: str,
    client: AsyncHTTPHandler,
    cache: InMemoryCache,
    api_base: str | None = None,
    authenticator: Authenticator | None = None,
) -> Mapping[str, object]:
    auth: Final = authenticator if authenticator is not None else Authenticator()
    try:
        token: Final = await asyncio.to_thread(auth.get_access_token, allow_device_login=False)
        account: Final = await asyncio.to_thread(auth.get_account_id)
        url: Final = f"{(api_base or auth.get_api_base()).rstrip('/')}/models?client_version={_CATALOG_CLIENT_VERSION}"
        headers: Final = {
            **get_chatgpt_default_headers(token, account),
            "accept": "application/json",
        }
        cache_key: Final = (
            "chatgpt_model_info:" + hashlib.sha256(json.dumps((url, sorted(headers.items()))).encode()).hexdigest()
        )
        cached: Final[object] = cache.get_cache(cache_key)
        if isinstance(cached, _ChatGPTCatalog):
            return next((card.metadata() for card in cached.models if card.slug == model), _EMPTY_METADATA)
        response: Final = await client.get(
            url=url,
            headers=headers,
            timeout=httpx.Timeout(5.0),
            follow_redirects=False,
            max_response_bytes=2 * 1024 * 1024,
        )
        response.raise_for_status()
        catalog: Final = _ChatGPTCatalog.model_validate_json(response.content)
        cache.set_cache(cache_key, catalog, ttl=MODEL_INFO_REFRESH_SECONDS)
        return next((card.metadata() for card in catalog.models if card.slug == model), _EMPTY_METADATA)
    except Exception:  # noqa: BLE001  # optional metadata discovery must not interrupt proxy startup
        verbose_logger.debug("Could not discover the active ChatGPT account model metadata")
        return _EMPTY_METADATA
