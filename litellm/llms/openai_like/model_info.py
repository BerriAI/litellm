import hashlib
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Final, TypeAlias

import httpx
from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.proxy.model_inventory import SupplierInventoryUnavailable, SupplierModelInventory
from litellm.types.proxy.model_metadata import GatewayModelMetadata
from litellm.utils import _add_path_to_api_base  # pyright: ignore[reportPrivateUsage]  # shared provider URL helper

MODEL_INFO_REFRESH_SECONDS: Final = 300
MODEL_INFO_REFRESH_CONCURRENCY: Final = 8
MODEL_INFO_DISCOVERY_PROVIDERS: Final = frozenset(
    {"hosted_vllm", "openai", "text-completion-openai", "openai_like", "openrouter", "vercel_ai_gateway"}
)
_EMPTY_LIMITS: Final[Mapping[str, object]] = MappingProxyType({})


def _positive_limit(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


_TokenLimit: TypeAlias = Annotated[int | None, BeforeValidator(_positive_limit)]


class _Modalities(BaseModel):
    input: Sequence[str] | None = None
    output: Sequence[str] | None = None


class _Architecture(BaseModel):
    input_modalities: Sequence[str] | None = None
    output_modalities: Sequence[str] | None = None


class _TopProvider(BaseModel):
    context_length: _TokenLimit = None
    max_completion_tokens: _TokenLimit = None


class _Reasoning(BaseModel):
    supported_efforts: Sequence[str] | None = None
    default_effort: str | None = None


class _ReasoningOption(BaseModel):
    type: str
    values: Sequence[str] | None = None


class _ModelCard(GatewayModelMetadata):
    id: str
    type: str | None = None
    mode: str | None = None
    max_model_len: _TokenLimit = None
    context_length: _TokenLimit = None
    max_tokens: _TokenLimit = None
    supported_parameters: Sequence[str] | None = None
    modalities: _Modalities | None = None
    architecture: _Architecture | None = None
    top_provider: _TopProvider | None = None
    reasoning: _Reasoning | None = None
    reasoning_options: tuple[_ReasoningOption, ...] = ()

    def token_limits(self, provider: str) -> Mapping[str, object]:
        context: Final = self.context_window or self.max_model_len or self.context_length
        parameters: Final = self.supported_parameters
        supported_tools: Final = "tools" in parameters if parameters is not None else None
        supported_reasoning: Final = (
            "reasoning" in parameters or "reasoning_effort" in parameters if parameters is not None else None
        )
        metadata: Final = GatewayModelMetadata(
            context_window=(
                self.top_provider.context_length or context
                if provider == "openrouter" and self.top_provider is not None
                else context
            ),
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=(
                self.max_output_tokens
                or (self.max_tokens if provider == "vercel_ai_gateway" and self.type in (None, "language") else None)
                or (
                    self.top_provider.max_completion_tokens
                    if provider == "openrouter" and self.top_provider is not None
                    else None
                )
            ),
            supports_function_calling=self.supports_function_calling
            if self.supports_function_calling is not None
            else supported_tools,
            supports_parallel_function_calling=self.supports_parallel_function_calling,
            supports_reasoning=self.supports_reasoning if self.supports_reasoning is not None else supported_reasoning,
            reasoning_effort_levels=(
                self.reasoning_effort_levels
                if self.reasoning_effort_levels is not None
                else (
                    self.reasoning.supported_efforts
                    if provider == "openrouter" and self.reasoning is not None
                    else next((option.values for option in self.reasoning_options if option.type == "effort"), None)
                    if provider == "vercel_ai_gateway"
                    else None
                )
            ),
            default_reasoning_effort=self.default_reasoning_effort
            or (self.reasoning.default_effort if provider == "openrouter" and self.reasoning is not None else None),
            supported_endpoints=self.supported_endpoints,
            request_defaults=self.request_defaults,
            supported_modalities=self.supported_modalities
            if self.supported_modalities is not None
            else (
                self.architecture.input_modalities
                if provider == "openrouter" and self.architecture is not None
                else self.modalities.input
                if provider == "vercel_ai_gateway" and self.modalities is not None
                else None
            ),
            supported_output_modalities=self.supported_output_modalities
            if self.supported_output_modalities is not None
            else (
                self.architecture.output_modalities
                if provider == "openrouter" and self.architecture is not None
                else self.modalities.output
                if provider == "vercel_ai_gateway" and self.modalities is not None
                else None
            ),
        )
        mode: Final = (
            self.mode
            if self.mode in ("chat", "completion", "embedding")
            else (
                "embedding"
                if provider == "vercel_ai_gateway" and self.type == "embedding"
                else "chat"
                if provider == "vercel_ai_gateway" and self.type == "language"
                else None
            )
        )
        return MappingProxyType(
            {**metadata.model_dump(mode="json", exclude_none=True), **({"mode": mode} if mode is not None else {})}
        )


class _ModelList(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_ModelCard, ...]
    error: object | None = None


async def get_openai_compatible_model_info(
    *,
    model: str,
    api_base: str,
    provider: str = "openai",
    headers: Mapping[str, str],
    client: AsyncHTTPHandler,
    cache: InMemoryCache,
) -> Mapping[str, object]:
    inventory: Final = await get_openai_compatible_model_inventory(
        api_base=api_base, provider=provider, headers=headers, client=client, cache=cache
    )
    return (
        inventory.models.get(model, _EMPTY_LIMITS) if isinstance(inventory, SupplierModelInventory) else _EMPTY_LIMITS
    )


async def get_openai_compatible_model_inventory(
    *,
    api_base: str,
    provider: str = "openai",
    headers: Mapping[str, str],
    client: AsyncHTTPHandler,
    cache: InMemoryCache,
    force_refresh: bool = False,
) -> SupplierModelInventory | SupplierInventoryUnavailable:
    url: Final = _add_path_to_api_base(api_base, "/v1/models")
    cache_key: Final = (
        "upstream_model_info:"
        + hashlib.sha256(json.dumps((provider, url, sorted(headers.items()))).encode()).hexdigest()
    )
    cached: Final[object] = cache.get_cache(cache_key)
    if not force_refresh and isinstance(cached, (SupplierModelInventory, SupplierInventoryUnavailable)):
        return cached

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
        ids: Final = tuple(card.id for card in models.data)
        if models.error is not None or any(not model_id for model_id in ids) or len(frozenset(ids)) != len(ids):
            unavailable: Final = SupplierInventoryUnavailable("malformed")
            cache.set_cache(cache_key, unavailable, ttl=60)
            return unavailable
    except httpx.HTTPStatusError:
        http_failure: Final = SupplierInventoryUnavailable("http")
        cache.set_cache(cache_key, http_failure, ttl=60)
        return http_failure
    except ValidationError:
        malformed: Final = SupplierInventoryUnavailable("malformed")
        cache.set_cache(cache_key, malformed, ttl=60)
        return malformed
    except Exception:  # noqa: BLE001  # optional upstream discovery must not interrupt proxy refresh
        transport: Final = SupplierInventoryUnavailable("transport")
        cache.set_cache(cache_key, transport, ttl=60)
        return transport

    inventory: Final = SupplierModelInventory(
        MappingProxyType({card.id: card.token_limits(provider) for card in models.data})
    )
    cache.set_cache(cache_key, inventory, ttl=MODEL_INFO_REFRESH_SECONDS)
    return inventory
