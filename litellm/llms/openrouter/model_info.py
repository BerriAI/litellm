from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from urllib.parse import quote

import httpx
from pydantic import ConfigDict, ValidationError, ValidationInfo, field_validator

from litellm.llms.openai_like.model_info import MODEL_INFO_REFRESH_SECONDS, model_inventory_cache_key
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openrouter import OpenRouterProviderPolicy
from litellm.types.proxy.model_inventory import (
    ModelInventoryCache,
    ModelInventoryHTTPClient,
    SupplierInventoryUnavailable,
    SupplierModelEndpoint,
    SupplierModelInventory,
)
from litellm.types.router import ModelMetadata, common_model_metadata
from litellm.utils import add_path_to_api_base


class _Endpoint(ModelMetadata):
    tag: str
    context_length: int | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    supported_parameters: tuple[str, ...] | None = None

    @field_validator("context_length", "max_prompt_tokens", "max_completion_tokens", mode="before")
    @classmethod
    def known_limit(cls, value: object) -> object:
        if value is None or type(value) is int and value >= 0:
            return value
        raise ValueError("Endpoint limits must be nonnegative integers or null")

    @field_validator("*", mode="before")
    @classmethod
    def strict_metadata(cls, value: object, info: ValidationInfo) -> object:
        return ModelMetadata.validate_supplier_field(info.field_name, value)

    def normalized(self) -> SupplierModelEndpoint:
        parameters: Final = self.supported_parameters
        return SupplierModelEndpoint(
            self.tag,
            ModelMetadata.model_validate(
                {
                    **self.model_dump(exclude_none=True),
                    "context_window": self.context_length,
                    "max_input_tokens": self.max_prompt_tokens,
                    "max_output_tokens": self.max_completion_tokens,
                    "supports_function_calling": self.supports_function_calling
                    if self.supports_function_calling is not None
                    else "tools" in parameters
                    if parameters is not None
                    else None,
                    "supports_parallel_function_calling": self.supports_parallel_function_calling
                    if self.supports_parallel_function_calling is not None
                    else True
                    if parameters is not None and "parallel_tool_calls" in parameters
                    else None,
                    "supports_reasoning": self.supports_reasoning
                    if self.supports_reasoning is not None
                    else bool({"reasoning", "reasoning_effort"}.intersection(parameters))
                    if parameters is not None
                    else None,
                }
            ),
            parameters,
        )


class _EndpointData(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    endpoints: tuple[_Endpoint, ...]


class _EndpointResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    data: _EndpointData
    error: object | None = None


async def _fetch_endpoints(
    model: str,
    *,
    api_base: str,
    headers: Mapping[str, str],
    client: ModelInventoryHTTPClient,
    cache: ModelInventoryCache,
    force_refresh: bool,
) -> tuple[SupplierModelEndpoint, ...] | SupplierInventoryUnavailable:
    if len(model.split("/")) != 2 or any(part in ("", ".", "..") for part in model.split("/")):
        return SupplierInventoryUnavailable("malformed")
    url: Final = add_path_to_api_base(api_base, f"/v1/models/{quote(model, safe='/')}/endpoints")
    cache_key: Final = model_inventory_cache_key("openrouter-endpoints", url, headers)
    cached: Final[object] = cache.get_cache(cache_key)
    if not force_refresh and isinstance(cached, (_EndpointData, SupplierInventoryUnavailable)):
        return (
            tuple(endpoint.normalized() for endpoint in cached.endpoints)
            if isinstance(cached, _EndpointData)
            else cached
        )
    try:
        response: Final = await client.get(
            url=url,
            headers=dict(headers),
            timeout=httpx.Timeout(5.0),
            follow_redirects=False,
            max_response_bytes=2 * 1024 * 1024,
        )
        response.raise_for_status()
        parsed: Final = _EndpointResponse.model_validate_json(response.content)
        data: Final = parsed.data
        if (
            parsed.error is not None
            or data.id != model
            or any(not endpoint.tag.strip() or endpoint.tag != endpoint.tag.strip() for endpoint in data.endpoints)
        ):
            raise ValueError("Endpoint inventory must identify the requested model and provider slugs")
    except httpx.HTTPStatusError:
        unavailable: Final = SupplierInventoryUnavailable("http")
        cache.set_cache(cache_key, unavailable, ttl=60)
        return unavailable
    except (ValidationError, ValueError):
        malformed: Final = SupplierInventoryUnavailable("malformed")
        cache.set_cache(cache_key, malformed, ttl=60)
        return malformed
    except Exception:  # noqa: BLE001  # optional metadata must not interrupt the last-valid serving generation
        transport: Final = SupplierInventoryUnavailable("transport")
        cache.set_cache(cache_key, transport, ttl=60)
        return transport
    cache.set_cache(cache_key, data, ttl=MODEL_INFO_REFRESH_SECONDS)
    return tuple(endpoint.normalized() for endpoint in data.endpoints)


async def get_openrouter_endpoint_inventory(
    *,
    inventory: SupplierModelInventory,
    models: tuple[str, ...],
    previous: SupplierModelInventory | None,
    api_base: str,
    headers: Mapping[str, str],
    client: ModelInventoryHTTPClient,
    cache: ModelInventoryCache,
    force_refresh: bool,
) -> SupplierModelInventory:
    selected: Final = tuple(dict.fromkeys(model for model in models if model in inventory.models))
    retained: Final[Mapping[str, tuple[SupplierModelEndpoint, ...]]] = (
        previous.endpoints
        if previous is not None
        and inventory.credential_scope is not None
        and previous.credential_scope == inventory.credential_scope
        else MappingProxyType({})
    )
    results: Final = tuple(
        [
            await _fetch_endpoints(
                model, api_base=api_base, headers=headers, client=client, cache=cache, force_refresh=force_refresh
            )
            for model in selected
        ]
    )
    return SupplierModelInventory(
        inventory.models,
        inventory.credential_scope,
        MappingProxyType(
            {
                model: result if isinstance(result, tuple) else retained.get(model, ())
                for model, result in zip(selected, results)
            }
        ),
    )


def openrouter_endpoint_metadata(
    model_info: Mapping[str, object], endpoints: tuple[SupplierModelEndpoint, ...], policy: OpenRouterProviderPolicy
) -> Mapping[str, object]:
    eligible: Final = tuple(endpoint for endpoint in endpoints if policy.allows_endpoint(endpoint.tag))
    model: Final = ModelMetadata.model_validate(model_info)
    sources: Final = tuple(_endpoint_metadata(model, endpoint) for endpoint in eligible)
    return MappingProxyType(common_model_metadata(sources).model_dump(mode="json", exclude_none=True))


def _endpoint_metadata(model: ModelMetadata, endpoint: SupplierModelEndpoint) -> ModelMetadata:
    supports_effort: Final = (
        endpoint.supported_parameters is not None and "reasoning_effort" in endpoint.supported_parameters
    )
    endpoint_levels: Final = endpoint.metadata.reasoning_effort_levels
    levels: Final = (
        model.reasoning_effort_levels
        if endpoint_levels is None
        else endpoint_levels
        if model.reasoning_effort_levels is None
        else tuple(level for level in model.reasoning_effort_levels if level in endpoint_levels)
    )
    return ModelMetadata.model_validate(
        {
            **endpoint.metadata.model_dump(exclude_none=True),
            "reasoning_effort_levels": levels if supports_effort else None,
            "default_reasoning_effort": (endpoint.metadata.default_reasoning_effort or model.default_reasoning_effort)
            if supports_effort
            else None,
        }
    )
