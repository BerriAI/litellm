"""Supplier-specific inventory dispatch shared by router and selected-offering discovery."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # native HTTP/cache protocol boundary

from pydantic import TypeAdapter

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.chatgpt.model_info import (
    chatgpt_model_inventory_api_base,
    chatgpt_model_inventory_identity,
    get_chatgpt_model_inventory,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai_like.model_info import MODEL_INFO_DISCOVERY_PROVIDERS, get_openai_compatible_model_inventory
from litellm.llms.openrouter.model_info import get_openrouter_endpoint_inventory, openrouter_endpoint_metadata
from litellm.types.llms.openrouter import OpenRouterProviderPolicy
from litellm.types.proxy.model_inventory import (
    ModelInventoryCache,
    ModelInventoryHTTPClient,
    SupplierInventoryUnavailable,
    SupplierModelInventory,
)
from litellm.types.router import LiteLLM_Params

_EMPTY_METADATA: Final[Mapping[str, object]] = MappingProxyType({})
_HEADERS: Final = TypeAdapter(Mapping[str, str])


def model_inventory_api_base(provider: str, api_base: str | None) -> str | None:
    if provider == "chatgpt":
        return chatgpt_model_inventory_api_base(api_base)
    if provider == "openrouter":
        return litellm.OpenrouterConfig.get_api_base(api_base)
    if provider == "vercel_ai_gateway":
        return litellm.VercelAIGatewayConfig.get_api_base(api_base)
    return api_base


def model_inventory_identity(provider: str) -> str | None:
    return chatgpt_model_inventory_identity() if provider == "chatgpt" else None


async def get_provider_model_inventory(
    *,
    provider: str,
    api_base: str | None,
    headers: Mapping[str, str],
    client: AsyncHTTPHandler,
    cache: InMemoryCache,
    force_refresh: bool = False,
    endpoint_models: tuple[str, ...] = (),
    previous: SupplierModelInventory | None = None,
) -> SupplierModelInventory | SupplierInventoryUnavailable:
    if provider == "chatgpt":
        return await get_chatgpt_model_inventory(
            api_base=api_base,
            client=client,
            cache=cache,
            force_refresh=force_refresh,
        )
    resolved_base: Final = model_inventory_api_base(provider, api_base)
    if resolved_base is None:
        return SupplierInventoryUnavailable("malformed")
    inventory: Final = await get_openai_compatible_model_inventory(
        provider=provider,
        api_base=resolved_base,
        headers=headers,
        client=client,
        cache=cache,
        force_refresh=force_refresh,
    )
    if provider != "openrouter" or not isinstance(inventory, SupplierModelInventory) or not endpoint_models:
        return inventory
    return await get_openrouter_endpoint_inventory(
        inventory=inventory,
        models=endpoint_models,
        previous=previous,
        api_base=resolved_base,
        headers=headers,
        client=cast(ModelInventoryHTTPClient, client),  # cast-ok: native handler implements bounded inventory GET
        cache=cast(ModelInventoryCache, cache),  # cast-ok: native in-memory cache implements inventory protocol
        force_refresh=force_refresh,
    )


def get_inventory_model_metadata(
    inventory: SupplierModelInventory, model: str, policy: OpenRouterProviderPolicy | None = None
) -> Mapping[str, object]:
    metadata: Final = inventory.models.get(model, _EMPTY_METADATA)
    return openrouter_endpoint_metadata(metadata, inventory.endpoints.get(model, ()), policy) if policy else metadata


async def get_deployment_model_metadata(
    *,
    params: LiteLLM_Params,
    client: AsyncHTTPHandler,
    cache: InMemoryCache,
) -> Mapping[str, object]:
    metadata: Final = TypeAdapter(Mapping[str, object]).validate_python(params.model_dump())
    if metadata.get("use_clientside_credentials") or "*" in params.model:
        return _EMPTY_METADATA
    connection: Final = (
        (params.model.removeprefix("chatgpt/"), "chatgpt", None, params.api_base)
        if params.model.startswith("chatgpt/")
        else litellm.get_llm_provider(model=params.model, litellm_params=params)
    )
    model, provider, dynamic_api_key, api_base = connection
    if provider != "chatgpt" and provider not in MODEL_INFO_DISCOVERY_PROVIDERS:
        return _EMPTY_METADATA
    api_key: Final = params.api_key or dynamic_api_key
    headers: Final = _HEADERS.validate_python(metadata.get("extra_headers") or metadata.get("headers") or {})
    inventory: Final = await get_provider_model_inventory(
        provider=provider,
        api_base=api_base,
        client=client,
        cache=cache,
        headers=MappingProxyType(
            {
                **({"authorization": f"Bearer {api_key}"} if api_key else {}),
                **{key.lower(): value for key, value in headers.items()},
            }
        ),
    )
    return (
        inventory.models.get(model, _EMPTY_METADATA)
        if isinstance(inventory, SupplierModelInventory)
        else _EMPTY_METADATA
    )
