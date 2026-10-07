"""Supplier-specific inventory dispatch shared by router and selected-offering discovery."""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.chatgpt.authenticator import prevent_device_login
from litellm.llms.chatgpt.model_info import (
    chatgpt_model_inventory_api_base,
    chatgpt_model_inventory_identity,
    get_chatgpt_model_inventory,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai_like.model_info import MODEL_INFO_DISCOVERY_PROVIDERS, get_openai_compatible_model_inventory
from litellm.types.proxy.model_inventory import SupplierInventoryUnavailable, SupplierModelInventory
from litellm.types.router import LiteLLM_Params

_EMPTY_METADATA: Final[Mapping[str, object]] = MappingProxyType({})
_HEADERS: Final = TypeAdapter(Mapping[str, str])


@contextmanager
def prevent_supplier_device_login() -> Generator[None]:
    with prevent_device_login():
        yield


def model_inventory_api_base(provider: str, api_base: str | None) -> str | None:
    if provider == "chatgpt":
        return chatgpt_model_inventory_api_base(api_base)
    return api_base or {
        "openrouter": "https://openrouter.ai/api/v1",
        "vercel_ai_gateway": "https://ai-gateway.vercel.sh/v1",
    }.get(provider)


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
    return await get_openai_compatible_model_inventory(
        provider=provider,
        api_base=resolved_base,
        headers=headers,
        client=client,
        cache=cache,
        force_refresh=force_refresh,
    )


async def get_deployment_model_metadata(
    *,
    params: LiteLLM_Params,
    client: AsyncHTTPHandler,
    cache: InMemoryCache,
) -> Mapping[str, object]:
    if params.get("use_clientside_credentials") or "*" in params.model:
        return _EMPTY_METADATA
    if params.model.startswith("chatgpt/"):
        model: str = params.model.removeprefix("chatgpt/")
        provider: str = "chatgpt"
        api_base: str | None = params.api_base
        dynamic_api_key: str | None = None
    else:
        model, provider, dynamic_api_key, api_base = litellm.get_llm_provider(model=params.model, litellm_params=params)
        if provider not in MODEL_INFO_DISCOVERY_PROVIDERS:
            return _EMPTY_METADATA
    api_key: Final = params.api_key or dynamic_api_key
    headers: Final = _HEADERS.validate_python(params.get("extra_headers") or params.get("headers") or {})
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
