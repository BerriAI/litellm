import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # typed native cache and HTTP boundaries

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError, ValidationInfo, field_validator

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai_like.model_info import MODEL_INFO_REFRESH_SECONDS
from litellm.types.proxy.model_inventory import (
    ModelInventoryCache,
    ModelInventoryHTTPClient,
    SupplierInventoryUnavailable,
    SupplierModelInventory,
)
from litellm.types.proxy.model_metadata import GatewayModelMetadata

from .authenticator import Authenticator
from .common_utils import get_chatgpt_default_headers

_CATALOG_CLIENT_VERSION: Final = "0.159.3"
_EMPTY_METADATA: Final[Mapping[str, object]] = MappingProxyType({})


class _ReasoningLevel(BaseModel):
    model_config = ConfigDict(frozen=True)

    effort: str


class _ChatGPTModel(GatewayModelMetadata):
    slug: str
    max_context_window: int | None = None
    default_reasoning_level: str | None = None
    supported_reasoning_levels: tuple[_ReasoningLevel, ...] | None = None
    input_modalities: Sequence[str] | None = None
    supports_parallel_tool_calls: bool | None = None
    multi_agent_reasoning_effort: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def strict_supplier_metadata(cls, value: object, info: ValidationInfo) -> object:
        normalized: Final = (
            None
            if info.field_name in ("context_window", "max_input_tokens", "max_output_tokens", "max_context_window")
            and type(value) is int
            and value == 0
            else value
        )
        if info.field_name in GatewayModelMetadata.model_fields or info.field_name == "max_context_window":
            cls.validate_field_input(
                "context_window" if info.field_name == "max_context_window" else info.field_name, normalized
            )
        if info.field_name == "supports_parallel_tool_calls":
            cls.validate_field_input("supports_parallel_function_calling", normalized)
        return normalized

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
            else None
            if native_default == "ultra"
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

    models: tuple[_ChatGPTModel, ...]
    error: object | None = None


def chatgpt_model_inventory_api_base(api_base: str | None = None, *, authenticator: Authenticator | None = None) -> str:
    """Saved server OAuth credentials may only be sent to the server-configured backend."""
    auth: Final = authenticator if authenticator is not None else Authenticator()
    trusted_base: Final = auth.get_api_base().rstrip("/")
    if api_base is not None and api_base.rstrip("/") != trusted_base:
        raise ValueError("Native model discovery requires the server-configured API base")
    return trusted_base


def chatgpt_model_inventory_identity() -> str | None:
    try:
        return Authenticator().get_account_id()
    except Exception:  # noqa: BLE001  # unreadable server auth is not proof of an account change
        return None


async def get_chatgpt_model_info(
    *,
    model: str,
    client: AsyncHTTPHandler,
    cache: InMemoryCache,
    api_base: str | None = None,
    authenticator: Authenticator | None = None,
) -> Mapping[str, object]:
    inventory: Final = await get_chatgpt_model_inventory(
        client=client, cache=cache, api_base=api_base, authenticator=authenticator
    )
    return (
        inventory.models.get(model, _EMPTY_METADATA)
        if isinstance(inventory, SupplierModelInventory)
        else _EMPTY_METADATA
    )


async def get_chatgpt_model_inventory(
    *,
    client: AsyncHTTPHandler,
    cache: InMemoryCache,
    api_base: str | None = None,
    authenticator: Authenticator | None = None,
    force_refresh: bool = False,
) -> SupplierModelInventory | SupplierInventoryUnavailable:
    inventory_cache: Final = cast(ModelInventoryCache, cache)  # cast-ok: native cache implements this protocol
    inventory_client: Final = cast(ModelInventoryHTTPClient, client)  # cast-ok: native HTTP handler implements get
    auth: Final = authenticator if authenticator is not None else Authenticator()
    try:
        trusted_base: Final = chatgpt_model_inventory_api_base(api_base, authenticator=auth)
        account: Final = await asyncio.to_thread(auth.get_account_id)
        url: Final = f"{trusted_base}/models?client_version={_CATALOG_CLIENT_VERSION}"
        credential_scope: Final = hashlib.sha256(json.dumps((url, account)).encode()).hexdigest() if account else None
    except Exception:  # noqa: BLE001  # malformed native auth is unavailable, never an authoritative empty inventory
        return SupplierInventoryUnavailable("authentication")
    try:
        token: Final = await asyncio.to_thread(auth.get_access_token, allow_device_login=False)
    except Exception:  # noqa: BLE001  # unavailable existing OAuth must never initiate a device login
        return SupplierInventoryUnavailable("authentication", credential_scope)
    try:
        headers: Final = {
            **get_chatgpt_default_headers(token, account),
            "accept": "application/json",
        }
        cache_key: Final = (
            "chatgpt_model_info:" + hashlib.sha256(json.dumps((url, sorted(headers.items()))).encode()).hexdigest()
        )
        cached: Final[object] = inventory_cache.get_cache(cache_key)
        if not force_refresh and isinstance(cached, SupplierModelInventory):
            return cached
        response: Final = await inventory_client.get(
            url=url,
            headers=headers,
            timeout=httpx.Timeout(5.0),
            follow_redirects=False,
            max_response_bytes=2 * 1024 * 1024,
        )
        response.raise_for_status()
        catalog: Final = _ChatGPTCatalog.model_validate_json(response.content)
        ids: Final = tuple(card.slug for card in catalog.models)
        if (
            catalog.error is not None
            or any(not model_id.strip() or model_id != model_id.strip() for model_id in ids)
            or len(frozenset(ids)) != len(ids)
        ):
            return SupplierInventoryUnavailable("malformed", credential_scope)
        inventory: Final = SupplierModelInventory(
            MappingProxyType({card.slug: card.metadata() for card in catalog.models}), credential_scope
        )
        inventory_cache.set_cache(cache_key, inventory, ttl=MODEL_INFO_REFRESH_SECONDS)
    except httpx.HTTPStatusError:
        return SupplierInventoryUnavailable("http", credential_scope)
    except ValidationError:
        return SupplierInventoryUnavailable("malformed", credential_scope)
    except Exception:  # noqa: BLE001  # optional metadata discovery must not interrupt proxy startup
        return SupplierInventoryUnavailable("transport", credential_scope)
    return inventory
