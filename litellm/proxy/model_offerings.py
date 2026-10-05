import asyncio
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import yaml

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.chatgpt.authenticator import Authenticator, prevent_device_login
from litellm.llms.chatgpt.model_info import get_chatgpt_model_inventory
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai_like.model_info import MODEL_INFO_REFRESH_CONCURRENCY, get_openai_compatible_model_inventory
from litellm.proxy.offering_router import OfferingRouterView, OfferingServingSnapshot
from litellm.router import Router
from litellm.types.proxy.model_inventory import SupplierInventoryUnavailable, SupplierModelInventory
from litellm.types.proxy.model_offerings import ModelOffering, ModelOfferingsConfig, SupplierConnection


@dataclass(frozen=True, slots=True)
class _ResolvedConnection:
    provider: str
    api_base: str | None
    api_key: str | None
    headers: Mapping[str, str]
    fingerprint: str
    native_account_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ProviderState:
    connection: _ResolvedConnection
    inventory: SupplierModelInventory | None
    credential_scope: str | None = None


@dataclass(frozen=True, slots=True)
class _LoadedConfig:
    config: ModelOfferingsConfig
    fingerprint: str
    connections: Mapping[str, _ResolvedConnection]


@dataclass(frozen=True, slots=True)
class _InvalidConfig:
    error_kind: str


def _resolve_secret(value: str) -> str:
    if not value.startswith("os.environ/"):
        return value
    resolved: Final = os.environ.get(value.removeprefix("os.environ/"))
    if resolved is None:
        raise ValueError("Offering connection references an unset environment variable")
    return resolved


def _native_account_id() -> str | None:
    try:
        return Authenticator().get_account_id()
    except Exception:  # noqa: BLE001  # an unreadable native auth file is not proof of an account change
        return None


def _resolve_connection(connection: SupplierConnection) -> _ResolvedConnection:
    api_base: Final = (
        str(connection.api_base).rstrip("/")
        if connection.api_base is not None
        else {
            "openrouter": "https://openrouter.ai/api/v1",
            "vercel_ai_gateway": "https://ai-gateway.vercel.sh/v1",
        }.get(connection.provider)
    )
    api_key: Final = _resolve_secret(connection.api_key.get_secret_value()) if connection.api_key is not None else None
    configured_headers: Final = {
        name.lower(): _resolve_secret(value.get_secret_value()) for name, value in connection.headers.items()
    }
    headers: Final = MappingProxyType(
        {**({"authorization": f"Bearer {api_key}"} if api_key is not None else {}), **configured_headers}
    )
    fingerprint: Final = hashlib.sha256(
        json.dumps((connection.provider, api_base, api_key, sorted(headers.items()))).encode()
    ).hexdigest()
    return _ResolvedConnection(
        connection.provider,
        api_base,
        api_key,
        headers,
        fingerprint,
        _native_account_id() if connection.provider == "chatgpt" else None,
    )


def _load_config(path: Path) -> _LoadedConfig | _InvalidConfig:
    try:
        content: Final = path.read_bytes()
        if len(content) > 1024 * 1024:
            return _InvalidConfig("ConfigTooLarge")
        config: Final = ModelOfferingsConfig.model_validate(yaml.safe_load(content))
        connections: Final = MappingProxyType(
            {name: _resolve_connection(connection) for name, connection in config.providers.items()}
        )
        fingerprint: Final = hashlib.sha256(
            json.dumps(
                (
                    hashlib.sha256(content).hexdigest(),
                    tuple(
                        (connection.fingerprint, connection.native_account_id) for connection in connections.values()
                    ),
                )
            ).encode()
        ).hexdigest()
        return _LoadedConfig(config, fingerprint, connections)
    except Exception as error:  # noqa: BLE001  # a bad operator edit must leave the complete last-valid snapshot intact
        return _InvalidConfig(type(error).__name__)


def _provider_after_fetch(
    previous: _ProviderState, fetched: SupplierModelInventory | SupplierInventoryUnavailable
) -> _ProviderState:
    if isinstance(fetched, SupplierModelInventory):
        return _ProviderState(previous.connection, fetched, fetched.credential_scope)
    if fetched.credential_scope is not None and previous.credential_scope != fetched.credential_scope:
        return _ProviderState(previous.connection, None, fetched.credential_scope)
    return previous


def _unavailable_reason(offering: ModelOffering, state: _ProviderState) -> str | None:
    if not offering.enabled:
        return "disabled in the offering configuration"
    if offering.source == "manual":
        return None
    if state.inventory is None:
        return "no successful inventory is available for the configured supplier connection"
    if offering.upstream_model not in state.inventory.models:
        return "the model is absent from the latest successful supplier inventory"
    return None


def _deployment(offering: ModelOffering, state: _ProviderState) -> Mapping[str, object]:
    connection: Final = state.connection
    reason: Final = _unavailable_reason(offering, state)
    supplier_info: Final[Mapping[str, object]] = (
        state.inventory.models.get(offering.upstream_model, {})
        if offering.source == "auto" and state.inventory is not None
        else {}
    )
    metadata: Final = {**supplier_info, **offering.model_info.model_dump(mode="json", exclude_none=True)}
    deployment_id: Final = (
        "offering-"
        + hashlib.sha256(
            json.dumps((offering.model_name, connection.fingerprint, offering.upstream_model)).encode()
        ).hexdigest()
    )
    return MappingProxyType(
        {
            "model_name": offering.model_name,
            "litellm_params": {
                "model": f"{connection.provider}/{offering.upstream_model}",
                **({"api_base": connection.api_base} if connection.api_base is not None else {}),
                **({"api_key": connection.api_key} if connection.api_key is not None else {}),
                **({"extra_headers": dict(connection.headers)} if connection.headers else {}),
            },
            "model_info": {**metadata, "id": deployment_id, "blocked": reason is not None},
        }
    )


class ModelOfferingsManager:
    def __init__(
        self,
        *,
        path: Path,
        template: Router,
        client: AsyncHTTPHandler,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = path
        self.template = template
        self.client = client
        self.clock = clock
        self.cache = InMemoryCache(max_size_in_memory=64)
        self.config: ModelOfferingsConfig | None = None
        self.config_fingerprint: str | None = None
        self.providers: Mapping[str, _ProviderState] = MappingProxyType({})
        self.deployments: tuple[Mapping[str, object], ...] = ()
        self.next_inventory_refresh = 0.0
        self.last_error: str | None = None
        self.lock = asyncio.Lock()
        self.inventory_semaphore = asyncio.Semaphore(MODEL_INFO_REFRESH_CONCURRENCY)
        self.router = OfferingRouterView(
            OfferingServingSnapshot(template, frozenset(), MappingProxyType({}), frozenset())
        )

    async def _fetch_provider(self, state: _ProviderState) -> _ProviderState:
        connection: Final = state.connection
        try:
            fetched: Final = (
                await get_chatgpt_model_inventory(
                    client=self.client, cache=self.cache, api_base=connection.api_base, force_refresh=True
                )
                if connection.provider == "chatgpt"
                else await get_openai_compatible_model_inventory(
                    api_base=connection.api_base or "",
                    provider=connection.provider,
                    headers=connection.headers,
                    client=self.client,
                    cache=self.cache,
                    force_refresh=True,
                )
            )
        except Exception:  # noqa: BLE001  # supplier failures cannot replace a successful inventory with absence
            return state
        if isinstance(fetched, SupplierInventoryUnavailable):
            verbose_proxy_logger.warning(
                "Supplier inventory refresh unavailable (%s); retaining the last successful inventory when connection identity is unchanged",
                fetched.reason,
            )
        return _provider_after_fetch(state, fetched)

    async def _bounded_fetch_provider(self, state: _ProviderState) -> _ProviderState:
        async with self.inventory_semaphore:
            return await self._fetch_provider(state)

    async def _refresh_providers(self, providers: Mapping[str, _ProviderState]) -> Mapping[str, _ProviderState]:
        names: Final = tuple(providers)
        results: Final = await asyncio.gather(*(self._bounded_fetch_provider(providers[name]) for name in names))
        return MappingProxyType(dict(zip(names, results)))

    def _apply(self, loaded: _LoadedConfig, providers: Mapping[str, _ProviderState]) -> bool:
        deployments: Final = tuple(
            _deployment(offering, providers[offering.provider])
            for offering in loaded.config.offerings
            if _unavailable_reason(offering, providers[offering.provider]) is None
        )
        try:
            with prevent_device_login():
                candidate: Final = (
                    self.template.snapshot_with_model_list(deployments, metadata_authoritative=True)
                    if deployments != self.deployments
                    else self.router.latest_snapshot().router
                )
            if len(candidate.get_model_ids()) != len(deployments):
                self._config_error("DeploymentValidationFailed")
                return False
        except Exception as error:  # noqa: BLE001  # a rejected native deployment must not partially replace serving state
            self._config_error(type(error).__name__)
            return False
        available: Final = frozenset(
            offering.model_name
            for offering in loaded.config.offerings
            if _unavailable_reason(offering, providers[offering.provider]) is None
        )
        unavailable: Final = MappingProxyType(
            {
                offering.model_name: reason
                for offering in loaded.config.offerings
                if (reason := _unavailable_reason(offering, providers[offering.provider])) is not None
            }
        )
        allowed_deployments: Final = frozenset(
            f"{providers[offering.provider].connection.provider}/{offering.upstream_model}"
            for offering in loaded.config.offerings
            if offering.model_name in available
        )
        snapshot: Final = OfferingServingSnapshot(candidate, available, unavailable, allowed_deployments)
        self.config = loaded.config
        self.config_fingerprint = loaded.fingerprint
        self.providers = providers
        self.deployments = deployments
        self.last_error = None
        self.router.publish_snapshot(snapshot)
        return True

    def _config_error(self, kind: str) -> None:
        if self.last_error != kind:
            verbose_proxy_logger.error(
                "External offering configuration rejected (%s); retaining the complete last-valid serving configuration",
                kind,
            )
        self.last_error = kind

    async def reload(self, *, initial: bool = False, force_inventory: bool = False) -> bool:
        async with self.lock:
            loaded: Final = await asyncio.to_thread(_load_config, self.path)
            if isinstance(loaded, _InvalidConfig):
                self._config_error(loaded.error_kind)
                return False
            now: Final = self.clock()
            config_changed: Final = loaded.fingerprint != self.config_fingerprint
            if not config_changed and not force_inventory and now < self.next_inventory_refresh:
                return True
            providers: Final = MappingProxyType(
                {
                    name: previous
                    if (previous := self.providers.get(name)) is not None
                    and previous.connection.fingerprint == connection.fingerprint
                    and (
                        connection.native_account_id is None
                        or previous.connection.native_account_id == connection.native_account_id
                    )
                    else _ProviderState(connection, None)
                    for name, connection in loaded.connections.items()
                }
            )
            connection_changed: Final = any(
                name not in self.providers or self.providers[name].connection != state.connection
                for name, state in providers.items()
            )
            refresh_due: Final = initial or force_inventory or connection_changed or now >= self.next_inventory_refresh
            refreshed: Final = await self._refresh_providers(providers) if refresh_due else providers
            if not self._apply(loaded, refreshed):
                return False
            if refresh_due:
                self.next_inventory_refresh = now + loaded.config.inventory_poll_seconds
            return True

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.config.config_poll_seconds if self.config is not None else 5)
            await self.reload()
