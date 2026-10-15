from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

import httpx

from litellm.types.router import ModelMetadata


@dataclass(frozen=True, slots=True)
class SupplierModelEndpoint:
    tag: str
    metadata: ModelMetadata
    supported_parameters: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class SupplierModelInventory:
    models: Mapping[str, Mapping[str, object]]
    credential_scope: str | None = None
    endpoints: Mapping[str, tuple[SupplierModelEndpoint, ...]] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class SupplierInventoryUnavailable:
    reason: Literal["authentication", "http", "transport", "malformed"]
    credential_scope: str | None = None


class ModelInventoryCache(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: object, *, ttl: int) -> None: ...


class ModelInventoryHTTPClient(Protocol):
    async def get(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        timeout: httpx.Timeout,
        follow_redirects: bool,
        max_response_bytes: int,
    ) -> httpx.Response: ...
