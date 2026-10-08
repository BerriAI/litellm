from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx


@dataclass(frozen=True, slots=True)
class SupplierModelInventory:
    models: Mapping[str, Mapping[str, object]]
    credential_scope: str | None = None


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
