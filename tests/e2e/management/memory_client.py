import hashlib
from dataclasses import dataclass
from typing import Final, Literal
from urllib.parse import quote

from e2e_http import NoBody, Result, unwrap
from models import (
    MemoryCaptureBody,
    MemoryEntriesData,
    MemoryEntryData,
    MemoryEntryParams,
    MemoryLegacyParams,
    MemoryLegacyRows,
    MemoryPolicyBody,
    MemoryPolicyData,
    MemoryPreferenceBody,
    MemoryStatusData,
)
from proxy_client import ProxyClient


@dataclass(frozen=True)
class MemoryClient:
    proxy: ProxyClient

    def set_policy(self, body: MemoryPolicyBody, *, caller: str | None = None) -> Result[MemoryPolicyData]:
        return self.proxy.transport.put(
            "/v2/memory/policies",
            headers=self.proxy.transport.bearer(caller) if caller else self.proxy.transport.master,
            json=body,
            response_type=MemoryPolicyData,
        )

    def policy_for_key(self, key: str, activation: Literal["disabled", "opt_in", "automatic"]) -> MemoryPolicyData:
        return unwrap(
            self.set_policy(
                MemoryPolicyBody(
                    target_type="key",
                    target_id=hashlib.sha256(key.encode()).hexdigest(),
                    activation=activation,
                )
            )
        )

    def delete_policy(self, policy_id: str) -> None:
        unwrap(
            self.proxy.transport.delete(
                f"/v2/memory/policies/{policy_id}",
                headers=self.proxy.transport.master,
                json=NoBody(),
                response_type=NoBody,
            )
        )

    def preference(self, key: str, enabled: bool) -> MemoryPreferenceBody:
        return unwrap(
            self.proxy.transport.put(
                "/v2/memory/preference",
                headers=self.proxy.transport.bearer(key),
                json=MemoryPreferenceBody(enabled=enabled),
                response_type=MemoryPreferenceBody,
            )
        )

    def status(self, key: str) -> MemoryStatusData:
        return unwrap(
            self.proxy.transport.get(
                "/v2/memory/status",
                headers=self.proxy.transport.bearer(key),
                params=NoBody(),
                response_type=MemoryStatusData,
            )
        )

    def entries(self, key: str, params: MemoryEntryParams = MemoryEntryParams()) -> list[MemoryEntryData]:
        return unwrap(
            self.proxy.transport.get(
                "/v2/memory/entries",
                headers=self.proxy.transport.bearer(key),
                params=params,
                response_type=MemoryEntriesData,
            )
        ).root

    def capture(self, key: str, body: MemoryCaptureBody) -> Result[MemoryEntryData]:
        return self.proxy.transport.post(
            "/v2/memory/entries",
            headers=self.proxy.transport.bearer(key),
            json=body,
            response_type=MemoryEntryData,
        )

    def delete_entry(self, key: str, memory_id: str) -> Result[NoBody]:
        return self.proxy.transport.delete(
            f"/v2/memory/entries/{memory_id}",
            headers=self.proxy.transport.bearer(key),
            json=NoBody(),
            response_type=NoBody,
        )

    def cleanup_user_entries(self, user_id: str) -> None:
        first: Final = unwrap(
            self.proxy.transport.get(
                "/v1/memory",
                headers=self.proxy.transport.master,
                params=MemoryLegacyParams(),
                response_type=MemoryLegacyRows,
            )
        )
        remaining: Final = tuple(
            unwrap(
                self.proxy.transport.get(
                    "/v1/memory",
                    headers=self.proxy.transport.master,
                    params=MemoryLegacyParams(page=page),
                    response_type=MemoryLegacyRows,
                )
            )
            for page in range(2, (first.total + 499) // 500 + 1)
        )
        rows: Final = tuple(row for page in (first, *remaining) for row in page.memories if row.user_id == user_id)
        for row in rows:
            unwrap(
                self.proxy.transport.delete(
                    f"/v1/memory/{quote(row.key, safe='')}",
                    headers=self.proxy.transport.master,
                    json=NoBody(),
                    response_type=NoBody,
                )
            )
