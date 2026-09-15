from dataclasses import dataclass

from e2e_http import NoBody, Result, unwrap
from models import (
    MemoryCaptureBody,
    MemoryEntriesData,
    MemoryEntryData,
    MemoryEntryParams,
    MemorySettingsBody,
    MemoryStatusData,
)
from proxy_client import ProxyClient


@dataclass(frozen=True)
class MemoryClient:
    proxy: ProxyClient

    def settings(self) -> MemorySettingsBody:
        return unwrap(
            self.proxy.transport.get(
                "/memory/v2/settings",
                headers=self.proxy.transport.master,
                params=NoBody(),
                response_type=MemorySettingsBody,
            )
        )

    def set_settings(self, body: MemorySettingsBody, *, caller: str | None = None) -> Result[MemorySettingsBody]:
        return self.proxy.transport.put(
            "/memory/v2/settings",
            headers=self.proxy.transport.bearer(caller) if caller else self.proxy.transport.master,
            json=body,
            response_type=MemorySettingsBody,
        )

    def read(self, key: str, memory_id: str) -> Result[MemoryEntryData]:
        return self.proxy.transport.get(
            f"/memory/v2/entries/{memory_id}",
            headers=self.proxy.transport.bearer(key),
            params=NoBody(),
            response_type=MemoryEntryData,
        )

    def update(self, key: str, memory_id: str, body: MemoryCaptureBody) -> Result[MemoryEntryData]:
        return self.proxy.transport.put(
            f"/memory/v2/entries/{memory_id}",
            headers=self.proxy.transport.bearer(key),
            json=body,
            response_type=MemoryEntryData,
        )

    def status(self, key: str) -> MemoryStatusData:
        return unwrap(
            self.proxy.transport.get(
                "/memory/v2/status",
                headers=self.proxy.transport.bearer(key),
                params=NoBody(),
                response_type=MemoryStatusData,
            )
        )

    def entries(self, key: str, params: MemoryEntryParams = MemoryEntryParams()) -> list[MemoryEntryData]:
        return unwrap(
            self.proxy.transport.get(
                "/memory/v2/entries",
                headers=self.proxy.transport.bearer(key),
                params=params,
                response_type=MemoryEntriesData,
            )
        ).root

    def capture(self, key: str, body: MemoryCaptureBody) -> Result[MemoryEntryData]:
        return self.proxy.transport.post(
            "/memory/v2/entries",
            headers=self.proxy.transport.bearer(key),
            json=body,
            response_type=MemoryEntryData,
        )

    def delete_entry(self, key: str, memory_id: str) -> Result[NoBody]:
        return self.proxy.transport.delete(
            f"/memory/v2/entries/{memory_id}",
            headers=self.proxy.transport.bearer(key),
            json=NoBody(),
            response_type=NoBody,
        )

    def cleanup_user_entries(self, user_id: str) -> None:
        while True:
            page = unwrap(
                self.proxy.transport.get(
                    "/memory/v2/entries",
                    headers=self.proxy.transport.master,
                    params=MemoryEntryParams(user_id=user_id),
                    response_type=MemoryEntriesData,
                )
            ).root
            if not page:
                return
            for entry in page:
                unwrap(
                    self.proxy.transport.delete(
                        f"/memory/v2/entries/{entry.memory_id}",
                        headers=self.proxy.transport.master,
                        json=NoBody(),
                        response_type=NoBody,
                    )
                )
