from typing import Protocol

from typing_extensions import ReadOnly, TypedDict


class NativeClientIdentity(TypedDict):
    cert_pem: ReadOnly[str]
    key_pem: ReadOnly[str]


class NativeDatabaseEvents(Protocol):
    def record(
        self,
        call_type: str,
        table: str | None,
        operation: str | None,
        started_at: float,
        duration_seconds: float,
        failure: str | None,
    ) -> None: ...
