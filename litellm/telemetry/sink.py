from enum import Enum
from typing import Protocol

from litellm.telemetry.records import AttemptRecord, InstanceInfo, RequestRecord, UIEvent
from litellm.telemetry.report import Report


class TelemetrySink(Protocol):
    def set_instance(self, info: InstanceInfo) -> None: ...

    def record_request(self, record: RequestRecord) -> None: ...

    def record_attempt(self, record: AttemptRecord) -> None: ...

    def record_ui_event(self, event: UIEvent) -> None: ...

    async def flush(self) -> None: ...


class NoopSink:
    def set_instance(self, info: InstanceInfo) -> None:
        return None

    def record_request(self, record: RequestRecord) -> None:
        return None

    def record_attempt(self, record: AttemptRecord) -> None:
        return None

    def record_ui_event(self, event: UIEvent) -> None:
        return None

    async def flush(self) -> None:
        return None


class ExportOutcome(str, Enum):
    SENT = "sent"
    REJECTED = "rejected"
    RETRY = "retry"


class Exporter(Protocol):
    async def export(self, report: Report) -> ExportOutcome: ...
