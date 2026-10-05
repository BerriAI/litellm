from typing import Final

from litellm.rust_bridge.trace.storage import ClickHouseStorage

OTEL_TRACES_TABLE: Final = "otel_traces"
TRACE_ROLLUP_TABLE: Final = "trace_rollup"
SPEND_LOGS_TABLE: Final = "spend_logs"


async def ensure_schema(storage: ClickHouseStorage) -> None:
    await storage.ensure_schema()
