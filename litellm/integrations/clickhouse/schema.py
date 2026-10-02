from typing import Final

from litellm.rust_bridge.traces import ClickHouseStorage

OTEL_TRACES_TABLE: Final = "otel_traces"
AGENT_TRACES_BY_KEY_TABLE: Final = "agent_traces_by_key"
SPEND_LOGS_TABLE: Final = "spend_logs"


async def ensure_schema(storage: ClickHouseStorage) -> None:
    await storage.ensure_schema()
