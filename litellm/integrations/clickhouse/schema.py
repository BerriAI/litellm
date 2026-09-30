from typing import Final

from litellm.rust_bridge.traces import TraceStorage

OTEL_TRACES_TABLE: Final = "otel_traces"
AGENT_TRACES_TABLE: Final = "agent_traces"
AGENT_TRACES_BY_KEY_TABLE: Final = "agent_traces_by_key"
SPEND_LOGS_TABLE: Final = "spend_logs"


async def ensure_schema(storage: TraceStorage, trace_retention_days: int, spend_log_retention_days: int) -> None:
    await storage.ensure_schema(trace_retention_days, spend_log_retention_days)
