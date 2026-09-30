from typing import Final

from litellm.integrations.clickhouse.clickhouse_client import ClickHouseClient
from litellm.rust_bridge.traces import schema_statements

OTEL_TRACES_TABLE: Final = "otel_traces"
AGENT_TRACES_TABLE: Final = "agent_traces"
SPEND_LOGS_TABLE: Final = "spend_logs"


async def ensure_schema(client: ClickHouseClient, trace_retention_days: int, spend_log_retention_days: int) -> None:
    for statement in schema_statements(client.database, trace_retention_days, spend_log_retention_days):
        await client.execute(statement)
