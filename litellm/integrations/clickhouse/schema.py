"""
ClickHouse DDL for agent tracing.

- otel_traces:  one row per span. Standard columns match the OTel Collector
                clickhouseexporter, so a collector can write here too.
- agent_traces: one row per trace, maintained by a materialized view.
- spend_logs:   one row per LiteLLM request (written by the `clickhouse` callback).

Join: otel_traces.LiteLLMRequestId = spend_logs.response_id
"""

from litellm.integrations.clickhouse.clickhouse_client import ClickHouseClient

OTEL_TRACES_TABLE = "otel_traces"
AGENT_TRACES_TABLE = "agent_traces"
SPEND_LOGS_TABLE = "spend_logs"

_ATTR_MAP = "Map(LowCardinality(String), String)"


def schema_statements(database: str, trace_retention_days: int, spend_log_retention_days: int) -> list[str]:
    db = database
    return [
        f"CREATE DATABASE IF NOT EXISTS {db}",
        f"""
CREATE TABLE IF NOT EXISTS {db}.{OTEL_TRACES_TABLE}
(
    Timestamp          DateTime64(9) CODEC(Delta, ZSTD(1)),
    TraceId            String CODEC(ZSTD(1)),
    SpanId             String CODEC(ZSTD(1)),
    ParentSpanId       String CODEC(ZSTD(1)),
    TraceState         String CODEC(ZSTD(1)),
    SpanName           LowCardinality(String) CODEC(ZSTD(1)),
    SpanKind           LowCardinality(String) CODEC(ZSTD(1)),
    ServiceName        LowCardinality(String) CODEC(ZSTD(1)),
    ResourceAttributes {_ATTR_MAP} CODEC(ZSTD(1)),
    ScopeName          String CODEC(ZSTD(1)),
    ScopeVersion       String CODEC(ZSTD(1)),
    SpanAttributes     {_ATTR_MAP} CODEC(ZSTD(1)),
    Duration           UInt64 CODEC(ZSTD(1)),
    StatusCode         LowCardinality(String) CODEC(ZSTD(1)),
    StatusMessage      String CODEC(ZSTD(1)),
    `Events.Timestamp`  Array(DateTime64(9)) CODEC(ZSTD(1)),
    `Events.Name`       Array(LowCardinality(String)) CODEC(ZSTD(1)),
    `Events.Attributes` Array({_ATTR_MAP}) CODEC(ZSTD(1)),
    `Links.TraceId`     Array(String) CODEC(ZSTD(1)),
    `Links.SpanId`      Array(String) CODEC(ZSTD(1)),
    `Links.TraceState`  Array(String) CODEC(ZSTD(1)),
    `Links.Attributes`  Array({_ATTR_MAP}) CODEC(ZSTD(1)),
    TeamId             LowCardinality(String) DEFAULT ResourceAttributes['litellm.team_id'],
    ApiKeyHash         String DEFAULT ResourceAttributes['litellm.api_key_hash'],
    ObservationType    LowCardinality(String) DEFAULT multiIf(
                           ParentSpanId = '', 'agent',
                           SpanAttributes['gen_ai.operation.name'] = 'invoke_agent', 'agent',
                           SpanAttributes['gen_ai.operation.name'] IN ('chat', 'text_completion', 'generate_content'), 'llm',
                           SpanAttributes['gen_ai.operation.name'] = 'execute_tool', 'tool',
                           'chain'),
    AgentName          LowCardinality(String) DEFAULT SpanAttributes['gen_ai.agent.name'],
    LiteLLMRequestId   String DEFAULT SpanAttributes['gen_ai.response.id'],
    Model              LowCardinality(String) DEFAULT SpanAttributes['gen_ai.request.model'],
    InputTokens        UInt32 DEFAULT toUInt32OrZero(SpanAttributes['gen_ai.usage.input_tokens']),
    OutputTokens       UInt32 DEFAULT toUInt32OrZero(SpanAttributes['gen_ai.usage.output_tokens']),
    Input              String CODEC(ZSTD(3)),
    Output             String CODEC(ZSTD(3)),
    InputPreview       String DEFAULT substring(Input, 1, 240),
    INDEX idx_trace_id TraceId          TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_req_id   LiteLLMRequestId TYPE bloom_filter(0.01)  GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toDate(Timestamp)
ORDER BY (TeamId, ServiceName, toDateTime(Timestamp), TraceId)
TTL toDateTime(Timestamp) + INTERVAL {trace_retention_days} DAY
SETTINGS ttl_only_drop_parts = 1
""",
        f"""
CREATE TABLE IF NOT EXISTS {db}.{AGENT_TRACES_TABLE}
(
    TeamId        LowCardinality(String),
    TraceId       String,
    StartTs       SimpleAggregateFunction(min, DateTime64(9)),
    EndTs         SimpleAggregateFunction(max, DateTime64(9)),
    ServiceName   SimpleAggregateFunction(any, LowCardinality(String)),
    RootName      SimpleAggregateFunction(anyLast, String),
    RootInput     SimpleAggregateFunction(anyLast, String),
    RootStatus    SimpleAggregateFunction(anyLast, String),
    SpanCount     SimpleAggregateFunction(sum, UInt64),
    AgentCount    SimpleAggregateFunction(sum, UInt64),
    LlmCount      SimpleAggregateFunction(sum, UInt64),
    ToolCount     SimpleAggregateFunction(sum, UInt64),
    ErrorCount    SimpleAggregateFunction(sum, UInt64),
    InputTokens   SimpleAggregateFunction(sum, UInt64),
    OutputTokens  SimpleAggregateFunction(sum, UInt64),
    Models        SimpleAggregateFunction(groupUniqArrayArray, Array(String)),
    RequestIds    SimpleAggregateFunction(groupArrayArray, Array(String))
)
ENGINE = AggregatingMergeTree
PARTITION BY toDate(StartTs)
ORDER BY (TeamId, TraceId)
TTL toDateTime(StartTs) + INTERVAL {trace_retention_days} DAY
""",
        f"""
CREATE MATERIALIZED VIEW IF NOT EXISTS {db}.{AGENT_TRACES_TABLE}_mv TO {db}.{AGENT_TRACES_TABLE} AS
SELECT
    TeamId, TraceId,
    min(Timestamp)                                         AS StartTs,
    max(Timestamp + toIntervalNanosecond(Duration))        AS EndTs,
    any(ServiceName)                                       AS ServiceName,
    anyLastIf(SpanName, ParentSpanId = '')                 AS RootName,
    anyLastIf(InputPreview, ParentSpanId = '')             AS RootInput,
    anyLastIf(StatusCode, ParentSpanId = '')               AS RootStatus,
    count()                                                AS SpanCount,
    countIf(ObservationType = 'agent')                     AS AgentCount,
    countIf(ObservationType = 'llm')                       AS LlmCount,
    countIf(ObservationType = 'tool')                      AS ToolCount,
    countIf(StatusCode = 'STATUS_CODE_ERROR')              AS ErrorCount,
    sum(InputTokens)                                       AS InputTokens,
    sum(OutputTokens)                                      AS OutputTokens,
    groupUniqArrayIf(toString(Model), Model != '')         AS Models,
    groupArrayIf(LiteLLMRequestId, LiteLLMRequestId != '') AS RequestIds
FROM {db}.{OTEL_TRACES_TABLE}
GROUP BY TeamId, TraceId
""",
        f"""
CREATE TABLE IF NOT EXISTS {db}.{SPEND_LOGS_TABLE}
(
    request_id            String,
    response_id           String,
    call_type             LowCardinality(String),
    api_key               String,
    key_alias             String,
    team_id               LowCardinality(String),
    team_alias            String,
    organization_id       String,
    user                  String,
    end_user              String,
    model                 LowCardinality(String),
    model_group           LowCardinality(String),
    model_id              String,
    custom_llm_provider   LowCardinality(String),
    api_base              String,
    spend                 Float64,
    prompt_tokens         UInt32,
    completion_tokens     UInt32,
    total_tokens          UInt32,
    cache_read_tokens     UInt32,
    cache_write_tokens    UInt32,
    start_time            DateTime64(3),
    end_time              DateTime64(3),
    completion_start_time Nullable(DateTime64(3)),
    status                LowCardinality(String),
    error_str             String,
    cache_hit             Bool,
    session_id            String,
    trace_id              String,
    span_id               String,
    request_tags          Array(String),
    metadata              String CODEC(ZSTD(3)),
    messages              String CODEC(ZSTD(3)),
    response              String CODEC(ZSTD(3)),
    INDEX idx_response_id response_id TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_trace_id    trace_id    TYPE bloom_filter(0.001) GRANULARITY 1
)
ENGINE = ReplacingMergeTree(end_time)
PARTITION BY toYYYYMM(start_time)
ORDER BY (team_id, toDateTime(start_time), request_id)
TTL toDateTime(start_time) + INTERVAL {spend_log_retention_days} DAY
""",
    ]


async def ensure_schema(client: ClickHouseClient, trace_retention_days: int, spend_log_retention_days: int) -> None:
    for statement in schema_statements(client.database, trace_retention_days, spend_log_retention_days):
        await client.execute(statement)
