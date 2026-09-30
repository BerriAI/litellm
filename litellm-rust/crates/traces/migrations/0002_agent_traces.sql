CREATE TABLE IF NOT EXISTS {database}.agent_traces
(
    TeamId        LowCardinality(String),
    TraceId       String,
    StartTs       SimpleAggregateFunction(min, DateTime64(9)),
    EndTs         SimpleAggregateFunction(max, DateTime64(9)),
    ServiceName   SimpleAggregateFunction(any, LowCardinality(String)),
    RootName      SimpleAggregateFunction(anyLast, Nullable(String)),
    RootInput     SimpleAggregateFunction(anyLast, Nullable(String)),
    RootStatus    SimpleAggregateFunction(anyLast, Nullable(String)),
    SpanCount     SimpleAggregateFunction(sum, UInt64),
    AgentCount    SimpleAggregateFunction(sum, UInt64),
    LlmCount      SimpleAggregateFunction(sum, UInt64),
    ToolCount     SimpleAggregateFunction(sum, UInt64),
    ErrorCount    SimpleAggregateFunction(sum, UInt64),
    InputTokens   SimpleAggregateFunction(sum, UInt64),
    OutputTokens  SimpleAggregateFunction(sum, UInt64),
    Models        SimpleAggregateFunction(groupUniqArrayArray, Array(String)),
    AgentNames    SimpleAggregateFunction(groupUniqArrayArray, Array(String)),
    RequestIds    SimpleAggregateFunction(groupArrayArray, Array(String))
)
ENGINE = AggregatingMergeTree
PARTITION BY toDate(StartTs)
ORDER BY (TeamId, TraceId)
TTL toDateTime(StartTs) + INTERVAL {trace_retention_days} DAY
