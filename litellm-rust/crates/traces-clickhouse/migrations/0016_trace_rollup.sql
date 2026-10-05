CREATE TABLE IF NOT EXISTS {database}.trace_rollup
(
    TeamId          LowCardinality(String),
    StartHour       DateTime('UTC'),
    ApiKeyHash      String,
    TraceId         String,
    UserId          String,
    TraceRef        SimpleAggregateFunction(any, String),
    ReceivedMs      SimpleAggregateFunction(min, UInt64),
    StartTs         SimpleAggregateFunction(min, DateTime64(9)),
    EndTs           SimpleAggregateFunction(max, DateTime64(9)),
    Name            AggregateFunction(argMin, String, Tuple(UInt8, DateTime64(9), String)),
    Service         AggregateFunction(argMin, String, Tuple(DateTime64(9), String)),
    RootInput       AggregateFunction(argMin, String, Tuple(UInt8, DateTime64(9), String)),
    AgentInput      AggregateFunction(argMin, String, Tuple(UInt8, DateTime64(9), String)),
    RootStatus      AggregateFunction(argMin, String, Tuple(UInt8, DateTime64(9), String)),
    SpanCount       SimpleAggregateFunction(sum, UInt64),
    AgentSpans      SimpleAggregateFunction(sum, UInt64),
    LlmSpans        SimpleAggregateFunction(sum, UInt64),
    ToolSpans       SimpleAggregateFunction(sum, UInt64),
    ErrorSpans      SimpleAggregateFunction(sum, UInt64),
    InputTokens     SimpleAggregateFunction(sum, UInt64),
    OutputTokens    SimpleAggregateFunction(sum, UInt64),
    Models          SimpleAggregateFunction(groupUniqArrayArray, Array(String)),
    AgentLabels     SimpleAggregateFunction(groupUniqArrayArray, Array(String)),
    AgentIdentities SimpleAggregateFunction(groupUniqArrayArray, Array(String)),
    Frameworks      SimpleAggregateFunction(groupUniqArrayArray, Array(String)),
    INDEX idx_trace_id  TraceId  TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_trace_ref TraceRef TYPE bloom_filter(0.001) GRANULARITY 1
)
ENGINE = AggregatingMergeTree
PARTITION BY toDate(StartHour)
ORDER BY (TeamId, StartHour, ApiKeyHash, TraceId, UserId)
SETTINGS ttl_only_drop_parts = 1, materialize_ttl_recalculate_only = 1, non_replicated_deduplication_window = 1000
