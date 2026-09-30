CREATE TABLE IF NOT EXISTS {database}.otel_traces
(
    Timestamp          DateTime64(9) CODEC(Delta, ZSTD(1)),
    TraceId            String CODEC(ZSTD(1)),
    SpanId             String CODEC(ZSTD(1)),
    ParentSpanId       String CODEC(ZSTD(1)),
    TraceState         String CODEC(ZSTD(1)),
    SpanName           LowCardinality(String) CODEC(ZSTD(1)),
    SpanKind           LowCardinality(String) CODEC(ZSTD(1)),
    ServiceName        LowCardinality(String) CODEC(ZSTD(1)),
    ResourceAttributes Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    ScopeName          String CODEC(ZSTD(1)),
    ScopeVersion       String CODEC(ZSTD(1)),
    SpanAttributes     Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    Duration           UInt64 CODEC(ZSTD(1)),
    StatusCode         LowCardinality(String) CODEC(ZSTD(1)),
    StatusMessage      String CODEC(ZSTD(1)),
    `Events.Timestamp`  Array(DateTime64(9)) CODEC(ZSTD(1)),
    `Events.Name`       Array(LowCardinality(String)) CODEC(ZSTD(1)),
    `Events.Attributes` Array(Map(LowCardinality(String), String)) CODEC(ZSTD(1)),
    `Links.TraceId`     Array(String) CODEC(ZSTD(1)),
    `Links.SpanId`      Array(String) CODEC(ZSTD(1)),
    `Links.TraceState`  Array(String) CODEC(ZSTD(1)),
    `Links.Attributes`  Array(Map(LowCardinality(String), String)) CODEC(ZSTD(1)),
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
