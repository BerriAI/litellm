CREATE VIEW IF NOT EXISTS {database}.spans
SQL SECURITY INVOKER
AS SELECT
    TeamId AS team_id,
    ApiKeyHash AS api_key_hash,
    UserId AS user_id,
    TraceId AS trace_id,
    hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS trace_ref,
    SpanId AS span_id,
    ParentSpanId AS parent_span_id,
    SpanName AS name,
    SpanKind AS span_kind,
    ServiceName AS service,
    Timestamp AS start_time,
    Duration AS duration_ns,
    Duration / 1000000.0 AS duration_ms,
    multiIf(StatusCode = 'STATUS_CODE_ERROR', 'error', StatusCode = 'STATUS_CODE_OK', 'ok', 'unset') AS status,
    StatusCode AS status_code,
    StatusMessage AS status_message,
    ObservationType AS observation_type,
    AgentName AS agent_name,
    Framework AS framework,
    Model AS model,
    InputTokens AS input_tokens,
    OutputTokens AS output_tokens,
    ResourceAttributes AS resource_attributes,
    SpanAttributes AS span_attributes,
    AgentMetadata AS agent_metadata,
    Input AS input,
    Output AS output,
    InputPreview AS input_preview,
    WrapperCandidate AS wrapper_candidate,
    LiteLLMRequestId AS request_id,
    CallKeys AS call_keys,
    CallEvidence AS call_evidence,
    ToolCallId AS tool_call_id,
    EngineReceivedMs AS ingested_at_ms
FROM (
    SELECT * FROM {database}.otel_traces
    ORDER BY Timestamp, EngineReceivedMs, StatusMessage, Duration, StatusCode
    LIMIT 1 BY TeamId, ApiKeyHash, TraceId, SpanId
)
