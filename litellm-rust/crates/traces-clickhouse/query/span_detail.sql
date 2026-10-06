scoped AS (
    SELECT * FROM otel_traces
    WHERE TraceId = {trace_id:String}
      AND ({all_teams:UInt8} = 1
           OR ({user_id:String} != '' AND UserId = {user_id:String})
           OR has({team_ids:Array(String)}, TeamId))
      AND ({trace_ref:String} = '' OR
           hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) = {trace_ref:String})
),
selected AS (
    SELECT *, coalesce(nullIf(SpanAttributes['gen_ai.tool.call.id'], ''), SpanAttributes['tool_use_id']) AS call_id,
           SpanAttributes['session.id'] AS session_id,
           (Framework IN ('claude-code', 'claude-agent-sdk') AND ObservationType = 'tool' AND call_id != '') AS native_tool
    FROM scoped WHERE has(requested_ids, SpanId)
    ORDER BY Timestamp DESC LIMIT 1 BY TeamId, ApiKeyHash, SpanId
),
inputs AS (
    SELECT TeamId, ApiKeyHash, SpanAttributes['session.id'] AS session_id,
           SpanAttributes['tool_use_id'] AS call_id,
           groupUniqArray(2)(Input) AS values, min(SpanId) AS source_span
    FROM scoped
    WHERE SpanName = 'claude_code.tool_result' AND Framework IN ('claude-code', 'claude-agent-sdk')
      AND Input != '' AND call_id != '' AND call_id IN (SELECT call_id FROM selected WHERE native_tool)
    GROUP BY TeamId, ApiKeyHash, session_id, call_id
),
outputs AS (
    SELECT TeamId, ApiKeyHash, SpanAttributes['session.id'] AS session_id,
           JSONExtractString(result, 'id') AS call_id,
           groupUniqArray(2)(tuple(JSONExtractString(result, 'content'), JSONExtractBool(result, 'is_error'))) AS values,
           min(SpanId) AS source_span
    FROM scoped ARRAY JOIN JSONExtractArrayRaw(Output, 'tool_results') AS result
    WHERE SpanName = 'claude_code.api_request_body' AND Framework IN ('claude-code', 'claude-agent-sdk')
      AND call_id != '' AND call_id IN (SELECT call_id FROM selected WHERE native_tool)
      AND JSONType(result, 'content') = 'String'
    GROUP BY TeamId, ApiKeyHash, session_id, call_id
),
executions AS (
    SELECT TeamId, ApiKeyHash, SpanAttributes['session.id'] AS session_id,
           groupUniqArray(coalesce(nullIf(SpanAttributes['gen_ai.tool.call.id'], ''), SpanAttributes['tool_use_id'])) AS call_ids
    FROM scoped
    WHERE ObservationType = 'tool' AND Framework IN ('claude-code', 'claude-agent-sdk')
    GROUP BY TeamId, ApiKeyHash, session_id
),
answers AS (
    SELECT TeamId, ApiKeyHash, ParentSpanId AS parent_span_id, argMax(Output, Timestamp) AS output,
           argMax(SpanId, Timestamp) AS source_span
    FROM scoped
    WHERE has(requested_ids, ParentSpanId) AND ObservationType = 'llm' AND Output != ''
    GROUP BY TeamId, ApiKeyHash, ParentSpanId
)
SELECT o.SpanId AS span_id,
       if(o.native_tool AND length(i.values) = 1, i.values[1], o.Input) AS input,
       multiIf(o.Output != '', o.Output,
               o.native_tool AND length(r.values) = 1, r.values[1].1,
               o.ObservationType = 'agent', a.output, '') AS output,
       mapUpdate(o.SpanAttributes, map(
           'lens.content.unexecuted_tool_results', if(o.SpanName = 'claude_code.api_request_body',
               concat('[', arrayStringConcat(arrayFilter(result -> JSONExtractString(result, 'id') != '' AND NOT has(e.call_ids, JSONExtractString(result, 'id')),
                   JSONExtractArrayRaw(o.Output, 'tool_results')), ','), ']'), '[]'),
           'lens.content.input_source', multiIf(o.native_tool AND length(i.values) = 1, i.source_span, o.Input != '', o.SpanId, ''),
           'lens.content.output_source', multiIf(o.Output != '', o.SpanId, o.native_tool AND length(r.values) = 1, r.source_span, o.ObservationType = 'agent' AND a.output != '', a.source_span, ''),
           'lens.content.input_status', multiIf(o.native_tool AND length(i.values) > 1, 'conflicting', input != '', 'recorded', 'not_recorded'),
           'lens.content.output_status', multiIf(o.Output != '', 'recorded', o.native_tool AND length(r.values) > 1, 'conflicting', o.native_tool AND length(r.values) = 1, 'recorded', output != '', 'recorded', 'not_recorded'),
           'lens.content.tool_result_is_error', if(o.native_tool AND length(r.values) = 1, toString(r.values[1].2), '')
       )) AS attributes
FROM selected AS o
LEFT JOIN inputs AS i ON o.native_tool AND o.TeamId = i.TeamId AND o.ApiKeyHash = i.ApiKeyHash AND o.session_id = i.session_id AND o.call_id = i.call_id
LEFT JOIN outputs AS r ON o.native_tool AND o.TeamId = r.TeamId AND o.ApiKeyHash = r.ApiKeyHash AND o.session_id = r.session_id AND o.call_id = r.call_id
LEFT JOIN executions AS e ON e.TeamId = o.TeamId AND e.ApiKeyHash = o.ApiKeyHash AND e.session_id = o.session_id
LEFT JOIN answers AS a ON a.parent_span_id = o.SpanId AND a.TeamId = o.TeamId AND a.ApiKeyHash = o.ApiKeyHash
ORDER BY indexOf(requested_ids, o.SpanId)
LIMIT length(requested_ids)
