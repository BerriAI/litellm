SELECT SpanId AS span_id,
       substringUTF8(StatusMessage, {error_offset:UInt64} + 1, 16384) AS message,
       lengthUTF8(StatusMessage) AS total_chars,
       hex(SHA256(StatusMessage)) AS version
FROM otel_traces
WHERE TraceId = {trace_id:String} AND SpanId = {span_id:String}
  AND ({all_teams:UInt8} = 1
       OR ({user_id:String} != '' AND UserId = {user_id:String})
       OR has({team_ids:Array(String)}, TeamId)
       OR ({api_key_hash:String} != '' AND ApiKeyHash = {api_key_hash:String}))
  AND ({trace_ref:String} = '' OR
       hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) = {trace_ref:String})
  AND ({error_version:String} = '' OR hex(SHA256(StatusMessage)) = {error_version:String})
ORDER BY Timestamp, EngineReceivedMs, StatusMessage
LIMIT 1
