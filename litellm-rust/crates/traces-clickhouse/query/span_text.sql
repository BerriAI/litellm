SELECT span_id,
       multiIf({range:String} = 'last', substringUTF8(part, toUInt64(greatest(toInt64(lengthUTF8(part)) - toInt64({chars:UInt64}), 0)) + 1),
               {bounded:UInt8} = 1, substringUTF8(part, {offset:UInt64} + 1, {chars:UInt64}),
               substringUTF8(part, {offset:UInt64} + 1)) AS text,
       lengthUTF8(part) AS total_chars,
       hex(SHA256(part)) AS version,
       {needle:String} != '' AND position(part, {needle:String}) > 0 AS contains
FROM (
    SELECT SpanId AS span_id,
           multiIf({part:String} = 'input', Input,
                   {part:String} = 'output', Output,
                   {part:String} = 'error', StatusMessage,
                   toJSONString(SpanAttributes)) AS part
    FROM owned_spans
    WHERE TraceId = {trace_id:String} AND SpanId IN {span_ids:Array(String)}
      AND hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) = {trace_ref:String}
ORDER BY Timestamp, EngineReceivedMs, StatusMessage, Duration, StatusCode
    LIMIT 1 BY SpanId
)
