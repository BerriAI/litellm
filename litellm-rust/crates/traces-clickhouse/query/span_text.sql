SELECT if({bounded:UInt8} = 1, substringUTF8(part, {offset:UInt64} + 1, {max_chars:UInt64}),
          substringUTF8(part, {offset:UInt64} + 1)) AS text,
       lengthUTF8(part) AS total_chars,
       hex(SHA256(part)) AS version
FROM (
    SELECT multiIf({part:String} = 'input', Input,
                   {part:String} = 'output', Output,
                   {part:String} = 'error', StatusMessage,
                   toJSONString(SpanAttributes)) AS part
    FROM owned_spans
    WHERE TraceId = {trace_id:String} AND SpanId = {span_id:String}
      AND hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) = {trace_ref:String}
    ORDER BY Timestamp, EngineReceivedMs, StatusMessage
    LIMIT 1
)
