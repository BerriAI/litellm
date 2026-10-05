SELECT bucket, failed, value, uniqExact(team_id, api_key_hash, trace_id) AS runs
FROM (
    SELECT runs.team_id AS team_id, runs.api_key_hash AS api_key_hash, runs.trace_id AS trace_id,
           if({buckets:UInt32} = 0, toUInt32(0),
              toUInt32(intDiv((runs.start_ms - {start_ms:Int64} + 1) * {buckets:UInt32} - 1, {end_ms:Int64} - {start_ms:Int64}))) AS bucket,
           toUInt8({by_failed:UInt8} = 1 AND runs.error_count > 0) AS failed,
           value
    FROM owned_spans AS spans
    INNER JOIN runs ON spans.TeamId = runs.team_id AND spans.ApiKeyHash = runs.api_key_hash
                   AND spans.TraceId = runs.trace_id
    ARRAY JOIN if({attribute_key:String} = '',
                  arrayConcat(mapKeys(spans.ResourceAttributes), mapKeys(spans.SpanAttributes)),
                  [spans.ResourceAttributes[{attribute_key:String}], spans.SpanAttributes[{attribute_key:String}]]) AS value
    WHERE spans.Timestamp >= fromUnixTimestamp64Milli({start_ms:Int64})
      AND spans.EngineReceivedMs <= {as_of_ms:UInt64}
      AND value != '' AND value ILIKE {contains:String}
)
GROUP BY bucket, failed, value
ORDER BY runs DESC, bucket, failed, value
LIMIT {limit:UInt64}
