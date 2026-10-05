WHERE Timestamp >= fromUnixTimestamp64Milli({start_ms:Int64})
  AND Timestamp < fromUnixTimestamp64Milli({end_ms:Int64})
  AND TraceId IN {trace_ids:Array(String)}
  AND hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) IN {trace_refs:Array(String)}
