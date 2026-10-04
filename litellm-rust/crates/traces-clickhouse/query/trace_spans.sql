WHERE TraceId = {trace_id:String}
  AND hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) = {trace_ref:String}
