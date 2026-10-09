SELECT TeamId AS team_id, ApiKeyHash AS key_hash,
       hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS trace_ref
FROM agent_traces_by_key
WHERE TraceId = {trace_id:String}
  AND ({all_teams:UInt8}=1 OR TeamId={team:String})
  AND ({key_hash:String}='' OR ApiKeyHash={key_hash:String})
  AND ({trace_ref:String}='' OR hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId)))={trace_ref:String})
GROUP BY TeamId, ApiKeyHash, TraceId
LIMIT 2
