SELECT TraceId AS trace_id,
       hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS trace_ref,
       count() AS count, avg(Score) AS average, min(Score) AS lowest
FROM lens_feedback FINAL
WHERE TraceId IN {trace_ids:Array(String)}
  AND ({all_teams:UInt8}=1 OR TeamId={team:String})
  AND ({key_hash:String}='' OR ApiKeyHash={key_hash:String})
  AND IsDeleted = 0
GROUP BY TeamId, ApiKeyHash, TraceId
