SELECT TraceId AS trace_id,
       hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS trace_ref,
       Author AS author, Score AS score, Comment AS comment,
       formatDateTime(CreatedAt, '%FT%T.%fZ', 'UTC') AS created_at,
       formatDateTime(UpdatedAt, '%FT%T.%fZ', 'UTC') AS updated_at
FROM lens_feedback FINAL
WHERE TraceId = {trace_id:String}
  AND hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) = {trace_ref:String}
  AND ({all_teams:UInt8}=1 OR TeamId={team:String})
  AND ({key_hash:String}='' OR ApiKeyHash={key_hash:String})
  AND IsDeleted = 0
ORDER BY UpdatedAt DESC, Author
