SELECT DISTINCT AgentName AS agent_name
FROM otel_traces
WHERE AgentName != ''
  AND ({all_teams:UInt8}=1 OR TeamId={team:String})
  AND ({key_hash:String}='' OR ApiKeyHash={key_hash:String})
ORDER BY agent_name
