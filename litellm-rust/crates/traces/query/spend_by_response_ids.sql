SELECT request_id, response_id, team_id, api_key, spend,
       toUnixTimestamp64Milli(start_time) AS start_ms
FROM spend_logs FINAL
WHERE response_id IN {response_ids:Array(String)}
  AND start_time >= fromUnixTimestamp64Milli({start_ms:Int64})
  AND start_time < fromUnixTimestamp64Milli({end_ms:Int64})
  AND (empty({team_ids:Array(String)}) OR team_id IN {team_ids:Array(String)})
  AND ({api_key_hash:String} = '' OR api_key = {api_key_hash:String})
ORDER BY start_time DESC
