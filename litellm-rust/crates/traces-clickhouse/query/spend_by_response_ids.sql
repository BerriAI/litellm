SELECT request_id, response_id, team_id, api_key, user, spend,
       toUnixTimestamp64Milli(start_time) AS start_ms
FROM spend_logs FINAL
WHERE response_id IN {response_ids:Array(String)}
  AND start_time >= fromUnixTimestamp64Milli({start_ms:Int64})
  AND start_time < fromUnixTimestamp64Milli({end_ms:Int64})
  AND ({all_teams:UInt8} = 1
       OR ({user_id:String} != '' AND user = {user_id:String})
       OR has({team_ids:Array(String)}, team_id))
ORDER BY start_time DESC
