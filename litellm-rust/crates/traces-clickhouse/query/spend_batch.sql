SELECT * FROM (
SELECT request_id, litellm_call_id, response_id, upstream_response_id, team_id, spend,
       toUnixTimestamp64Milli(start_time) AS start_ms
FROM (
    SELECT *,
           -- A chat request served through the Responses API returns the upstream `resp_` id to the
           -- client but logs LiteLLM's managed `resp_<base64>` id, which embeds it.
           if(startsWith(response_id, 'resp_'),
              extract(tryBase64Decode(substring(response_id, 6)), 'response_id:([^;]+)'),
              '') AS upstream_response_id
    FROM spend_logs FINAL
    WHERE start_time >= fromUnixTimestamp64Milli({start_ms:Int64})
      AND start_time < fromUnixTimestamp64Milli({end_ms:Int64})
      AND ({all_teams:UInt8} = 1
           OR ({user_id:String} != '' AND user = {user_id:String})
           OR has({team_ids:Array(String)}, team_id))
)
WHERE response_id IN {response_ids:Array(String)}
   OR upstream_response_id IN {response_ids:Array(String)}
   OR (litellm_call_id != '' AND litellm_call_id IN {call_ids:Array(String)})
ORDER BY start_time DESC
)
WHERE {has_cursor:UInt8} = 0
   OR (team_id, start_ms, request_id) > ({after_team:String}, {after_ms:Int64}, {after_id:String})
ORDER BY team_id, start_ms, request_id
LIMIT {page_size:UInt32}
