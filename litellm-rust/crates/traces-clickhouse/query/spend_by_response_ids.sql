WITH candidates AS (
    SELECT DISTINCT request_id, team_id, start_time
    FROM lens_call_costs
    WHERE start_time >= fromUnixTimestamp64Milli({start_ms:Int64})
      AND start_time < fromUnixTimestamp64Milli({end_ms:Int64})
      AND ({all_teams:UInt8} = 1
           OR ({user_id:String} != '' AND user = {user_id:String})
           OR has({team_ids:Array(String)}, team_id))
      AND ((key_kind = 'provider_request' AND key_value IN {provider_request_ids:Array(String)})
        OR (key_kind = 'provider_response' AND key_value IN {response_ids:Array(String)})
        OR (key_kind = 'litellm_request' AND key_value IN {request_ids:Array(String)})
        OR (key_kind = 'transport' AND key_value IN {trace_ids:Array(String)}))
)
SELECT request_id, litellm_call_id, response_id, upstream_response_id, provider_request_id,
       trace_id, span_id, team_id, api_key, user, spend,
       toUnixTimestamp64Milli(start_time) AS start_ms
FROM lens_call_costs FINAL
WHERE key_kind = 'canonical'
  AND (key_value, team_id, start_time) IN (SELECT request_id, team_id, start_time FROM candidates)
  AND start_time >= fromUnixTimestamp64Milli({start_ms:Int64})
  AND start_time < fromUnixTimestamp64Milli({end_ms:Int64})
  AND ({all_teams:UInt8} = 1
       OR ({user_id:String} != '' AND user = {user_id:String})
       OR has({team_ids:Array(String)}, team_id))
  AND (provider_request_id IN {provider_request_ids:Array(String)}
    OR response_id IN {response_ids:Array(String)}
    OR upstream_response_id IN {response_ids:Array(String)}
    OR litellm_call_id IN {request_ids:Array(String)}
    OR (litellm_call_id = '' AND request_id IN {request_ids:Array(String)})
    OR (trace_id != '' AND trace_id IN {trace_ids:Array(String)}))
ORDER BY start_time DESC
