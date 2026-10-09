SELECT request_id, litellm_call_id, response_id, upstream_response_id, provider_request_id, trace_id, span_id, team_id,
       api_key, user, spend, toUnixTimestamp64Milli(start_time) AS start_ms
FROM (
    SELECT request_id, litellm_call_id, response_id, upstream_response_id, provider_request_id, trace_id, span_id,
           team_id, api_key, user, spend, start_time
    FROM spend_logs
    WHERE start_time >= fromUnixTimestamp64Milli({start_ms:Int64})
      AND start_time < fromUnixTimestamp64Milli({end_ms:Int64})
      AND (team_id, start_time, request_id) IN (
          SELECT team_id, start_time, request_id
          FROM spend_logs
          WHERE start_time >= fromUnixTimestamp64Milli({start_ms:Int64})
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
      )
    -- Picks the row FINAL keeps: highest end_time, then the newest part, then the last row inserted in it.
    ORDER BY end_time DESC, _block_number DESC, _part_offset DESC
    LIMIT 1 BY team_id, start_time, request_id
)
WHERE ({all_teams:UInt8} = 1
       OR ({user_id:String} != '' AND user = {user_id:String})
       OR has({team_ids:Array(String)}, team_id))
  AND (provider_request_id IN {provider_request_ids:Array(String)}
       OR response_id IN {response_ids:Array(String)}
       OR upstream_response_id IN {response_ids:Array(String)}
       OR litellm_call_id IN {request_ids:Array(String)}
       OR (litellm_call_id = '' AND request_id IN {request_ids:Array(String)})
       OR (trace_id != '' AND trace_id IN {trace_ids:Array(String)}))
  AND ({has_cursor:UInt8} = 0
       OR (team_id, start_ms, request_id) > ({after_team:String}, {after_ms:Int64}, {after_id:String}))
ORDER BY team_id, start_ms, request_id
LIMIT {page_size:UInt32}
