SELECT * FROM (
SELECT request_id, litellm_call_id, response_id, upstream_response_id, trace_id, span_id, team_id, api_key, user, spend,
       toUnixTimestamp64Milli(start_time) AS start_ms
FROM (
    SELECT *,
           -- A chat request served through the Responses API returns the upstream `resp_` id to the
           -- client but logs LiteLLM's managed `resp_<base64>` id, which embeds it.
           if(startsWith(response_id, 'resp_'),
              extract(tryBase64Decode(substring(response_id, 6)), 'response_id:([^;]+)'),
              '') AS upstream_response_id
    FROM owned_calls
    WHERE EngineReceivedMs <= {as_of_ms:UInt64}
      AND start_time >= fromUnixTimestamp64Milli({start_ms:Int64})
      AND start_time < fromUnixTimestamp64Milli({end_ms:Int64})
)
WHERE response_id IN {response_ids:Array(String)}
   OR upstream_response_id IN {response_ids:Array(String)}
   OR litellm_call_id IN {request_ids:Array(String)}
   OR (litellm_call_id = '' AND request_id IN {request_ids:Array(String)})
   OR (trace_id != '' AND trace_id IN {trace_ids:Array(String)})
)
WHERE {has_cursor:UInt8} = 0
   OR (team_id, start_ms, request_id) > ({after_team:String}, {after_ms:Int64}, {after_id:String})
ORDER BY team_id, start_ms, request_id
LIMIT {limit:UInt32}
