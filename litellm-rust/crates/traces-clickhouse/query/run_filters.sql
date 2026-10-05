
   AND (({trace_id:String} != '' AND trace_id = {trace_id:String})
    OR ({trace_ref:String} != '' AND trace_ref = {trace_ref:String})
    OR ({trace_id:String} = '' AND {trace_ref:String} = ''
        AND min(StartTs) >= fromUnixTimestamp64Milli({start_ms:Int64})
        AND min(StartTs) < fromUnixTimestamp64Milli({end_ms:Int64})
        AND min(StartTs) >= fromUnixTimestamp64Milli({range_start_ms:Int64})
        AND min(StartTs) < fromUnixTimestamp64Milli({range_end_ms:Int64})
        AND arrayAll(t -> trace_id ILIKE t OR input_preview ILIKE t OR name ILIKE t, {text:Array(String)})
        AND arrayAll((f, p, m) -> (m = 'exclude') != multiIf(
                f = 'name', name ILIKE p,
                f = 'agent', arrayExists(a -> a ILIKE p, search_agents),
                f = 'root_status', search_root_status ILIKE p,
                f = 'has_error', search_has_error ILIKE p,
                f = 'model', arrayExists(x -> x ILIKE p, models),
                f = 'input', input_preview ILIKE p,
                f = 'trace_id', trace_id ILIKE p,
                f = 'service', service ILIKE p,
                f = 'team', team_id ILIKE p,
                false),
              {filter_fields:Array(String)}, {filter_patterns:Array(String)}, {filter_modes:Array(String)})
        AND arrayAll((i, m) -> (m = 'exclude') != has(matched_attributes, i),
              arrayEnumerate({attribute_keys:Array(String)}), {attribute_modes:Array(String)})
        AND (empty({trace_refs:Array(String)}) OR trace_ref IN {trace_refs:Array(String)})))
