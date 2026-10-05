SELECT if({buckets:UInt32} = 0, toUInt32(0),
          toUInt32(intDiv((toInt128(start_ms) - toInt128({start_ms:Int64}) + 1) * {buckets:UInt32} - 1, toInt128({end_ms:Int64}) - toInt128({start_ms:Int64})))) AS bucket,
       toUInt8({by_failed:UInt8} = 1 AND error_count > 0) AS failed,
       value,
       count() AS runs
FROM runs
ARRAY JOIN multiIf(
    {value:String} = '', [''],
    {value:String} = 'primary_agent', [ifNull(nullIf(arrayElement(search_agents, 1), ''), service)],
    {value:String} = 'name', [name],
    {value:String} = 'agent', search_agents,
    {value:String} = 'root_status', [search_root_status],
    {value:String} = 'has_error', [search_has_error],
    {value:String} = 'model', models,
    {value:String} = 'input', [input_preview],
    {value:String} = 'trace_id', [trace_id],
    {value:String} = 'service', [service],
    {value:String} = 'team', [team_id],
    []) AS value
WHERE ({value:String} IN ('', 'primary_agent') OR value != '') AND value ILIKE {contains:String}
GROUP BY bucket, failed, value
ORDER BY runs DESC, bucket, failed, value
LIMIT {limit:UInt64}
