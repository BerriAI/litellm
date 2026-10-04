SELECT if({buckets:UInt32} = 0, toUInt32(0),
          toUInt32(intDiv((start_ms - {start_ms:Int64}) * {buckets:UInt32}, {end_ms:Int64} - {start_ms:Int64}))) AS bucket,
       toUInt8({by_failed:UInt8} = 1 AND error_count > 0) AS failed,
       value,
       count() AS runs
FROM runs
ARRAY JOIN multiIf(
    {value:String} = '', [''],
    {value:String} = 'primary_agent', [ifNull(nullIf(arrayElement(search_agents, 1), ''), service)],
    {value:String} = 'name', [name],
    {value:String} = 'agent', search_agents,
    {value:String} = 'status', [search_status],
    {value:String} = 'model', models,
    {value:String} = 'input', [input_preview],
    {value:String} = 'trace_id', [trace_id],
    []) AS value
WHERE ({value:String} = '' OR value != '') AND value ILIKE {contains:String}
GROUP BY bucket, failed, value
ORDER BY runs DESC, bucket, failed, value
LIMIT {limit:UInt64}
