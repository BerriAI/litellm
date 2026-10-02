WITH concat(leftPad(toString(cityHash64(concat(source,team_id,trace_ref,trace_id))),20,'0'),
    hex(concat(source,char(0),team_id,char(0),trace_ref,char(0),trace_id))) AS selection_key
SELECT *, selection_key FROM (
    SELECT *, if({sample_cap:UInt64}=0, ceiling(eligible*{sample_percent:Float64}/100),
        least(toFloat64({sample_cap:UInt64}),ceiling(eligible*{sample_percent:Float64}/100))) AS selected
    FROM (
        SELECT *, count() OVER () AS eligible,
            row_number() OVER (ORDER BY selection_key) AS position
        FROM (
    SELECT 'traces' AS source, TraceId AS trace_id, TeamId AS team_id, hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId))) AS trace_ref,
        coalesce(nullIf(argMin(ResourceAttributes['run.name'], Timestamp), ''),
            argMin(SpanName, Timestamp)) AS name, toString(min(Timestamp)) AS start_time,
        uniqExact(SpanId) AS span_count, countIf(ParentSpanId='') > 0 AS root_seen,
        argMin(ServiceName, Timestamp) AS service,
        arrayZip(mapKeys(argMin(mapConcat(ResourceAttributes, SpanAttributes), tuple(ParentSpanId!='',Timestamp))),
            mapValues(argMin(mapConcat(ResourceAttributes, SpanAttributes), tuple(ParentSpanId!='',Timestamp)))) AS attributes
    FROM otel_traces
    WHERE {source:String} IN ('traces','both')
      AND ({all_teams:UInt8}=1 OR TeamId={team:String})
      AND ({key_hash:String}='' OR ApiKeyHash={key_hash:String})
      AND (TeamId,ApiKeyHash,TraceId) IN (
          SELECT TeamId,ApiKeyHash,TraceId FROM otel_traces
          WHERE ({all_teams:UInt8}=1 OR TeamId={team:String})
            AND ({key_hash:String}='' OR ApiKeyHash={key_hash:String})
            AND if(EngineReceivedMs>0,toInt64(EngineReceivedMs),
                toUnixTimestamp64Milli(Timestamp)+toInt64(intDiv(Duration,1000000))) >= {start:UInt64}
      )
    GROUP BY TeamId,ApiKeyHash,TraceId
    HAVING max(EngineReceivedMs) < {end:UInt64}
       AND max(toUnixTimestamp64Milli(Timestamp)+toInt64(intDiv(Duration,1000000))) < {end:UInt64}
       AND ({agent_name:String}='' OR countIf(AgentName={agent_name:String}) > 0)
       AND countIf(arrayAll((k,v) -> ResourceAttributes[k]=v OR SpanAttributes[k]=v,
           {filter_keys:Array(String)},{filter_values:Array(String)})
           AND ({service:String}='' OR ServiceName={service:String})) > 0
    UNION ALL
    SELECT 'requests' AS source, request_id AS trace_id, team_id, '' AS trace_ref, model AS name,
        toString(start_time) AS start_time, toUInt64(1) AS span_count, toUInt8(1) AS root_seen,
        model_group AS service,
        arrayConcat(JSONExtractKeysAndValues(metadata, 'requester_metadata', 'String'),
            arrayMap(t -> tuple('tag', t), request_tags)) AS attributes
    FROM spend_logs FINAL
    WHERE {source:String} IN ('requests','both')
      AND ({all_teams:UInt8}=1 OR team_id={team:String})
      AND ({key_hash:String}='' OR api_key={key_hash:String})
      AND if(EngineReceivedMs>0,toInt64(EngineReceivedMs),toUnixTimestamp64Milli(end_time)) >= {start:UInt64}
      AND EngineReceivedMs < {end:UInt64}
      AND toUnixTimestamp64Milli(end_time) < {end:UInt64}
      AND arrayAll((k,v) -> JSONExtractString(metadata,k)=v
          OR JSONExtractString(metadata,'requester_metadata',k)=v OR (k='tag' AND has(request_tags,v)),
          {filter_keys:Array(String)},{filter_values:Array(String)})
      AND ({service:String}='' OR model_group={service:String})
      AND {agent_name:String}=''
      AND NOT JSONExtractBool(metadata,'litellm_lens_internal')
      AND ({source:String}!='both' OR (team_id,api_key,response_id) NOT IN (
          SELECT TeamId,ApiKeyHash,LiteLLMRequestId FROM otel_traces
          WHERE ({all_teams:UInt8}=1 OR TeamId={team:String})
            AND ({key_hash:String}='' OR ApiKeyHash={key_hash:String}) AND LiteLLMRequestId!=''
      ))
)
WHERE ({selected_team:String}='' OR team_id={selected_team:String})
  AND (empty({execution_ids:Array(String)}) OR has({execution_ids:Array(String)},
      concat(source,char(0),team_id,char(0),if(trace_ref='',trace_id,trace_ref))))
)
)
WHERE ({preview:UInt8}=1 OR position <= selected)
  AND selection_key > {after:String}
ORDER BY selection_key LIMIT {limit:UInt32} OFFSET {offset:UInt64}
