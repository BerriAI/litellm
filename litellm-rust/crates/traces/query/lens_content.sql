SELECT * FROM (
    SELECT SpanId AS span_id, ParentSpanId AS parent_span_id, SpanName AS name,
        ObservationType AS kind,
        substringUTF8(concat('Input: ',Input,'\nOutput: ',Output,'\nStatus: ',StatusCode,' ',StatusMessage),
            {offset:UInt32},8000) AS content,
        lengthUTF8(concat('Input: ',Input,'\nOutput: ',Output,'\nStatus: ',StatusCode,' ',StatusMessage))
            >= {offset:UInt32}+8000 AS truncated
    FROM otel_traces WHERE {source:String}='traces'
      AND ({all_teams:UInt8}=1 OR TeamId={team:String})
      AND ({key_hash:String}='' OR ApiKeyHash={key_hash:String})
      AND ({trace_ref:String}='' OR hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId)))={trace_ref:String})
      AND TraceId={id:String} AND TeamId={record_team:String} AND SpanId > {cursor:String}
    ORDER BY SpanId LIMIT 1 BY SpanId LIMIT 40
)
UNION ALL
SELECT * FROM (
    SELECT request_id AS span_id, '' AS parent_span_id, model AS name, 'llm' AS kind,
        substringUTF8(concat('Input: ',messages,'\nOutput: ',response,'\nError: ',error_str),
            {offset:UInt32},8000) AS content,
        lengthUTF8(concat('Input: ',messages,'\nOutput: ',response,'\nError: ',error_str))
            >= {offset:UInt32}+8000 AS truncated
    FROM spend_logs FINAL WHERE {source:String}='requests'
      AND ({all_teams:UInt8}=1 OR team_id={team:String})
      AND ({key_hash:String}='' OR api_key={key_hash:String})
      AND request_id={id:String} AND team_id={record_team:String} LIMIT 1
)
