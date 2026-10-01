WITH greatest(toInt64({offset:UInt32})-1,1) AS content_offset,
(value, budget) -> if(lengthUTF8(value) <= budget, value,
    concat(substringUTF8(value, 1, intDiv(budget, 3)), '\n[... content omitted ...]\n',
        substringUTF8(value, -(budget - intDiv(budget, 3))))) AS excerpt
SELECT * FROM (
    SELECT SpanId AS span_id, ParentSpanId AS parent_span_id, SpanName AS name,
        ObservationType AS kind,
        if({offset:UInt32}=1 AND lengthUTF8(concat('Input: ',Input,'\nOutput: ',Output,'\nStatus: ',StatusCode,' ',StatusMessage))>8000,
            concat('Input: ',excerpt(Input,2000),'\nOutput: ',excerpt(Output,5000),
                '\nStatus: ',StatusCode,' ',excerpt(StatusMessage,500)),
            substringUTF8(concat('Input: ',Input,'\nOutput: ',Output,'\nStatus: ',StatusCode,' ',StatusMessage),
                content_offset,8000)) AS content,
        lengthUTF8(concat('Input: ',Input,'\nOutput: ',Output,'\nStatus: ',StatusCode,' ',StatusMessage))
            >= content_offset+8000 AS truncated
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
        if({offset:UInt32}=1 AND lengthUTF8(concat('Input: ',messages,'\nOutput: ',response,'\nError: ',error_str))>8000,
            concat('Input: ',excerpt(messages,2000),'\nOutput: ',excerpt(response,5000),'\nError: ',excerpt(error_str,500)),
            substringUTF8(concat('Input: ',messages,'\nOutput: ',response,'\nError: ',error_str),
                content_offset,8000)) AS content,
        lengthUTF8(concat('Input: ',messages,'\nOutput: ',response,'\nError: ',error_str))
            >= content_offset+8000 AS truncated
    FROM spend_logs FINAL WHERE {source:String}='requests'
      AND ({all_teams:UInt8}=1 OR team_id={team:String})
      AND ({key_hash:String}='' OR api_key={key_hash:String})
      AND request_id={id:String} AND team_id={record_team:String} LIMIT 1
)
