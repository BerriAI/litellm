SELECT sum(matches) AS count FROM (
    SELECT count() AS matches FROM otel_traces WHERE {source:String}='traces'
      AND ({all_teams:UInt8}=1 OR TeamId={team:String})
      AND ({key_hash:String}='' OR ApiKeyHash={key_hash:String})
      AND ({trace_ref:String}='' OR hex(SHA256(concat(TeamId, char(0), ApiKeyHash, char(0), TraceId)))={trace_ref:String})
      AND TraceId={id:String} AND TeamId={record_team:String} AND SpanId={span:String}
      AND position(concat('Input: ',Input,'\nOutput: ',Output,'\nStatus: ',StatusCode,' ',StatusMessage),{quote:String})>0
    UNION ALL
    SELECT count() AS matches FROM spend_logs FINAL WHERE {source:String}='requests'
      AND ({all_teams:UInt8}=1 OR team_id={team:String})
      AND ({key_hash:String}='' OR api_key={key_hash:String})
      AND request_id={id:String} AND team_id={record_team:String} AND request_id={span:String}
      AND position(concat('Input: ',messages,'\nOutput: ',response,'\nError: ',error_str),{quote:String})>0
)
