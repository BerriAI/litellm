SELECT
    EXISTS(SELECT 1 FROM otel_traces
        WHERE ({all_teams:UInt8}=1 OR TeamId={team:String})
          AND ({key_hash:String}='' OR ApiKeyHash={key_hash:String})) AS traces,
    EXISTS(SELECT 1 FROM spend_logs
        WHERE ({all_teams:UInt8}=1 OR team_id={team:String})
          AND ({key_hash:String}='' OR api_key={key_hash:String})
          AND NOT JSONExtractBool(metadata,'litellm_lens_internal')) AS requests
