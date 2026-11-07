-- Run once in Athena, in the database you will connect to Hex.
-- Replace YOUR_BUCKET and litellm-spend with HEX_S3_BUCKET_NAME and HEX_S3_PATH.
-- Keep query results in a separate prefix, outside this table's location.
-- Enable the LiteLLM callback with: litellm_settings: {callbacks: [hex]}.
-- Grant the proxy PutObject on the log prefix; grant Hex read-only access to
-- that prefix plus Athena/Glue query permissions and its query-results prefix.
-- Hex supports IAM role authentication; no Hex API key is needed by the proxy.
CREATE EXTERNAL TABLE IF NOT EXISTS litellm_hex_spend (
    id string,
    start_time double,
    end_time double,
    model string,
    model_group string,
    custom_llm_provider string,
    call_type string,
    status string,
    spend double,
    prompt_tokens bigint,
    completion_tokens bigint,
    total_tokens bigint,
    response_time double,
    cache_hit boolean,
    team_id string,
    organization_id string,
    user_id string,
    key_alias string
)
PARTITIONED BY (log_date string)
ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe'
STORED AS TEXTFILE
LOCATION 's3://YOUR_BUCKET/litellm-spend/'
TBLPROPERTIES (
    'projection.enabled' = 'true',
    'projection.log_date.type' = 'date',
    'projection.log_date.range' = '2020-01-01,NOW',
    'projection.log_date.format' = 'yyyy-MM-dd',
    'projection.log_date.interval' = '1',
    'projection.log_date.interval.unit' = 'DAYS',
    'storage.location.template' = 's3://YOUR_BUCKET/litellm-spend/${log_date}/'
);
