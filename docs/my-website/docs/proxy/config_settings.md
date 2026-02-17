# All settings

```yaml
environment_variables: {}

model_list:
  - model_name: string
    litellm_params: {}
    model_info:
      id: string
      mode: embedding
      input_cost_per_token: 0
      output_cost_per_token: 0
      max_tokens: 2048
      base_model: gpt-4-1106-preview
      additionalProp1: {}

litellm_settings:
  # Logging/Callback settings
  success_callback: ["langfuse"]  # list of success callbacks
  failure_callback: ["sentry"]  # list of failure callbacks
  callbacks: ["otel"]  # list of callbacks - runs on success and failure
  service_callbacks: ["datadog", "prometheus"]  # logs redis, postgres failures on datadog, prometheus
  turn_off_message_logging: boolean  # prevent the messages and responses from being logged to on your callbacks, but request metadata will still be logged. Useful for privacy/compliance when handling sensitive data.
  redact_user_api_key_info: boolean  # Redact information about the user api key (hashed token, user_id, team id, etc.), from logs. Currently supported for Langfuse, OpenTelemetry, Logfire, ArizeAI logging.
  langfuse_default_tags: ["cache_hit", "cache_key", "proxy_base_url", "user_api_key_alias", "user_api_key_user_id", "user_api_key_user_email", "user_api_key_team_alias", "semantic-similarity", "proxy_base_url"] # default tags for Langfuse Logging
  # Networking settings
  request_timeout: 10 # (int) llm requesttimeout in seconds. Raise Timeout error if call takes longer than 10s. Sets litellm.request_timeout
  force_ipv4: boolean # If true, litellm will force ipv4 for all LLM requests. Some users have seen httpx ConnectionError when using ipv6 + Anthropic API
  
  # Debugging - see debugging docs for more options
  # Use `--debug` or `--detailed_debug` CLI flags, or set LITELLM_LOG env var to "INFO", "DEBUG", or "ERROR"
  json_logs: boolean # if true, logs will be in json format

  # Fallbacks, reliability
  default_fallbacks: ["claude-opus"] # set default_fallbacks, in case a specific model group is misconfigured / bad.
  content_policy_fallbacks: [{ "gpt-3.5-turbo-small": ["claude-opus"] }] # fallbacks for ContentPolicyErrors
  context_window_fallbacks: [{ "gpt-3.5-turbo-small": ["gpt-3.5-turbo-large", "claude-opus"] }] # fallbacks for ContextWindowExceededErrors

  # MCP Aliases - Map aliases to MCP server names for easier tool access
  mcp_aliases: {
      "github": "github_mcp_server",
      "zapier": "zapier_mcp_server",
      "deepwiki": "deepwiki_mcp_server",
    } # Maps friendly aliases to MCP server names. Only the first alias for each server is used

  # Caching settings
  cache: true
  cache_params: # set cache params for redis
    type: redis # type of cache to initialize (options: "local", "redis", "s3", "gcs")

    # Optional - Redis Settings
    host: "localhost" # The host address for the Redis cache. Required if type is "redis".
    port: 6379 # The port number for the Redis cache. Required if type is "redis".
    password: "your_password" # The password for the Redis cache. Required if type is "redis".
    namespace: "litellm.caching.caching" # namespace for redis cache
    max_connections: 100  # [OPTIONAL] Set Maximum number of Redis connections. Passed directly to redis-py. 
    # Optional - Redis Cluster Settings
    redis_startup_nodes: [{ "host": "127.0.0.1", "port": "7001" }]

    # Optional - Redis Sentinel Settings
    service_name: "mymaster"
    sentinel_nodes: [["localhost", 26379]]

    # Optional - GCP IAM Authentication for Redis
    gcp_service_account: "projects/-/serviceAccounts/your-sa@project.iam.gserviceaccount.com" # GCP service account for IAM authentication
    gcp_ssl_ca_certs: "./server-ca.pem" # Path to SSL CA certificate file for GCP Memorystore Redis
    ssl: true # Enable SSL for secure connections
    ssl_cert_reqs: null # Set to null for self-signed certificates
    ssl_check_hostname: false # Set to false for self-signed certificates

    # Optional - Qdrant Semantic Cache Settings
    qdrant_semantic_cache_embedding_model: openai-embedding # the model should be defined on the model_list
    qdrant_collection_name: test_collection
    qdrant_quantization_config: binary
    similarity_threshold: 0.8 # similarity threshold for semantic cache

    # Optional - S3 Cache Settings
    s3_bucket_name: cache-bucket-litellm # AWS Bucket Name for S3
    s3_region_name: us-west-2 # AWS Region Name for S3
    s3_aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID # us os.environ/<variable name> to pass environment variables. This is AWS Access Key ID for S3
    s3_aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY # AWS Secret Access Key for S3
    s3_endpoint_url: https://s3.amazonaws.com # [OPTIONAL] S3 endpoint URL, if you want to use Backblaze/cloudflare s3 bucket

    # Optional - GCS Cache Settings
    gcs_bucket_name: cache-bucket-litellm # GCS Bucket Name for caching
    gcs_path_service_account: os.environ/GCS_PATH_SERVICE_ACCOUNT # Path to GCS service account JSON file
    gcs_path: cache/ # [OPTIONAL] GCS path prefix for cache objects

    # Common Cache settings
    # Optional - Supported call types for caching
    supported_call_types:
      ["acompletion", "atext_completion", "aembedding", "atranscription"]
      # /chat/completions, /completions, /embeddings, /audio/transcriptions
    mode: default_off # if default_off, you need to opt in to caching on a per call basis
    ttl: 600 # ttl for caching
    disable_copilot_system_to_assistant: False # DEPRECATED - GitHub Copilot API supports system prompts.

callback_settings:
  otel:
    message_logging: boolean # OTEL logging callback specific settings

general_settings:
  completion_model: string
  store_prompts_in_spend_logs: boolean
  forward_client_headers_to_llm_api: boolean
  disable_spend_logs: boolean  # turn off writing each transaction to the db
  disable_master_key_return: boolean  # turn off returning master key on UI (checked on '/user/info' endpoint)
  disable_retry_on_max_parallel_request_limit_error: boolean  # turn off retries when max parallel request limit is reached
  disable_reset_budget: boolean  # turn off reset budget scheduled task
  disable_adding_master_key_hash_to_db: boolean  # turn off storing master key hash in db, for spend tracking
  disable_responses_id_security: boolean  # turn off response ID security checks that prevent users from accessing other users' responses
  enable_jwt_auth: boolean  # allow proxy admin to auth in via jwt tokens with 'litellm_proxy_admin' in claims
  enforce_user_param: boolean  # requires all openai endpoint requests to have a 'user' param
  reject_clientside_metadata_tags: boolean  # if true, rejects requests with client-side 'metadata.tags' to prevent users from influencing budgets
  allowed_routes: ["route1", "route2"]  # list of allowed proxy API routes - a user can access. (currently JWT-Auth only)
  key_management_system: google_kms  # either google_kms or azure_kms
  master_key: string
  maximum_spend_logs_retention_period: 30d # The maximum time to retain spend logs before deletion.
  maximum_spend_logs_retention_interval: 1d # interval in which the spend log cleanup task should run in.
  user_mcp_management_mode: restricted  # or "view_all"

  # Database Settings
  database_url: string
  database_connection_pool_limit: 0  # default 10
  database_connection_timeout: 0  # default 60s
  allow_requests_on_db_unavailable: boolean  # if true, will allow requests that can not connect to the DB to verify Virtual Key to still work 

  custom_auth: string
  max_parallel_requests: 0 # the max parallel requests allowed per deployment
  global_max_parallel_requests: 0 # the max parallel requests allowed on the proxy all up
  infer_model_from_keys: true
  background_health_checks: true
  health_check_interval: 300
  alerting: ["slack", "email"]
  alerting_threshold: 0
  use_client_credentials_pass_through_routes: boolean  # use client credentials for all pass through routes like "/vertex-ai", /bedrock/. When this is True Virtual Key auth will not be applied on these endpoints

router_settings:
  routing_strategy: simple-shuffle # Literal["simple-shuffle", "least-busy", "usage-based-routing","latency-based-routing"], default="simple-shuffle" - RECOMMENDED for best performance
  redis_host: <your-redis-host>           # string
  redis_password: <your-redis-password>   # string
  redis_port: <your-redis-port>           # string
  enable_pre_call_checks: true            # bool - Before call is made check if a call is within model context window 
  allowed_fails: 3 # cooldown model if it fails > 1 call in a minute. 
  cooldown_time: 30 # (in seconds) how long to cooldown model if fails/min > allowed_fails
  disable_cooldowns: True                  # bool - Disable cooldowns for all models 
  enable_tag_filtering: True                # bool - Use tag based routing for requests
  tag_filtering_match_any: True             # bool - Tag matching behavior (only when enable_tag_filtering=true). `true`: match if deployment has ANY requested tag; `false`: match only if deployment has ALL requested tags
  retry_policy: {                          # Dict[str, int]: retry policy for different types of exceptions
    "AuthenticationErrorRetries": 3,
    "TimeoutErrorRetries": 3,
    "RateLimitErrorRetries": 3,
    "ContentPolicyViolationErrorRetries": 4,
    "InternalServerErrorRetries": 4
  }
  allowed_fails_policy: {
    "BadRequestErrorAllowedFails": 1000, # Allow 1000 BadRequestErrors before cooling down a deployment
    "AuthenticationErrorAllowedFails": 10, # int 
    "TimeoutErrorAllowedFails": 12, # int 
    "RateLimitErrorAllowedFails": 10000, # int 
    "ContentPolicyViolationErrorAllowedFails": 15, # int 
    "InternalServerErrorAllowedFails": 20, # int 
  }
  content_policy_fallbacks=[{"claude-2": ["my-fallback-model"]}] # List[Dict[str, List[str]]]: Fallback model for content policy violations
  fallbacks=[{"claude-2": ["my-fallback-model"]}] # List[Dict[str, List[str]]]: Fallback model for all errors

```

### litellm_settings - Reference

| Name | Type | Description |
|------|------|-------------|
| success_callback | array of strings | List of success callbacks. [Doc Proxy logging callbacks](logging), [Doc Metrics](prometheus) |
| failure_callback | array of strings | List of failure callbacks [Doc Proxy logging callbacks](logging), [Doc Metrics](prometheus) |
| callbacks | array of strings | List of callbacks - runs on success and failure [Doc Proxy logging callbacks](logging), [Doc Metrics](prometheus) |
| service_callbacks | array of strings | System health monitoring - Logs redis, postgres failures on specified services (e.g. datadog, prometheus) [Doc Metrics](prometheus) |
| turn_off_message_logging | boolean | If true, prevents messages and responses from being logged to callbacks, but request metadata will still be logged. Useful for privacy/compliance when handling sensitive data [Proxy Logging](logging) |
| modify_params | boolean | If true, allows modifying the parameters of the request before it is sent to the LLM provider |
| enable_preview_features | boolean | If true, enables preview features - e.g. Azure O1 Models with streaming support.|
| LITELLM_DISABLE_STOP_SEQUENCE_LIMIT | Disable validation for stop sequence limit (default: 4) |  
| redact_user_api_key_info | boolean | If true, redacts information about the user api key from logs [Proxy Logging](logging#redacting-userapikeyinfo) |
| mcp_aliases | object | Maps friendly aliases to MCP server names for easier tool access. Only the first alias for each server is used. [MCP Aliases](../mcp#mcp-aliases) |
| langfuse_default_tags | array of strings | Default tags for Langfuse Logging. Use this if you want to control which LiteLLM-specific fields are logged as tags by the LiteLLM proxy. By default LiteLLM Proxy logs no LiteLLM-specific fields as tags. [Further docs](./logging#litellm-specific-tags-on-langfuse---cache_hit-cache_key) |
| set_verbose | boolean | [DEPRECATED - see debugging docs](./debugging) Use `--debug` or `--detailed_debug` CLI flags, or set `LITELLM_LOG` env var to "INFO", "DEBUG", or "ERROR" instead. |
| json_logs | boolean | If true, logs will be in json format. If you need to store the logs as JSON, just set the `litellm.json_logs = True`. We currently just log the raw POST request from litellm as a JSON [Further docs](./debugging) |
| default_fallbacks | array of strings | List of fallback models to use if a specific model group is misconfigured / bad. [Further docs](./reliability#default-fallbacks) |
| request_timeout | integer | The timeout for requests in seconds. If not set, the default value is `6000 seconds`. [For reference OpenAI Python SDK defaults to `600 seconds`.](https://github.com/openai/openai-python/blob/main/src/openai/_constants.py) |
| force_ipv4 | boolean | If true, litellm will force ipv4 for all LLM requests. Some users have seen httpx ConnectionError when using ipv6 + Anthropic API |
| content_policy_fallbacks | array of objects | Fallbacks to use when a ContentPolicyViolationError is encountered. [Further docs](./reliability#content-policy-fallbacks) |
| context_window_fallbacks | array of objects | Fallbacks to use when a ContextWindowExceededError is encountered. [Further docs](./reliability#context-window-fallbacks) |
| cache | boolean | If true, enables caching. [Further docs](./caching) |
| cache_params | object | Parameters for the cache. [Further docs](./caching#supported-cache_params-on-proxy-configyaml) |
| disable_end_user_cost_tracking | boolean | If true, turns off end user cost tracking on prometheus metrics + litellm spend logs table on proxy. |
| disable_end_user_cost_tracking_prometheus_only | boolean | If true, turns off end user cost tracking on prometheus metrics only. |
| key_generation_settings | object | Restricts who can generate keys. [Further docs](./virtual_keys.md#restricting-key-generation) |
| disable_add_transform_inline_image_block | boolean | For Fireworks AI models - if true, turns off the auto-add of `#transform=inline` to the url of the image_url, if the model is not a vision model. |
| disable_hf_tokenizer_download | boolean | If true, it defaults to using the openai tokenizer for all models (including huggingface models). |
| enable_json_schema_validation | boolean | If true, enables json schema validation for all requests. |
| disable_copilot_system_to_assistant | boolean | **DEPRECATED** - GitHub Copilot API supports system prompts. |

### general_settings - Reference

| Name | Type | Description |
|------|------|-------------|
| completion_model | string | The default model to use for completions when `model` is not specified in the request |
| disable_spend_logs | boolean | If true, turns off writing each transaction to the database |
| disable_spend_updates | boolean | If true, turns off all spend updates to the DB. Including key/user/team spend updates. |
| disable_master_key_return | boolean | If true, turns off returning master key on UI. (checked on '/user/info' endpoint) |
| disable_retry_on_max_parallel_request_limit_error | boolean | If true, turns off retries when max parallel request limit is reached |
| disable_reset_budget | boolean | If true, turns off reset budget scheduled task |
| disable_adding_master_key_hash_to_db | boolean | If true, turns off storing master key hash in db |
| disable_responses_id_security | boolean | If true, disables response ID security checks that prevent users from accessing response IDs from other users. When false (default), response IDs are encrypted with user information to ensure users can only access their own responses. Applies to /v1/responses endpoints |
| enable_jwt_auth | boolean | allow proxy admin to auth in via jwt tokens with 'litellm_proxy_admin' in claims. [Doc on JWT Tokens](token_auth) |
| enforce_user_param | boolean | If true, requires all OpenAI endpoint requests to have a 'user' param. [Doc on call hooks](call_hooks)|
| reject_clientside_metadata_tags | boolean | If true, rejects requests that contain client-side 'metadata.tags' to prevent users from influencing budgets by sending different tags. Tags can only be inherited from the API key metadata. |
| allowed_routes | array of strings | List of allowed proxy API routes a user can access [Doc on controlling allowed routes](enterprise#control-available-public-private-routes)|
| key_management_system | string | Specifies the key management system. [Doc Secret Managers](../secret) |
| master_key | string | The master key for the proxy [Set up Virtual Keys](virtual_keys) |
| database_url | string | The URL for the database connection [Set up Virtual Keys](virtual_keys) |
| database_connection_pool_limit | integer | The limit for database connection pool [Setting DB Connection Pool limit](#configure-db-pool-limits--connection-timeouts) |
| database_connection_timeout | integer | The timeout for database connections in seconds [Setting DB Connection Pool limit, timeout](#configure-db-pool-limits--connection-timeouts) |
| allow_requests_on_db_unavailable | boolean | If true, allows requests to succeed even if DB is unreachable. **Only use this if running LiteLLM in your VPC** This will allow requests to work even when LiteLLM cannot connect to the DB to verify a Virtual Key [Doc on graceful db unavailability](prod#5-if-running-litellm-on-vpc-gracefully-handle-db-unavailability) |
| custom_auth | string | Write your own custom authentication logic [Doc Custom Auth](virtual_keys#custom-auth) |
| max_parallel_requests | integer | The max parallel requests allowed per deployment |
| global_max_parallel_requests | integer | The max parallel requests allowed on the proxy overall |
| infer_model_from_keys | boolean | If true, infers the model from the provided keys |
| background_health_checks | boolean | If true, enables background health checks. [Doc on health checks](health) |
| health_check_interval | integer | The interval for health checks in seconds [Doc on health checks](health) |
| alerting | array of strings | List of alerting methods [Doc on Slack Alerting](alerting) |
| alerting_threshold | integer | The threshold for triggering alerts [Doc on Slack Alerting](alerting) |
| use_client_credentials_pass_through_routes | boolean | If true, uses client credentials for all pass-through routes. [Doc on pass through routes](pass_through) |
| health_check_details | boolean | If false, hides health check details (e.g. remaining rate limit). [Doc on health checks](health) |
| public_routes | List[str] | (Enterprise Feature) Control list of public routes |
| alert_types | List[str] | Control list of alert types to send to slack (Doc on alert types)[./alerting.md] |
| enforced_params | List[str] | (Enterprise Feature) List of params that must be included in all requests to the proxy |
| enable_oauth2_auth | boolean | (Enterprise Feature) If true, enables oauth2.0 authentication |
| use_x_forwarded_for | str | If true, uses the X-Forwarded-For header to get the client IP address |
| service_account_settings | List[Dict[str, Any]] | Set `service_account_settings` if you want to create settings that only apply to service account keys (Doc on service accounts)[./service_accounts.md] | 
| image_generation_model | str | The default model to use for image generation - ignores model set in request |
| store_model_in_db | boolean | If true, enables storing model + credential information in the DB. |
| supported_db_objects | List[str] | Fine-grained control over which object types to load from the database when `store_model_in_db` is True. Available types: `"models"`, `"mcp"`, `"guardrails"`, `"vector_stores"`, `"pass_through_endpoints"`, `"prompts"`, `"model_cost_map"`. If not set, all object types are loaded (default behavior). Example: `supported_db_objects: ["mcp"]` to only load MCP servers from DB. |
| user_mcp_management_mode | string | Controls what non-admins can see on the MCP dashboard. `restricted` (default) only lists MCP servers that the user’s teams are explicitly allowed to access. `view_all` lets every user see the full MCP server list. Tool list/call always respects per-key permissions, so users still cannot run MCP calls without access. |
| store_prompts_in_spend_logs | boolean | If true, allows prompts and responses to be stored in the spend logs table. |
| max_request_size_mb | int | The maximum size for requests in MB. Requests above this size will be rejected. |
| max_response_size_mb | int | The maximum size for responses in MB. LLM Responses above this size will not be sent. |
| proxy_budget_rescheduler_min_time | int | The minimum time (in seconds) to wait before checking db for budget resets. **Default is 597 seconds** |
| proxy_budget_rescheduler_max_time | int | The maximum time (in seconds) to wait before checking db for budget resets. **Default is 605 seconds** |
| proxy_batch_write_at | int | Time (in seconds) to wait before batch writing spend logs to the db. **Default is 10 seconds** |
| proxy_batch_polling_interval | int | Time (in seconds) to wait before polling a batch, to check if it's completed. **Default is 6000 seconds (1 hour)** |
| alerting_args | dict | Args for Slack Alerting [Doc on Slack Alerting](./alerting.md) |
| custom_key_generate | str | Custom function for key generation [Doc on custom key generation](./virtual_keys.md#custom--key-generate) |
| allowed_ips | List[str] | List of IPs allowed to access the proxy. If not set, all IPs are allowed. |
| embedding_model | str | The default model to use for embeddings - ignores model set in request |
| default_team_disabled | boolean | If true, users cannot create 'personal' keys (keys with no team_id). |
| alert_to_webhook_url | Dict[str] | [Specify a webhook url for each alert type.](./alerting.md#set-specific-slack-channels-per-alert-type) |
| key_management_settings | List[Dict[str, Any]] | Settings for key management system (e.g. AWS KMS, Azure Key Vault) [Doc on key management](../secret.md) |
| allow_user_auth | boolean | (Deprecated) old approach for user authentication. |
| user_api_key_cache_ttl | int | The time (in seconds) to cache user api keys in memory. |
| disable_prisma_schema_update | boolean | If true, turns off automatic schema updates to DB |
| litellm_key_header_name | str | If set, allows passing LiteLLM keys as a custom header. [Doc on custom headers](./virtual_keys.md#custom-headers) |
| moderation_model | str | The default model to use for moderation. |
| custom_sso | str | Path to a python file that implements custom SSO logic. [Doc on custom SSO](./custom_sso.md) |
| allow_client_side_credentials | boolean | If true, allows passing client side credentials to the proxy. (Useful when testing finetuning models) [Doc on client side credentials](./virtual_keys.md#client-side-credentials) |
| admin_only_routes | List[str] | (Enterprise Feature) List of routes that are only accessible to admin users. [Doc on admin only routes](./enterprise#control-available-public-private-routes) |
| use_azure_key_vault | boolean | If true, load keys from azure key vault | 
| use_google_kms | boolean | If true, load keys from google kms |
| spend_report_frequency | str | Specify how often you want a Spend Report to be sent (e.g. "1d", "2d", "30d") [More on this](./alerting.md#spend-report-frequency) |
| ui_access_mode | Literal["admin_only"] | If set, restricts access to the UI to admin users only. [Docs](./ui.md#restrict-ui-access) |
| litellm_jwtauth | Dict[str, Any] | Settings for JWT authentication. [Docs](./token_auth.md) |
| litellm_license | str | The license key for the proxy. [Docs](../enterprise.md#how-does-deployment-with-enterprise-license-work) |
| oauth2_config_mappings | Dict[str, str] | Define the OAuth2 config mappings | 
| pass_through_endpoints | List[Dict[str, Any]] | Define the pass through endpoints. [Docs](./pass_through) |
| enable_oauth2_proxy_auth | boolean | (Enterprise Feature) If true, enables oauth2.0 authentication |
| forward_openai_org_id | boolean | If true, forwards the OpenAI Organization ID to the backend LLM call (if it's OpenAI). |
| forward_client_headers_to_llm_api | boolean | If true, forwards the client headers (any `x-` headers and `anthropic-beta` headers) to the backend LLM call |
| maximum_spend_logs_retention_period               | str                   | Used to set the max retention time for spend logs in the db, after which they will be auto-purged                                                                                                                                                                                                                             |
| maximum_spend_logs_retention_interval             | str                   | Used to set the interval in which the spend log cleanup task should run in.                                                                                                                                                                                                                                                   |

### router_settings - Reference

:::info

Most values can also be set via `litellm_settings`. If you see overlapping values, settings on
`router_settings` will override those on `litellm_settings`. :::

```yaml
router_settings:
  routing_strategy: simple-shuffle # Literal["simple-shuffle", "least-busy", "usage-based-routing","latency-based-routing"], default="simple-shuffle" - RECOMMENDED for best performance
  redis_host: <your-redis-host>           # string
  redis_password: <your-redis-password>   # string
  redis_port: <your-redis-port>           # string
  enable_pre_call_checks: true            # bool - Before call is made check if a call is within model context window
  allowed_fails: 3 # cooldown model if it fails > 1 call in a minute.
  cooldown_time: 30 # (in seconds) how long to cooldown model if fails/min > allowed_fails
  disable_cooldowns: True                  # bool - Disable cooldowns for all models
  enable_tag_filtering: True                # bool - Use tag based routing for requests
  tag_filtering_match_any: True             # bool - Tag matching behavior (only when enable_tag_filtering=true). `true`: match if deployment has ANY requested tag; `false`: match only if deployment has ALL requested tags
  retry_policy: {                          # Dict[str, int]: retry policy for different types of exceptions
    "AuthenticationErrorRetries": 3,
    "TimeoutErrorRetries": 3,
    "RateLimitErrorRetries": 3,
    "ContentPolicyViolationErrorRetries": 4,
    "InternalServerErrorRetries": 4
  }
  allowed_fails_policy: {
    "BadRequestErrorAllowedFails": 1000, # Allow 1000 BadRequestErrors before cooling down a deployment
    "AuthenticationErrorAllowedFails": 10, # int
    "TimeoutErrorAllowedFails": 12, # int
    "RateLimitErrorAllowedFails": 10000, # int
    "ContentPolicyViolationErrorAllowedFails": 15, # int
    "InternalServerErrorAllowedFails": 20, # int
  }
  content_policy_fallbacks=[{"claude-2": ["my-fallback-model"]}] # List[Dict[str, List[str]]]: Fallback model for content policy violations
  fallbacks=[{"claude-2": ["my-fallback-model"]}] # List[Dict[str, List[str]]]: Fallback model for all errors
```

| Name | Type | Description |
|------|------|-------------|
| routing_strategy | string | The strategy used for routing requests. Options: "simple-shuffle", "least-busy", "usage-based-routing", "latency-based-routing". Default is "simple-shuffle". [More information here](../routing) |
| redis_host | string | The host address for the Redis server. **Only set this if you have multiple instances of LiteLLM Proxy and want current tpm/rpm tracking to be shared across them** |
| redis_password | string | The password for the Redis server. **Only set this if you have multiple instances of LiteLLM Proxy and want current tpm/rpm tracking to be shared across them** |
| redis_port | string | The port number for the Redis server. **Only set this if you have multiple instances of LiteLLM Proxy and want current tpm/rpm tracking to be shared across them**|
| redis_db | int | The database number for the Redis server. **Only set this if you have multiple instances of LiteLLM Proxy and want current tpm/rpm tracking to be shared across them**|
| enable_pre_call_check | boolean | If true, checks if a call is within the model's context window before making the call. [More information here](reliability) |
| content_policy_fallbacks | array of objects | Specifies fallback models for content policy violations. [More information here](reliability) |
| fallbacks | array of objects | Specifies fallback models for all types of errors. [More information here](reliability) |
| enable_tag_filtering | boolean | If true, uses tag based routing for requests [Tag Based Routing](tag_routing) |
| tag_filtering_match_any | boolean | Tag matching behavior (only when enable_tag_filtering=true). `true`: match if deployment has ANY requested tag; `false`: match only if deployment has ALL requested tags |
| cooldown_time | integer | The duration (in seconds) to cooldown a model if it exceeds the allowed failures. |
| disable_cooldowns | boolean | If true, disables cooldowns for all models. [More information here](reliability) |
| retry_policy | object | Specifies the number of retries for different types of exceptions. [More information here](reliability) |
| allowed_fails | integer | The number of failures allowed before cooling down a model. [More information here](reliability) |
| allowed_fails_policy | object | Specifies the number of allowed failures for different error types before cooling down a deployment. [More information here](reliability) |
| default_max_parallel_requests | Optional[int] | The default maximum number of parallel requests for a deployment. |
| default_priority | (Optional[int]) | The default priority for a request. Only for '.scheduler_acompletion()'. Default is None. | 
| polling_interval | (Optional[float]) | frequency of polling queue. Only for '.scheduler_acompletion()'. Default is 3ms. |
| max_fallbacks | Optional[int] | The maximum number of fallbacks to try before exiting the call. Defaults to 5. |
| default_litellm_params | Optional[dict] | The default litellm parameters to add to all requests (e.g. `temperature`, `max_tokens`). |
| timeout | Optional[float] | The default timeout for a request. Default is 10 minutes. |
| stream_timeout | Optional[float] | The default timeout for a streaming request. If not set, the 'timeout' value is used. |
| debug_level | Literal["DEBUG", "INFO"] | The debug level for the logging library in the router. Defaults to "INFO". |
| client_ttl | int | Time-to-live for cached clients in seconds. Defaults to 3600. |
| cache_kwargs | dict | Additional keyword arguments for the cache initialization. Use this for non-string Redis parameters that may fail when set via `REDIS_*` environment variables. |
| routing_strategy_args | dict | Additional keyword arguments for the routing strategy - e.g. lowest latency routing default ttl |
| model_group_alias | dict | Model group alias mapping. E.g. `{"claude-3-haiku": "claude-3-haiku-20240229"}` |
| num_retries | int | Number of retries for a request. Defaults to 3. |
| default_fallbacks | Optional[List[str]] | Fallbacks to try if no model group-specific fallbacks are defined. |
| caching_groups | Optional[List[tuple]] | List of model groups for caching across model groups. Defaults to None. - e.g. caching_groups=[("openai-gpt-3.5-turbo", "azure-gpt-3.5-turbo")]|
| alerting_config | AlertingConfig | [SDK-only arg] Slack alerting configuration. Defaults to None. [Further Docs](../routing.md#alerting-) |
| assistants_config | AssistantsConfig | Set on proxy via `assistant_settings`. [Further docs](../assistants.md) |
| set_verbose | boolean | [DEPRECATED PARAM - see debug docs](./debugging) If true, sets the logging level to verbose. |
| retry_after | int | Time to wait before retrying a request in seconds. Defaults to 0. If `x-retry-after` is received from LLM API, this value is overridden. |
| provider_budget_config | ProviderBudgetConfig | Provider budget configuration. Use this to set llm_provider budget limits. example $100/day to OpenAI, $100/day to Azure, etc. Defaults to None. [Further Docs](./provider_budget_routing.md) |
| enable_pre_call_checks | boolean | If true, checks if a call is within the model's context window before making the call. [More information here](reliability) |
| model_group_retry_policy | Dict[str, RetryPolicy] | [SDK-only arg] Set retry policy for model groups. |
| context_window_fallbacks | List[Dict[str, List[str]]] | Fallback models for context window violations. |
| redis_url | str | URL for Redis server. **Known performance issue with Redis URL.** |
| cache_responses | boolean | Flag to enable caching LLM Responses, if cache set under `router_settings`. If true, caches responses. Defaults to False. |
| router_general_settings | RouterGeneralSettings | [SDK-Only] Router general settings - contains optimizations like 'async_only_mode'. [Docs](../routing.md#router-general-settings) |
| optional_pre_call_checks | List[str] | List of pre-call checks to add to the router. Currently supported: 'router_budget_limiting', 'prompt_caching' |
| ignore_invalid_deployments | boolean | If true, ignores invalid deployments. Default for proxy is True - to prevent invalid models from blocking other models from being loaded. |
| search_tools | List[SearchToolTypedDict] | List of search tool configurations for Search API integration. Each tool specifies a search_tool_name and litellm_params with search_provider, api_key, api_base, etc. [Further Docs](../search.md) |
| guardrail_list | List[GuardrailTypedDict] | List of guardrail configurations for guardrail load balancing. Enables load balancing across multiple guardrail deployments with the same guardrail_name. [Further Docs](./guardrails/guardrail_load_balancing.md) |


### environment variables - Reference

| Name | Description |
|------|-------------|
| ACTIONS_ID_TOKEN_REQUEST_TOKEN | Token for requesting ID in GitHub Actions
| ACTIONS_ID_TOKEN_REQUEST_URL | URL for requesting ID token in GitHub Actions
| AGENTOPS_ENVIRONMENT | Environment for AgentOps logging integration
| AGENTOPS_API_KEY | API Key for AgentOps logging integration
| AGENTOPS_SERVICE_NAME | Service Name for AgentOps logging integration
| AISPEND_ACCOUNT_ID | Account ID for AI Spend
| AISPEND_API_KEY | API Key for AI Spend
| AIOHTTP_CONNECTOR_LIMIT | Connection limit for aiohttp connector. When set to 0, no limit is applied. **Default is 0**
| AIOHTTP_CONNECTOR_LIMIT_PER_HOST | Connection limit per host for aiohttp connector. When set to 0, no limit is applied. **Default is 0**
| AIOHTTP_KEEPALIVE_TIMEOUT | Keep-alive timeout for aiohttp connections in seconds. **Default is 120**
| AIOHTTP_TRUST_ENV | Flag to enable aiohttp trust environment. When this is set to True, aiohttp will respect HTTP(S)_PROXY env vars. **Default is False**
| AIOHTTP_TTL_DNS_CACHE | DNS cache time-to-live for aiohttp in seconds. **Default is 300**
| ALLOWED_EMAIL_DOMAINS | List of email domains allowed for access
| APSCHEDULER_COALESCE | Whether to combine multiple pending executions of a job into one. **Default is False**
| APSCHEDULER_MAX_INSTANCES | Maximum number of concurrent instances of each job. **Default is 1**
| APSCHEDULER_MISFIRE_GRACE_TIME | Grace time in seconds for misfired jobs. **Default is 1**
| APSCHEDULER_REPLACE_EXISTING | Whether to replace existing jobs with the same ID. **Default is False**
| ARIZE_API_KEY | API key for Arize platform integration
| ARIZE_SPACE_KEY | Space key for Arize platform
| ARGILLA_BATCH_SIZE | Batch size for Argilla logging
| ARGILLA_API_KEY | API key for Argilla platform
| ARGILLA_SAMPLING_RATE | Sampling rate for Argilla logging
| ARGILLA_DATASET_NAME | Dataset name for Argilla logging
| ARGILLA_BASE_URL | Base URL for Argilla service
| ATHINA_API_KEY | API key for Athina service
| ATHINA_BASE_URL | Base URL for Athina service (defaults to `https://log.athina.ai`)
| AUTH_STRATEGY | Strategy used for authentication (e.g., OAuth, API key)
| AUTO_REDIRECT_UI_LOGIN_TO_SSO | Flag to enable automatic redirect of UI login page to SSO when SSO is configured. Default is **false**
| AUDIO_SPEECH_CHUNK_SIZE | Chunk size for audio speech processing. Default is 1024
| ANTHROPIC_API_KEY | API key for Anthropic service
| ANTHROPIC_API_BASE | Base URL for Anthropic API. Default is https://api.anthropic.com
| ANTHROPIC_TOKEN_COUNTING_BETA_VERSION | Beta version header for Anthropic token counting API. Default is `token-counting-2024-11-01`
| AWS_ACCESS_KEY_ID | Access Key ID for AWS services
| AWS_BATCH_ROLE_ARN | ARN of the AWS IAM role for batch operations
| AWS_DEFAULT_REGION | Default AWS region for service interactions when AWS_REGION is not set
| AWS_PROFILE_NAME | AWS CLI profile name to be used
| AWS_REGION | AWS region for service interactions (takes precedence over AWS_DEFAULT_REGION)
| AWS_REGION_NAME | Default AWS region for service interactions
| AWS_ROLE_ARN | ARN of the AWS IAM role to assume for authentication
| AWS_ROLE_NAME | Role name for AWS IAM usage
| AWS_S3_BUCKET_NAME | Name of the AWS S3 bucket for file operations
| AWS_S3_OUTPUT_BUCKET_NAME | Name of the AWS S3 output bucket for batch operations
| AWS_SECRET_ACCESS_KEY | Secret Access Key for AWS services
| AWS_SESSION_NAME | Name for AWS session
| AWS_WEB_IDENTITY_TOKEN | Web identity token for AWS
| AWS_WEB_IDENTITY_TOKEN_FILE | Path to file containing web identity token for AWS
| AZURE_API_VERSION | Version of the Azure API being used
| AZURE_AI_API_BASE | Base URL for Azure AI services (e.g., Azure AI Anthropic)
| AZURE_AI_API_KEY | API key for Azure AI services (e.g., Azure AI Anthropic)
| AZURE_AUTHORITY_HOST | Azure authority host URL
| AZURE_CERTIFICATE_PASSWORD | Password for Azure OpenAI certificate
| AZURE_CLIENT_ID | Client ID for Azure services
| AZURE_CLIENT_SECRET | Client secret for Azure services
| AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS | Input cost per 1K tokens for Azure Computer Use service
| AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS | Output cost per 1K tokens for Azure Computer Use service
| AZURE_DEFAULT_RESPONSES_API_VERSION | Version of the Azure Default Responses API being used. Default is "preview"
| AZURE_DOCUMENT_INTELLIGENCE_API_VERSION | API version for Azure Document Intelligence service
| AZURE_DOCUMENT_INTELLIGENCE_DEFAULT_DPI | Default DPI (dots per inch) setting for Azure Document Intelligence service
| AZURE_TENANT_ID | Tenant ID for Azure Active Directory
| AZURE_USERNAME | Username for Azure services, use in conjunction with AZURE_PASSWORD for azure ad token with basic username/password workflow
| AZURE_PASSWORD | Password for Azure services, use in conjunction with AZURE_USERNAME for azure ad token with basic username/password workflow
| AZURE_FEDERATED_TOKEN_FILE | File path to Azure federated token
| AZURE_FILE_SEARCH_COST_PER_GB_PER_DAY | Cost per GB per day for Azure File Search service
| AZURE_SCOPE | For EntraID Auth, Scope for Azure services, defaults to "https://cognitiveservices.azure.com/.default"
| AZURE_SENTINEL_DCR_IMMUTABLE_ID | Immutable ID of the Data Collection Rule for Azure Sentinel logging
| AZURE_SENTINEL_STREAM_NAME | Stream name for Azure Sentinel logging
| AZURE_SENTINEL_CLIENT_SECRET | Client secret for Azure Sentinel authentication
| AZURE_SENTINEL_ENDPOINT | Endpoint for Azure Sentinel logging
| AZURE_SENTINEL_TENANT_ID | Tenant ID for Azure Sentinel authentication
| AZURE_SENTINEL_CLIENT_ID | Client ID for Azure Sentinel authentication
| AZURE_KEY_VAULT_URI | URI for Azure Key Vault
| AZURE_OPERATION_POLLING_TIMEOUT | Timeout in seconds for Azure operation polling
| AZURE_STORAGE_ACCOUNT_KEY | The Azure Storage Account Key to use for Authentication to Azure Blob Storage logging
| AZURE_STORAGE_ACCOUNT_NAME | Name of the Azure Storage Account to use for logging to Azure Blob Storage
| AZURE_STORAGE_FILE_SYSTEM | Name of the Azure Storage File System to use for logging to Azure Blob Storage.  (Typically the Container name)
| AZURE_STORAGE_TENANT_ID | The Application Tenant ID to use for Authentication to Azure Blob Storage logging
| AZURE_STORAGE_CLIENT_ID | The Application Client ID to use for Authentication to Azure Blob Storage logging
| AZURE_STORAGE_CLIENT_SECRET | The Application Client Secret to use for Authentication to Azure Blob Storage logging
| AZURE_VECTOR_STORE_COST_PER_GB_PER_DAY | Cost per GB per day for Azure Vector Store service
| BATCH_STATUS_POLL_INTERVAL_SECONDS | Interval in seconds for polling batch status. Default is 3600 (1 hour)
| BATCH_STATUS_POLL_MAX_ATTEMPTS | Maximum number of attempts for polling batch status. Default is 24 (for 24 hours)
| BEDROCK_MAX_POLICY_SIZE | Maximum size for Bedrock policy. Default is 75
| BEDROCK_MIN_THINKING_BUDGET_TOKENS | Minimum thinking budget in tokens for Bedrock reasoning models. Bedrock returns a 400 error if budget_tokens is below this value. Requests with lower values are clamped to this minimum. Default is 1024
| BERRISPEND_ACCOUNT_ID | Account ID for BerriSpend service
| BRAINTRUST_API_KEY | API key for Braintrust integration
| BRAINTRUST_API_BASE | Base URL for Braintrust API. Default is https://api.braintrustdata.com/v1
| BRAINTRUST_MOCK | Enable mock mode for Braintrust integration testing. When set to true, intercepts Braintrust API calls and returns mock responses without making actual network calls. Default is false
| BRAINTRUST_MOCK_LATENCY_MS | Mock latency in milliseconds for Braintrust API calls when mock mode is enabled. Simulates network round-trip time. Default is 100ms
| CACHED_STREAMING_CHUNK_DELAY | Delay in seconds for cached streaming chunks. Default is 0.02
| CHATGPT_API_BASE | Base URL for ChatGPT API. Default is https://chatgpt.com/backend-api/codex
| CHATGPT_AUTH_FILE | Filename for ChatGPT authentication data. Default is "auth.json"
| CHATGPT_DEFAULT_INSTRUCTIONS | Default system instructions for ChatGPT provider
| CHATGPT_ORIGINATOR | Originator identifier for ChatGPT API requests. Default is "codex_cli_rs"
| CHATGPT_TOKEN_DIR | Directory to store ChatGPT authentication tokens. Default is "~/.config/litellm/chatgpt"
| CHATGPT_USER_AGENT | Custom user agent string for ChatGPT API requests
| CHATGPT_USER_AGENT_SUFFIX | Suffix to append to the ChatGPT user agent string
| CIRCLE_OIDC_TOKEN | OpenID Connect token for CircleCI
| CIRCLE_OIDC_TOKEN_V2 | Version 2 of the OpenID Connect token for CircleCI
| CLI_JWT_EXPIRATION_HOURS | Expiration time in hours for CLI-generated JWT tokens. Default is 24 hours. Can also be set via LITELLM_CLI_JWT_EXPIRATION_HOURS
| CLOUDZERO_API_KEY | CloudZero API key for authentication
| CLOUDZERO_CONNECTION_ID | CloudZero connection ID for data submission
| CLOUDZERO_EXPORT_INTERVAL_MINUTES | Interval in minutes for CloudZero data export operations
| CLOUDZERO_MAX_FETCHED_DATA_RECORDS | Maximum number of data records to fetch from CloudZero
| CLOUDZERO_TIMEZONE | Timezone for date handling (default: UTC)
| CONFIG_FILE_PATH | File path for configuration file
| CYBERARK_ACCOUNT | CyberArk account name for secret management
| CYBERARK_API_BASE | Base URL for CyberArk API
| CYBERARK_API_KEY | API key for CyberArk secret management service
| CYBERARK_CLIENT_CERT | Path to client certificate for CyberArk authentication
| CYBERARK_CLIENT_KEY | Path to client key for CyberArk authentication
| CYBERARK_USERNAME | Username for CyberArk authentication
| CYBERARK_SSL_VERIFY | Flag to enable or disable SSL certificate verification for CyberArk. Default is True
| CONFIDENT_API_KEY | API key for DeepEval integration
| CUSTOM_TIKTOKEN_CACHE_DIR | Custom directory for Tiktoken cache
| CONFIDENT_API_KEY | API key for Confident AI (Deepeval) Logging service
| COHERE_API_BASE | Base URL for Cohere API. Default is https://api.cohere.com
| DATABASE_HOST | Hostname for the database server
| DATABASE_NAME | Name of the database
| DATABASE_PASSWORD | Password for the database user
| DATABASE_PORT | Port number for database connection
| DATABASE_SCHEMA | Schema name used in the database
| DATABASE_URL | Connection URL for the database
| DATABASE_USER | Username for database connection
| DATABASE_USERNAME | Alias for database user
| DATABRICKS_API_BASE | Base URL for Databricks API
| DATABRICKS_CLIENT_ID | Client ID for Databricks OAuth M2M authentication (Service Principal application ID)
| DATABRICKS_CLIENT_SECRET | Client secret for Databricks OAuth M2M authentication
| DATABRICKS_USER_AGENT | Custom user agent string for Databricks API requests. Used for partner telemetry attribution
| DAYS_IN_A_MONTH | Days in a month for calculation purposes. Default is 28
| DAYS_IN_A_WEEK | Days in a week for calculation purposes. Default is 7
| DAYS_IN_A_YEAR | Days in a year for calculation purposes. Default is 365
| DYNAMOAI_API_KEY | API key for DynamoAI Guardrails service
| DYNAMOAI_API_BASE | Base URL for DynamoAI API. Default is https://api.dynamo.ai
| DYNAMOAI_MODEL_ID | Model ID for DynamoAI tracking/logging purposes
| DYNAMOAI_POLICY_IDS | Comma-separated list of DynamoAI policy IDs to apply
| DD_BASE_URL | Base URL for Datadog integration
| DATADOG_BASE_URL | (Alternative to DD_BASE_URL) Base URL for Datadog integration
| _DATADOG_BASE_URL | (Alternative to DD_BASE_URL) Base URL for Datadog integration
| DD_AGENT_HOST | Hostname or IP of DataDog agent (e.g., "localhost"). When set, logs are sent to agent instead of direct API
| DD_AGENT_PORT | Port of DataDog agent for log intake. Default is 10518
| DD_API_KEY | API key for Datadog integration
| DD_APP_KEY | Application key for Datadog Cost Management integration. Required along with DD_API_KEY for cost metrics
| DD_SITE | Site URL for Datadog (e.g., datadoghq.com)
| DD_SOURCE | Source identifier for Datadog logs
| DD_TRACER_STREAMING_CHUNK_YIELD_RESOURCE | Resource name for Datadog tracing of streaming chunk yields. Default is "streaming.chunk.yield"
| DD_ENV | Environment identifier for Datadog logs. Only supported for `datadog_llm_observability` callback
| DD_SERVICE | Service identifier for Datadog logs. Defaults to "litellm-server"
| DD_VERSION | Version identifier for Datadog logs. Defaults to "unknown"
| DATADOG_MOCK | Enable mock mode for Datadog integration testing. When set to true, intercepts Datadog API calls and returns mock responses without making actual network calls. Default is false
| DATADOG_MOCK_LATENCY_MS | Mock latency in milliseconds for Datadog API calls when mock mode is enabled. Simulates network round-trip time. Default is 100ms
| DEBUG_OTEL | Enable debug mode for OpenTelemetry
| DEFAULT_ALLOWED_FAILS | Maximum failures allowed before cooling down a model. Default is 3
| DEFAULT_A2A_AGENT_TIMEOUT | Default timeout in seconds for A2A (Agent-to-Agent) protocol requests. Default is 6000
| DEFAULT_ACCESS_GROUP_CACHE_TTL | Time-to-live in seconds for cached access group information. Default is 600 (10 minutes)
| DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS | Default maximum tokens for Anthropic chat completions. Default is 4096
| DEFAULT_BATCH_SIZE | Default batch size for operations. Default is 512
| DEFAULT_CHUNK_OVERLAP | Default chunk overlap for RAG text splitters. Default is 200
| DEFAULT_CHUNK_SIZE | Default chunk size for RAG text splitters. Default is 1000
| DEFAULT_CLIENT_DISCONNECT_CHECK_TIMEOUT_SECONDS | Timeout in seconds for checking client disconnection. Default is 1
| DEFAULT_COOLDOWN_TIME_SECONDS | Duration in seconds to cooldown a model after failures. Default is 5
| DEFAULT_CRON_JOB_LOCK_TTL_SECONDS | Time-to-live for cron job locks in seconds. Default is 60 (1 minute)
| DEFAULT_DATAFORSEO_LOCATION_CODE | Default location code for DataForSEO search API. Default is 2250 (France)
| DEFAULT_FAILURE_THRESHOLD_PERCENT | Threshold percentage of failures to cool down a deployment. Default is 0.5 (50%)
| DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS | Minimum number of requests before applying error rate cooldown. Prevents cooldown from triggering on first failure. Default is 5
| DEFAULT_FLUSH_INTERVAL_SECONDS | Default interval in seconds for flushing operations. Default is 5
| DEFAULT_HEALTH_CHECK_INTERVAL | Default interval in seconds for health checks. Default is 300 (5 minutes)
| DEFAULT_HEALTH_CHECK_PROMPT | Default prompt used during health checks for non-image models. Default is "test from litellm"
| DEFAULT_IMAGE_HEIGHT | Default height for images. Default is 300
| DEFAULT_IMAGE_TOKEN_COUNT | Default token count for images. Default is 250
| DEFAULT_IMAGE_WIDTH | Default width for images. Default is 300
| DEFAULT_IN_MEMORY_TTL | Default time-to-live for in-memory cache in seconds. Default is 5
| DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL | Default time-to-live in seconds for management objects (User, Team, Key, Organization) in memory cache. Default is 60 seconds.
| DEFAULT_MAX_LRU_CACHE_SIZE | Default maximum size for LRU cache. Default is 16
| DEFAULT_MAX_RECURSE_DEPTH | Default maximum recursion depth. Default is 100
| DEFAULT_MAX_RECURSE_DEPTH_SENSITIVE_DATA_MASKER | Default maximum recursion depth for sensitive data masker. Default is 10
| DEFAULT_MAX_RETRIES | Default maximum retry attempts. Default is 2
| DEFAULT_MAX_TOKENS | Default maximum tokens for LLM calls. Default is 4096
| DEFAULT_MAX_TOKENS_FOR_TRITON | Default maximum tokens for Triton models. Default is 2000
| DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE | Default maximum size for redis batch cache. Default is 1000
| DEFAULT_MCP_SEMANTIC_FILTER_EMBEDDING_MODEL | Default embedding model for MCP semantic tool filtering. Default is "text-embedding-3-small"
| DEFAULT_MCP_SEMANTIC_FILTER_SIMILARITY_THRESHOLD | Default similarity threshold for MCP semantic tool filtering. Default is 0.3
| DEFAULT_MCP_SEMANTIC_FILTER_TOP_K | Default number of top results to return for MCP semantic tool filtering. Default is 10
| MCP_NPM_CACHE_DIR | Directory for npm cache used by STDIO MCP servers. In containers the default (~/.npm) may not exist or be read-only. Default is `/tmp/.npm_mcp_cache`
| MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL | Default TTL in seconds for MCP OAuth2 token cache. Default is 3600
| MCP_OAUTH2_TOKEN_CACHE_MAX_SIZE | Maximum number of entries in MCP OAuth2 token cache. Default is 200
| MCP_OAUTH2_TOKEN_CACHE_MIN_TTL | Minimum TTL in seconds for MCP OAuth2 token cache. Default is 10
| MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS | Seconds to subtract from token expiry when computing cache TTL. Default is 60
| DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT | Default token count for mock response completions. Default is 20
| DEFAULT_MOCK_RESPONSE_PROMPT_TOKEN_COUNT | Default token count for mock response prompts. Default is 10
| DEFAULT_MODEL_CREATED_AT_TIME | Default creation timestamp for models. Default is 1677610602
| DEFAULT_NUM_WORKERS_LITELLM_PROXY | Default number of workers for LiteLLM proxy when `NUM_WORKERS` is not set. Default is 1. **We strongly recommend setting NUM_WORKERS to the number of vCPUs available** (e.g. `NUM_WORKERS=8` or `--num_workers 8`).
| DEFAULT_PROMPT_INJECTION_SIMILARITY_THRESHOLD | Default threshold for prompt injection similarity. Default is 0.7
| DEFAULT_POLLING_INTERVAL | Default polling interval for schedulers in seconds. Default is 0.03
| DEFAULT_REASONING_EFFORT_DISABLE_THINKING_BUDGET | Default reasoning effort disable thinking budget. Default is 0
| DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET | Default high reasoning effort thinking budget. Default is 4096
| DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET | Default low reasoning effort thinking budget. Default is 1024
| DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET | Default medium reasoning effort thinking budget. Default is 2048
| DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET | Default minimal reasoning effort thinking budget. Default is 512
| DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_FLASH | Default minimal reasoning effort thinking budget for Gemini 2.5 Flash. Default is 512
| DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_FLASH_LITE | Default minimal reasoning effort thinking budget for Gemini 2.5 Flash Lite. Default is 512
| DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET_GEMINI_2_5_PRO | Default minimal reasoning effort thinking budget for Gemini 2.5 Pro. Default is 512
| DEFAULT_REDIS_MAJOR_VERSION | Default Redis major version to assume when version cannot be determined. Default is 7
| DEFAULT_REDIS_SYNC_INTERVAL | Default Redis synchronization interval in seconds. Default is 1
| DEFAULT_REPLICATE_GPU_PRICE_PER_SECOND | Default price per second for Replicate GPU. Default is 0.001400
| DEFAULT_REPLICATE_POLLING_DELAY_SECONDS | Default delay in seconds for Replicate polling. Default is 1
| DEFAULT_REPLICATE_POLLING_RETRIES | Default number of retries for Replicate polling. Default is 5
| DEFAULT_SQS_BATCH_SIZE | Default batch size for SQS logging. Default is 512
| DEFAULT_SQS_FLUSH_INTERVAL_SECONDS | Default flush interval for SQS logging. Default is 10
| DEFAULT_S3_BATCH_SIZE | Default batch size for S3 logging. Default is 512
| DEFAULT_S3_FLUSH_INTERVAL_SECONDS | Default flush interval for S3 logging. Default is 10
| DEFAULT_SLACK_ALERTING_THRESHOLD | Default threshold for Slack alerting. Default is 300
| DEFAULT_SOFT_BUDGET | Default soft budget for LiteLLM proxy keys. Default is 50.0
| DEFAULT_TRIM_RATIO | Default ratio of tokens to trim from prompt end. Default is 0.75
| DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS | Default duration for video generation in seconds in google. Default is 8
| DIRECT_URL | Direct URL for service endpoint
| DISABLE_ADMIN_UI | Toggle to disable the admin UI
| DISABLE_AIOHTTP_TRANSPORT | Flag to disable aiohttp transport. When this is set to True, litellm will use httpx instead of aiohttp. **Default is False**
| DISABLE_AIOHTTP_TRUST_ENV | Flag to disable aiohttp trust environment. When this is set to True, litellm will not trust the environment for aiohttp eg. `HTTP_PROXY` and `HTTPS_PROXY` environment variables will not be used when this is set to True. **Default is False**
| DISABLE_SCHEMA_UPDATE | Toggle to disable schema updates
| DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE | Threshold for deployment failures per minute before enforcing rate limits in parallel request limiter. Default is 1
| DOCS_DESCRIPTION | Description text for documentation pages
| DOCS_FILTERED | Flag indicating filtered documentation
| DOCS_TITLE | Title of the documentation pages
| DOCS_URL | The path to the Swagger API documentation. **By default this is "/"**
| EMAIL_LOGO_URL | URL for the logo used in emails
| EMAIL_BUDGET_ALERT_TTL | Time-to-live for email budget alerts in seconds
| EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE | Maximum spend percentage for triggering email budget alerts
| EMAIL_SUPPORT_CONTACT | Support contact email address
| EMAIL_SIGNATURE | Custom HTML footer/signature for all emails. Can include HTML tags for formatting and links.
| EMAIL_SUBJECT_INVITATION | Custom subject template for invitation emails. 
| EMAIL_SUBJECT_KEY_CREATED | Custom subject template for key creation emails. 
| EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE | Percentage of max budget that triggers alerts (as decimal: 0.8 = 80%). Default is 0.8
| EMAIL_BUDGET_ALERT_TTL | Time-to-live for budget alert deduplication in seconds. Default is 86400 (24 hours)
| ENKRYPTAI_API_BASE | Base URL for EnkryptAI Guardrails API. **Default is https://api.enkryptai.com**
| ENKRYPTAI_API_KEY | API key for EnkryptAI Guardrails service
| FIREWORKS_AI_4_B | Size parameter for Fireworks AI 4B model. Default is 4
| FIREWORKS_AI_16_B | Size parameter for Fireworks AI 16B model. Default is 16
| FIREWORKS_AI_56_B_MOE | Size parameter for Fireworks AI 56B MOE model. Default is 56
| FIREWORKS_AI_80_B | Size parameter for Fireworks AI 80B model. Default is 80
| FIREWORKS_AI_176_B_MOE | Size parameter for Fireworks AI 176B MOE model. Default is 176
| FOCUS_PROVIDER | Destination provider for Focus exports (e.g., `s3`). Defaults to `s3`.
| FOCUS_FORMAT | Output format for Focus exports. Defaults to `parquet`.
| FOCUS_FREQUENCY | Frequency for scheduled Focus exports (`hourly`, `daily`, or `interval`). Defaults to `hourly`.
| FOCUS_CRON_OFFSET | Minute offset used when scheduling hourly/daily Focus exports. Defaults to `5` minutes.
| FOCUS_INTERVAL_SECONDS | Interval (in seconds) for Focus exports when `frequency` is `interval`.
| FOCUS_PREFIX | Object key prefix (or folder) used when uploading Focus export files. Defaults to `focus_exports`.
| FOCUS_S3_BUCKET_NAME | S3 bucket to upload Focus export files when using the S3 destination.
| FOCUS_S3_REGION_NAME | AWS region for the Focus export S3 bucket.
| FOCUS_S3_ENDPOINT_URL | Custom endpoint for the Focus export S3 client (optional; useful for S3-compatible storage).
| FOCUS_S3_ACCESS_KEY | AWS access key ID used by the Focus export S3 client.
| FOCUS_S3_SECRET_KEY | AWS secret access key used by the Focus export S3 client.
| FOCUS_S3_SESSION_TOKEN | AWS session token used by the Focus export S3 client (optional).
| FUNCTION_DEFINITION_TOKEN_COUNT | Token count for function definitions. Default is 9
| GALILEO_BASE_URL | Base URL for Galileo platform
| GALILEO_PASSWORD | Password for Galileo authentication
| GALILEO_PROJECT_ID | Project ID for Galileo usage
| GALILEO_USERNAME | Username for Galileo authentication
| GOOGLE_SECRET_MANAGER_PROJECT_ID | Project ID for Google Secret Manager
| GCS_BUCKET_NAME | Name of the Google Cloud Storage bucket
| GCS_MOCK | Enable mock mode for GCS integration testing. When set to true, intercepts GCS API calls and returns mock responses without making actual network calls. Default is false
| GCS_MOCK_LATENCY_MS | Mock latency in milliseconds for GCS API calls when mock mode is enabled. Simulates network round-trip time. Default is 150ms
| GCS_PATH_SERVICE_ACCOUNT | Path to the Google Cloud service account JSON file
| GCS_FLUSH_INTERVAL | Flush interval for GCS logging (in seconds). Specify how often you want a log to be sent to GCS. **Default is 20 seconds**
| GCS_BATCH_SIZE | Batch size for GCS logging. Specify after how many logs you want to flush to GCS. If `BATCH_SIZE` is set to 10, logs are flushed every 10 logs. **Default is 2048**
| GCS_USE_BATCHED_LOGGING | Enable batched logging for GCS. When enabled (default), multiple log payloads are combined into single GCS object uploads (NDJSON format), dramatically reducing API calls. When disabled, sends each log individually as separate GCS objects (legacy behavior). **Default is true**
| GCS_PUBSUB_TOPIC_ID | PubSub Topic ID to send LiteLLM SpendLogs to.
| GCS_PUBSUB_PROJECT_ID | PubSub Project ID to send LiteLLM SpendLogs to.
| GENERIC_AUTHORIZATION_ENDPOINT | Authorization endpoint for generic OAuth providers
| GENERIC_CLIENT_ID | Client ID for generic OAuth providers
| GENERIC_CLIENT_SECRET | Client secret for generic OAuth providers
| GENERIC_CLIENT_STATE | State parameter for generic client authentication
| GENERIC_CLIENT_USE_PKCE | Enable PKCE (Proof Key for Code Exchange) for generic OAuth providers. Set to "true" when your OAuth provider requires PKCE. **Default is false**
| GENERIC_SSO_HEADERS | Comma-separated list of additional headers to add to the request - e.g. Authorization=Bearer `<token>`, Content-Type=application/json, etc.
| GENERIC_INCLUDE_CLIENT_ID | Include client ID in requests for OAuth
| GENERIC_SCOPE | Scope settings for generic OAuth providers
| GENERIC_TOKEN_ENDPOINT | Token endpoint for generic OAuth providers
| GENERIC_USER_DISPLAY_NAME_ATTRIBUTE | Attribute for user's display name in generic auth
| GENERIC_USER_EMAIL_ATTRIBUTE | Attribute for user's email in generic auth
| GENERIC_USER_EXTRA_ATTRIBUTES | Comma-separated list of additional fields to extract from generic SSO provider response (e.g., "department,employee_id,groups"). Accessible via `CustomOpenID.extra_fields` in custom SSO handlers. Supports dot notation for nested fields
| GENERIC_USER_FIRST_NAME_ATTRIBUTE | Attribute for user's first name in generic auth
| GENERIC_USER_ID_ATTRIBUTE | Attribute for user ID in generic auth
| GENERIC_USER_LAST_NAME_ATTRIBUTE | Attribute for user's last name in generic auth
| GENERIC_USER_PROVIDER_ATTRIBUTE | Attribute specifying the user's provider
| GENERIC_USER_ROLE_ATTRIBUTE | Attribute specifying the user's role
| GENERIC_USERINFO_ENDPOINT | Endpoint to fetch user information in generic OAuth
| GENERIC_LOGGER_ENDPOINT | Endpoint URL for the Generic Logger callback to send logs to
| GENERIC_LOGGER_HEADERS | JSON string of headers to include in Generic Logger callback requests
| GENERIC_ROLE_MAPPINGS_DEFAULT_ROLE | Default LiteLLM role to assign when no role mapping matches in generic SSO. Used with GENERIC_ROLE_MAPPINGS_ROLES
| GENERIC_ROLE_MAPPINGS_GROUP_CLAIM | The claim/attribute name in the SSO token that contains the user's groups. Used for role mapping
| GENERIC_ROLE_MAPPINGS_ROLES | Python dict string mapping LiteLLM roles to SSO group names. Example: `{"proxy_admin": ["admin-group"], "internal_user": ["users"]}`
| GENERIC_USER_ROLE_MAPPINGS | Alternative to GENERIC_ROLE_MAPPINGS_ROLES for configuring user role mappings from SSO
| GEMINI_API_BASE | Base URL for Gemini API. Default is https://generativelanguage.googleapis.com
| GALILEO_BASE_URL | Base URL for Galileo platform
| GALILEO_PASSWORD | Password for Galileo authentication
| GALILEO_PROJECT_ID | Project ID for Galileo usage
| GALILEO_USERNAME | Username for Galileo authentication
| GITHUB_COPILOT_TOKEN_DIR | Directory to store GitHub Copilot token for `github_copilot` llm provider
| GITHUB_COPILOT_API_KEY_FILE | File to store GitHub Copilot API key for `github_copilot` llm provider
| GITHUB_COPILOT_ACCESS_TOKEN_FILE | File to store GitHub Copilot access token for `github_copilot` llm provider
| GREENSCALE_API_KEY | API key for Greenscale service
| GREENSCALE_ENDPOINT | Endpoint URL for Greenscale service
| GRAYSWAN_API_BASE | Base URL for GraySwan API. Default is https://api.grayswan.ai
| GRAYSWAN_API_KEY | API key for GraySwan Cygnal service
| GRAYSWAN_REASONING_MODE | Reasoning mode for GraySwan guardrail
| GRAYSWAN_VIOLATION_THRESHOLD | Violation threshold for GraySwan guardrail
| GOOGLE_APPLICATION_CREDENTIALS | Path to Google Cloud credentials JSON file
| GOOGLE_CLIENT_ID | Client ID for Google OAuth
| GOOGLE_CLIENT_SECRET | Client secret for Google OAuth
| GOOGLE_KMS_RESOURCE_NAME | Name of the resource in Google KMS
| GUARDRAILS_AI_API_BASE | Base URL for Guardrails AI API
| HEALTH_CHECK_TIMEOUT_SECONDS | Timeout in seconds for health checks. Default is 60
| HEROKU_API_BASE | Base URL for Heroku API
| HEROKU_API_KEY | API key for Heroku services
| HF_API_BASE | Base URL for Hugging Face API
| HCP_VAULT_ADDR | Address for [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault)
| HCP_VAULT_APPROLE_MOUNT_PATH | Mount path for AppRole authentication in [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault). Default is "approle"
| HCP_VAULT_APPROLE_ROLE_ID | Role ID for AppRole authentication in [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault)
| HCP_VAULT_APPROLE_SECRET_ID | Secret ID for AppRole authentication in [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault)
| HCP_VAULT_CLIENT_CERT | Path to client certificate for [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault)
| HCP_VAULT_CLIENT_KEY | Path to client key for [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault)
| HCP_VAULT_MOUNT_NAME | Mount name for [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault)
| HCP_VAULT_NAMESPACE | Namespace for [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault)
| HCP_VAULT_PATH_PREFIX | Path prefix for [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault)
| HCP_VAULT_TOKEN | Token for [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault)
| HCP_VAULT_CERT_ROLE | Role for [Hashicorp Vault Secret Manager Auth](../secret.md#hashicorp-vault)
| HELICONE_API_KEY | API key for Helicone service
| HELICONE_API_BASE | Base URL for Helicone service, defaults to `https://api.helicone.ai`
| HELICONE_MOCK | Enable mock mode for Helicone integration testing. When set to true, intercepts Helicone API calls and returns mock responses without making actual network calls. Default is false
| HELICONE_MOCK_LATENCY_MS | Mock latency in milliseconds for Helicone API calls when mock mode is enabled. Simulates network round-trip time. Default is 100ms
| HOSTNAME | Hostname for the server, this will be [emitted to `datadog` logs](https://docs.litellm.ai/docs/proxy/logging#datadog)
| HOURS_IN_A_DAY | Hours in a day for calculation purposes. Default is 24
| HIDDENLAYER_API_BASE | Base URL for HiddenLayer API. Defaults to `https://api.hiddenlayer.ai`
| HIDDENLAYER_AUTH_URL | Authentication URL for HiddenLayer. Defaults to `https://auth.hiddenlayer.ai`
| HIDDENLAYER_CLIENT_ID | Client ID for HiddenLayer SaaS authentication
| HIDDENLAYER_CLIENT_SECRET | Client secret for HiddenLayer SaaS authentication
| HUGGINGFACE_API_BASE | Base URL for Hugging Face API
| HUGGINGFACE_API_KEY | API key for Hugging Face API
| HUMANLOOP_PROMPT_CACHE_TTL_SECONDS | Time-to-live in seconds for cached prompts in Humanloop. Default is 60
| IAM_TOKEN_DB_AUTH | IAM token for database authentication
| IBM_GUARDRAILS_API_BASE | Base URL for IBM Guardrails API
| IBM_GUARDRAILS_AUTH_TOKEN | Authorization bearer token for IBM Guardrails API
| INITIAL_RETRY_DELAY | Initial delay in seconds for retrying requests. Default is 0.5
| JITTER | Jitter factor for retry delay calculations. Default is 0.75
| JSON_LOGS | Enable JSON formatted logging
| JWT_AUDIENCE | Expected audience for JWT tokens
| JWT_PUBLIC_KEY_URL | URL to fetch public key for JWT verification
| LAGO_API_BASE | Base URL for Lago API
| LAGO_API_CHARGE_BY | Parameter to determine charge basis in Lago
| LAGO_API_EVENT_CODE | Event code for Lago API events
| LAGO_API_KEY | API key for accessing Lago services
| LANGFUSE_DEBUG | Toggle debug mode for Langfuse
| LANGFUSE_FLUSH_INTERVAL | Interval for flushing Langfuse logs
| LANGFUSE_TRACING_ENVIRONMENT | Environment for Langfuse tracing
| LANGFUSE_HOST | Host URL for Langfuse service
| LANGFUSE_MOCK | Enable mock mode for Langfuse integration testing. When set to true, intercepts Langfuse API calls and returns mock responses without making actual network calls. Default is false
| LANGFUSE_MOCK_LATENCY_MS | Mock latency in milliseconds for Langfuse API calls when mock mode is enabled. Simulates network round-trip time. Default is 100ms
| LANGFUSE_PUBLIC_KEY | Public key for Langfuse authentication
| LANGFUSE_RELEASE | Release version of Langfuse integration
| LANGFUSE_SECRET_KEY | Secret key for Langfuse authentication
| LANGFUSE_PROPAGATE_TRACE_ID | Flag to enable propagating trace ID to Langfuse. Default is False
| LANGSMITH_API_KEY | API key for Langsmith platform
| LANGSMITH_BASE_URL | Base URL for Langsmith service
| LANGSMITH_BATCH_SIZE | Batch size for operations in Langsmith
| LANGSMITH_DEFAULT_RUN_NAME | Default name for Langsmith run
| LANGSMITH_PROJECT | Project name for Langsmith integration
| LANGSMITH_SAMPLING_RATE | Sampling rate for Langsmith logging
| LANGSMITH_TENANT_ID | Tenant ID for Langsmith multi-tenant deployments
| LANGSMITH_MOCK | Enable mock mode for Langsmith integration testing. When set to true, intercepts Langsmith API calls and returns mock responses without making actual network calls. Default is false
| LANGSMITH_MOCK_LATENCY_MS | Mock latency in milliseconds for Langsmith API calls when mock mode is enabled. Simulates network round-trip time. Default is 100ms
| LANGTRACE_API_KEY | API key for Langtrace service
| LASSO_API_BASE | Base URL for Lasso API
| LASSO_API_KEY | API key for Lasso service
| LASSO_USER_ID | User ID for Lasso service
| LASSO_CONVERSATION_ID | Conversation ID for Lasso service
| LENGTH_OF_LITELLM_GENERATED_KEY | Length of keys generated by LiteLLM. Default is 16
| LEGACY_MULTI_INSTANCE_RATE_LIMITING | Flag to enable legacy multi-instance rate limiting. **Default is False**
| LITERAL_API_KEY | API key for Literal integration
| LITERAL_API_URL | API URL for Literal service
| LITERAL_BATCH_SIZE | Batch size for Literal operations
| LITELLM_ANTHROPIC_BETA_HEADERS_URL | Custom URL for fetching Anthropic beta headers configuration. Default is the GitHub main branch URL
| LITELLM_ANTHROPIC_DISABLE_URL_SUFFIX | Disable automatic URL suffix appending for Anthropic API base URLs. When set to `true`, prevents LiteLLM from automatically adding `/v1/messages` or `/v1/complete` to custom Anthropic API endpoints
| LITELLM_ASSETS_PATH | Path to directory for UI assets and logos. Used when running with read-only filesystem (e.g., Kubernetes). Default is `/var/lib/litellm/assets` in Docker.
| LITELLM_CLI_JWT_EXPIRATION_HOURS | Expiration time in hours for CLI-generated JWT tokens. Default is 24 hours
| LITELLM_DD_AGENT_HOST | Hostname or IP of DataDog agent for LiteLLM-specific logging. When set, logs are sent to agent instead of direct API
| LITELLM_DEPLOYMENT_ENVIRONMENT | Environment name for the deployment (e.g., "production", "staging"). Used as a fallback when OTEL_ENVIRONMENT_NAME is not set. Sets the `environment` tag in telemetry data
| LITELLM_DD_AGENT_PORT | Port of DataDog agent for LiteLLM-specific log intake. Default is 10518
| LITELLM_DD_LLM_OBS_PORT | Port for Datadog LLM Observability agent. Default is 8126
| LITELLM_DONT_SHOW_FEEDBACK_BOX | Flag to hide feedback box in LiteLLM UI
| LITELLM_DROP_PARAMS | Parameters to drop in LiteLLM requests
| LITELLM_MODIFY_PARAMS | Parameters to modify in LiteLLM requests
| LITELLM_EMAIL | Email associated with LiteLLM account
| LITELLM_GLOBAL_MAX_PARALLEL_REQUEST_RETRIES | Maximum retries for parallel requests in LiteLLM
| LITELLM_GLOBAL_MAX_PARALLEL_REQUEST_RETRY_TIMEOUT | Timeout for retries of parallel requests in LiteLLM
| LITELLM_DISABLE_LAZY_LOADING | When set to "1", "true", "yes", or "on", disables lazy loading of attributes (currently only affects encoding/tiktoken). This ensures encoding is initialized before VCR starts recording HTTP requests, fixing VCR cassette creation issues. See [issue #18659](https://github.com/BerriAI/litellm/issues/18659)
| LITELLM_MIGRATION_DIR | Custom migrations directory for prisma migrations, used for baselining db in read-only file systems.
| LITELLM_HOSTED_UI | URL of the hosted UI for LiteLLM
| LITELLM_UI_API_DOC_BASE_URL | Optional override for the API Reference base URL (used in sample code/docs) when the admin UI runs on a different host than the proxy. Defaults to `PROXY_BASE_URL` when unset.
| LITELLM_UI_PATH | Path to directory for Admin UI files. Used when running with read-only filesystem (e.g., Kubernetes). Default is `/var/lib/litellm/ui` in Docker.
| LITELM_ENVIRONMENT | Environment of LiteLLM Instance, used by logging services. Currently only used by DeepEval.
| LITELLM_KEY_ROTATION_ENABLED | Enable auto-key rotation for LiteLLM (boolean). Default is false.
| LITELLM_KEY_ROTATION_CHECK_INTERVAL_SECONDS | Interval in seconds for how often to run job that auto-rotates keys. Default is 86400 (24 hours).
| LITELLM_KEY_ROTATION_GRACE_PERIOD | Duration to keep old key valid after rotation (e.g. "24h", "2d"). Default is empty (immediate revoke). Used for scheduled rotations and as fallback when not specified in regenerate request.
| LITELLM_LICENSE | License key for LiteLLM usage
| LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS | Set to `True` to use the local bundled Anthropic beta headers config only, disabling remote fetching. Default is `False`
| LITELLM_LOCAL_MODEL_COST_MAP | Local configuration for model cost mapping in LiteLLM
| LITELLM_LOCAL_POLICY_TEMPLATES | When set to "true", uses local backup policy templates instead of fetching from GitHub. Policy templates are fetched from https://raw.githubusercontent.com/BerriAI/litellm/main/policy_templates.json by default, with automatic fallback to local backup on failure
| LITELLM_LOG | Enable detailed logging for LiteLLM
| LITELLM_MODEL_COST_MAP_URL | URL for fetching model cost map data. Default is https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
| LITELLM_LOG_FILE | File path to write LiteLLM logs to. When set, logs will be written to both console and the specified file
| LITELLM_LOGGER_NAME | Name for OTEL logger 
| LITELLM_METER_NAME | Name for OTEL Meter 
| LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS | Optionally enable semantic logs for OTEL
| LITELLM_OTEL_INTEGRATION_ENABLE_METRICS | Optionally enable emantic metrics for OTEL
| LITELLM_ENABLE_PYROSCOPE | If true, enables Pyroscope CPU profiling. Profiles are sent to PYROSCOPE_SERVER_ADDRESS. Off by default. See [Pyroscope profiling](/proxy/pyroscope_profiling).
| PYROSCOPE_APP_NAME | Application name reported to Pyroscope. Required when LITELLM_ENABLE_PYROSCOPE is true. No default.
| PYROSCOPE_SERVER_ADDRESS | Pyroscope server URL to send profiles to. Required when LITELLM_ENABLE_PYROSCOPE is true. No default.
| PYROSCOPE_SAMPLE_RATE | Optional. Sample rate for Pyroscope profiling (integer). No default; when unset, the pyroscope-io library default is used.
| LITELLM_MASTER_KEY | Master key for proxy authentication
| LITELLM_MODE | Operating mode for LiteLLM (e.g., production, development)
| LITELLM_NON_ROOT | Flag to run LiteLLM in non-root mode for enhanced security in Docker containers
| LITELLM_RATE_LIMIT_WINDOW_SIZE | Rate limit window size for LiteLLM. Default is 60
| LITELLM_REASONING_AUTO_SUMMARY | If set to "true", automatically enables detailed reasoning summaries for reasoning models (e.g., o1, o3-mini, deepseek-reasoner). When enabled, adds `summary: "detailed"` to reasoning effort configurations. Default is "false"
| LITELLM_SALT_KEY | Salt key for encryption in LiteLLM
| LITELLM_SSL_CIPHERS | SSL/TLS cipher configuration for faster handshakes. Controls cipher suite preferences for OpenSSL connections.
| LITELLM_SECRET_AWS_KMS_LITELLM_LICENSE | AWS KMS encrypted license for LiteLLM
| LITELLM_TOKEN | Access token for LiteLLM integration
| LITELLM_USER_AGENT | Custom user agent string for LiteLLM API requests. Used for partner telemetry attribution
| LITELLM_PRINT_STANDARD_LOGGING_PAYLOAD | If true, prints the standard logging payload to the console - useful for debugging
| LITELM_ENVIRONMENT | Environment for LiteLLM Instance. This is currently only logged to DeepEval to determine the environment for DeepEval integration.
| LITELLM_ASYNCIO_QUEUE_MAXSIZE | Maximum size for asyncio queues (e.g. log queues, spend update queues, and cookbook examples such as realtime audio in `nova_sonic_realtime.py`). Bounds in-memory growth to prevent OOM. Default is 1000.
| LOGFIRE_TOKEN | Token for Logfire logging service
| LOGFIRE_BASE_URL | Base URL for Logfire logging service (useful for self hosted deployments)
| LOGGING_WORKER_CONCURRENCY | Maximum number of concurrent coroutine slots for the logging worker on the asyncio event loop. Default is 100. Setting too high will flood the event loop with logging tasks which will lower the overall latency of the requests.
| LOGGING_WORKER_MAX_QUEUE_SIZE | Maximum size of the logging worker queue. When the queue is full, the worker aggressively clears tasks to make room instead of dropping logs. Default is 50,000
| LOGGING_WORKER_MAX_TIME_PER_COROUTINE | Maximum time in seconds allowed for each coroutine in the logging worker before timing out. Default is 20.0
| LOGGING_WORKER_CLEAR_PERCENTAGE | Percentage of the queue to extract when clearing. Default is 50% 
| MAX_EXCEPTION_MESSAGE_LENGTH | Maximum length for exception messages. Default is 2000
| MAX_ITERATIONS_TO_CLEAR_QUEUE | Maximum number of iterations to attempt when clearing the logging worker queue during shutdown. Default is 200
| MAX_TIME_TO_CLEAR_QUEUE | Maximum time in seconds to spend clearing the logging worker queue during shutdown. Default is 5.0
| LOGGING_WORKER_AGGRESSIVE_CLEAR_COOLDOWN_SECONDS | Cooldown time in seconds before allowing another aggressive clear operation when the queue is full. Default is 0.5 
| MAX_STRING_LENGTH_PROMPT_IN_DB | Maximum length for strings in spend logs when sanitizing request bodies. Strings longer than this will be truncated. Default is 1000
| MAX_IN_MEMORY_QUEUE_FLUSH_COUNT | Maximum count for in-memory queue flush operations. Default is 1000
| MAX_IMAGE_URL_DOWNLOAD_SIZE_MB | Maximum size in MB for downloading images from URLs. Prevents memory issues from downloading very large images. Images exceeding this limit will be rejected before download. Set to 0 to completely disable image URL handling (all image_url requests will be blocked). Default is 50MB (matching [OpenAI's limit](https://platform.openai.com/docs/guides/images-vision?api-mode=chat#image-input-requirements))
| MAX_LONG_SIDE_FOR_IMAGE_HIGH_RES | Maximum length for the long side of high-resolution images. Default is 2000
| MAX_REDIS_BUFFER_DEQUEUE_COUNT | Maximum count for Redis buffer dequeue operations. Default is 100
| MAX_SHORT_SIDE_FOR_IMAGE_HIGH_RES | Maximum length for the short side of high-resolution images. Default is 768
| MAX_SIZE_IN_MEMORY_QUEUE | Maximum size for in-memory queue. Default is 10000
| MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB | Maximum size in KB for each item in memory cache. Default is 512 or 1024
| MAX_SPENDLOG_ROWS_TO_QUERY | Maximum number of spend log rows to query. Default is 1,000,000
| MAX_TEAM_LIST_LIMIT | Maximum number of teams to list. Default is 20
| MAX_TILE_HEIGHT | Maximum height for image tiles. Default is 512
| MAX_TILE_WIDTH | Maximum width for image tiles. Default is 512
| MAX_TOKEN_TRIMMING_ATTEMPTS | Maximum number of attempts to trim a token message. Default is 10
| MAXIMUM_TRACEBACK_LINES_TO_LOG | Maximum number of lines to log in traceback in LiteLLM Logs UI. Default is 100
| MAX_RETRY_DELAY | Maximum delay in seconds for retrying requests. Default is 8.0
| MAX_LANGFUSE_INITIALIZED_CLIENTS | Maximum number of Langfuse clients to initialize on proxy. Default is 50. This is set since langfuse initializes 1 thread everytime a client is initialized. We've had an incident in the past where we reached 100% cpu utilization because Langfuse was initialized several times.
| MAX_MCP_SEMANTIC_FILTER_TOOLS_HEADER_LENGTH | Maximum header length for MCP semantic filter tools. Default is 150
| MAX_POLICY_ESTIMATE_IMPACT_ROWS | Maximum number of rows returned when estimating the impact of a policy. Default is 1000
| MIN_NON_ZERO_TEMPERATURE | Minimum non-zero temperature value. Default is 0.0001
| MINIMUM_PROMPT_CACHE_TOKEN_COUNT | Minimum token count for caching a prompt. Default is 1024
| MISTRAL_API_BASE | Base URL for Mistral API. Default is https://api.mistral.ai
| MISTRAL_API_KEY | API key for Mistral API
| MICROSOFT_AUTHORIZATION_ENDPOINT | Custom authorization endpoint URL for Microsoft SSO (overrides default Microsoft OAuth authorization endpoint)
| MICROSOFT_CLIENT_ID | Client ID for Microsoft services
| MICROSOFT_CLIENT_SECRET | Client secret for Microsoft services
| MICROSOFT_SERVICE_PRINCIPAL_ID | Service Principal ID for Microsoft Enterprise Application. (This is an advanced feature if you want litellm to auto-assign members to Litellm Teams based on their Microsoft Entra ID Groups)
| MICROSOFT_TENANT | Tenant ID for Microsoft Azure
| MICROSOFT_TOKEN_ENDPOINT | Custom token endpoint URL for Microsoft SSO (overrides default Microsoft OAuth token endpoint)
| MICROSOFT_USER_DISPLAY_NAME_ATTRIBUTE | Field name for user display name in Microsoft SSO response. Default is `displayName`
| MICROSOFT_USER_EMAIL_ATTRIBUTE | Field name for user email in Microsoft SSO response. Default is `userPrincipalName`
| MICROSOFT_USER_FIRST_NAME_ATTRIBUTE | Field name for user first name in Microsoft SSO response. Default is `givenName`
| MICROSOFT_USER_ID_ATTRIBUTE | Field name for user ID in Microsoft SSO response. Default is `id`
| MICROSOFT_USER_LAST_NAME_ATTRIBUTE | Field name for user last name in Microsoft SSO response. Default is `surname`
| MICROSOFT_USERINFO_ENDPOINT | Custom userinfo endpoint URL for Microsoft SSO (overrides default Microsoft Graph userinfo endpoint)
| MODEL_COST_MAP_MAX_SHRINK_RATIO | Maximum allowed shrinkage ratio when validating a fetched model cost map against the local backup. Rejects the fetched map if it is smaller than this fraction of the backup. Default is 0.5
| MODEL_COST_MAP_MIN_MODEL_COUNT | Minimum number of models a fetched cost map must contain to be considered valid. Default is 50
| NO_DOCS | Flag to disable Swagger UI documentation
| NO_REDOC | Flag to disable Redoc documentation
| NO_PROXY | List of addresses to bypass proxy
| NON_LLM_CONNECTION_TIMEOUT | Timeout in seconds for non-LLM service connections. Default is 15
| OAUTH_TOKEN_INFO_ENDPOINT | Endpoint for OAuth token info retrieval
| OPENAI_BASE_URL | Base URL for OpenAI API
| OPENAI_API_BASE | Base URL for OpenAI API. Default is https://api.openai.com/
| OPENAI_API_KEY | API key for OpenAI services
| OPENAI_CHATGPT_API_BASE | Alternative to CHATGPT_API_BASE. Base URL for ChatGPT API
| OPENAI_FILE_SEARCH_COST_PER_1K_CALLS | Cost per 1000 calls for OpenAI file search. Default is 0.0025
| OPENAI_ORGANIZATION | Organization identifier for OpenAI
| OPENID_BASE_URL | Base URL for OpenID Connect services
| OPENID_CLIENT_ID | Client ID for OpenID Connect authentication
| OPENID_CLIENT_SECRET | Client secret for OpenID Connect authentication
| OPENMETER_API_ENDPOINT | API endpoint for OpenMeter integration
| OPENMETER_API_KEY | API key for OpenMeter services
| OPENMETER_EVENT_TYPE | Type of events sent to OpenMeter
| ONYX_API_BASE | Base URL for Onyx Security AI Guard service (defaults to https://ai-guard.onyx.security)
| ONYX_API_KEY | API key for Onyx Security AI Guard service
| ONYX_TIMEOUT | Timeout in seconds for Onyx Guard server requests. Default is 10
| OTEL_ENDPOINT | OpenTelemetry endpoint for traces
| OTEL_EXPORTER_OTLP_ENDPOINT | OpenTelemetry endpoint for traces
| OTEL_ENVIRONMENT_NAME | Environment name for OpenTelemetry
| OTEL_EXPORTER | Exporter type for OpenTelemetry
| OTEL_EXPORTER_OTLP_PROTOCOL | Exporter type for OpenTelemetry
| OTEL_HEADERS | Headers for OpenTelemetry requests
| OTEL_MODEL_ID | Model ID for OpenTelemetry tracing
| OTEL_EXPORTER_OTLP_HEADERS | Headers for OpenTelemetry requests
| OTEL_SERVICE_NAME | Service name identifier for OpenTelemetry
| OTEL_TRACER_NAME | Tracer name for OpenTelemetry tracing
| OTEL_LOGS_EXPORTER | Exporter type for OpenTelemetry logs (e.g., console)
| PAGERDUTY_API_KEY | API key for PagerDuty Alerting
| PANW_PRISMA_AIRS_API_KEY | API key for PANW Prisma AIRS service
| PANW_PRISMA_AIRS_API_BASE | Base URL for PANW Prisma AIRS service
| PHOENIX_API_KEY | API key for Arize Phoenix
| PHOENIX_COLLECTOR_ENDPOINT | API endpoint for Arize Phoenix
| PHOENIX_COLLECTOR_HTTP_ENDPOINT | API http endpoint for Arize Phoenix
| PILLAR_API_BASE | Base URL for Pillar API Guardrails
| PILLAR_API_KEY | API key for Pillar API Guardrails
| PILLAR_ON_FLAGGED_ACTION | Action to take when content is flagged ('block' or 'monitor')
| POD_NAME | Pod name for the server, this will be [emitted to `datadog` logs](https://docs.litellm.ai/docs/proxy/logging#datadog) as `POD_NAME` 
| POSTHOG_API_KEY | API key for PostHog analytics integration
| POSTHOG_API_URL | Base URL for PostHog API (defaults to https://us.i.posthog.com)
| POSTHOG_MOCK | Enable mock mode for PostHog integration testing. When set to true, intercepts PostHog API calls and returns mock responses without making actual network calls. Default is false
| POSTHOG_MOCK_LATENCY_MS | Mock latency in milliseconds for PostHog API calls when mock mode is enabled. Simulates network round-trip time. Default is 100ms
| PREDIBASE_API_BASE | Base URL for Predibase API
| PRESIDIO_ANALYZER_API_BASE | Base URL for Presidio Analyzer service
| PRESIDIO_ANONYMIZER_API_BASE | Base URL for Presidio Anonymizer service
| PROMETHEUS_BUDGET_METRICS_REFRESH_INTERVAL_MINUTES | Refresh interval in minutes for Prometheus budget metrics. Default is 5
| PROMETHEUS_FALLBACK_STATS_SEND_TIME_HOURS | Fallback time in hours for sending stats to Prometheus. Default is 9
| PROMETHEUS_URL | URL for Prometheus service
| PROMPTLAYER_API_KEY | API key for PromptLayer integration
| PROXY_ADMIN_ID | Admin identifier for proxy server
| PROXY_BASE_URL | Base URL for proxy service
| PROXY_BATCH_WRITE_AT | Time in seconds to wait before batch writing spend logs to the database. Default is 10
| PROXY_BATCH_POLLING_INTERVAL | Time in seconds to wait before polling a batch, to check if it's completed. Default is 6000s (1 hour)
| PROXY_BUDGET_RESCHEDULER_MAX_TIME | Maximum time in seconds to wait before checking database for budget resets. Default is 605
| PROXY_BUDGET_RESCHEDULER_MIN_TIME | Minimum time in seconds to wait before checking database for budget resets. Default is 597
| PYTHON_GC_THRESHOLD | GC thresholds ('gen0,gen1,gen2', e.g. '1000,50,50'); defaults to Python’s values.
| PROXY_LOGOUT_URL | URL for logging out of the proxy service
| QDRANT_API_BASE | Base URL for Qdrant API
| QDRANT_API_KEY | API key for Qdrant service
| QDRANT_SCALAR_QUANTILE | Scalar quantile for Qdrant operations. Default is 0.99
| QDRANT_URL | Connection URL for Qdrant database
| QDRANT_VECTOR_SIZE | Vector size for Qdrant operations. Default is 1536
| REDIS_CONNECTION_POOL_TIMEOUT | Timeout in seconds for Redis connection pool. Default is 5
| REDIS_HOST | Hostname for Redis server
| REDIS_PASSWORD | Password for Redis service
| REDIS_PORT | Port number for Redis server
| REDIS_SOCKET_TIMEOUT | Timeout in seconds for Redis socket operations. Default is 0.1
| REDIS_GCP_SERVICE_ACCOUNT | GCP service account for IAM authentication with Redis. Format: "projects/-/serviceAccounts/name@project.iam.gserviceaccount.com"
| REDIS_GCP_SSL_CA_CERTS | Path to SSL CA certificate file for secure GCP Memorystore Redis connections
| REDOC_URL | The path to the Redoc Fast API documentation. **By default this is "/redoc"**
| REPEATED_STREAMING_CHUNK_LIMIT | Limit for repeated streaming chunks to detect looping. Default is 100
| REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES | Maximum size in bytes for WebSocket messages in realtime connections. Default is None.
| REPLICATE_MODEL_NAME_WITH_ID_LENGTH | Length of Replicate model names with ID. Default is 64
| REPLICATE_POLLING_DELAY_SECONDS | Delay in seconds for Replicate polling operations. Default is 0.5
| REQUEST_TIMEOUT | Timeout in seconds for requests. Default is 6000
| ROOT_REDIRECT_URL | URL to redirect root path (/) to when DOCS_URL is set to something other than "/" (DOCS_URL is "/" by default)
| ROUTER_MAX_FALLBACKS | Maximum number of fallbacks for router. Default is 5
| RUNWAYML_DEFAULT_API_VERSION | Default API version for RunwayML service. Default is "2024-11-06"
| RUNWAYML_POLLING_TIMEOUT | Timeout in seconds for RunwayML image generation polling. Default is 600 (10 minutes)
| S3_VECTORS_DEFAULT_DIMENSION | Default vector dimension for S3 Vectors RAG ingestion. Default is 1024
| S3_VECTORS_DEFAULT_DISTANCE_METRIC | Default distance metric for S3 Vectors RAG ingestion. Options: "cosine", "euclidean". Default is "cosine"
| SECRET_MANAGER_REFRESH_INTERVAL | Refresh interval in seconds for secret manager. Default is 86400 (24 hours)
| SEPARATE_HEALTH_APP | If set to '1', runs health endpoints on a separate ASGI app and port. Default: '0'.
| SEPARATE_HEALTH_PORT | Port for the separate health endpoints app. Only used if SEPARATE_HEALTH_APP=1. Default: 4001.
| SUPERVISORD_STOPWAITSECS | Upper bound timeout in seconds for graceful shutdown when SEPARATE_HEALTH_APP=1. Default: 3600 (1 hour).
| SERVER_ROOT_PATH | Root path for the server application
| SEND_USER_API_KEY_ALIAS | Flag to send user API key alias to Zscaler AI Guard. Default is False
| SEND_USER_API_KEY_TEAM_ID | Flag to send user API key team ID to Zscaler AI Guard. Default is False
| SEND_USER_API_KEY_USER_ID | Flag to send user API key user ID to Zscaler AI Guard. Default is False
| SET_VERBOSE | [DEPRECATED] Use `LITELLM_LOG` instead with values "INFO", "DEBUG", or "ERROR". See [debugging docs](./debugging)
| SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD | Minimum number of requests to consider "reasonable traffic" for single-deployment cooldown logic. Default is 1000
| SLACK_DAILY_REPORT_FREQUENCY | Frequency of daily Slack reports (e.g., daily, weekly)
| SLACK_WEBHOOK_URL | Webhook URL for Slack integration
| SMTP_HOST | Hostname for the SMTP server
| SMTP_PASSWORD | Password for SMTP authentication (do not set if SMTP does not require auth)
| SMTP_PORT | Port number for SMTP server
| SMTP_SENDER_EMAIL | Email address used as the sender in SMTP transactions
| SMTP_SENDER_LOGO | Logo used in emails sent via SMTP
| SMTP_TLS | Flag to enable or disable TLS for SMTP connections
| SMTP_USERNAME | Username for SMTP authentication (do not set if SMTP does not require auth)
| SENDGRID_API_KEY | API key for SendGrid email service
| RESEND_API_KEY | API key for Resend email service
| SENDGRID_SENDER_EMAIL | Email address used as the sender in SendGrid email transactions 
| SPEND_LOGS_URL | URL for retrieving spend logs
| SPEND_LOG_CLEANUP_BATCH_SIZE | Number of logs deleted per batch during cleanup. Default is 1000
| SSL_CERTIFICATE | Path to the SSL certificate file
| SSL_ECDH_CURVE | ECDH curve for SSL/TLS key exchange (e.g., 'X25519' to disable PQC).
| SSL_SECURITY_LEVEL | [BETA] Security level for SSL/TLS connections. E.g. `DEFAULT@SECLEVEL=1`
| SSL_VERIFY | Flag to enable or disable SSL certificate verification
| SSL_CERT_FILE | Path to the SSL certificate file for custom CA bundle
| SUPABASE_KEY | API key for Supabase service
| SUPABASE_URL | Base URL for Supabase instance
| STORE_MODEL_IN_DB | If true, enables storing model + credential information in the DB. 
| SYSTEM_MESSAGE_TOKEN_COUNT | Token count for system messages. Default is 4
| TEST_EMAIL_ADDRESS | Email address used for testing purposes
| TOGETHER_AI_4_B | Size parameter for Together AI 4B model. Default is 4
| TOGETHER_AI_8_B | Size parameter for Together AI 8B model. Default is 8
| TOGETHER_AI_21_B | Size parameter for Together AI 21B model. Default is 21
| TOGETHER_AI_41_B | Size parameter for Together AI 41B model. Default is 41
| TOGETHER_AI_80_B | Size parameter for Together AI 80B model. Default is 80
| TOGETHER_AI_110_B | Size parameter for Together AI 110B model. Default is 110
| TOGETHER_AI_EMBEDDING_150_M | Size parameter for Together AI 150M embedding model. Default is 150
| TOGETHER_AI_EMBEDDING_350_M | Size parameter for Together AI 350M embedding model. Default is 350
| TOOL_CHOICE_OBJECT_TOKEN_COUNT | Token count for tool choice objects. Default is 4
| UI_LOGO_PATH | Path to the logo image used in the UI
| UI_PASSWORD | Password for accessing the UI
| UI_USERNAME | Username for accessing the UI
| UPSTREAM_LANGFUSE_DEBUG | Flag to enable debugging for upstream Langfuse
| UPSTREAM_LANGFUSE_HOST | Host URL for upstream Langfuse service
| UPSTREAM_LANGFUSE_PUBLIC_KEY | Public key for upstream Langfuse authentication
| UPSTREAM_LANGFUSE_RELEASE | Release version identifier for upstream Langfuse
| UPSTREAM_LANGFUSE_SECRET_KEY | Secret key for upstream Langfuse authentication
| USE_AWS_KMS | Flag to enable AWS Key Management Service for encryption
| USE_PRISMA_MIGRATE | Flag to use prisma migrate instead of prisma db push. Recommended for production environments.
| WANDB_API_KEY | API key for Weights & Biases (W&B) logging integration
| WANDB_HOST | Host URL for Weights & Biases (W&B) service
| WANDB_PROJECT_ID | Project ID for Weights & Biases (W&B) logging integration
| WEBHOOK_URL | URL for receiving webhooks from external services
| SPEND_LOG_RUN_LOOPS | Constant for setting how many runs of 1000 batch deletes should spend_log_cleanup task run
| SPEND_LOG_CLEANUP_BATCH_SIZE | Number of logs deleted per batch during cleanup. Default is 1000
| SPEND_LOG_QUEUE_POLL_INTERVAL | Polling interval in seconds for spend log queue. Default is 2.0
| SPEND_LOG_QUEUE_SIZE_THRESHOLD | Threshold for spend log queue size before processing. Default is 100
| COROUTINE_CHECKER_MAX_SIZE_IN_MEMORY | Maximum size for CoroutineChecker in-memory cache. Default is 1000
| DEFAULT_SHARED_HEALTH_CHECK_TTL | Time-to-live in seconds for cached health check results in shared health check mode. Default is 300 (5 minutes)
| DEFAULT_SHARED_HEALTH_CHECK_LOCK_TTL | Time-to-live in seconds for health check lock in shared health check mode. Default is 60 (1 minute)
| ZSCALER_AI_GUARD_API_KEY | API key for Zscaler AI Guard service
| ZSCALER_AI_GUARD_POLICY_ID | Policy ID for Zscaler AI Guard guardrails
| ZSCALER_AI_GUARD_URL | Base URL for Zscaler AI Guard API. Default is https://api.us1.zseclipse.net/v1/detection/execute-policy
