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
  turn_off_message_logging: boolean  # prevent the messages and responses from being logged to on your callbacks, but request metadata will still be logged.
  redact_user_api_key_info: boolean  # Redact information about the user api key (hashed token, user_id, team id, etc.), from logs. Currently supported for Langfuse, OpenTelemetry, Logfire, ArizeAI logging.
  langfuse_default_tags: ["cache_hit", "cache_key", "proxy_base_url", "user_api_key_alias", "user_api_key_user_id", "user_api_key_user_email", "user_api_key_team_alias", "semantic-similarity", "proxy_base_url"] # default tags for Langfuse Logging
  
  # Networking settings
  request_timeout: 10 # (int) llm requesttimeout in seconds. Raise Timeout error if call takes longer than 10s. Sets litellm.request_timeout 
  force_ipv4: boolean # If true, litellm will force ipv4 for all LLM requests. Some users have seen httpx ConnectionError when using ipv6 + Anthropic API
  
  set_verbose: boolean # sets litellm.set_verbose=True to view verbose debug logs. DO NOT LEAVE THIS ON IN PRODUCTION
  json_logs: boolean # if true, logs will be in json format

  # Fallbacks, reliability
  default_fallbacks: ["claude-opus"] # set default_fallbacks, in case a specific model group is misconfigured / bad.
  content_policy_fallbacks: [{"gpt-3.5-turbo-small": ["claude-opus"]}] # fallbacks for ContentPolicyErrors
  context_window_fallbacks: [{"gpt-3.5-turbo-small": ["gpt-3.5-turbo-large", "claude-opus"]}] # fallbacks for ContextWindowExceededErrors



  # Caching settings
  cache: true 
  cache_params:        # set cache params for redis
    type: redis        # type of cache to initialize

    # Optional - Redis Settings
    host: "localhost"  # The host address for the Redis cache. Required if type is "redis".
    port: 6379  # The port number for the Redis cache. Required if type is "redis".
    password: "your_password"  # The password for the Redis cache. Required if type is "redis".
    namespace: "litellm.caching.caching" # namespace for redis cache
  
    # Optional - Redis Cluster Settings
    redis_startup_nodes: [{"host": "127.0.0.1", "port": "7001"}] 

    # Optional - Redis Sentinel Settings
    service_name: "mymaster"
    sentinel_nodes: [["localhost", 26379]]

    # Optional - Qdrant Semantic Cache Settings
    qdrant_semantic_cache_embedding_model: openai-embedding # the model should be defined on the model_list
    qdrant_collection_name: test_collection
    qdrant_quantization_config: binary
    similarity_threshold: 0.8   # similarity threshold for semantic cache

    # Optional - S3 Cache Settings
    s3_bucket_name: cache-bucket-litellm   # AWS Bucket Name for S3
    s3_region_name: us-west-2              # AWS Region Name for S3
    s3_aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID  # us os.environ/<variable name> to pass environment variables. This is AWS Access Key ID for S3
    s3_aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY  # AWS Secret Access Key for S3
    s3_endpoint_url: https://s3.amazonaws.com  # [OPTIONAL] S3 endpoint URL, if you want to use Backblaze/cloudflare s3 bucket

    # Common Cache settings
    # Optional - Supported call types for caching
    supported_call_types: ["acompletion", "atext_completion", "aembedding", "atranscription"]
                          # /chat/completions, /completions, /embeddings, /audio/transcriptions
    mode: default_off # if default_off, you need to opt in to caching on a per call basis
    ttl: 600 # ttl for caching


callback_settings:
  otel:
    message_logging: boolean  # OTEL logging callback specific settings

general_settings:
  completion_model: string
  disable_spend_logs: boolean  # turn off writing each transaction to the db
  disable_master_key_return: boolean  # turn off returning master key on UI (checked on '/user/info' endpoint)
  disable_retry_on_max_parallel_request_limit_error: boolean  # turn off retries when max parallel request limit is reached
  disable_reset_budget: boolean  # turn off reset budget scheduled task
  disable_adding_master_key_hash_to_db: boolean  # turn off storing master key hash in db, for spend tracking
  enable_jwt_auth: boolean  # allow proxy admin to auth in via jwt tokens with 'litellm_proxy_admin' in claims
  enforce_user_param: boolean  # requires all openai endpoint requests to have a 'user' param
  allowed_routes: ["route1", "route2"]  # list of allowed proxy API routes - a user can access. (currently JWT-Auth only)
  key_management_system: google_kms  # either google_kms or azure_kms
  master_key: string

  # Database Settings
  database_url: string
  database_connection_pool_limit: 0  # default 100
  database_connection_timeout: 0  # default 60s
  allow_requests_on_db_unavailable: boolean  # if true, will allow requests that can not connect to the DB to verify Virtual Key to still work 

  custom_auth: string
  max_parallel_requests: 0  # the max parallel requests allowed per deployment 
  global_max_parallel_requests: 0  # the max parallel requests allowed on the proxy all up 
  infer_model_from_keys: true
  background_health_checks: true
  health_check_interval: 300
  alerting: ["slack", "email"]
  alerting_threshold: 0
  use_client_credentials_pass_through_routes: boolean  # use client credentials for all pass through routes like "/vertex-ai", /bedrock/. When this is True Virtual Key auth will not be applied on these endpoints
```

### litellm_settings - Reference

| Name | Type | Description |
|------|------|-------------|
| success_callback | array of strings | List of success callbacks. [Doc Proxy logging callbacks](logging), [Doc Metrics](prometheus) |
| failure_callback | array of strings | List of failure callbacks [Doc Proxy logging callbacks](logging), [Doc Metrics](prometheus) |
| callbacks | array of strings | List of callbacks - runs on success and failure [Doc Proxy logging callbacks](logging), [Doc Metrics](prometheus) |
| service_callbacks | array of strings | System health monitoring - Logs redis, postgres failures on specified services (e.g. datadog, prometheus) [Doc Metrics](prometheus) |
| turn_off_message_logging | boolean | If true, prevents messages and responses from being logged to callbacks, but request metadata will still be logged [Proxy Logging](logging) |
| modify_params | boolean | If true, allows modifying the parameters of the request before it is sent to the LLM provider |
| enable_preview_features | boolean | If true, enables preview features - e.g. Azure O1 Models with streaming support.|
| redact_user_api_key_info | boolean | If true, redacts information about the user api key from logs [Proxy Logging](logging#redacting-userapikeyinfo) |
| langfuse_default_tags | array of strings | Default tags for Langfuse Logging. Use this if you want to control which LiteLLM-specific fields are logged as tags by the LiteLLM proxy. By default LiteLLM Proxy logs no LiteLLM-specific fields as tags. [Further docs](./logging#litellm-specific-tags-on-langfuse---cache_hit-cache_key) |
| set_verbose | boolean | If true, sets litellm.set_verbose=True to view verbose debug logs. DO NOT LEAVE THIS ON IN PRODUCTION |
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

### general_settings - Reference

| Name | Type | Description |
|------|------|-------------|
| completion_model | string | The default model to use for completions when `model` is not specified in the request |
| disable_spend_logs | boolean | If true, turns off writing each transaction to the database |
| disable_master_key_return | boolean | If true, turns off returning master key on UI. (checked on '/user/info' endpoint) |
| disable_retry_on_max_parallel_request_limit_error | boolean | If true, turns off retries when max parallel request limit is reached |
| disable_reset_budget | boolean | If true, turns off reset budget scheduled task |
| disable_adding_master_key_hash_to_db | boolean | If true, turns off storing master key hash in db |
| enable_jwt_auth | boolean | allow proxy admin to auth in via jwt tokens with 'litellm_proxy_admin' in claims. [Doc on JWT Tokens](token_auth) |
| enforce_user_param | boolean | If true, requires all OpenAI endpoint requests to have a 'user' param. [Doc on call hooks](call_hooks)|
| allowed_routes | array of strings | List of allowed proxy API routes a user can access [Doc on controlling allowed routes](enterprise#control-available-public-private-routes)|
| key_management_system | string | Specifies the key management system. [Doc Secret Managers](../secret) |
| master_key | string | The master key for the proxy [Set up Virtual Keys](virtual_keys) |
| database_url | string | The URL for the database connection [Set up Virtual Keys](virtual_keys) |
| database_connection_pool_limit | integer | The limit for database connection pool [Setting DB Connection Pool limit](#configure-db-pool-limits--connection-timeouts) |
| database_connection_timeout | integer | The timeout for database connections in seconds [Setting DB Connection Pool limit, timeout](#configure-db-pool-limits--connection-timeouts) |
| allow_requests_on_db_unavailable | boolean | If true, allows requests to succeed even if DB is unreachable. **Only use this if running LiteLLM in your VPC** This will allow requests to work even when LiteLLM cannot connect to the DB to verify a Virtual Key |
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
| store_model_in_db | boolean | If true, allows `/model/new` endpoint to store model information in db. Endpoint disabled by default. [Doc on `/model/new` endpoint](./model_management.md#create-a-new-model) |
| store_prompts_in_spend_logs | boolean | If true, allows prompts and responses to be stored in the spend logs table. |
| max_request_size_mb | int | The maximum size for requests in MB. Requests above this size will be rejected. |
| max_response_size_mb | int | The maximum size for responses in MB. LLM Responses above this size will not be sent. |
| proxy_budget_rescheduler_min_time | int | The minimum time (in seconds) to wait before checking db for budget resets. **Default is 597 seconds** |
| proxy_budget_rescheduler_max_time | int | The maximum time (in seconds) to wait before checking db for budget resets. **Default is 605 seconds** |
| proxy_batch_write_at | int | Time (in seconds) to wait before batch writing spend logs to the db. **Default is 10 seconds** |
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
| forward_client_headers_to_llm_api | boolean | If true, forwards the client headers (any `x-` headers) to the backend LLM call |

### router_settings - Reference

:::info

Most values can also be set via `litellm_settings`. If you see overlapping values, settings on `router_settings` will override those on `litellm_settings`.
:::

```yaml
router_settings:
  routing_strategy: usage-based-routing-v2 # Literal["simple-shuffle", "least-busy", "usage-based-routing","latency-based-routing"], default="simple-shuffle"
  redis_host: <your-redis-host>           # string
  redis_password: <your-redis-password>   # string
  redis_port: <your-redis-port>           # string
  enable_pre_call_checks: true            # bool - Before call is made check if a call is within model context window 
  allowed_fails: 3 # cooldown model if it fails > 1 call in a minute. 
  cooldown_time: 30 # (in seconds) how long to cooldown model if fails/min > allowed_fails
  disable_cooldowns: True                  # bool - Disable cooldowns for all models 
  enable_tag_filtering: True                # bool - Use tag based routing for requests
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
| enable_pre_call_check | boolean | If true, checks if a call is within the model's context window before making the call. [More information here](reliability) |
| content_policy_fallbacks | array of objects | Specifies fallback models for content policy violations. [More information here](reliability) |
| fallbacks | array of objects | Specifies fallback models for all types of errors. [More information here](reliability) |
| enable_tag_filtering | boolean | If true, uses tag based routing for requests [Tag Based Routing](tag_routing) |
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
| cache_kwargs | dict | Additional keyword arguments for the cache initialization. |
| routing_strategy_args | dict | Additional keyword arguments for the routing strategy - e.g. lowest latency routing default ttl |
| model_group_alias | dict | Model group alias mapping. E.g. `{"claude-3-haiku": "claude-3-haiku-20240229"}` |
| num_retries | int | Number of retries for a request. Defaults to 3. |
| default_fallbacks | Optional[List[str]] | Fallbacks to try if no model group-specific fallbacks are defined. |
| caching_groups | Optional[List[tuple]] | List of model groups for caching across model groups. Defaults to None. - e.g. caching_groups=[("openai-gpt-3.5-turbo", "azure-gpt-3.5-turbo")]|
| alerting_config | AlertingConfig | [SDK-only arg] Slack alerting configuration. Defaults to None. [Further Docs](../routing.md#alerting-) |
| assistants_config | AssistantsConfig | Set on proxy via `assistant_settings`. [Further docs](../assistants.md) |
| set_verbose | boolean | [DEPRECATED PARAM - see debug docs](./debugging.md) If true, sets the logging level to verbose. |
| retry_after | int | Time to wait before retrying a request in seconds. Defaults to 0. If `x-retry-after` is received from LLM API, this value is overridden. |
| provider_budget_config | ProviderBudgetConfig | Provider budget configuration. Use this to set llm_provider budget limits. example $100/day to OpenAI, $100/day to Azure, etc. Defaults to None. [Further Docs](./provider_budget_routing.md) |
| enable_pre_call_checks | boolean | If true, checks if a call is within the model's context window before making the call. [More information here](reliability) |
| model_group_retry_policy | Dict[str, RetryPolicy] | [SDK-only arg] Set retry policy for model groups. |
| context_window_fallbacks | List[Dict[str, List[str]]] | Fallback models for context window violations. |
| redis_url | str | URL for Redis server. **Known performance issue with Redis URL.** |
| cache_responses | boolean | Flag to enable caching LLM Responses, if cache set under `router_settings`. If true, caches responses. Defaults to False. |
| router_general_settings | RouterGeneralSettings | [SDK-Only] Router general settings - contains optimizations like 'async_only_mode'. [Docs](../routing.md#router-general-settings) |
| optional_pre_call_checks | List[str] | List of pre-call checks to add to the router. Currently supported: 'router_budget_limiting', 'prompt_caching' |


### environment variables - Reference

| Name | Description |
|------|-------------|
| ACTIONS_ID_TOKEN_REQUEST_TOKEN | Token for requesting ID in GitHub Actions
| ACTIONS_ID_TOKEN_REQUEST_URL | URL for requesting ID token in GitHub Actions
| AISPEND_ACCOUNT_ID | Account ID for AI Spend
| AISPEND_API_KEY | API Key for AI Spend
| ALLOWED_EMAIL_DOMAINS | List of email domains allowed for access
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
| AWS_ACCESS_KEY_ID | Access Key ID for AWS services
| AWS_PROFILE_NAME | AWS CLI profile name to be used
| AWS_REGION_NAME | Default AWS region for service interactions
| AWS_ROLE_NAME | Role name for AWS IAM usage
| AWS_SECRET_ACCESS_KEY | Secret Access Key for AWS services
| AWS_SESSION_NAME | Name for AWS session
| AWS_WEB_IDENTITY_TOKEN | Web identity token for AWS
| AZURE_API_VERSION | Version of the Azure API being used
| AZURE_AUTHORITY_HOST | Azure authority host URL
| AZURE_CLIENT_ID | Client ID for Azure services
| AZURE_CLIENT_SECRET | Client secret for Azure services
| AZURE_FEDERATED_TOKEN_FILE | File path to Azure federated token
| AZURE_KEY_VAULT_URI | URI for Azure Key Vault
| AZURE_STORAGE_ACCOUNT_KEY | The Azure Storage Account Key to use for Authentication to Azure Blob Storage logging
| AZURE_STORAGE_ACCOUNT_NAME | Name of the Azure Storage Account to use for logging to Azure Blob Storage
| AZURE_STORAGE_FILE_SYSTEM | Name of the Azure Storage File System to use for logging to Azure Blob Storage.  (Typically the Container name)
| AZURE_STORAGE_TENANT_ID | The Application Tenant ID to use for Authentication to Azure Blob Storage logging
| AZURE_STORAGE_CLIENT_ID | The Application Client ID to use for Authentication to Azure Blob Storage logging
| AZURE_STORAGE_CLIENT_SECRET | The Application Client Secret to use for Authentication to Azure Blob Storage logging
| AZURE_TENANT_ID | Tenant ID for Azure Active Directory
| BERRISPEND_ACCOUNT_ID | Account ID for BerriSpend service
| BRAINTRUST_API_KEY | API key for Braintrust integration
| CIRCLE_OIDC_TOKEN | OpenID Connect token for CircleCI
| CIRCLE_OIDC_TOKEN_V2 | Version 2 of the OpenID Connect token for CircleCI
| CONFIG_FILE_PATH | File path for configuration file
| CUSTOM_TIKTOKEN_CACHE_DIR | Custom directory for Tiktoken cache
| DATABASE_HOST | Hostname for the database server
| DATABASE_NAME | Name of the database
| DATABASE_PASSWORD | Password for the database user
| DATABASE_PORT | Port number for database connection
| DATABASE_SCHEMA | Schema name used in the database
| DATABASE_URL | Connection URL for the database
| DATABASE_USER | Username for database connection
| DATABASE_USERNAME | Alias for database user
| DATABRICKS_API_BASE | Base URL for Databricks API
| DD_BASE_URL | Base URL for Datadog integration
| DATADOG_BASE_URL | (Alternative to DD_BASE_URL) Base URL for Datadog integration
| _DATADOG_BASE_URL | (Alternative to DD_BASE_URL) Base URL for Datadog integration
| DD_API_KEY | API key for Datadog integration
| DD_SITE | Site URL for Datadog (e.g., datadoghq.com)
| DD_SOURCE | Source identifier for Datadog logs
| DD_ENV | Environment identifier for Datadog logs. Only supported for `datadog_llm_observability` callback
| DD_SERVICE | Service identifier for Datadog logs. Defaults to "litellm-server"
| DD_VERSION | Version identifier for Datadog logs. Defaults to "unknown"
| DEBUG_OTEL | Enable debug mode for OpenTelemetry
| DIRECT_URL | Direct URL for service endpoint
| DISABLE_ADMIN_UI | Toggle to disable the admin UI
| DISABLE_SCHEMA_UPDATE | Toggle to disable schema updates
| DOCS_DESCRIPTION | Description text for documentation pages
| DOCS_FILTERED | Flag indicating filtered documentation
| DOCS_TITLE | Title of the documentation pages
| DOCS_URL | The path to the Swagger API documentation. **By default this is "/"**
| EMAIL_SUPPORT_CONTACT | Support contact email address
| GCS_BUCKET_NAME | Name of the Google Cloud Storage bucket
| GCS_PATH_SERVICE_ACCOUNT | Path to the Google Cloud service account JSON file
| GCS_FLUSH_INTERVAL | Flush interval for GCS logging (in seconds). Specify how often you want a log to be sent to GCS. **Default is 20 seconds**
| GCS_BATCH_SIZE | Batch size for GCS logging. Specify after how many logs you want to flush to GCS. If `BATCH_SIZE` is set to 10, logs are flushed every 10 logs. **Default is 2048**
| GCS_PUBSUB_TOPIC_ID | PubSub Topic ID to send LiteLLM SpendLogs to.
| GCS_PUBSUB_PROJECT_ID | PubSub Project ID to send LiteLLM SpendLogs to.
| GENERIC_AUTHORIZATION_ENDPOINT | Authorization endpoint for generic OAuth providers
| GENERIC_CLIENT_ID | Client ID for generic OAuth providers
| GENERIC_CLIENT_SECRET | Client secret for generic OAuth providers
| GENERIC_CLIENT_STATE | State parameter for generic client authentication
| GENERIC_INCLUDE_CLIENT_ID | Include client ID in requests for OAuth
| GENERIC_SCOPE | Scope settings for generic OAuth providers
| GENERIC_TOKEN_ENDPOINT | Token endpoint for generic OAuth providers
| GENERIC_USER_DISPLAY_NAME_ATTRIBUTE | Attribute for user's display name in generic auth
| GENERIC_USER_EMAIL_ATTRIBUTE | Attribute for user's email in generic auth
| GENERIC_USER_FIRST_NAME_ATTRIBUTE | Attribute for user's first name in generic auth
| GENERIC_USER_ID_ATTRIBUTE | Attribute for user ID in generic auth
| GENERIC_USER_LAST_NAME_ATTRIBUTE | Attribute for user's last name in generic auth
| GENERIC_USER_PROVIDER_ATTRIBUTE | Attribute specifying the user's provider
| GENERIC_USER_ROLE_ATTRIBUTE | Attribute specifying the user's role
| GENERIC_USERINFO_ENDPOINT | Endpoint to fetch user information in generic OAuth
| GALILEO_BASE_URL | Base URL for Galileo platform
| GALILEO_PASSWORD | Password for Galileo authentication
| GALILEO_PROJECT_ID | Project ID for Galileo usage
| GALILEO_USERNAME | Username for Galileo authentication
| GREENSCALE_API_KEY | API key for Greenscale service
| GREENSCALE_ENDPOINT | Endpoint URL for Greenscale service
| GOOGLE_APPLICATION_CREDENTIALS | Path to Google Cloud credentials JSON file
| GOOGLE_CLIENT_ID | Client ID for Google OAuth
| GOOGLE_CLIENT_SECRET | Client secret for Google OAuth
| GOOGLE_KMS_RESOURCE_NAME | Name of the resource in Google KMS
| HF_API_BASE | Base URL for Hugging Face API
| HCP_VAULT_ADDR | Address for [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault)
| HCP_VAULT_CLIENT_CERT | Path to client certificate for [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault)
| HCP_VAULT_CLIENT_KEY | Path to client key for [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault)
| HCP_VAULT_NAMESPACE | Namespace for [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault)
| HCP_VAULT_TOKEN | Token for [Hashicorp Vault Secret Manager](../secret.md#hashicorp-vault)
| HCP_VAULT_CERT_ROLE | Role for [Hashicorp Vault Secret Manager Auth](../secret.md#hashicorp-vault)
| HELICONE_API_KEY | API key for Helicone service
| HOSTNAME | Hostname for the server, this will be [emitted to `datadog` logs](https://docs.litellm.ai/docs/proxy/logging#datadog)
| HUGGINGFACE_API_BASE | Base URL for Hugging Face API
| IAM_TOKEN_DB_AUTH | IAM token for database authentication
| JSON_LOGS | Enable JSON formatted logging
| JWT_AUDIENCE | Expected audience for JWT tokens
| JWT_PUBLIC_KEY_URL | URL to fetch public key for JWT verification
| LAGO_API_BASE | Base URL for Lago API
| LAGO_API_CHARGE_BY | Parameter to determine charge basis in Lago
| LAGO_API_EVENT_CODE | Event code for Lago API events
| LAGO_API_KEY | API key for accessing Lago services
| LANGFUSE_DEBUG | Toggle debug mode for Langfuse
| LANGFUSE_FLUSH_INTERVAL | Interval for flushing Langfuse logs
| LANGFUSE_HOST | Host URL for Langfuse service
| LANGFUSE_PUBLIC_KEY | Public key for Langfuse authentication
| LANGFUSE_RELEASE | Release version of Langfuse integration
| LANGFUSE_SECRET_KEY | Secret key for Langfuse authentication
| LANGSMITH_API_KEY | API key for Langsmith platform
| LANGSMITH_BASE_URL | Base URL for Langsmith service
| LANGSMITH_BATCH_SIZE | Batch size for operations in Langsmith
| LANGSMITH_DEFAULT_RUN_NAME | Default name for Langsmith run
| LANGSMITH_PROJECT | Project name for Langsmith integration
| LANGSMITH_SAMPLING_RATE | Sampling rate for Langsmith logging
| LANGTRACE_API_KEY | API key for Langtrace service
| LITERAL_API_KEY | API key for Literal integration
| LITERAL_API_URL | API URL for Literal service
| LITERAL_BATCH_SIZE | Batch size for Literal operations
| LITELLM_DONT_SHOW_FEEDBACK_BOX | Flag to hide feedback box in LiteLLM UI
| LITELLM_DROP_PARAMS | Parameters to drop in LiteLLM requests
| LITELLM_EMAIL | Email associated with LiteLLM account
| LITELLM_GLOBAL_MAX_PARALLEL_REQUEST_RETRIES | Maximum retries for parallel requests in LiteLLM
| LITELLM_GLOBAL_MAX_PARALLEL_REQUEST_RETRY_TIMEOUT | Timeout for retries of parallel requests in LiteLLM
| LITELLM_HOSTED_UI | URL of the hosted UI for LiteLLM
| LITELLM_LICENSE | License key for LiteLLM usage
| LITELLM_LOCAL_MODEL_COST_MAP | Local configuration for model cost mapping in LiteLLM
| LITELLM_LOG | Enable detailed logging for LiteLLM
| LITELLM_MODE | Operating mode for LiteLLM (e.g., production, development)
| LITELLM_SALT_KEY | Salt key for encryption in LiteLLM
| LITELLM_SECRET_AWS_KMS_LITELLM_LICENSE | AWS KMS encrypted license for LiteLLM
| LITELLM_TOKEN | Access token for LiteLLM integration
| LITELLM_PRINT_STANDARD_LOGGING_PAYLOAD | If true, prints the standard logging payload to the console - useful for debugging
| LOGFIRE_TOKEN | Token for Logfire logging service
| MICROSOFT_CLIENT_ID | Client ID for Microsoft services
| MICROSOFT_CLIENT_SECRET | Client secret for Microsoft services
| MICROSOFT_TENANT | Tenant ID for Microsoft Azure
| NO_DOCS | Flag to disable documentation generation
| NO_PROXY | List of addresses to bypass proxy
| OAUTH_TOKEN_INFO_ENDPOINT | Endpoint for OAuth token info retrieval
| OPENAI_API_BASE | Base URL for OpenAI API
| OPENAI_API_KEY | API key for OpenAI services
| OPENAI_ORGANIZATION | Organization identifier for OpenAI
| OPENID_BASE_URL | Base URL for OpenID Connect services
| OPENID_CLIENT_ID | Client ID for OpenID Connect authentication
| OPENID_CLIENT_SECRET | Client secret for OpenID Connect authentication
| OPENMETER_API_ENDPOINT | API endpoint for OpenMeter integration
| OPENMETER_API_KEY | API key for OpenMeter services
| OPENMETER_EVENT_TYPE | Type of events sent to OpenMeter
| OTEL_ENDPOINT | OpenTelemetry endpoint for traces
| OTEL_ENVIRONMENT_NAME | Environment name for OpenTelemetry
| OTEL_EXPORTER | Exporter type for OpenTelemetry
| OTEL_HEADERS | Headers for OpenTelemetry requests
| OTEL_SERVICE_NAME | Service name identifier for OpenTelemetry
| OTEL_TRACER_NAME | Tracer name for OpenTelemetry tracing
| PAGERDUTY_API_KEY | API key for PagerDuty Alerting
| POD_NAME | Pod name for the server, this will be [emitted to `datadog` logs](https://docs.litellm.ai/docs/proxy/logging#datadog) as `POD_NAME` 
| PREDIBASE_API_BASE | Base URL for Predibase API
| PRESIDIO_ANALYZER_API_BASE | Base URL for Presidio Analyzer service
| PRESIDIO_ANONYMIZER_API_BASE | Base URL for Presidio Anonymizer service
| PROMETHEUS_URL | URL for Prometheus service
| PROMPTLAYER_API_KEY | API key for PromptLayer integration
| PROXY_ADMIN_ID | Admin identifier for proxy server
| PROXY_BASE_URL | Base URL for proxy service
| PROXY_LOGOUT_URL | URL for logging out of the proxy service
| PROXY_MASTER_KEY | Master key for proxy authentication
| QDRANT_API_BASE | Base URL for Qdrant API
| QDRANT_API_KEY | API key for Qdrant service
| QDRANT_URL | Connection URL for Qdrant database
| REDIS_HOST | Hostname for Redis server
| REDIS_PASSWORD | Password for Redis service
| REDIS_PORT | Port number for Redis server
| REDOC_URL | The path to the Redoc Fast API documentation. **By default this is "/redoc"**
| SERVER_ROOT_PATH | Root path for the server application
| SET_VERBOSE | Flag to enable verbose logging
| SLACK_DAILY_REPORT_FREQUENCY | Frequency of daily Slack reports (e.g., daily, weekly)
| SLACK_WEBHOOK_URL | Webhook URL for Slack integration
| SMTP_HOST | Hostname for the SMTP server
| SMTP_PASSWORD | Password for SMTP authentication
| SMTP_PORT | Port number for SMTP server
| SMTP_SENDER_EMAIL | Email address used as the sender in SMTP transactions
| SMTP_SENDER_LOGO | Logo used in emails sent via SMTP
| SMTP_TLS | Flag to enable or disable TLS for SMTP connections
| SMTP_USERNAME | Username for SMTP authentication
| SPEND_LOGS_URL | URL for retrieving spend logs
| SSL_CERTIFICATE | Path to the SSL certificate file
| SSL_VERIFY | Flag to enable or disable SSL certificate verification
| SUPABASE_KEY | API key for Supabase service
| SUPABASE_URL | Base URL for Supabase instance
| TEST_EMAIL_ADDRESS | Email address used for testing purposes
| UI_LOGO_PATH | Path to the logo image used in the UI
| UI_PASSWORD | Password for accessing the UI
| UI_USERNAME | Username for accessing the UI
| UPSTREAM_LANGFUSE_DEBUG | Flag to enable debugging for upstream Langfuse
| UPSTREAM_LANGFUSE_HOST | Host URL for upstream Langfuse service
| UPSTREAM_LANGFUSE_PUBLIC_KEY | Public key for upstream Langfuse authentication
| UPSTREAM_LANGFUSE_RELEASE | Release version identifier for upstream Langfuse
| UPSTREAM_LANGFUSE_SECRET_KEY | Secret key for upstream Langfuse authentication
| USE_AWS_KMS | Flag to enable AWS Key Management Service for encryption
| WEBHOOK_URL | URL for receiving webhooks from external services

