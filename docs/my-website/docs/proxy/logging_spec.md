
# StandardLoggingPayload Specification

Found under `kwargs["standard_logging_object"]`. This is a standard payload, logged for every successful and failed response.

## StandardLoggingPayload

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier |
| `trace_id` | `str` | Trace multiple LLM calls belonging to same overall request |
| `call_type` | `str` | Type of call |
| `response_cost` | `float` | Cost of the response in USD ($) |
| `response_cost_failure_debug_info` | `StandardLoggingModelCostFailureDebugInformation` | Debug information if cost tracking fails |
| `status` | `StandardLoggingPayloadStatus` | Status of the payload |
| `total_tokens` | `int` | Total number of tokens |
| `prompt_tokens` | `int` | Number of prompt tokens |
| `completion_tokens` | `int` | Number of completion tokens |
| `startTime` | `float` | Start time of the call |
| `endTime` | `float` | End time of the call |
| `completionStartTime` | `float` | Time to first token for streaming requests |
| `response_time` | `float` | Total response time. If streaming, this is the time to first token |
| `model_map_information` | `StandardLoggingModelInformation` | Model mapping information |
| `model` | `str` | Model name sent in request |
| `model_id` | `Optional[str]` | Model ID of the deployment used |
| `model_group` | `Optional[str]` | `model_group` used for the request |
| `api_base` | `str` | LLM API base URL |
| `metadata` | `StandardLoggingMetadata` | Metadata information |
| `cache_hit` | `Optional[bool]` | Whether cache was hit |
| `cache_key` | `Optional[str]` | Optional cache key |
| `saved_cache_cost` | `float` | Cost saved by cache |
| `request_tags` | `list` | List of request tags |
| `end_user` | `Optional[str]` | Optional end user identifier |
| `requester_ip_address` | `Optional[str]` | Optional requester IP address |
| `messages` | `Optional[Union[str, list, dict]]` | Messages sent in the request |
| `response` | `Optional[Union[str, list, dict]]` | LLM response |
| `error_str` | `Optional[str]` | Optional error string |
| `error_information` | `Optional[StandardLoggingPayloadErrorInformation]` | Optional error information |
| `model_parameters` | `dict` | Model parameters |
| `hidden_params` | `StandardLoggingHiddenParams` | Hidden parameters |

## StandardLoggingUserAPIKeyMetadata

| Field | Type | Description |
|-------|------|-------------|
| `user_api_key_hash` | `Optional[str]` | Hash of the litellm virtual key |
| `user_api_key_alias` | `Optional[str]` | Alias of the API key |
| `user_api_key_org_id` | `Optional[str]` | Organization ID associated with the key |
| `user_api_key_team_id` | `Optional[str]` | Team ID associated with the key |
| `user_api_key_user_id` | `Optional[str]` | User ID associated with the key |
| `user_api_key_team_alias` | `Optional[str]` | Team alias associated with the key |

## StandardLoggingMetadata

Inherits from `StandardLoggingUserAPIKeyMetadata` and adds:

| Field | Type | Description |
|-------|------|-------------|
| `spend_logs_metadata` | `Optional[dict]` | Key-value pairs for spend logging |
| `requester_ip_address` | `Optional[str]` | Requester's IP address |
| `requester_metadata` | `Optional[dict]` | Additional requester metadata |

## StandardLoggingAdditionalHeaders

| Field | Type | Description |
|-------|------|-------------|
| `x_ratelimit_limit_requests` | `int` | Rate limit for requests |
| `x_ratelimit_limit_tokens` | `int` | Rate limit for tokens |
| `x_ratelimit_remaining_requests` | `int` | Remaining requests in rate limit |
| `x_ratelimit_remaining_tokens` | `int` | Remaining tokens in rate limit |

## StandardLoggingHiddenParams

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | `Optional[str]` | Optional model ID |
| `cache_key` | `Optional[str]` | Optional cache key |
| `api_base` | `Optional[str]` | Optional API base URL |
| `response_cost` | `Optional[str]` | Optional response cost |
| `additional_headers` | `Optional[StandardLoggingAdditionalHeaders]` | Additional headers |

## StandardLoggingModelInformation

| Field | Type | Description |
|-------|------|-------------|
| `model_map_key` | `str` | Model map key |
| `model_map_value` | `Optional[ModelInfo]` | Optional model information |

## StandardLoggingModelCostFailureDebugInformation

| Field | Type | Description |
|-------|------|-------------|
| `error_str` | `str` | Error string |
| `traceback_str` | `str` | Traceback string |
| `model` | `str` | Model name |
| `cache_hit` | `Optional[bool]` | Whether cache was hit |
| `custom_llm_provider` | `Optional[str]` | Optional custom LLM provider |
| `base_model` | `Optional[str]` | Optional base model |
| `call_type` | `str` | Call type |
| `custom_pricing` | `Optional[bool]` | Whether custom pricing was used |

## StandardLoggingPayloadErrorInformation

| Field | Type | Description |
|-------|------|-------------|
| `error_code` | `Optional[str]` | Optional error code (eg. "429") |
| `error_class` | `Optional[str]` | Optional error class (eg. "RateLimitError") |
| `llm_provider` | `Optional[str]` | LLM provider that returned the error (eg. "openai")` |

## StandardLoggingPayloadStatus

A literal type with two possible values:
- `"success"`
- `"failure"`