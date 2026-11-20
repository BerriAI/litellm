# Kafka message format

Each Kafka record value is one UTF-8 JSON object containing the existing
`StandardLoggingPayload`. There is no `standard_logging_object` wrapper, batch
array, schema ID prefix or Kafka-specific envelope. Kafka compression is handled
by the client. Consumers decode the value as UTF-8 and parse JSON normally.
The key is the UTF-8 `litellm_call_id`, falling back to `id`. When neither is a
string, the key is unset. Kafka topic, partition, offset and broker timestamp
are record metadata, not fields added to this JSON.

## Complete example event

This is a complete illustrative non-streaming chat event with message redaction
enabled. It follows an SDK-generated event, with synthetic identifiers, model,
timestamps and charges. The charges are example data, not provider prices.
The same JSON encoding applies to streaming final events and failures. This
example is not an exhaustive schema. Optional fields from other routes and
future standard-payload versions can appear. Consumers must tolerate missing
fields, null values, empty strings, empty objects and additional fields.

```json
{
  "id": "chatcmpl-example",
  "litellm_call_id": "e6cf53d6-636a-49ab-a45c-b4ae320d73b4",
  "trace_id": "trace-example",
  "session_id": "",
  "call_type": "acompletion",
  "cache_hit": null,
  "stream": false,
  "status": "success",
  "status_fields": {
    "llm_api_status": "success",
    "guardrail_status": "not_run"
  },
  "custom_llm_provider": "openai",
  "saved_cache_cost": 0.0,
  "startTime": 1735689600.0,
  "endTime": 1735689601.2,
  "completionStartTime": 1735689600.8,
  "response_time": 1.2,
  "model": "example-model",
  "metadata": {
    "user_api_key_hash": null,
    "user_api_key_alias": "analytics-app",
    "user_api_key_spend": null,
    "user_api_key_max_budget": null,
    "user_api_key_budget_reset_at": null,
    "user_api_key_user_spend": null,
    "user_api_key_user_max_budget": null,
    "user_api_key_team_spend": null,
    "user_api_key_team_max_budget": null,
    "user_api_key_team_id": "team-example",
    "user_api_key_org_id": null,
    "user_api_key_org_alias": null,
    "user_api_key_project_id": null,
    "user_api_key_project_alias": null,
    "user_api_key_user_id": null,
    "user_api_key_team_alias": null,
    "user_api_key_user_email": null,
    "user_api_key_end_user_id": null,
    "user_api_key_customer_member_id": null,
    "user_api_key_customer_team_id": null,
    "user_api_key_customer_org_id": null,
    "user_api_key_request_route": null,
    "spend_logs_metadata": null,
    "requester_ip_address": null,
    "user_agent": null,
    "requester_metadata": null,
    "prompt_management_metadata": null,
    "applied_guardrails": [],
    "mcp_tool_call_metadata": null,
    "vector_store_request_metadata": null,
    "routing_decision": null,
    "usage_object": {
      "completion_tokens": 4,
      "prompt_tokens": 10,
      "total_tokens": 14,
      "completion_tokens_details": null,
      "prompt_tokens_details": null
    },
    "requester_custom_headers": null,
    "cold_storage_object_key": null,
    "user_api_key_auth_metadata": null,
    "team_alias": null,
    "team_id": null,
    "kubernetes_pod_routing": null,
    "used_client_oauth_token": null
  },
  "cache_key": null,
  "cost_finalized": true,
  "response_cost": 0.018,
  "cost_breakdown": {
    "total_cost": 0.018,
    "service_tier": null,
    "data_residency": null,
    "vertex_location": null,
    "input_cost": 0.01,
    "output_cost": 0.008,
    "tool_usage_cost": 0.0,
    "original_cost": 0.018,
    "discount_percent": 0.0,
    "discount_amount": 0.0,
    "margin_percent": 0.0,
    "margin_fixed_amount": 0.0,
    "margin_total_amount": 0.0
  },
  "autorouter_savings": null,
  "autorouter_savings_estimate": null,
  "autorouter_baseline_observation": null,
  "total_tokens": 14,
  "prompt_tokens": 10,
  "completion_tokens": 4,
  "request_tags": [],
  "request_model_access_groups": [],
  "end_user": null,
  "api_base": "",
  "model_group": "",
  "model_id": "",
  "requester_ip_address": null,
  "user_agent": null,
  "messages": [
    {
      "role": "user",
      "content": "redacted-by-litellm"
    }
  ],
  "response": {
    "id": "chatcmpl-example",
    "created": 1735689601,
    "model": "example-model",
    "object": "chat.completion",
    "choices": [
      {
        "finish_reason": "stop",
        "index": 0,
        "message": {
          "content": "redacted-by-litellm",
          "role": "assistant",
          "tool_calls": null,
          "function_call": null
        }
      }
    ],
    "usage": {
      "completion_tokens": 4,
      "prompt_tokens": 10,
      "total_tokens": 14,
      "completion_tokens_details": null,
      "prompt_tokens_details": null
    }
  },
  "model_parameters": {},
  "hidden_params": {
    "model_id": null,
    "cache_key": null,
    "api_base": "https://provider.example/v1",
    "response_cost": 0.018,
    "additional_headers": {},
    "litellm_overhead_time_ms": null,
    "batch_models": null,
    "batch_successful_requests": null,
    "batch_failed_requests": null,
    "litellm_model_name": "openai/example-model",
    "usage_object": null
  },
  "model_map_information": {
    "model_map_key": null,
    "model_map_value": null
  },
  "error_str": null,
  "error_information": {
    "error_code": "",
    "error_class": "",
    "llm_provider": "",
    "traceback": "",
    "error_message": "",
    "error_provider_request_id": null,
    "error_rate_limit_category": null,
    "error_rate_limit_type": null,
    "error_budget_entity_type": null,
    "error_budget_entity_id": null,
    "error_budget_limit": null,
    "error_budget_spend": null,
    "error_cooldown_deployments": null,
    "normalized_error": null
  },
  "response_cost_failure_debug_info": null,
  "zero_cost_diagnostic": null,
  "guardrail_information": null,
  "standard_built_in_tools_params": {
    "web_search_options": null,
    "file_search": null
  }
}
```

## Fields and missing values

The authoritative definitions are `StandardLoggingPayload`,
`StandardLoggingMetadata` and their nested types in
[`litellm/types/utils.py`](../../types/utils.py). This exporter serializes the
standard event after exclusions and redaction. It does not make every declared
field mandatory, flatten metadata, rename fields or fill missing values.
Non-finite numbers such as NaN are serialized as JSON null.

| Field or group | Meaning and availability |
| --- | --- |
| `id`, `litellm_call_id` | Response and LiteLLM call identifiers. A response ID can be unavailable on failures. The call ID can be null or absent. Preserve both for correlation. |
| `trace_id`, `session_id` | Trace across attempts and application session. Session can be an empty string when not supplied. |
| `call_type`, `stream` | SDK operation and streaming indicator. `stream` can be null if the event did not set it explicitly. |
| `status`, `status_fields` | Overall outcome and separate API/guardrail outcomes. Do not infer success from token counts. |
| `model`, `custom_llm_provider` | Model and provider selected for the call. Provider can be null or unavailable for some routes. |
| `model_group`, `model_id`, `api_base` | Proxy model group, deployment identifier and endpoint. SDK calls can have empty strings or null identifiers without proxy context. |
| `model_map_information` | Model-map lookup key and entry, possibly null. Entries can be large and vary with the installed cost map. |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | Normalized input, output and total usage. Failed or interrupted calls can have absent, zero or incomplete usage. Zero alone does not prove no billable work occurred. |
| `metadata.usage_object` | Detailed usage, possibly null. Provider-specific cached, reasoning, audio and other token details can be absent or null. Do not assume every usage field is additive. |
| `response_cost`, `cost_breakdown`, `cost_finalized` | Computed cost in USD, optional breakdown and optional finalization flag. Cost can be unavailable, estimated or zero when pricing/usage is unavailable. Check finalization and diagnostics before reconciliation. |
| `response_cost_failure_debug_info`, `zero_cost_diagnostic` | Optional pricing diagnostics. Usually null or absent when not applicable. |
| `autorouter_savings`, `autorouter_savings_estimate`, `autorouter_baseline_observation` | Optional routing savings and baseline information. Null when not available. |
| `startTime`, `endTime`, `completionStartTime` | Unix timestamps in seconds, including fractional seconds. Completion timing depends on the route and streaming state. |
| `response_time` | Elapsed response time in seconds. It is not milliseconds. |
| `cache_hit`, `cache_key`, `saved_cache_cost` | Cache result, optional cache identity and estimated avoided cost. Hit/key can be null when no cache context exists. |
| `request_tags`, `request_model_access_groups` | Request labels and optional access groups. They can be empty or absent. |
| `end_user`, `requester_ip_address`, `user_agent` | Caller context, possibly null. These fields can contain personal information. |
| `messages`, `response` | Content shaped by the API surface, possibly null or absent. Chat, Responses and other routes have different response structures. Redaction is described below. |
| `model_parameters` | Model invocation parameters. It can be empty and can contain request content unless redaction is enabled. |
| `hidden_params` | Additional LiteLLM details, such as deployment, endpoint, headers and usage. Values can be null. Treat this as diagnostic data, not a stable warehouse column set. |
| `error_str`, `error_information` | Failure details. Success can have a null error string and an object with empty/null fields, as shown above. Error code, class, provider, message, traceback and rate-limit/budget details depend on the failure. |
| `guardrail_information` | Optional guardrail outcomes, possibly null. Prompt text is redacted when message logging is disabled. |
| `standard_built_in_tools_params` | Optional built-in tool configuration, possibly containing null entries. |
| Classifier audit fields | Additional classifier-specific fields can appear. Existing message redaction removes classifier audit content. |

### Metadata

| Metadata path | Meaning and availability |
| --- | --- |
| `user_api_key_alias`, `user_api_key_hash` | Virtual-key alias and hash. The field is not named `key_alias`, and the hash is not the raw API key. Either can be null without virtual-key context. |
| `user_api_key_team_id`, `user_api_key_team_alias` | Team attribution, possibly null. |
| `user_api_key_org_id`, `user_api_key_org_alias` | Organization attribution, possibly null. |
| `user_api_key_project_id`, `user_api_key_project_alias` | Project attribution, possibly null or absent. |
| `user_api_key_user_id`, `user_api_key_user_email`, `user_api_key_end_user_id` | User attribution, possibly null. Email and other identities are not removed by message redaction. |
| `user_api_key_app_id`, `user_api_key_app_alias` | Optional application attribution. These can be absent. |
| `user_api_key_customer_*` | Optional hierarchy identifiers, aliases, scopes and hierarchy data. They can be absent or null. |
| `user_api_key_spend`, `user_api_key_max_budget`, `user_api_key_budget_reset_at` | Available key spend and budget context. Can be null. Treat as captured context rather than an atomic ledger balance. |
| `user_api_key_user_spend`, `user_api_key_user_max_budget`, `user_api_key_team_spend`, `user_api_key_team_max_budget` | Optional user/team budget context, possibly null. |
| `user_api_key_request_route`, `user_api_key_auth_metadata` | Optional request route and authentication metadata. Custom values may be sensitive. |
| `spend_logs_metadata`, `requester_metadata`, `requester_custom_headers` | Optional custom metadata and headers. Their keys are application-dependent and message redaction does not sanitize arbitrary values. |
| `requester_ip_address`, `user_agent`, `team_id`, `team_alias` | Additional caller/team context, possibly null. Availability differs across request paths. |
| `prompt_management_metadata`, `mcp_tool_call_metadata`, `vector_store_request_metadata` | Optional prompt, MCP and retrieval details, null when unused. |
| `routing_decision`, `kubernetes_pod_routing` | Optional routing and pod information, null when unavailable. |
| `applied_guardrails` | Guardrail names, often an empty array when none apply. |
| `cold_storage_object_key` | Optional object-storage reference, null when unused. |
| `used_client_oauth_token`, `is_silent_experiment` | Optional request context flags. The OAuth flag is not a token and does not enable Kafka OAuth authentication. |

## Message redaction

`kafka_params.turn_off_message_logging: true` redacts the Kafka copy. It does not
turn off PostgreSQL prompt storage or modify the shared event for other loggers.
Global `litellm.turn_off_message_logging` and supported per-request redaction also
apply. Kafka does not opt out of a global or per-request redaction decision.
Message logging remains enabled unless one of these settings disables it.

The existing helper replaces prompt content with `redacted-by-litellm`, redacts
response content and tool/function arguments in supported API shapes, redacts
request content in model parameters, and removes classifier audit content and
guardrail prompt text. Redaction usually retains the surrounding response shape,
usage and identifiers. It does not necessarily remove `messages` or `response`.

This is not a general personal-data or secret scrubber. Key aliases and hashes,
user/team identities, email addresses, IP addresses, endpoint URLs, custom
metadata, headers and error messages are not all erased by message redaction.
Review those values before exporting them. Do not put secrets in custom metadata.

## Removing fields and configuration scope

```yaml
litellm_settings:
  callbacks: ["kafka"]
  standard_logging_payload_excluded_fields:
    - messages
    - response
    - model_parameters
    - hidden_params
    - model_map_information
  kafka_params:
    bootstrap_servers: os.environ/KAFKA_BOOTSTRAP_SERVERS
    topic: litellm_events
    security_protocol: SSL
    turn_off_message_logging: true
```

`standard_logging_payload_excluded_fields` is a global LiteLLM setting. It removes
matching top-level fields for loggers using the shared standard-payload exclusion
helper, including Kafka. It is not a Kafka-only projection and should not be
used to promise identical behavior from every integration or from PostgreSQL
persistence. PostgreSQL spend-log settings and budget/accounting updates are
separate. Kafka's copy does not mutate the shared payload, but other loggers
that apply the same global setting will also exclude these fields.

The names are exact top-level keys, not dotted paths. Listing
`metadata.user_api_key_user_email` does not remove the nested email field.
Listing `metadata` removes the entire metadata object, including key alias,
team attribution and detailed usage. No Kafka-specific allowlist, nested-field
projection or rename configuration is provided. Exclusions happen before JSON
serialization. Removing `litellm_call_id` can change the Kafka key to the fallback
`id`, and removing both identifiers leaves the Kafka key unset.

Adding arbitrary request metadata does not guarantee it appears as a new
top-level event field. Use the existing supported metadata paths, or update the
shared standard-event producer when introducing a new standard field. Prefer
warehouse projections for analysis column names instead of changing the wire
format for individual cloud platforms.
