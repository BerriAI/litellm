# Kafka message format

Each record is one UTF-8 JSON object containing the existing
`StandardLoggingPayload`, without a wrapper, batch array or schema header.
The Kafka key is `litellm_call_id`, falling back to `id`. It is unset if neither
is a string. Topic, partition, offset and broker timestamp remain Kafka metadata.

The authoritative schema is `StandardLoggingPayload` and its nested types in
[`litellm/types/utils.py`](../../types/utils.py). Consumers must tolerate missing,
null and additional fields. Non-finite numbers are serialized as JSON null.

## JSON example

This synthetic event shows a successful request with message logging disabled.
It includes a selection of fields from the standard payload. Other fields can
be present depending on the API, provider and logging configuration.

```json
{
  "id": "chatcmpl-example",
  "litellm_call_id": "call-example",
  "status": "success",
  "model": "example-model",
  "custom_llm_provider": "openai",
  "prompt_tokens": 12,
  "completion_tokens": 8,
  "total_tokens": 20,
  "response_cost": 0.00002,
  "startTime": 1735689600.0,
  "endTime": 1735689600.5,
  "response_time": 0.5,
  "messages": [{"role": "user", "content": "redacted-by-litellm"}],
  "metadata": {
    "user_api_key_alias": "analytics-service",
    "user_api_key_team_id": null,
    "usage_object": {
      "prompt_tokens": 12,
      "completion_tokens": 8,
      "total_tokens": 20
    }
  }
}
```

The cost is illustrative. Redaction can preserve message and response wrappers
with their content replaced, so consumers must accept the supported API shapes.
An absent field, an explicit `null` and an empty value can have different meanings.

## Selected fields

| Event path | Meaning |
| --- | --- |
| `id`, `litellm_call_id` | Response and request correlation identifiers |
| `status`, `error_information` | Outcome and available failure details |
| `model`, `custom_llm_provider` | Model and provider |
| `metadata.user_api_key_alias`, `metadata.user_api_key_hash` | Virtual-key attribution, not the raw API key |
| `metadata.user_api_key_team_id` | Team attribution |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | Normalized usage |
| `metadata.usage_object` | Available provider-specific usage details |
| `response_cost`, `cost_finalized` | Computed cost and available finalization status |
| `startTime`, `endTime`, `completionStartTime` | Unix timestamps in seconds |
| `response_time` | Duration in seconds |
| `messages`, `response` | Content, shaped by the API surface and redaction policy |

SDK calls without proxy virtual-key context can lack alias and team fields.
Failed or interrupted calls can have incomplete usage or cost. A zero or
missing value does not establish that no billable work occurred. Streaming
exports a final event rather than one record per chunk.

## Redaction and exclusions

`kafka_params.turn_off_message_logging: true` redacts the Kafka copy using
LiteLLM's existing message-redaction helpers. Global and supported per-request
redaction also apply. The Kafka copy does not mutate another destination's
event or change PostgreSQL prompt storage.

Message redaction covers supported prompt/response shapes and tool arguments,
and omits the response object, model parameters, free-text error messages and tracebacks. Response log probabilities can reveal answer tokens. Model parameters can contain rerank documents, queries or prediction content. It does not cover arbitrary metadata, headers or identifiers. Review custom
metadata before exporting it and do not store secrets there.

`standard_logging_payload_excluded_fields` removes exact top-level keys for
destinations using the shared standard-payload helper. It is a global setting,
not a Kafka-only projection. For example, excluding `metadata` also removes key
alias and team attribution. Nested paths such as
`metadata.user_api_key_user_email` are unsupported. Removing both correlation
identifiers also removes the Kafka key. Prefer downstream projections for renaming or selecting analysis columns.
