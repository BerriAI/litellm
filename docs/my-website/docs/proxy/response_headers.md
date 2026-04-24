# Response Headers

When you make a request to the proxy, the proxy will return the following headers:

## Rate Limit Headers
[OpenAI-compatible headers](https://platform.openai.com/docs/guides/rate-limits/rate-limits-in-headers):

| Header | Type | Description |
|--------|------|-------------|
| `x-ratelimit-remaining-requests` | Optional[int] | The remaining number of requests that are permitted before exhausting the rate limit |
| `x-ratelimit-remaining-tokens` | Optional[int] | The remaining number of tokens that are permitted before exhausting the rate limit |
| `x-ratelimit-limit-requests` | Optional[int] | The maximum number of requests that are permitted before exhausting the rate limit |
| `x-ratelimit-limit-tokens` | Optional[int] | The maximum number of tokens that are permitted before exhausting the rate limit |
| `x-ratelimit-reset-requests` | Optional[int] | The time at which the rate limit will reset |
| `x-ratelimit-reset-tokens` | Optional[int] | The time at which the rate limit will reset |

### How Rate Limit Headers work

**If key has rate limits set**

The proxy will return the [remaining rate limits for that key](https://github.com/BerriAI/litellm/blob/bfa95538190575f7f317db2d9598fc9a82275492/litellm/proxy/hooks/parallel_request_limiter.py#L778).

**If key does not have rate limits set**

The proxy returns the remaining requests/tokens returned by the backend provider. (LiteLLM will standardize the backend provider's response headers to match the OpenAI format)

If the backend provider does not return these headers, the value will be `None`.

These headers are useful for clients to understand the current rate limit status and adjust their request rate accordingly.


## Latency Headers
| Header | Type | Description |
|--------|------|-------------|
| `x-litellm-response-duration-ms` | float | Total duration from the moment that a request gets to LiteLLM Proxy to the moment it gets returned to the client. |
| `x-litellm-overhead-duration-ms` | float | LiteLLM processing overhead in milliseconds |

## Retry, Fallback Headers
| Header | Type | Description |
|--------|------|-------------|
| `x-litellm-attempted-retries` | int | Number of retry attempts made |
| `x-litellm-attempted-fallbacks` | int | Number of fallback attempts made |
| `x-litellm-max-fallbacks` | int | Maximum number of fallback attempts allowed |

## Cost Tracking Headers
| Header | Type | Description | Available on Pass-Through Endpoints |
|--------|------|-------------|-------------|
| `x-litellm-response-cost` | float | Cost of the API call | |
| `x-litellm-key-spend` | float | Total spend for the API key | ✅ |

## LiteLLM Specific Headers
| Header | Type | Description | Available on Pass-Through Endpoints |
|--------|------|-------------|-------------|
| `x-litellm-call-id` | string | Id for this request | ✅ |
| `x-litellm-model-id` | string | Deployment id (`model_info.id`) | |
| `x-litellm-model-api-base` | string | API base URL | ✅ |
| `x-litellm-version` | string | LiteLLM version | |
| `x-litellm-model-group` | string | Routed `model_list[].model_name` (client `model`) | |

### Example

```yaml
model_list:
  - model_name: my-chat-model          # clients call this
    litellm_params:
      model: gpt-4o-mini               # LiteLLM calls this upstream
    model_info:
      id: "7c9f2a1b3d8e4f0a2c6b5d9e1f3a7b8c"   # optional; auto-generated if omitted
```

| Header | Example | Notes |
|--------|---------|-------|
| `x-litellm-model-group` | `my-chat-model` | `model_name` / request `model`; not `litellm_params.model`. |
| `x-litellm-model-id` | `7c9f2a1b3d8e4f0a2c6b5d9e1f3a7b8c` | Which deployment row; use with `/v1/model/info?litellm_model_id=...`. |
| Response body `model` | often `my-chat-model` | Often restamped to match the client; upstream id stays in config. |

### More examples (illustrative)

| Header | Example | Meaning |
|--------|---------|---------|
| `x-litellm-response-cost` | `0.000214` | This call (USD). |
| `x-litellm-key-spend` | `12.847` | Key total after this call. |
| `x-litellm-response-duration-ms` | `842.3` | Proxy end-to-end (ms). |
| `x-litellm-overhead-duration-ms` | `15.1` | LiteLLM overhead (ms). |
| `x-litellm-attempted-retries` | `0` | Retries. |
| `x-litellm-attempted-fallbacks` | `1` | Fallbacks to another deployment. |
| `x-litellm-call-id` | `019b2c4d-e5f6-7890-abcd-ef1234567890` | Logs / tracing. |
| `x-litellm-version` | `1.55.3` | Version. |
| `x-litellm-model-api-base` | `https://api.openai.com/v1` | Provider base (no query string). |

## Response headers from LLM providers

LiteLLM also returns the original response headers from the LLM provider. These headers are prefixed with `llm_provider-` to distinguish them from LiteLLM's headers.

Example response headers:
```
llm_provider-openai-processing-ms: 256
llm_provider-openai-version: 2020-10-01
llm_provider-x-ratelimit-limit-requests: 30000
llm_provider-x-ratelimit-limit-tokens: 150000000
```

