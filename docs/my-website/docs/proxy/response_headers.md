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
| `x-litellm-response-duration-ms` | float | Total duration of the API response in milliseconds |
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
| `x-litellm-call-id` | string | Unique identifier for the API call | ✅ |
| `x-litellm-model-id` | string | Unique identifier for the model used | |
| `x-litellm-model-api-base` | string | Base URL of the API endpoint | ✅ |
| `x-litellm-version` | string | Version of LiteLLM being used | |
| `x-litellm-model-group` | string | Model group identifier | |

## Response headers from LLM providers

LiteLLM also returns the original response headers from the LLM provider. These headers are prefixed with `llm_provider-` to distinguish them from LiteLLM's headers.

Example response headers:
```
llm_provider-openai-processing-ms: 256
llm_provider-openai-version: 2020-10-01
llm_provider-x-ratelimit-limit-requests: 30000
llm_provider-x-ratelimit-limit-tokens: 150000000
```

