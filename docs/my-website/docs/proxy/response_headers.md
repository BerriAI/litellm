# Rate Limit Headers

When you make a request to the proxy, the proxy will return the following [OpenAI-compatible headers](https://platform.openai.com/docs/guides/rate-limits/rate-limits-in-headers):

- `x-ratelimit-remaining-requests` - Optional[int]: The remaining number of requests that are permitted before exhausting the rate limit.
- `x-ratelimit-remaining-tokens` - Optional[int]: The remaining number of tokens that are permitted before exhausting the rate limit.
- `x-ratelimit-limit-requests` - Optional[int]: The maximum number of requests that are permitted before exhausting the rate limit.
- `x-ratelimit-limit-tokens` - Optional[int]: The maximum number of tokens that are permitted before exhausting the rate limit.
- `x-ratelimit-reset-requests` - Optional[int]: The time at which the rate limit will reset.    
- `x-ratelimit-reset-tokens` - Optional[int]: The time at which the rate limit will reset.

These headers are useful for clients to understand the current rate limit status and adjust their request rate accordingly.

## How are these headers calculated?

**If key has rate limits set**

The proxy will return the [remaining rate limits for that key](https://github.com/BerriAI/litellm/blob/bfa95538190575f7f317db2d9598fc9a82275492/litellm/proxy/hooks/parallel_request_limiter.py#L778).

**If key does not have rate limits set**

The proxy returns the remaining requests/tokens returned by the backend provider. 

If the backend provider does not return these headers, the value will be `None`.
