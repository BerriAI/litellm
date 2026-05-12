"""Shared util for emitting v3 parallel_request_limiter rate limit headers
in `x-ratelimit-{descriptor}-{remaining,limit}-{type}` form.

`parallel_request_limiter_v3.async_post_call_success_hook` populates these
headers via `response._hidden_params.additional_headers`, but two paths
silently drop them:

  1. Streaming responses — the post-call hook fires *after* SSE chunk
     transmission has already started, so headers set on the response
     object never reach the client.
  2. Plain-dict responses (e.g. `/v1/messages` non-streaming) — the hook
     short-circuits when `hasattr(response, "_hidden_params")` is False.

`common_request_processing.py` calls this util at two seed sites
(pre-stream branch and the non-streaming response builder) so the
rate-limit response cached on `self.data` is reflected in the outgoing
response headers regardless of which code path produced the response.
"""

from typing import Optional


def apply_rate_limit_statuses_to_headers(
    headers: dict,
    rate_limit_response: Optional[dict],
) -> None:
    """Populate x-ratelimit-* entries on `headers` from a v3 limiter response.

    Uses setdefault so existing keys (e.g. set by an earlier hook on the same
    request) are preserved. No-op when rate_limit_response is None or empty.
    """
    if not rate_limit_response:
        return
    for status in rate_limit_response.get("statuses", []):
        descriptor_key = status.get("descriptor_key")
        rate_limit_type = status.get("rate_limit_type")
        if descriptor_key is None or rate_limit_type is None:
            continue
        prefix = f"x-ratelimit-{descriptor_key}"
        headers.setdefault(
            f"{prefix}-remaining-{rate_limit_type}",
            status.get("limit_remaining"),
        )
        headers.setdefault(
            f"{prefix}-limit-{rate_limit_type}",
            status.get("current_limit"),
        )
