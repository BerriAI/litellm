"""Upstream-reported cost and usage for pass-through endpoints.

Some pass-through targets invoke several models internally, so LiteLLM cannot
price the request from the response body. Those targets report the totals for
the whole HTTP request in ``x-litellm-*`` response headers instead; LiteLLM
records the reported values without recomputing them.
"""

import math
from dataclasses import dataclass
from typing import Final

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.utils import Usage

UPSTREAM_RESPONSE_COST_HEADER: Final = "x-litellm-response-cost"
UPSTREAM_TOTAL_TOKENS_HEADER: Final = "x-litellm-total-tokens"

# model_call_details key holding what the upstream reported, so later stages of
# the success path can tell an upstream-reported cost apart from one LiteLLM
# derived itself.
UPSTREAM_REPORTED_USAGE_KEY: Final = "_litellm_upstream_reported_usage"


@dataclass(frozen=True, slots=True)
class UpstreamReportedUsage:
    """Totals an upstream pass-through target reported for one HTTP request.

    ``None`` means the upstream did not report a usable value, either because
    the header was absent or because it could not be parsed.
    """

    response_cost: float | None
    total_tokens: int | None


def _parse_response_cost(raw_value: str | None) -> float | None:
    if raw_value is None:
        verbose_proxy_logger.warning(
            "pass_through_endpoint: upstream did not send %s; recording 0 cost for this request",
            UPSTREAM_RESPONSE_COST_HEADER,
        )
        return None
    try:
        response_cost: Final = float(raw_value)
    except ValueError:
        verbose_proxy_logger.warning(
            "pass_through_endpoint: upstream sent unparseable %s=%r; recording 0 cost for this request",
            UPSTREAM_RESPONSE_COST_HEADER,
            raw_value,
        )
        return None
    if not math.isfinite(response_cost) or response_cost < 0:
        verbose_proxy_logger.warning(
            "pass_through_endpoint: upstream sent out-of-range %s=%r; recording 0 cost for this request",
            UPSTREAM_RESPONSE_COST_HEADER,
            raw_value,
        )
        return None
    return response_cost


def _parse_total_tokens(raw_value: str | None) -> int | None:
    if raw_value is None:
        verbose_proxy_logger.warning(
            "pass_through_endpoint: upstream did not send %s; recording 0 tokens for this request",
            UPSTREAM_TOTAL_TOKENS_HEADER,
        )
        return None
    try:
        total_tokens: Final = int(raw_value)
    except ValueError:
        verbose_proxy_logger.warning(
            "pass_through_endpoint: upstream sent unparseable %s=%r; recording 0 tokens for this request",
            UPSTREAM_TOTAL_TOKENS_HEADER,
            raw_value,
        )
        return None
    if total_tokens < 0:
        verbose_proxy_logger.warning(
            "pass_through_endpoint: upstream sent negative %s=%r; recording 0 tokens for this request",
            UPSTREAM_TOTAL_TOKENS_HEADER,
            raw_value,
        )
        return None
    return total_tokens


def parse_upstream_reported_usage(headers: httpx.Headers) -> UpstreamReportedUsage | None:
    """Read the reported totals off an upstream pass-through response.

    Returns ``None`` when neither header is present, which is the normal case
    for a target that does not speak this contract (e.g. Anthropic or Vertex,
    whose cost LiteLLM derives from the response body instead).
    """
    raw_response_cost: Final = headers.get(UPSTREAM_RESPONSE_COST_HEADER)
    raw_total_tokens: Final = headers.get(UPSTREAM_TOTAL_TOKENS_HEADER)
    if raw_response_cost is None and raw_total_tokens is None:
        return None
    return UpstreamReportedUsage(
        response_cost=_parse_response_cost(raw_response_cost),
        total_tokens=_parse_total_tokens(raw_total_tokens),
    )


def apply_upstream_reported_usage(
    logging_obj: LiteLLMLoggingObj,
    headers: httpx.Headers,
) -> UpstreamReportedUsage | None:
    """Record the upstream's reported totals on the request's logging object.

    Only the values the upstream actually reported are written, so a target
    that reports cost but not tokens keeps the token count LiteLLM derived on
    its own rather than having it zeroed.
    """
    reported: Final = parse_upstream_reported_usage(headers)
    if reported is None:
        return None
    logging_obj.model_call_details[UPSTREAM_REPORTED_USAGE_KEY] = reported
    if reported.response_cost is not None:
        logging_obj.model_call_details["response_cost"] = reported.response_cost
    if reported.total_tokens is not None:
        logging_obj.model_call_details["combined_usage_object"] = Usage(total_tokens=reported.total_tokens)
    return reported


def has_upstream_reported_usage(logging_obj: LiteLLMLoggingObj) -> bool:
    """Whether the upstream spoke this contract on the request's response.

    True even when the value it sent was unusable: a target that reports its
    own totals owns the cost for the request, and a header we could not parse
    means zero, never a fallback to someone else's estimate.
    """
    return isinstance(logging_obj.model_call_details.get(UPSTREAM_REPORTED_USAGE_KEY), UpstreamReportedUsage)
