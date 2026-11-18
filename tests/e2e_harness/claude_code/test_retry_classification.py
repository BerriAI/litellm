"""Unit tests for the retry-shape classification in `cli_driver`.

Markerless harness tests: they exercise driver plumbing over hand-built
outcomes, not a product feature, so they run without a proxy and carry no
`e2e` marker.

The pairing that matters is that a saturated upstream is retryable but is not
rate-limit-shaped. litellm-e2e-pr build 182 failed a green cell on a Bedrock
503 that no pattern matched, while feeding a 503 to the rate-limit summary
would tell the rate-limiter's binary search to lower a request rate that was
never the problem.
"""

from __future__ import annotations

import pytest

from claude_code.cli_driver import (
    ClaudeCLIError,
    DriverResult,
    is_rate_limit_shaped,
    is_retryable_shaped,
    is_transient_upstream_shaped,
)

_BEDROCK_503 = (
    "[claude-opus-4-7-bedrock-converse] tool_search probe failed: status 503: "
    '{"error":{"message":"litellm.ServiceUnavailableError: BedrockException - '
    '{\\"message\\":\\"Bedrock is unable to process your request.\\"}"}}'
)
_ANTHROPIC_529 = "status 529: {\"type\":\"overloaded_error\"}"
_OPENAI_429 = 'status 429: {"error":{"message":"Rate limit reached"}}'


def _failed(text: str) -> DriverResult:
    return DriverResult(text=text, exit_code=1)


@pytest.mark.parametrize(
    "text, rate_limit, transient",
    [
        (_BEDROCK_503, False, True),
        (_ANTHROPIC_529, False, True),
        ("status 503 service unavailable", False, True),
        ("upstream overloaded, try again later", False, True),
        (_OPENAI_429, True, False),
        ("throttling exception from provider", True, False),
        ("claude CLI timed out after 120s", True, False),
        ('status 400: {"error":"bad request"}', False, False),
    ],
)
def test_shapes_are_classified_independently(text: str, rate_limit: bool, transient: bool) -> None:
    outcome = _failed(text)
    assert is_rate_limit_shaped(outcome) is rate_limit
    assert is_transient_upstream_shaped(outcome) is transient
    assert is_retryable_shaped(outcome) is (rate_limit or transient)


def test_bedrock_503_is_retryable_but_not_rate_limit_shaped() -> None:
    outcome = _failed(_BEDROCK_503)
    assert is_retryable_shaped(outcome)
    assert not is_rate_limit_shaped(outcome)


def test_passing_outcome_is_never_retryable() -> None:
    passed = DriverResult(text=_BEDROCK_503, exit_code=0)
    assert not is_retryable_shaped(passed)
    assert not is_transient_upstream_shaped(passed)


def test_driver_error_message_is_classified() -> None:
    assert is_transient_upstream_shaped(ClaudeCLIError("upstream returned 503"))
    assert is_rate_limit_shaped(ClaudeCLIError("claude CLI timed out"))
    assert not is_retryable_shaped(ClaudeCLIError("binary not found"))
