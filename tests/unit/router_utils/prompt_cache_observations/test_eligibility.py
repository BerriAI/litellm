import pytest

from litellm.router_utils.prompt_cache_observations.eligibility import (
    AwaitingProviderCount,
    BelowMinimum,
    CountedPrefixTokens,
    CountFailure,
    Eligibility,
    EligibilityUnassessed,
    EligibilityUnassessedReason,
    Eligible,
    PrefixCount,
    accepted_prefix_count,
    assess_eligibility,
    write_proves_prefix_eligible,
)
from litellm.router_utils.prompt_cache_observations.types import CacheablePrefix, CacheCounters
from litellm.types.token_count import TokenCountSource
from litellm.types.utils import TokenCountResponse


def _count_response(
    total_tokens: int = 2048,
    tokenizer_type: str = "anthropic_api",
    error: bool = False,
    count_source: TokenCountSource | None = None,
) -> TokenCountResponse:
    return TokenCountResponse(
        total_tokens=total_tokens,
        request_model="claude-test",
        model_used="claude-test",
        tokenizer_type=tokenizer_type,
        error=error,
        count_source=count_source,
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (_count_response(), CountedPrefixTokens(tokens=2048)),
        (_count_response(count_source="provider"), CountedPrefixTokens(tokens=2048)),
        (None, CountFailure.NO_RESPONSE),
        (_count_response(error=True), CountFailure.PROVIDER_ERROR),
        (_count_response(total_tokens=0), CountFailure.PROVIDER_ERROR),
        (_count_response(tokenizer_type="bedrock_mantle_api"), CountFailure.WRONG_SOURCE),
        (_count_response(count_source="local_estimate"), CountFailure.WRONG_SOURCE),
    ],
)
def test_only_a_count_from_the_deployments_own_provider_api_is_accepted(
    response: TokenCountResponse | None, expected: PrefixCount
) -> None:
    assert accepted_prefix_count(response, "anthropic_api") == expected


@pytest.mark.parametrize(
    ("minimum", "activity_seen", "count", "expected"),
    [
        (None, True, None, Eligible(proven_by_provider_activity=True)),
        (1024, True, CountedPrefixTokens(tokens=10), Eligible(proven_by_provider_activity=True)),
        (
            None,
            False,
            CountedPrefixTokens(tokens=4096),
            EligibilityUnassessed(EligibilityUnassessedReason.MINIMUM_UNKNOWN),
        ),
        (1024, False, None, AwaitingProviderCount(minimum=1024)),
        (1024, False, CountedPrefixTokens(tokens=1024), Eligible(proven_by_provider_activity=False)),
        (1024, False, CountedPrefixTokens(tokens=1023), BelowMinimum(counted=1023, minimum=1024)),
        (1024, False, CountFailure.WRONG_SOURCE, EligibilityUnassessed(EligibilityUnassessedReason.COUNT_FAILED)),
    ],
)
def test_eligibility_needs_provider_evidence_against_the_exact_minimum(
    minimum: int | None, activity_seen: bool, count: PrefixCount | None, expected: Eligibility
) -> None:
    assert assess_eligibility(minimum, activity_seen, count) == expected


@pytest.mark.parametrize(
    ("breakpoints", "counters", "proves"),
    [
        (1, CacheCounters(read_tokens=0, write_tokens=2048), True),
        (1, CacheCounters(read_tokens=2048, write_tokens=0), False),
        (2, CacheCounters(read_tokens=0, write_tokens=2048), False),
        (3, CacheCounters(read_tokens=4096, write_tokens=10), False),
        (1, CacheCounters(read_tokens=0, write_tokens=0), False),
    ],
)
def test_only_a_write_on_a_single_breakpoint_proves_the_prefix_met_the_minimum(
    breakpoints: int, counters: CacheCounters, proves: bool
) -> None:
    prefix = CacheablePrefix(fingerprint="prefix", ttl_seconds=300, breakpoints=breakpoints)
    assert write_proves_prefix_eligible(prefix, counters) is proves
