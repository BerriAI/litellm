from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from litellm.types.utils import TokenCountResponse


class CountFailure(StrEnum):
    NO_RESPONSE = "no_response"
    PROVIDER_ERROR = "provider_error"
    WRONG_SOURCE = "wrong_source"


@dataclass(frozen=True, slots=True)
class CountedPrefixTokens:
    tokens: int


PrefixCount: TypeAlias = CountedPrefixTokens | CountFailure


class EligibilityUnassessedReason(StrEnum):
    MINIMUM_UNKNOWN = "minimum_unknown"
    COUNT_FAILED = "count_failed"


@dataclass(frozen=True, slots=True)
class Eligible:
    proven_by_provider_activity: bool


@dataclass(frozen=True, slots=True)
class AwaitingProviderCount:
    minimum: int


@dataclass(frozen=True, slots=True)
class BelowMinimum:
    counted: int
    minimum: int


@dataclass(frozen=True, slots=True)
class EligibilityUnassessed:
    reason: EligibilityUnassessedReason


Eligibility: TypeAlias = Eligible | AwaitingProviderCount | BelowMinimum | EligibilityUnassessed


def accepted_prefix_count(response: TokenCountResponse | None, expected_tokenizer_type: str) -> PrefixCount:
    """Only a count the deployment's own provider API returned qualifies. A counter that fell back to
    another endpoint (Bedrock Runtime to Mantle) or to a local estimate reports a different tokenizer."""
    if response is None:
        return CountFailure.NO_RESPONSE
    if response.error or response.total_tokens <= 0:
        return CountFailure.PROVIDER_ERROR
    if response.tokenizer_type != expected_tokenizer_type or response.count_source not in (None, "provider"):
        return CountFailure.WRONG_SOURCE
    return CountedPrefixTokens(tokens=response.total_tokens)


def assess_eligibility(minimum: int | None, provider_activity_seen: bool, count: PrefixCount | None) -> Eligibility:
    """A cache read or write on this prefix and configuration is the provider's own proof that the prefix
    meets its minimum, so it needs no count. Otherwise only a provider count against the model's exact
    minimum decides, and a generic default minimum never stands in for an unknown one."""
    if provider_activity_seen:
        return Eligible(proven_by_provider_activity=True)
    if minimum is None:
        return EligibilityUnassessed(EligibilityUnassessedReason.MINIMUM_UNKNOWN)
    match count:
        case None:
            return AwaitingProviderCount(minimum=minimum)
        case CountFailure():
            return EligibilityUnassessed(EligibilityUnassessedReason.COUNT_FAILED)
        case CountedPrefixTokens(tokens=tokens):
            return Eligible(proven_by_provider_activity=False) if tokens >= minimum else BelowMinimum(tokens, minimum)
