from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from .types import CacheablePrefix, CacheCounters


class EligibilityUnassessedReason(StrEnum):
    MINIMUM_UNKNOWN = "minimum_unknown"
    PREFIX_LENGTH_UNPROVEN = "prefix_length_unproven"


@dataclass(frozen=True, slots=True)
class Eligible:
    pass


@dataclass(frozen=True, slots=True)
class EligibilityUnassessed:
    reason: EligibilityUnassessedReason


Eligibility: TypeAlias = Eligible | EligibilityUnassessed


def write_proves_prefix_eligible(prefix: CacheablePrefix, counters: CacheCounters) -> bool:
    """Only a cache write on a request with a single breakpoint proves this prefix met the minimum. With
    several breakpoints the counters are a total over all of them, so an earlier, shorter prefix can account
    for every token, and a read can be served from an earlier entry without this prefix ever being cached."""
    return prefix.breakpoints == 1 and counters.write_tokens > 0


def assess_eligibility(minimum: int | None, provider_proof_seen: bool) -> Eligibility:
    """Only the provider's own cache write, as `write_proves_prefix_eligible` accepts it, proves the prefix met
    the model's minimum. A token count total is not the cacheable prefix length, since it also counts the
    request framing around the prefix, so no count stands in for that proof."""
    if provider_proof_seen:
        return Eligible()
    if minimum is None:
        return EligibilityUnassessed(EligibilityUnassessedReason.MINIMUM_UNKNOWN)
    return EligibilityUnassessed(EligibilityUnassessedReason.PREFIX_LENGTH_UNPROVEN)
