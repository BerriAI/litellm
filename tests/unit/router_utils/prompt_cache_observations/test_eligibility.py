import pytest

from litellm.router_utils.prompt_cache_observations.eligibility import write_proves_prefix_eligible
from litellm.router_utils.prompt_cache_observations.types import CacheablePrefix, CacheCounters


def _proof(breakpoints: int, counters: CacheCounters) -> bool:
    return write_proves_prefix_eligible(
        CacheablePrefix(fingerprint="prefix", ttl_seconds=300, breakpoints=breakpoints), counters
    )


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
    assert _proof(breakpoints, counters) is proves
