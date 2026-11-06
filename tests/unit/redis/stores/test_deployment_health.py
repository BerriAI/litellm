import ast
from dataclasses import dataclass
from typing import Final

from litellm.router_utils.health_state_cache import DeploymentHealthStateValue


@dataclass(frozen=True, slots=True)
class _WireStates:
    """The health states as main's sync ``RedisCache.set_cache`` writes them (``str(value)``) and
    ``RedisCache.get_cache`` reads them back (``json.loads``, then ``ast.literal_eval``). The same literals sit in
    ``litellm/rust/crates/redis-store/tests/deployment_health.rs``, where real Redis is checked to hold them
    after the Rust store writes."""

    states: dict[str, DeploymentHealthStateValue]
    str_dict: str
    ttl_seconds: float
    redis_ttl_seconds: int


_WIRE_STATES: Final = _WireStates(
    states={"dep-a": {"is_healthy": False, "timestamp": 1.5, "reason": "timeout"}},
    str_dict="{'dep-a': {'is_healthy': False, 'timestamp': 1.5, 'reason': 'timeout'}}",
    ttl_seconds=90.0,
    redis_ttl_seconds=90,
)


def test_the_wire_case_is_what_mains_codec_writes_and_reads() -> None:
    assert str(_WIRE_STATES.states) == _WIRE_STATES.str_dict
    assert ast.literal_eval(_WIRE_STATES.str_dict) == _WIRE_STATES.states
    assert int(_WIRE_STATES.ttl_seconds) == _WIRE_STATES.redis_ttl_seconds
