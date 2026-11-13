import ast
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.redis.stores.connection import connect
from litellm.redis.stores.deployment_health import DeploymentHealthStateValue


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


_LEFT_RECORDS: Final = TypeAdapter(dict[str, object])


@pytest.mark.parametrize(
    ("written", "read"),
    [
        pytest.param(
            "{'dep-a': {'is_healthy': False, 'timestamp': 1.5, 'reason': 'timeout', 'region': 'eu'}}",
            {"dep-a": {"is_healthy": False, "timestamp": 1.5, "reason": "timeout"}},
            id="extra_fields_are_ignored",
        ),
        pytest.param(
            "{'dep-a': {'is_healthy': 0, 'timestamp': 95, 'reason': 'timeout'}}",
            {"dep-a": {"is_healthy": False, "timestamp": 95, "reason": "timeout"}},
            id="a_falsy_flag_is_unhealthy",
        ),
        pytest.param(
            "{'dep-a': {'timestamp': 1.5}}",
            {"dep-a": {"is_healthy": True, "timestamp": 1.5, "reason": ""}},
            id="a_missing_flag_is_healthy",
        ),
        pytest.param(
            "{'dep-a': {'is_healthy': False, 'timestamp': 'soon', 'reason': 'timeout'}}",
            {"dep-a": {"is_healthy": False, "timestamp": 0.0, "reason": "timeout"}},
            id="a_timestamp_that_is_no_number_is_stale",
        ),
        pytest.param(
            "{'dep-a': {'is_healthy': False, 'timestamp': True, 'reason': 'timeout'}}",
            {"dep-a": {"is_healthy": False, "timestamp": 0.0, "reason": "timeout"}},
            id="a_bool_timestamp_is_no_number",
        ),
        pytest.param(
            "{'dep-a': 'not a record', 'dep-b': None, 'dep-c': {'is_healthy': False, 'timestamp': 1.5, "
            "'reason': 'timeout'}}",
            {"dep-c": {"is_healthy": False, "timestamp": 1.5, "reason": "timeout"}},
            id="what_is_no_record_is_dropped_alone",
        ),
    ],
)
def test_a_record_another_writer_left_is_read_the_way_mains_readers_read_it(
    written: str, read: Mapping[str, DeploymentHealthStateValue]
) -> None:
    """The same cases ``litellm/rust/crates/redis-store/tests/deployment_health.rs`` reads from real Redis, decoded
    here from what main's ``ast.literal_eval`` reads back."""
    store: Final = connect(None).deployment_health()

    store.set_sync(_LEFT_RECORDS.validate_python(ast.literal_eval(written)), 90.0)

    assert store.get_sync() == read
