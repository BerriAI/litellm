from datetime import timedelta

import pytest

from litellm.redis.transport.contract import ttl_ms


@pytest.mark.parametrize(
    ("ttl", "expected"),
    [
        pytest.param(timedelta(seconds=1.5), 1500, id="whole-ms"),
        pytest.param(timedelta(microseconds=1500), 2, id="rounds-up"),
        pytest.param(timedelta(microseconds=1), 1, id="sub-ms-is-not-zero"),
    ],
)
def test_ttl_ms_never_shortens_the_ttl_or_sends_zero(ttl: timedelta, expected: int) -> None:
    assert ttl_ms(ttl) == expected
