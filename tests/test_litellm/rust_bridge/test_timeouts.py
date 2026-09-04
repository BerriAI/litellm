import httpx
import pytest

from litellm.rust_bridge.timeouts import timeout_to_seconds


@pytest.mark.parametrize(
    ("timeout", "expected"),
    (
        pytest.param(None, None, id="none"),
        pytest.param(12.5, 12.5, id="float"),
        pytest.param(httpx.Timeout(30.0, read=42.0), 42.0, id="httpx-read-timeout"),
    ),
)
def test_timeout_to_seconds(timeout: float | httpx.Timeout | None, expected: float | None) -> None:
    assert timeout_to_seconds(timeout) == expected
