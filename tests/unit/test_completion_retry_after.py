from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.main import _retry_after_wait


def _retry_state(exception: Exception) -> SimpleNamespace:
    return SimpleNamespace(outcome=SimpleNamespace(exception=lambda: exception))


def _rate_limit_error(headers: dict[str, str]) -> litellm.RateLimitError:
    response = httpx.Response(
        status_code=429,
        headers=headers,
        request=httpx.Request("POST", "https://example.invalid/v1"),
    )
    return litellm.RateLimitError(
        message="rate limited",
        model="test-model",
        llm_provider="test-provider",
        response=response,
    )


@pytest.mark.parametrize(
    ("headers", "expected_wait"),
    [
        ({"retry-after": "20"}, 20),
        ({"retry-after-ms": "2500"}, 2.5),
        ({"retry-after": "120"}, 60),
    ],
)
def test_retry_after_wait_uses_and_caps_provider_hint(headers: dict[str, str], expected_wait: float) -> None:
    fallback_wait = Mock(return_value=1.0)

    wait = _retry_after_wait(_retry_state(_rate_limit_error(headers)), fallback_wait)

    assert wait == expected_wait
    fallback_wait.assert_not_called()


def test_retry_after_wait_preserves_fallback_strategy_without_hint() -> None:
    fallback_wait = Mock(return_value=3.0)

    wait = _retry_after_wait(_retry_state(RuntimeError("request failed")), fallback_wait)

    assert wait == 3.0
    fallback_wait.assert_called_once()
