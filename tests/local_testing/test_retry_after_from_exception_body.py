#### What this tests ####
#    Tests extracting a quota-reset ("cooldown until") timestamp from a
#    provider error body, e.g. Zhipu bigmodel 1308/1310:
#      "已达到 7 天使用上限，2026-09-15 09:21:11 后可继续使用。"
#    and wiring it into the router's cooldown-time decision chain.

from datetime import datetime, timedelta, timezone

import pytest

from litellm.exceptions import RateLimitError
from litellm.litellm_core_utils.exception_mapping_utils import (
    _get_retry_after_from_exception_body,
)

_CN_TZ = timezone(timedelta(hours=8))


def _zhipu_1310_exception(reset_at: datetime) -> RateLimitError:
    message = (
        '{"error":{"code":"1310","message":"已达到 7 天使用上限，'
        + reset_at.strftime("%Y-%m-%d %H:%M:%S")
        + ' 后可继续使用。如需超限额按量付费使用，可联系管理员开启超额按量付费。"}}'
    )
    return RateLimitError(message=message, llm_provider="hosted_vllm", model="GLM-5.3")


def test_future_timestamp_returns_seconds_until_reset():
    reset_at = datetime.now(_CN_TZ) + timedelta(hours=10)
    now = datetime.now(_CN_TZ).timestamp()
    exc = _zhipu_1310_exception(reset_at)
    seconds = _get_retry_after_from_exception_body(exc, now=now)
    assert seconds is not None
    assert seconds == pytest.approx(10 * 3600, abs=5)


def test_iso_timestamp_also_matched():
    reset_at = datetime.now(_CN_TZ) + timedelta(hours=2)
    exc = RateLimitError(
        message=f"quota resets at {reset_at.strftime('%Y-%m-%dT%H:%M:%S')}, retry then",
        llm_provider="openai", model="gpt-5.4",
    )
    seconds = _get_retry_after_from_exception_body(exc)
    assert seconds is not None
    assert 0 < seconds <= 2 * 3600 + 5


def test_past_timestamp_returns_none():
    reset_at = datetime.now(_CN_TZ) - timedelta(hours=1)
    exc = _zhipu_1310_exception(reset_at)
    assert _get_retry_after_from_exception_body(exc) is None


def test_no_timestamp_returns_none():
    exc = RateLimitError(
        message="Rate limit reached for requests", llm_provider="openai", model="gpt-5.4"
    )
    assert _get_retry_after_from_exception_body(exc) is None


def test_non_exception_input_returns_none():
    assert _get_retry_after_from_exception_body("not an exception") is None  # type: ignore[arg-type]


def test_garbage_timestamp_returns_none():
    exc = RateLimitError(
        message="reset at 9999-99-99 99:99:99 please", llm_provider="openai", model="gpt-5.4"
    )
    assert _get_retry_after_from_exception_body(exc) is None
