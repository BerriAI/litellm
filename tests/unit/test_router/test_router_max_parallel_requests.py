from __future__ import annotations

import datetime
from typing import Final
from unittest.mock import patch

import pytest

import litellm
from litellm.utils import calculate_max_parallel_requests

_ROUTER_CLOCK: Final = "litellm.router_strategy.lowest_tpm_rpm_v2.get_utc_datetime"
_FIRST_MINUTE: Final = datetime.datetime(2026, 1, 1, 12, 0, 59, tzinfo=datetime.timezone.utc)
_NEXT_MINUTE: Final = datetime.datetime(2026, 1, 1, 12, 1, 0, tzinfo=datetime.timezone.utc)


def _rpm_limited_router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "limited",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "test-key",
                    "rpm": 1,
                },
            }
        ],
        routing_strategy="usage-based-routing-v2",
        enable_pre_call_checks=True,
        num_retries=0,
    )


_RPM_LIMITED_REQUEST: Final = {
    "model": "limited",
    "messages": [{"role": "user", "content": "rate limit"}],
    "mock_response": "response",
}


@pytest.mark.parametrize(
    ("max_parallel_requests", "tpm", "rpm", "default_max_parallel_requests", "expected"),
    [
        pytest.param(10, 300_000, 30, 40, 10, id="explicit-limit"),
        pytest.param(None, 300_000, 30, 40, 30, id="rpm"),
        pytest.param(None, 300_000, None, 40, 1_800, id="tpm"),
        pytest.param(None, 20, None, 40, 1, id="minimum-tpm-limit"),
        pytest.param(None, None, None, 40, 40, id="router-default"),
        pytest.param(None, None, None, None, None, id="unset"),
    ],
)
def test_scenario(
    max_parallel_requests: int | None,
    tpm: int | None,
    rpm: int | None,
    default_max_parallel_requests: int | None,
    expected: int | None,
) -> None:
    calculated: Final = calculate_max_parallel_requests(
        max_parallel_requests=max_parallel_requests,
        rpm=rpm,
        tpm=tpm,
        default_max_parallel_requests=default_max_parallel_requests,
    )

    assert calculated == expected


@pytest.mark.asyncio
async def test_max_parallel_requests_rpm_rate_limiting() -> None:
    router: Final = _rpm_limited_router()

    with patch(_ROUTER_CLOCK, return_value=_FIRST_MINUTE):
        first_response: Final = await router.acompletion(**_RPM_LIMITED_REQUEST)

        assert first_response.choices[0].message.content == "response"
        with pytest.raises(litellm.RateLimitError):
            await router.acompletion(**_RPM_LIMITED_REQUEST)


@pytest.mark.asyncio
async def test_rpm_limit_resets_in_the_next_minute() -> None:
    router: Final = _rpm_limited_router()

    with patch(_ROUTER_CLOCK, return_value=_FIRST_MINUTE):
        await router.acompletion(**_RPM_LIMITED_REQUEST)
    with patch(_ROUTER_CLOCK, return_value=_NEXT_MINUTE):
        next_minute_response: Final = await router.acompletion(**_RPM_LIMITED_REQUEST)

    assert next_minute_response.choices[0].message.content == "response"


@pytest.mark.asyncio
async def test_max_parallel_requests_tpm_rate_limiting_base_case() -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "limited",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "test-key",
                    "tpm": 1,
                },
            }
        ],
        routing_strategy="usage-based-routing-v2",
        enable_pre_call_checks=True,
        num_retries=0,
    )
    request: Final = {
        "model": "limited",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "mock_response": "ok",
    }

    with pytest.raises(litellm.RateLimitError):
        await router.acompletion(**request)
