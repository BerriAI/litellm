import asyncio
from typing import Final

import pytest

import litellm
from litellm import Router
from litellm.router_utils.client_initalization_utils import MaxParallelRequestsLimit


def _limit(max_parallel_requests: int = 1) -> MaxParallelRequestsLimit:
    return MaxParallelRequestsLimit(
        max_parallel_requests=max_parallel_requests, model_id="deployment-1", model_group="gpt-5.6"
    )


async def _hold(limit: MaxParallelRequestsLimit, release: asyncio.Event) -> str:
    with limit:
        await release.wait()
        return "ok"


def _expect_rejection(limit: MaxParallelRequestsLimit) -> litellm.RateLimitError:
    with pytest.raises(litellm.RateLimitError) as excinfo:
        limit.acquire()
    return excinfo.value


@pytest.mark.asyncio
async def test_request_arriving_while_every_slot_is_in_use_gets_429_without_waiting():
    limit: Final = _limit(max_parallel_requests=2)
    release: Final = asyncio.Event()
    holders: Final = [asyncio.create_task(_hold(limit, release)) for _ in range(2)]
    await asyncio.sleep(0)
    assert limit.in_flight == 2

    rejection: Final = _expect_rejection(limit)

    assert rejection.status_code == 429
    assert "deployment-1" in rejection.message
    assert "gpt-5.6" in rejection.message
    assert "max_parallel_requests=2" in rejection.message
    assert limit.in_flight == 2

    release.set()
    assert await asyncio.wait_for(asyncio.gather(*holders), timeout=2) == ["ok", "ok"]
    assert limit.in_flight == 0
    with limit:
        assert limit.in_flight == 1
    assert limit.in_flight == 0


@pytest.mark.asyncio
async def test_burst_over_the_cap_admits_exactly_max_parallel_requests_and_rejects_the_rest():
    limit: Final = _limit(max_parallel_requests=3)
    release: Final = asyncio.Event()

    async def attempt() -> str:
        try:
            return await _hold(limit, release)
        except litellm.RateLimitError as e:
            return f"rejected:{e.status_code}"

    callers: Final = [asyncio.create_task(attempt()) for _ in range(10)]
    await asyncio.sleep(0)
    assert limit.in_flight == 3
    release.set()
    outcomes: Final = await asyncio.wait_for(asyncio.gather(*callers), timeout=2)
    assert outcomes.count("ok") == 3
    assert outcomes.count("rejected:429") == 7
    assert limit.in_flight == 0


def test_slot_is_released_when_the_held_call_raises():
    limit: Final = _limit()
    with pytest.raises(RuntimeError):
        with limit:
            raise RuntimeError("provider blew up")
    assert limit.in_flight == 0
    with limit:
        assert limit.in_flight == 1


async def _concurrent_outcomes(router: Router, calls: int) -> list[str]:
    async def attempt() -> str:
        try:
            await router.acompletion(
                model="gpt-5.6", messages=[{"role": "user", "content": "hi"}], mock_response="ok", mock_delay=0.3
            )
        except litellm.RateLimitError as e:
            return e.message
        return "ok"

    return await asyncio.gather(*[attempt() for _ in range(calls)])


@pytest.mark.parametrize(
    ("litellm_params", "expected_cap"),
    [
        ({"max_parallel_requests": 2, "rpm": 7, "tpm": 100_000}, 2),
        ({"rpm": 7, "tpm": 100_000}, 7),
        ({"tpm": 100_000}, 600),
        ({"tpm": 100}, 1),
    ],
)
@pytest.mark.asyncio
async def test_router_deployment_rejects_past_its_derived_cap(litellm_params: dict[str, int], expected_cap: int):
    router: Final = Router(
        model_list=[
            {"model_name": "gpt-5.6", "litellm_params": {"model": "openai/gpt-5.6", "api_key": "k", **litellm_params}}
        ],
        num_retries=0,
    )

    outcomes: Final = await _concurrent_outcomes(router, expected_cap + 1)

    assert outcomes.count("ok") == expected_cap
    rejections: Final = [outcome for outcome in outcomes if outcome != "ok"]
    assert len(rejections) == 1
    assert f"max_parallel_requests={expected_cap}" in rejections[0]


@pytest.mark.asyncio
async def test_router_without_any_concurrency_setting_has_no_limit():
    router: Final = Router(
        model_list=[{"model_name": "gpt-5.6", "litellm_params": {"model": "openai/gpt-5.6", "api_key": "k"}}],
        num_retries=0,
    )

    assert await _concurrent_outcomes(router, 20) == ["ok"] * 20
