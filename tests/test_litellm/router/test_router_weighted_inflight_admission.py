import asyncio
from unittest.mock import AsyncMock

import pytest

from litellm.router import Router
from litellm.router_utils.weighted_inflight_admission import (
    AdmissionClass,
    WeightedInFlightAdmission,
)


@pytest.fixture
def router() -> Router:
    instance = Router.__new__(Router)
    instance.weighted_inflight_admission = WeightedInFlightAdmission(
        capacity=1,
        classes=(AdmissionClass(name="interactive", reservation=1, priority=0),),
    )
    instance.weighted_inflight_admission_class = "interactive"
    instance.set_response_headers = AsyncMock(side_effect=lambda response, **_: response)
    return instance


@pytest.mark.asyncio
async def test_make_call_releases_lease_on_success(router: Router):
    async def original_function():
        return "response"

    assert await router.make_call(original_function) == "response"
    assert router.weighted_inflight_admission.active == 0


@pytest.mark.asyncio
async def test_make_call_releases_lease_on_error(router: Router):
    async def original_function():
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        await router.make_call(original_function)

    assert router.weighted_inflight_admission.active == 0


@pytest.mark.asyncio
async def test_make_call_keeps_stream_lease_until_exhausted(router: Router):
    async def original_function(**kwargs):
        async def stream():
            yield "chunk"

        return stream()

    response = await router.make_call(original_function, stream=True)
    assert router.weighted_inflight_admission.active == 1
    assert [chunk async for chunk in response] == ["chunk"]
    assert router.weighted_inflight_admission.active == 0


@pytest.mark.asyncio
async def test_make_call_waits_for_existing_lease(router: Router):
    started = asyncio.Event()
    finish = asyncio.Event()

    async def first_call():
        started.set()
        await finish.wait()
        return "first"

    first = asyncio.create_task(router.make_call(first_call))
    await started.wait()
    second = asyncio.create_task(router.make_call(lambda: "second"))
    await asyncio.sleep(0)
    assert not second.done()

    finish.set()
    assert await first == "first"
    assert await second == "second"


@pytest.mark.asyncio
async def test_request_cannot_select_admission_class_or_leak_control_kwarg(router: Router):
    observed: dict[str, object] = {}

    async def original_function(**kwargs: object):
        observed.update(kwargs)
        return "response"

    response = await router.make_call(
        original_function,
        admission_class="untrusted-background",
        metadata={"admission_class": "untrusted-background"},
    )
    assert response == "response"
    assert observed["metadata"] == {"admission_class": "untrusted-background"}
    assert "admission_class" not in observed
    assert router.weighted_inflight_admission.metrics.snapshot()["admitted"] == 1
