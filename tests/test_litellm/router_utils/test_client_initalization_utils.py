import asyncio
from typing import Final

import pytest
from pydantic import ValidationError

import litellm
from litellm import Router
from litellm.router_utils.client_initalization_utils import DeploymentSemaphore


def _semaphore(queue_size: int | None, max_parallel_requests: int = 1) -> DeploymentSemaphore:
    return DeploymentSemaphore(
        max_parallel_requests=max_parallel_requests,
        model_id="deployment-1",
        model_group="gpt-5.6",
        queue_size=queue_size,
    )


async def _hold(semaphore: DeploymentSemaphore, release: asyncio.Event) -> str:
    async with semaphore:
        await release.wait()
        return "ok"


async def _expect_rejection(semaphore: DeploymentSemaphore) -> litellm.RateLimitError:
    with pytest.raises(litellm.RateLimitError) as excinfo:
        await asyncio.wait_for(semaphore.acquire(), timeout=1)
    return excinfo.value


@pytest.mark.asyncio
async def test_queue_full_rejects_new_caller_while_queued_callers_still_complete():
    semaphore: Final = _semaphore(queue_size=2)
    release: Final = asyncio.Event()
    holder: Final = asyncio.create_task(_hold(semaphore, release))
    await asyncio.sleep(0)
    queued: Final = [asyncio.create_task(_hold(semaphore, release)) for _ in range(2)]
    await asyncio.sleep(0)
    assert semaphore.locked() and semaphore.waiting == 2

    rejection: Final = await _expect_rejection(semaphore)

    assert rejection.status_code == 429
    assert "deployment-1" in rejection.message
    assert "gpt-5.6" in rejection.message
    assert "max_parallel_requests=1" in rejection.message
    assert "max_parallel_requests_queue_size=2" in rejection.message
    assert semaphore.waiting == 2

    release.set()
    assert await asyncio.wait_for(asyncio.gather(holder, *queued), timeout=2) == ["ok", "ok", "ok"]
    assert semaphore.waiting == 0
    assert not semaphore.locked()


@pytest.mark.asyncio
async def test_zero_queue_size_rejects_as_soon_as_every_slot_is_busy():
    semaphore: Final = _semaphore(queue_size=0, max_parallel_requests=2)
    release: Final = asyncio.Event()
    holders: Final = [asyncio.create_task(_hold(semaphore, release)) for _ in range(2)]
    await asyncio.sleep(0)

    await _expect_rejection(semaphore)
    assert semaphore.waiting == 0

    release.set()
    assert await asyncio.wait_for(asyncio.gather(*holders), timeout=2) == ["ok", "ok"]


@pytest.mark.asyncio
async def test_unset_queue_size_parks_every_caller_until_a_slot_frees():
    semaphore: Final = _semaphore(queue_size=None)
    release: Final = asyncio.Event()
    callers: Final = [asyncio.create_task(_hold(semaphore, release)) for _ in range(50)]
    await asyncio.sleep(0)
    assert semaphore.waiting == 49

    release.set()
    assert await asyncio.wait_for(asyncio.gather(*callers), timeout=2) == ["ok"] * 50
    assert semaphore.waiting == 0


@pytest.mark.asyncio
async def test_cancelled_waiter_gives_its_queue_slot_back():
    semaphore: Final = _semaphore(queue_size=1)
    release: Final = asyncio.Event()
    holder: Final = asyncio.create_task(_hold(semaphore, release))
    await asyncio.sleep(0)
    cancelled: Final = asyncio.create_task(_hold(semaphore, release))
    await asyncio.sleep(0)
    assert semaphore.waiting == 1

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert semaphore.waiting == 0

    replacement: Final = asyncio.create_task(_hold(semaphore, release))
    await asyncio.sleep(0)
    assert semaphore.waiting == 1
    release.set()
    assert await asyncio.wait_for(asyncio.gather(holder, replacement), timeout=2) == ["ok", "ok"]


def _router_semaphore(router: Router, model_name: str) -> DeploymentSemaphore:
    deployment: Final = router.get_deployment_by_model_group_name(model_group_name=model_name)
    assert deployment is not None
    client: Final = router._get_client(deployment=deployment.model_dump(), kwargs={}, client_type="max_parallel_requests")
    assert isinstance(client, DeploymentSemaphore)
    return client


@pytest.mark.parametrize("invalid_queue_size", [-1, 2.5, True, "3"])
def test_invalid_queue_sizes_are_rejected_instead_of_coerced(invalid_queue_size: object):
    """A negative bound would reject every busy request and a fraction would be truncated, so
    neither may reach a semaphore, the router default, or a live update of that default."""
    with pytest.raises(ValidationError):
        _semaphore(queue_size=invalid_queue_size)
    model_list: Final = [{"model_name": "gpt-5.6", "litellm_params": {"model": "openai/gpt-5.6", "rpm": 1}}]
    with pytest.raises(ValidationError):
        Router(model_list=model_list, default_max_parallel_requests_queue_size=invalid_queue_size)
    with pytest.raises(ValidationError):
        Router(
            model_list=[
                {
                    "model_name": "gpt-5.6",
                    "litellm_params": {
                        "model": "openai/gpt-5.6",
                        "rpm": 1,
                        "max_parallel_requests_queue_size": invalid_queue_size,
                    },
                }
            ]
        )

    router: Final = Router(model_list=model_list, default_max_parallel_requests_queue_size=4)
    semaphore: Final = _router_semaphore(router, "gpt-5.6")
    with pytest.raises(ValidationError):
        router.update_settings(default_max_parallel_requests_queue_size=invalid_queue_size)
    assert router.default_max_parallel_requests_queue_size == 4
    assert semaphore.queue_size == 4


@pytest.mark.asyncio
async def test_deployment_queue_size_overrides_router_default_and_zero_is_honored():
    router: Final = Router(
        model_list=[
            {"model_name": "inherits-default", "litellm_params": {"model": "openai/gpt-5.6", "rpm": 1}},
            {
                "model_name": "no-queue",
                "litellm_params": {"model": "openai/gpt-5.6", "tpm": 100, "max_parallel_requests_queue_size": 0},
            },
        ],
        default_max_parallel_requests_queue_size=1,
    )
    release: Final = asyncio.Event()

    inherits: Final = _router_semaphore(router, "inherits-default")
    inherits_holder: Final = asyncio.create_task(_hold(inherits, release))
    await asyncio.sleep(0)
    inherits_waiter: Final = asyncio.create_task(_hold(inherits, release))
    await asyncio.sleep(0)
    assert "max_parallel_requests_queue_size=1" in (await _expect_rejection(inherits)).message

    no_queue: Final = _router_semaphore(router, "no-queue")
    no_queue_holder: Final = asyncio.create_task(_hold(no_queue, release))
    await asyncio.sleep(0)
    assert "max_parallel_requests_queue_size=0" in (await _expect_rejection(no_queue)).message

    release.set()
    await asyncio.wait_for(asyncio.gather(inherits_holder, inherits_waiter, no_queue_holder), timeout=2)


@pytest.mark.asyncio
async def test_router_without_queue_size_keeps_unbounded_queueing():
    router: Final = Router(
        model_list=[{"model_name": "gpt-5.6", "litellm_params": {"model": "openai/gpt-5.6", "max_parallel_requests": 1}}]
    )
    semaphore: Final = _router_semaphore(router, "gpt-5.6")
    release: Final = asyncio.Event()
    callers: Final = [asyncio.create_task(_hold(semaphore, release)) for _ in range(20)]
    await asyncio.sleep(0)
    assert semaphore.waiting == 19
    release.set()
    assert await asyncio.wait_for(asyncio.gather(*callers), timeout=2) == ["ok"] * 20


@pytest.mark.asyncio
async def test_update_settings_applies_default_queue_size_to_live_semaphores_without_an_override():
    router: Final = Router(
        model_list=[
            {"model_name": "inherits-default", "litellm_params": {"model": "openai/gpt-5.6", "rpm": 1}},
            {
                "model_name": "pinned",
                "litellm_params": {"model": "openai/gpt-5.6", "rpm": 1, "max_parallel_requests_queue_size": 5},
            },
        ],
    )
    inherits: Final = _router_semaphore(router, "inherits-default")
    pinned: Final = _router_semaphore(router, "pinned")
    assert router.get_settings()["default_max_parallel_requests_queue_size"] is None

    router.update_settings(default_max_parallel_requests_queue_size=0)
    assert router.get_settings()["default_max_parallel_requests_queue_size"] == 0
    assert (inherits.queue_size, pinned.queue_size) == (0, 5)

    release: Final = asyncio.Event()
    holder: Final = asyncio.create_task(_hold(inherits, release))
    await asyncio.sleep(0)
    assert "max_parallel_requests_queue_size=0" in (await _expect_rejection(inherits)).message

    router.update_settings(default_max_parallel_requests_queue_size=None)
    assert (inherits.queue_size, pinned.queue_size) == (None, 5)
    waiter: Final = asyncio.create_task(_hold(inherits, release))
    await asyncio.sleep(0)
    assert inherits.waiting == 1

    release.set()
    assert await asyncio.wait_for(asyncio.gather(holder, waiter), timeout=2) == ["ok", "ok"]
