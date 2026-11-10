"""
A registry read on another thread while an upsert writes router-core and reads its list back: the
read waits for the write, so it never resolves one registry version's positions against the
other's list (an IndexError past the list, or the neighbouring model's deployments after a removal).
"""

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Final

import pytest

import litellm
from litellm import Router

_DEPLOYMENTS: Final = 4


def _deployment(index: int, timeout: float = 10) -> dict[str, object]:  # mutable-ok: a fresh model_list entry
    return {
        "model_name": f"model-{index % _DEPLOYMENTS}",
        "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": f"key-{index}", "timeout": timeout},
        "model_info": {"id": f"deployment-{index}"},
    }


def _ids(router: Router, model: int) -> list[str]:  # mutable-ok: a fresh list
    rows: Final = router._hosted._core_get_all_deployments(model_name=f"model-{model}")  # pyright: ignore[reportPrivateUsage]  # the registry read under test
    return sorted(str(row["model_info"]["id"]) for row in rows)


@pytest.mark.parametrize("upserted", [0, 3, _DEPLOYMENTS + 1])
def test_read_on_another_thread_waits_for_an_upsert_to_read_its_list_back(upserted: int) -> None:
    router: Final = Router(model_list=[_deployment(index) for index in range(2 * _DEPLOYMENTS)])
    hosted: Final = router._hosted  # pyright: ignore[reportPrivateUsage]  # the registry under test
    write: Final[Callable[[Mapping[str, object]], None]] = hosted.changes_runner  # pyright: ignore[reportAssignmentType]  # set by the core
    pool: Final = ThreadPoolExecutor(max_workers=4)
    reads: Final[list[Future[Mapping[int, Sequence[str]]]]] = []  # mutable-ok: one per write

    def read_every_model() -> Mapping[int, Sequence[str]]:
        return {model: _ids(router, model) for model in range(_DEPLOYMENTS)}

    def write_with_a_read_waiting(changes: Mapping[str, object]) -> None:
        write(changes)
        # router-core holds the write; a read starting now must wait until the list is read back.
        read: Final = pool.submit(read_every_model)
        _ = wait([read], timeout=0.05)
        reads.append(read)

    hosted.changes_runner = write_with_a_read_waiting
    try:
        router.upsert_deployment(litellm.Deployment(**_deployment(upserted, timeout=11)))  # pyright: ignore[reportArgumentType]  # a deployment dict
    finally:
        hosted.changes_runner = write

    assert reads
    for read in reads:
        seen = read.result(timeout=5)
        for model in range(_DEPLOYMENTS):
            # Between the upsert's removal and its add, its model is served by the other deployment.
            assert seen[model], (model, seen)
            assert set(seen[model]) <= {f"deployment-{model}", f"deployment-{model + _DEPLOYMENTS}"}, (model, seen)
    pool.shutdown()
    assert _ids(router, upserted % _DEPLOYMENTS) == sorted(
        [f"deployment-{upserted % _DEPLOYMENTS}", f"deployment-{upserted % _DEPLOYMENTS + _DEPLOYMENTS}"]
    )
