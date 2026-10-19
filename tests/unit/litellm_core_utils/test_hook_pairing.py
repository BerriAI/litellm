import textwrap
from collections.abc import Callable
from typing import Final

import pytest

from litellm.litellm_core_utils.hook_pairing import HookPairing, run_held_hook_now
from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter


class _Logger:
    pass


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        (("sync", "end"), ["ran"]),
        (("end", "sync"), ["ran"]),
        (("sync", "reach", "end"), []),
        (("reach", "sync", "end"), []),
        (("sync", "end", "reach"), ["ran"]),
        (("sync",), []),
        (("end",), []),
    ],
)
def test_held_sync_hook_runs_only_when_both_runs_end_without_the_async_run_reaching_its_logger(
    steps: tuple[str, ...], expected: list[str]
) -> None:
    ran: Final[list[str]] = []
    logger: Final = _Logger()
    pairing: Final = HookPairing(run_held_hook_now)

    def holding_sync_run() -> None:
        with pairing.sync_run():
            pairing.hold(logger, lambda: ran.append("ran"))

    actions: Final[dict[str, Callable[[], None]]] = {
        "sync": holding_sync_run,
        "reach": lambda: pairing.reach(logger),
        "end": pairing.end_async_run,
    }

    for step in steps:
        actions[step]()

    assert ran == expected


def test_held_sync_hook_waits_for_every_open_sync_run_after_the_async_run_ends() -> None:
    ran: Final[list[str]] = []
    pairing: Final = HookPairing(run_held_hook_now)

    with pairing.sync_run():
        pairing.hold(_Logger(), lambda: ran.append("ran"))
        pairing.end_async_run()
        with pairing.sync_run():
            pass
        assert ran == []

    assert ran == ["ran"]


def test_async_run_discarded_before_it_starts_still_runs_the_held_hook() -> None:
    ran: Final[list[str]] = []
    pairing: Final = HookPairing(run_held_hook_now)

    async def never_started() -> None:
        ran.append("async run")

    with pairing.sync_run():
        pairing.hold(_Logger(), lambda: ran.append("ran"))
    pairing.async_run(never_started()).close()

    assert ran == ["ran"]


def test_async_run_still_queued_at_exit_is_left_to_the_exit_flush() -> None:
    script: Final = textwrap.dedent(
        """
        import asyncio
        import atexit

        from litellm.litellm_core_utils.hook_pairing import HookPairing, run_held_hook_now

        logger = object()
        pairing = HookPairing(run_held_hook_now)
        queued = []


        async def async_run():
            pairing.reach(logger)
            print("async_hook")
            pairing.end_async_run()


        atexit.register(lambda: asyncio.run(queued.pop()))
        with pairing.sync_run():
            pairing.hold(logger, lambda: print("sync_hook"))
        queued.append(pairing.async_run(async_run()))
        """
    )

    completed: Final = run_child_interpreter(script, timeout=120)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["async_hook"]


@pytest.mark.parametrize(("seconds_to_clear_queue", "expected"), [(0, ["sync_hook"]), (5, ["async_hook"])])
def test_async_run_the_exit_flush_cannot_start_still_runs_the_held_hook(
    seconds_to_clear_queue: int, expected: list[str]
) -> None:
    script: Final = textwrap.dedent(
        f"""
        import asyncio

        from litellm.litellm_core_utils import logging_worker
        from litellm.litellm_core_utils.hook_pairing import HookPairing, run_held_hook_now

        logging_worker.MAX_TIME_TO_CLEAR_QUEUE = {seconds_to_clear_queue}
        logger = object()
        pairing = HookPairing(run_held_hook_now)


        async def async_run():
            pairing.reach(logger)
            print("async_hook")
            pairing.end_async_run()


        async def main():
            with pairing.sync_run():
                pairing.hold(logger, lambda: print("sync_hook"))
            logging_worker.GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(pairing.async_run(async_run()))


        asyncio.run(main())
        """
    )

    completed: Final = run_child_interpreter(script, timeout=120)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == expected


def test_async_run_reaching_one_logger_leaves_another_loggers_held_hook_to_run() -> None:
    ran: Final[list[str]] = []
    reached: Final = _Logger()
    unreached: Final = _Logger()
    pairing: Final = HookPairing(run_held_hook_now)

    with pairing.sync_run():
        pairing.hold(reached, lambda: ran.append("reached"))
        pairing.hold(unreached, lambda: ran.append("unreached"))
    pairing.reach(reached)
    pairing.end_async_run()

    assert ran == ["unreached"]


def test_held_hooks_run_through_the_pairings_runner_and_a_raising_one_stays_non_blocking() -> None:
    ran: Final[list[str]] = []
    runner_calls: Final[list[str]] = []
    pairing: Final = HookPairing(lambda hook: (runner_calls.append("runner"), hook()))

    def raising() -> None:
        raise RuntimeError("logger is down")

    with pairing.sync_run():
        pairing.hold(_Logger(), raising)
        pairing.hold(_Logger(), lambda: ran.append("ran"))
    pairing.end_async_run()

    assert runner_calls == ["runner", "runner"]
    assert ran == ["ran"]
