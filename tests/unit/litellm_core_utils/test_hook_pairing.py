from collections.abc import Callable
from typing import Final

import pytest

from litellm.litellm_core_utils.hook_pairing import HookPairing, run_held_hook_now


class _Logger:
    pass


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        (("hold", "reach", "end"), []),
        (("reach", "hold", "end"), []),
        (("hold", "end"), ["ran"]),
        (("end", "hold"), ["ran"]),
        (("hold", "end", "reach"), ["ran"]),
        (("hold",), []),
    ],
)
def test_held_sync_hook_runs_only_when_the_async_run_ends_without_reaching_its_logger(
    steps: tuple[str, ...], expected: list[str]
) -> None:
    ran: Final[list[str]] = []
    logger: Final = _Logger()
    pairing: Final = HookPairing(run_held_hook_now)
    actions: Final[dict[str, Callable[[], None]]] = {
        "hold": lambda: pairing.hold(logger, lambda: ran.append("ran")),
        "reach": lambda: pairing.reach(logger),
        "end": pairing.end,
    }

    for step in steps:
        actions[step]()

    assert ran == expected


def test_async_run_reaching_one_logger_leaves_another_loggers_held_hook_to_run() -> None:
    ran: Final[list[str]] = []
    reached: Final = _Logger()
    unreached: Final = _Logger()
    pairing: Final = HookPairing(run_held_hook_now)

    pairing.hold(reached, lambda: ran.append("reached"))
    pairing.hold(unreached, lambda: ran.append("unreached"))
    pairing.reach(reached)
    pairing.end()

    assert ran == ["unreached"]


def test_held_hooks_run_through_the_pairings_runner_and_a_raising_one_stays_non_blocking() -> None:
    ran: Final[list[str]] = []
    runner_calls: Final[list[str]] = []
    pairing: Final = HookPairing(lambda hook: (runner_calls.append("runner"), hook()))

    def raising() -> None:
        raise RuntimeError("logger is down")

    pairing.hold(_Logger(), raising)
    pairing.hold(_Logger(), lambda: ran.append("ran"))
    pairing.end()

    assert runner_calls == ["runner", "runner"]
    assert ran == ["ran"]
