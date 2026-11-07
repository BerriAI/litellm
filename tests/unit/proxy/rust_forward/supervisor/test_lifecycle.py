import functools
import itertools
import signal
from dataclasses import dataclass, replace
from typing import Final

import pytest
from hypothesis import given
from hypothesis import strategies as st

from litellm.proxy.rust_forward.platform.posix import PosixPlatform
from litellm.proxy.rust_forward.supervisor.lifecycle import (
    FAILURE,
    Action,
    Caught,
    Child,
    ChildExited,
    Event,
    Lifecycle,
    Probe,
    Send,
    Spawn,
    boot,
    exit_status,
    step,
)

PY: Final = Child.PYTHON
RUST: Final = Child.RUST
LISTENING: Final = Probe.PYTHON_LISTENING
TERM: Final = signal.SIGTERM
INT: Final = signal.SIGINT
HUP: Final = signal.SIGHUP
STOPS: Final = PosixPlatform().stop_signals


def _replay(events: tuple[Event, ...]) -> tuple[tuple[tuple[Action, ...], ...], int | None]:
    transitions: Final = tuple(
        itertools.accumulate(events, lambda transition, event: step(transition[0], event, STOPS), initial=boot())
    )
    return tuple(actions for _, actions in transitions[1:]), exit_status(transitions[-1][0])


@pytest.mark.parametrize(
    ("trace", "status"),
    [
        pytest.param(
            (
                (LISTENING, (Spawn(RUST),)),
                (Caught(TERM), (Send(RUST, TERM), Send(PY, TERM))),
                (ChildExited(RUST, 0), ()),
                (ChildExited(PY, -15), ()),
            ),
            -15,
            id="TERM reaches Rust first, then Python, and the supervisor dies as Python did",
        ),
        pytest.param(
            (
                (LISTENING, (Spawn(RUST),)),
                (Caught(INT), (Send(RUST, INT), Send(PY, INT))),
                (Caught(INT), (Send(RUST, INT), Send(PY, INT))),
                (ChildExited(PY, 1), (Send(RUST, TERM),)),
                (ChildExited(RUST, 0), ()),
            ),
            1,
            id="a second Ctrl-C reaches Python again to force the exit",
        ),
        pytest.param(
            (
                (LISTENING, (Spawn(RUST),)),
                (Caught(HUP), (Send(PY, HUP),)),
                (Caught(signal.SIGTTIN), (Send(PY, signal.SIGTTIN),)),
                (ChildExited(PY, 0), (Send(RUST, TERM),)),
                (ChildExited(RUST, 0), ()),
            ),
            0,
            id="gunicorn's HUP and TTIN reach only Python and Rust keeps serving",
        ),
        pytest.param(
            (
                (LISTENING, (Spawn(RUST),)),
                (ChildExited(RUST, 0), (Send(PY, TERM),)),
                (ChildExited(PY, -15), ()),
            ),
            FAILURE,
            id="losing Rust while Python serves stops Python and fails",
        ),
        pytest.param(
            (
                (Caught(TERM), (Send(PY, TERM),)),
                (LISTENING, ()),
                (ChildExited(PY, -15), ()),
            ),
            -15,
            id="a stop before Python listens never starts Rust",
        ),
        pytest.param(((ChildExited(PY, 3), ()),), 3, id="Python failing to boot exits with its status"),
    ],
)
def test_supervisor_relays_signals_and_exits_as_python_did(
    trace: tuple[tuple[Event, tuple[Action, ...]], ...], status: int
) -> None:
    events: Final = tuple(event for event, _ in trace)
    assert _replay(events) == (tuple(actions for _, actions in trace), status)


@dataclass(frozen=True, slots=True)
class Observed:
    """What an outside observer of the two processes knows, without looking inside ``Lifecycle``."""

    state: Lifecycle
    alive: frozenset[Child] = frozenset({PY})
    rust_spawns: int = 0
    stop_requested: bool = False
    rust_lost: bool = False
    python_status: int | None = None


def _observe(seen: Observed, event: Event) -> Observed:
    state, actions = step(seen.state, event, STOPS)
    death: Final = event if isinstance(event, ChildExited) and event.child in seen.alive else None
    exited: Final = None if death is None else death.child
    running: Final = seen.alive - {exited}
    assert all(action.child in running for action in actions if isinstance(action, Send)), "signalled a dead child"
    if isinstance(event, Caught):
        to_rust: Final = (Send(RUST, event.signum),) if event.signum in STOPS and RUST in running else ()
        assert actions == (*to_rust, *((Send(PY, event.signum),) if PY in running else ()))
    spawned: Final = Spawn(RUST) in actions
    assert not spawned or (event is LISTENING and not seen.stop_requested), "started Rust out of turn"
    stop_requested: Final = seen.stop_requested or (isinstance(event, Caught) and event.signum in STOPS)
    after: Final = replace(
        seen,
        state=state,
        alive=running | ({RUST} if spawned else frozenset()),
        rust_spawns=seen.rust_spawns + int(spawned),
        stop_requested=stop_requested,
        rust_lost=seen.rust_lost or (exited is RUST and PY in running and not stop_requested),
        python_status=death.status if death is not None and exited is PY else seen.python_status,
    )
    settled: Final = None if after.alive or after.python_status is None else after.python_status
    assert exit_status(state) == (FAILURE if after.rust_lost and settled is not None else settled)
    return after


EVENTS: Final = st.lists(
    st.one_of(
        st.sampled_from(PosixPlatform().relayed_signals).map(Caught),
        st.just(LISTENING),
        st.builds(ChildExited, st.sampled_from(Child), st.integers(-15, 3)),
    ),
    max_size=40,
)


@given(EVENTS)
def test_any_event_order_keeps_the_supervisor_transparent_to_python(events: list[Event]) -> None:
    final: Final = functools.reduce(_observe, events, Observed(boot()[0]))
    assert final.rust_spawns <= 1
