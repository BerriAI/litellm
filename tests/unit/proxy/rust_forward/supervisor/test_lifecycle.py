import functools
import signal
from dataclasses import dataclass, replace
from typing import Final

import pytest
from hypothesis import example, given
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
    Phase,
    Probe,
    Send,
    Spawn,
    boot,
    exit_status,
    step,
)

TERM: Final = signal.SIGTERM
INT: Final = signal.SIGINT
POSIX: Final = PosixPlatform()


@dataclass(frozen=True, slots=True)
class World:
    """What the operating system knows about the children, kept independently of ``Lifecycle``."""

    alive: frozenset[Child] = frozenset({Child.PYTHON})
    spawned: frozenset[Child] = frozenset({Child.PYTHON})
    reported_listening: bool = False
    stop_requested: bool = False
    python_status: int | None = None
    rust_lost: bool = False


def _expected_exit(world: World) -> int | None:
    if world.alive or world.python_status is None:
        return None
    return FAILURE if world.rust_lost else world.python_status


def _expected_relay(world: World, signum: signal.Signals) -> tuple[Action, ...]:
    to_rust: Final = (Send(Child.RUST, signum),) if signum in STOPS and Child.RUST in world.alive else ()
    to_python: Final = (Send(Child.PYTHON, signum),) if Child.PYTHON in world.alive else ()
    return (*to_rust, *to_python)


def _expected_on_exit(world: World, child: Child) -> tuple[Action, ...]:
    match child:
        case Child.PYTHON:
            return (Send(Child.RUST, TERM),) if Child.RUST in world.alive else ()
        case Child.RUST:
            serving: Final = Child.PYTHON in world.alive and not world.stop_requested
            return (Send(Child.PYTHON, TERM),) if serving else ()


def _after(world: World, event: Event) -> World:
    match event:
        case Caught(signum):
            return replace(world, stop_requested=world.stop_requested or signum in STOPS)
        case Probe.PYTHON_LISTENING:
            return replace(world, reported_listening=world.reported_listening or Child.PYTHON in world.alive)
        case ChildExited(child, status) if child in world.alive:
            lost: Final = child is Child.RUST and Child.PYTHON in world.alive and not world.stop_requested
            return replace(
                world,
                alive=world.alive - {child},
                python_status=status if child is Child.PYTHON else world.python_status,
                rust_lost=world.rust_lost or lost,
            )
        case ChildExited():
            return world


def _spawned(world: World, actions: tuple[Action, ...]) -> World:
    started: Final = frozenset(action.child for action in actions if isinstance(action, Spawn))
    return replace(world, alive=world.alive | started, spawned=world.spawned | started)


SIGNALS: Final = POSIX.relayed_signals
STOPS: Final = POSIX.stop_signals
EVENTS: Final = st.lists(
    st.one_of(
        st.sampled_from(SIGNALS).map(Caught),
        st.just(Probe.PYTHON_LISTENING),
        st.builds(ChildExited, st.sampled_from(Child), st.integers(-15, 3)),
    ),
    max_size=40,
)


@given(EVENTS)
@example([Caught(TERM), Caught(signal.SIGHUP), Probe.PYTHON_LISTENING])
@example([Probe.PYTHON_LISTENING, Caught(TERM), Caught(signal.SIGWINCH), ChildExited(Child.RUST, 0)])
def test_lifecycle_relays_every_signal_and_ends_with_pythons_status(events: list[Event]) -> None:
    state, actions = boot()
    world = World()
    assert actions == (Spawn(Child.PYTHON),)
    for event in events:
        before = _after(world, event)
        state, actions = step(state, event, STOPS)
        match event:
            case Caught(signum):
                assert actions == _expected_relay(before, signum)
            case Probe.PYTHON_LISTENING:
                starts_rust = Child.PYTHON in world.alive and not world.reported_listening and not world.stop_requested
                assert actions == ((Spawn(Child.RUST),) if starts_rust else ())
            case ChildExited(child) if child in world.alive:
                assert actions == _expected_on_exit(world, child)
            case ChildExited():
                assert actions == ()
        assert all(action.child in before.alive for action in actions if isinstance(action, Send))
        assert all(action.child not in before.spawned for action in actions if isinstance(action, Spawn))
        world = _spawned(before, actions)
        assert exit_status(state) == _expected_exit(world)


@given(EVENTS)
def test_rust_starts_at_most_once_and_only_while_python_listens_unstopped(events: list[Event]) -> None:
    def advance(acc: tuple[Lifecycle, tuple[tuple[Event, Lifecycle, Action], ...]], event: Event):
        state, seen = acc
        following, actions = step(state, event, STOPS)
        return following, (*seen, *((event, state, action) for action in actions))

    _, history = functools.reduce(advance, events, (boot()[0], ()))
    spawns: Final = tuple((event, state) for event, state, action in history if isinstance(action, Spawn))
    assert len(spawns) <= 1
    assert all(
        event is Probe.PYTHON_LISTENING and state.python is Phase.SPAWNED and not state.stop_requested
        for event, state in spawns
    )


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ((Probe.PYTHON_LISTENING, Caught(TERM), ChildExited(Child.RUST, 0), ChildExited(Child.PYTHON, -15)), -15),
        (
            (
                Probe.PYTHON_LISTENING,
                Caught(INT),
                Caught(INT),
                ChildExited(Child.PYTHON, 1),
                ChildExited(Child.RUST, 0),
            ),
            1,
        ),
        (
            (Probe.PYTHON_LISTENING, Caught(signal.SIGHUP), ChildExited(Child.PYTHON, -1), ChildExited(Child.RUST, 0)),
            -1,
        ),
        ((Probe.PYTHON_LISTENING, ChildExited(Child.RUST, 0), ChildExited(Child.PYTHON, 0)), FAILURE),
        ((ChildExited(Child.PYTHON, 3),), 3),
    ],
)
def test_supervisor_exits_as_python_did_unless_rust_was_lost(events: tuple[Event, ...], expected: int) -> None:
    final: Final = functools.reduce(lambda state, event: step(state, event, STOPS)[0], events, boot()[0])
    assert exit_status(final) == expected
