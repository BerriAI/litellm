import signal
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Final, Literal, TypeAlias

from typing_extensions import assert_never


class Child(Enum):
    RUST = "rust"
    PYTHON = "python"


FAILURE: Final = 1


class Phase(Enum):
    NOT_STARTED = auto()
    SPAWNED = auto()
    LISTENING = auto()


@dataclass(frozen=True, slots=True)
class Exited:
    status: int


@dataclass(frozen=True, slots=True)
class Lost:
    """Rust exited while Python was serving and nobody had asked either to stop."""

    status: int


PythonState: TypeAlias = Literal[Phase.SPAWNED, Phase.LISTENING] | Exited
RustState: TypeAlias = Literal[Phase.NOT_STARTED, Phase.SPAWNED] | Exited | Lost


@dataclass(frozen=True, slots=True)
class Lifecycle:
    python: PythonState
    rust: RustState
    stop_requested: bool


@dataclass(frozen=True, slots=True)
class Caught:
    signum: signal.Signals


class Probe(Enum):
    PYTHON_LISTENING = auto()


@dataclass(frozen=True, slots=True)
class ChildExited:
    child: Child
    status: int


@dataclass(frozen=True, slots=True)
class Spawn:
    child: Child


@dataclass(frozen=True, slots=True)
class Send:
    child: Child
    signum: signal.Signals


Event: TypeAlias = Caught | Literal[Probe.PYTHON_LISTENING] | ChildExited
Action: TypeAlias = Spawn | Send
Transition: TypeAlias = tuple[Lifecycle, tuple[Action, ...]]


def boot() -> Transition:
    return Lifecycle(Phase.SPAWNED, Phase.NOT_STARTED, stop_requested=False), (Spawn(Child.PYTHON),)


def _relay(state: Lifecycle, signum: signal.Signals, stops: bool) -> Transition:
    to_rust: Final = (Send(Child.RUST, signum),) if stops and state.rust is Phase.SPAWNED else ()
    to_python: Final = (Send(Child.PYTHON, signum),) if not isinstance(state.python, Exited) else ()
    return replace(state, stop_requested=state.stop_requested or stops), (*to_rust, *to_python)


# Python alone takes no connections until it has booted, so Rust only opens the public port once Python listens.
def _python_listening(state: Lifecycle) -> Transition:
    if state.python is not Phase.SPAWNED:
        return state, ()
    if state.stop_requested:
        return replace(state, python=Phase.LISTENING), ()
    return replace(state, python=Phase.LISTENING, rust=Phase.SPAWNED), (Spawn(Child.RUST),)


def _python_exited(state: Lifecycle, status: int) -> Transition:
    if isinstance(state.python, Exited):
        return state, ()
    exited: Final = replace(state, python=Exited(status))
    return exited, ((Send(Child.RUST, signal.SIGTERM),) if state.rust is Phase.SPAWNED else ())


def _rust_exited(state: Lifecycle, status: int) -> Transition:
    if state.rust is not Phase.SPAWNED:
        return state, ()
    if state.stop_requested or isinstance(state.python, Exited):
        return replace(state, rust=Exited(status)), ()
    return replace(state, rust=Lost(status)), (Send(Child.PYTHON, signal.SIGTERM),)


def step(state: Lifecycle, event: Event, stop_signals: frozenset[signal.Signals]) -> Transition:
    """The supervisor never gives a signal or an exit a meaning of its own. Every signal reaches Python as sent,
    Python's exit status is the supervisor's, and Rust only follows Python, except that losing Rust stops Python."""
    match event:
        case Caught(signum):
            return _relay(state, signum, signum in stop_signals)
        case Probe.PYTHON_LISTENING:
            return _python_listening(state)
        case ChildExited(child, status):
            match child:
                case Child.PYTHON:
                    return _python_exited(state, status)
                case Child.RUST:
                    return _rust_exited(state, status)
                case _:
                    assert_never(child)
        case _:
            assert_never(event)


def exit_status(state: Lifecycle) -> int | None:
    match state.python, state.rust:
        case Exited(status), Phase.NOT_STARTED | Exited():
            return status
        case Exited(), Lost():
            return FAILURE
        case _:
            return None
