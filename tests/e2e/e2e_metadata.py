"""Per-test metadata for the e2e suite: the step log each test records as it runs.

`steps` is appended at runtime by `@step`-decorated harness helpers, in call
order, so the list IS the test's user story and its last element is where a
failing test died. Nothing about it is hand-written, so it cannot drift from
what the test actually did.

tests/e2e is a black-box HTTP suite that imports litellm in zero files and is
shipped to the runner image as tests/e2e alone, and every harness module imports
this one, so it imports only the stdlib and pydantic.
"""

from __future__ import annotations

import inspect
import os
import re
import string
import threading
from collections import deque
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import AbstractContextManager, contextmanager
from enum import Enum
from functools import reduce, wraps
from types import TracebackType
from typing import Final, ParamSpec, TypeVar, cast

from pydantic import BaseModel

_P = ParamSpec("_P")
_R = TypeVar("_R")
_Y = TypeVar("_Y")

MAX_STEPS: Final = 50
MAX_STEP_CHARS: Final = 200

SECRET_ENV_NAME: Final = re.compile(r"(^|_)(KEY|SECRET|TOKEN|PASSWORD|CREDENTIALS?)(_|$)", re.IGNORECASE)
MIN_SECRET_CHARS: Final = 8
MASK: Final = "***"


def environment_secrets(environ: Mapping[str, str] = os.environ) -> frozenset[str]:
    """The credentials a live run holds: every secret-named environment variable's
    value, long enough that masking it can't blank out ordinary words."""
    return frozenset(
        value for name, value in environ.items() if SECRET_ENV_NAME.search(name) and len(value) >= MIN_SECRET_CHARS
    )


def _masked(label: str, secrets: Iterable[str]) -> str:
    longest_first: Final = sorted(secrets, key=len, reverse=True)
    return reduce(lambda text, secret: text.replace(secret, MASK), longest_first, label)


STEP_FRAMES: Final = 1
"""Frames a `@step` wrapper puts between a helper and its caller. A decorated
helper that warns about its caller adds this to `stacklevel`
(`stacklevel=2 + STEP_FRAMES`), or the warning is reported at the wrapper."""


class StepRecorder:
    """The ordered step log for the running test.

    A plain lock-guarded list rather than a ContextVar: ContextVars do not
    propagate into worker threads, and several e2e helpers call out from
    threads. Under xdist each worker is its own process, so there is no
    cross-test bleed beyond what the per-test reset already handles.
    """

    def __init__(self, secrets: Callable[[], Iterable[str]] = environment_secrets) -> None:
        self._secrets = secrets
        self._lock = threading.Lock()
        self._steps: deque[str] = deque(maxlen=MAX_STEPS)
        self._dropped = 0

    def reset(self) -> None:
        """Called first thing in every test's setup phase, so each test starts
        empty."""
        with self._lock:
            self._steps.clear()
            self._dropped = 0

    def record(self, label: str) -> None:
        """Append `label`, unless it repeats the previous step.

        A retrying helper (poll_cost_row) or a load test calling a decorated
        helper in a loop would otherwise emit thousands of <property> entries per
        testcase: a consecutive repeat collapses, so a poll loop is one step in
        the story rather than fifty, and past MAX_STEPS the oldest step makes way.
        It is the oldest that goes because the last step is the one that has to
        survive: it is where a failing test died.

        Any credential the run holds is masked before the label is kept, however it
        got into the label, since the steps are published with the results.
        """
        cleaned = " ".join(_masked(label, self._secrets()).split())[:MAX_STEP_CHARS]
        if not cleaned:
            return
        with self._lock:
            if self._steps and self._steps[-1] == cleaned:
                return
            if len(self._steps) == MAX_STEPS:
                self._dropped += 1
            self._steps.append(cleaned)

    def taken(self) -> tuple[str, ...]:
        """The story so far, led by a line counting the steps a full log dropped,
        so a story that starts mid-test says so rather than reading as complete."""
        with self._lock:
            dropped: Final = (f"({self._dropped} earlier steps not recorded)",) if self._dropped else ()
            return dropped + tuple(self._steps)


STEPS: Final = StepRecorder()


def _joined(phrases: tuple[str, ...]) -> str:
    if len(phrases) <= 1:
        return "".join(phrases)
    return f"{', '.join(phrases[:-1])} and {phrases[-1]}"


def _model_phrase(model: BaseModel) -> str:
    """The fields the caller set, as "models: a, b and rpm limit: 3". A
    `Field(repr=False)` field, pydantic's flag for a secret, is never shown."""
    values: Final = (
        (name, cast("object", getattr(model, name)))
        for name, field in type(model).model_fields.items()
        if name in model.model_fields_set and field.repr
    )
    phrases: Final = tuple(f"{name.replace('_', ' ')}: {_phrase(value)}" for name, value in values if _given(value))
    return _joined(phrases) or "default settings"


def _given(value: object) -> bool:
    return value is not None and value != [] and value != ()


def _phrase(value: object) -> str:
    if isinstance(value, BaseModel):
        return _model_phrase(value)
    if isinstance(value, Enum):
        return _phrase(cast("object", value.value))
    if isinstance(value, Mapping):
        entries: Final = cast("Mapping[object, object]", value)
        return _joined(tuple(f"{str(key).replace('_', ' ')}: {_phrase(item)}" for key, item in entries.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return ", ".join(map(_phrase, cast("Iterable[object]", value)))
    return str(value)


_PLACEHOLDER: Final = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")


def _placeholders(label: str) -> frozenset[str]:
    return frozenset(field for _, field, _, _ in string.Formatter().parse(label) if field is not None)


def _resolved(field: str, arguments: Mapping[str, object]) -> object:
    """`body.litellm_params.model` is the `body` argument's `litellm_params.model`."""
    root, *attributes = field.split(".")
    return reduce(lambda value, attribute: cast("object", getattr(value, attribute)), attributes, arguments[root])


def _filled(label: str, bound: inspect.BoundArguments) -> str:
    bound.apply_defaults()
    arguments: Final = cast("Mapping[str, object]", bound.arguments)
    return "".join(
        literal + ("" if field is None else _phrase(_resolved(field, arguments)))
        for literal, field, _, _ in string.Formatter().parse(label)
    )


class _Nesting(threading.local):
    """Whether this thread is already inside a `@step` helper.

    Per thread, like the helpers themselves: a worker thread a step fans out to
    starts outside any step, so its own decorated calls still record."""

    def __init__(self) -> None:
        self.inside: bool = False


_NESTING: Final = _Nesting()


@contextmanager
def _inside_step() -> Generator[None]:
    """Hold the nesting guard for the duration, restoring whatever it was."""
    outer: Final = _NESTING.inside
    _NESTING.inside = True
    try:
        yield
    finally:
        _NESTING.inside = outer


class _StepContext(AbstractContextManager[_Y]):
    """A `@contextmanager` helper's context, entered and exited inside its step.

    Calling a `@contextmanager` function runs none of its body: the setup runs at
    `__enter__` and the cleanup at `__exit__`, both after the call has returned
    and so both outside the guard the call held. Here each runs inside it, so the
    helpers they call stay out of the story, while the `with` body in between --
    the test's own code -- still records. Without this, a test that died inside
    the `with` would have the cleanup's steps appended behind the one it died on.
    """

    def __init__(self, inner: AbstractContextManager[_Y]) -> None:
        self._inner: Final = inner

    def __enter__(self) -> _Y:
        with _inside_step():
            return self._inner.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        with _inside_step():
            return self._inner.__exit__(exc_type, exc, traceback)


def step(label: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Record `label` on the running test whenever this helper is called.

    Goes on HARNESS helpers (client methods, fixtures), never on tests. The
    label is recorded BEFORE the wrapped call, so a helper that raises still
    leaves its own label as the last element -- which is the whole point: the
    last step is where the test died.

    Only the outermost step records. Harness layers call each other --
    `ProxyClient.create_model` goes through `register_model`, a domain
    client wraps the shared `ProxyClient` -- so every layer can carry its own
    label without one action showing up in the story once per layer. The story
    reads at the level the test called in at, and the label of the helper the
    test called is still the last one when anything beneath it raises.

    On a `@contextmanager` helper `@step` goes ABOVE `@contextmanager`, and the
    setup and cleanup around its `yield` count as part of the step (see
    `_StepContext`). A bare generator function is refused where the decorator
    runs: its body only runs as the caller iterates, interleaved with the
    caller's own steps, so no single point in the story is where it happened.
    """

    def decorate(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        signature: Final = inspect.signature(fn)
        placeholders: Final = _placeholders(label)
        malformed: Final = sorted(field for field in placeholders if not _PLACEHOLDER.fullmatch(field))
        if malformed:
            raise TypeError(f"@step({label!r}) has {malformed}: a placeholder is a parameter or its dotted attribute")
        unknown: Final = {field.split(".")[0] for field in placeholders} - signature.parameters.keys()
        if unknown:
            raise TypeError(f"@step({label!r}) names {sorted(unknown)}, which {fn.__qualname__} doesn't take")
        static_label: Final = None if placeholders else label.format()
        if inspect.isgeneratorfunction(fn):
            raise TypeError(
                f"@step({label!r}) cannot wrap the generator function {fn!r}: put it on a helper that"
                " returns, or above @contextmanager on one that yields a context"
            )
        underlying: Final[object] = inspect.unwrap(fn)  # pyright: ignore[reportAny]  # inspect.unwrap is typed as returning Any
        opens_a_context: Final = inspect.isgeneratorfunction(underlying)

        @wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if not _NESTING.inside:
                STEPS.record(static_label or _filled(label, signature.bind(*args, **kwargs)))
            with _inside_step():
                result = fn(*args, **kwargs)
            if opens_a_context and isinstance(result, AbstractContextManager):
                context: Final = cast("AbstractContextManager[object]", result)
                return cast("_R", _StepContext(context))
            return result

        return wrapper

    return decorate


def step_properties() -> tuple[tuple[str, str], ...]:
    """The step log as repeated `step` properties. Appended after the setup and
    call phases, never at collection."""
    return tuple(("step", label) for label in STEPS.taken())
