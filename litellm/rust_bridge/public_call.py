"""Bind a public LiteLLM call to its legacy Python signature without running it."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeVar, cast  # noqa: TID251  # narrows caller-owned containers without copying them


def signature(legacy: Callable[..., object]) -> inspect.Signature:
    return inspect.signature(legacy)


def bind(
    legacy: inspect.Signature, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> Mapping[str, object] | None:
    try:
        bound: Final = legacy.bind(*args, **kwargs)
    except TypeError:
        return None
    bound.apply_defaults()
    return bound.arguments


def optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def optional_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return cast("Mapping[str, object]", value)  # cast-ok: the same caller-owned object is handed on unchanged


def optional_sequence(value: object) -> Sequence[object] | None:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        return None
    return cast("Sequence[object]", value)  # cast-ok: the same caller-owned object is handed on unchanged


def known_params(names: Collection[str], request: Mapping[str, object]) -> Mapping[str, object]:
    return {name: value for name, value in request.items() if name in names and value is not None}


_BAGS: Final = frozenset({inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD})


@dataclass(frozen=True, slots=True)
class NativeCall:
    """A public call as Python bound it.

    ``base`` is the positional arguments by name plus the signature defaults, with no caller
    keyword in it. ``kwargs`` is the caller's keyword dict, which callbacks may rewrite before
    the request is decoded. The call the public function sees is ``kwargs`` laid over ``base``.
    """

    args: tuple[object, ...]
    kwargs: Mapping[str, object]
    base: Mapping[str, object]

    @property
    def resolved(self) -> Mapping[str, object]:
        return MappingProxyType({**self.base, **self.kwargs})


def _without_bags(legacy: inspect.Signature, named: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({name: value for name, value in named.items() if legacy.parameters[name].kind not in _BAGS})


@dataclass(frozen=True, slots=True)
class AsyncFlag:
    """The keyword that makes a sync entry hand off to its async entry, as in ``completion(..., acompletion=True)``.

    ``is_set`` is the route's own reading of the flag. ``@client`` hands off on ``is True``, the Messages handler on truthiness.

    Completion uses ``AsyncFlag("acompletion", sync=signature(_PYTHON_COMPLETION))``.
    Messages uses ``AsyncFlag("is_async", sync=signature(_PYTHON_MESSAGES), is_set=bool)``.
    """

    name: str
    sync: inspect.Signature
    is_set: Callable[[object], bool] = lambda value: value is True

    def requested(self, kwargs: Mapping[str, object]) -> bool:
        return self.is_set(kwargs.get(self.name))

    def dropped(self, kwargs: Mapping[str, object]) -> Mapping[str, object]:
        return MappingProxyType({name: value for name, value in kwargs.items() if name != self.name})

    def keywords(self, args: tuple[object, ...], kwargs: Mapping[str, object]) -> Mapping[str, object]:
        """Name positionals by the sync signature, since the async entry may order its parameters differently."""
        named: Final = _without_bags(self.sync, self.sync.bind_partial(*args, **kwargs).arguments)
        return self.dropped({**named, **kwargs})


def native_call(legacy: inspect.Signature, args: tuple[object, ...], kwargs: Mapping[str, object]) -> NativeCall:
    positional: Final = legacy.bind_partial(*args)
    positional.apply_defaults()
    return NativeCall(args=args, kwargs=kwargs, base=_without_bags(legacy, positional.arguments))


NativeResultT: Final = TypeVar("NativeResultT")


def native_call_hook(
    hook: Callable[[NativeCall], NativeResultT],
    call: NativeCall,
    _args: tuple[object, ...],
    _kwargs: Mapping[str, object],
) -> NativeResultT:
    return hook(call)
