"""Bind a public LiteLLM call to its legacy Python signature without running it."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Final, cast  # noqa: TID251  # narrows caller-owned containers without copying them


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
