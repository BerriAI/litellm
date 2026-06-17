"""A tagged union the type checker can actually see.

``Ok`` and ``Error`` are separate classes joined by a ``Union`` alias, so reaching for
``result.ok`` before eliminating the ``Error`` arm (via ``isinstance`` or a ``match``
pattern) is a pyright error, not a runtime ``AttributeError``. The single-class
``expression.Result`` this replaces declared both payload fields on one class, which made
unguarded access invisible to tooling.

Both variants are covariant and frozen; the absent side defaults to ``Never`` so a bare
``Ok(value)`` or ``Error(err)`` infers fully and is assignable to any ``Result`` whose
matching side fits.

``is_ok``/``is_error`` are runtime predicates only; they do not narrow the union for the
type checker. Inside the strictly-checked v2 tree, discriminate with ``match`` or
``isinstance``.

Adopted verbatim from the sibling v2 package ``litellm/translation/result.py`` so the two
clean-room efforts share one ``Result`` shape and one rationale.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias

from typing_extensions import Never, TypeVar

_TOk_co = TypeVar("_TOk_co", covariant=True, default=Never)
_TError_co = TypeVar("_TError_co", covariant=True, default=Never)
_TMapped = TypeVar("_TMapped")
_TBindError = TypeVar("_TBindError")


@dataclass(frozen=True)
class Ok(Generic[_TOk_co, _TError_co]):
    ok: _TOk_co

    def is_ok(self) -> Literal[True]:
        return True

    def is_error(self) -> Literal[False]:
        return False

    def map(self, mapper: Callable[[_TOk_co], _TMapped]) -> Ok[_TMapped, _TError_co]:
        return Ok(mapper(self.ok))

    def bind(
        self, mapper: Callable[[_TOk_co], Result[_TMapped, _TBindError]]
    ) -> Result[_TMapped, _TBindError]:
        return mapper(self.ok)


@dataclass(frozen=True)
class Error(Generic[_TOk_co, _TError_co]):
    error: _TError_co

    def is_ok(self) -> Literal[False]:
        return False

    def is_error(self) -> Literal[True]:
        return True

    def map(self, mapper: Callable[[_TOk_co], _TMapped]) -> Error[_TMapped, _TError_co]:
        return Error(self.error)

    def bind(
        self, mapper: Callable[[_TOk_co], Result[_TMapped, _TBindError]]
    ) -> Error[_TMapped, _TError_co]:
        return Error(self.error)


Result: TypeAlias = Ok[_TOk_co, _TError_co] | Error[_TOk_co, _TError_co]
