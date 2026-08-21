"""A tagged-union ``Result`` the type checker can actually narrow.

``Ok`` and ``Error`` are separate frozen classes joined by a ``Union`` alias, so
reaching for ``result.ok`` before eliminating the ``Error`` arm (via ``isinstance``
or a ``match`` pattern) is a type error rather than a runtime ``AttributeError``. A
single class carrying both payload fields would make that unguarded access invisible
to the type checker.

Both variants are covariant and frozen; the absent side defaults to ``Never`` so a
bare ``Ok(value)`` or ``Error(err)`` infers fully and is assignable to any ``Result``
whose matching side fits.

``is_ok`` / ``is_error`` are runtime predicates that also narrow via their ``Literal``
returns; inside strictly typed code, discriminate with ``match`` or ``isinstance``.

This is the shared ``Result`` shape for the ``outbound_credentials`` resolver: every
seam returns ``Result[T, CredError]`` instead of raising, so each failure is a value
the caller must handle rather than an exception that can slip past the type checker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias

from typing_extensions import Never, TypeVar

_TOk_co = TypeVar("_TOk_co", covariant=True, default=Never)
_TError_co = TypeVar("_TError_co", covariant=True, default=Never)


@dataclass(frozen=True)
class Ok(Generic[_TOk_co, _TError_co]):
    ok: _TOk_co

    def is_ok(self) -> Literal[True]:
        return True

    def is_error(self) -> Literal[False]:
        return False


@dataclass(frozen=True)
class Error(Generic[_TOk_co, _TError_co]):
    error: _TError_co

    def is_ok(self) -> Literal[False]:
        return False

    def is_error(self) -> Literal[True]:
        return True


Result: TypeAlias = Ok[_TOk_co, _TError_co] | Error[_TOk_co, _TError_co]
