"""Failures as values.

Nothing in this package raises. Fallible steps return ``Result`` and the
FastAPI layer converts an error to an HTTP response at the edge. Every error
carries a ``.summary`` so the edge has one human-readable string to surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from expression import Result, case, tag, tagged_union
from expression.collections import Block

from .ir import Body, ChatRequest


@dataclass(frozen=True)
class BoundaryError:
    """An untyped payload that did not satisfy a typed boundary.

    ``failures`` lists every field-level problem; ``summary`` joins them so a
    caller never has to reach past the public surface to render the error.
    """

    summary: str
    failures: Block[str]

    @staticmethod
    def of(failures: Block[str]) -> "BoundaryError":
        summary = "; ".join(failures) if len(failures) > 0 else "invalid request"
        return BoundaryError(summary=summary, failures=failures)


@tagged_union(frozen=True)
class TranslationError:
    tag: Literal["boundary", "unsupported"] = tag()

    boundary: BoundaryError = case()
    unsupported: str = case()

    @staticmethod
    def of_boundary(value: BoundaryError) -> "TranslationError":
        return TranslationError(boundary=value)

    @staticmethod
    def of_unsupported(reason: str) -> "TranslationError":
        return TranslationError(unsupported=reason)

    @property
    def summary(self) -> str:
        match self:
            case TranslationError(tag="boundary", boundary=err):
                return err.summary
            case TranslationError(tag="unsupported", unsupported=reason):
                return reason
        return "unknown translation error"


ParseResult = Result[ChatRequest, BoundaryError]

TranslateResult = Result[Body, TranslationError]
