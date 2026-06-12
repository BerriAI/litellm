"""Failures as values.

Nothing in this package raises. Fallible steps return ``Result`` and the
dispatch seam converts an error at the edge: a ``boundary`` or ``unsupported``
error means "this request is outside v2's proven surface", and the seam falls
back to v1 so no feature is ever silently lost. Every error carries a
``.summary`` so the edge has one human-readable string to surface or log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from expression import Result, case, tag, tagged_union
from expression.collections import Block
from typing_extensions import assert_never

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
    def of(failures: Block[str]) -> BoundaryError:
        summary = "; ".join(failures) if len(failures) > 0 else "invalid request"
        return BoundaryError(summary=summary, failures=failures)


@tagged_union(frozen=True)
class TranslationError:
    tag: Literal["boundary", "unsupported"] = tag()

    boundary: BoundaryError = case()
    unsupported: str = case()

    @staticmethod
    def of_boundary(value: BoundaryError) -> TranslationError:
        return TranslationError(boundary=value)

    @staticmethod
    def of_unsupported(reason: str) -> TranslationError:
        return TranslationError(unsupported=reason)

    @property
    def summary(self) -> str:
        # Exhaustiveness: every Literal tag has an arm; the trailing
        # assert_never typechecks only while that stays true (the package's
        # standard pattern -- a `case never:` capture arm is flagged by
        # pyright strict as an unmatchable pattern).
        match self.tag:
            case "boundary":
                return self.boundary.summary
            case "unsupported":
                return self.unsupported
        assert_never(self.tag)


ParseResult = Result[ChatRequest, TranslationError]

TranslateResult = Result[Body, TranslationError]
