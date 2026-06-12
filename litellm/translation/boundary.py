"""The typed boundary.

Untyped data enters the package here and nowhere else: the FastAPI body dict
on the way in, provider response JSON on the way out. ``parse`` follows the
arktype calling convention over frozen pydantic v2 models: hand it a model
class and a raw payload and it returns the validated model or an error value
listing every field failure, never raising. Models are declared with
``extra="forbid"`` so an inbound field the schema does not account for is a
typed ``unsupported`` error (the dispatch seam falls back to v1 on it), never
a silent drop. ``as_plain_json`` admits opaque JSON sub-trees (tool arguments,
JSON schemas) by checking every leaf and deep-copying, so a ``JsonBlob`` never
aliases caller-owned data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TypeVar

from expression.collections import Block
from pydantic import BaseModel, ValidationError

from .errors import BoundaryError, TranslationError
from .ir import PlainJson
from .result import Error, Ok, Result

_TModel = TypeVar("_TModel", bound=BaseModel)

_UNSUPPORTED_ERROR_TYPES = frozenset({"extra_forbidden", "union_tag_invalid"})


def parse(
    model_cls: type[_TModel], raw: Mapping[str, object]
) -> Result[_TModel, TranslationError]:
    """Validate ``raw`` against ``model_cls``, accumulating every failure.

    Unknown fields (``extra_forbidden``) and unrecognized union tags become an
    ``unsupported`` error so the seam can fall back to v1; everything else is
    a ``boundary`` error. Both carry every individual failure, not just the
    first.
    """
    try:
        return Ok(model_cls.model_validate(raw))
    except ValidationError as exc:
        errors = exc.errors(include_url=False)
        locations = [
            (".".join(str(part) for part in err["loc"]), err) for err in errors
        ]
        unsupported = [
            location or "<root>"
            for location, err in locations
            if err["type"] in _UNSUPPORTED_ERROR_TYPES
        ]
        failures = [
            f"{location or '<root>'}: {err['msg']}"
            for location, err in locations
            if err["type"] not in _UNSUPPORTED_ERROR_TYPES
        ]
        if unsupported:
            fields = ", ".join(sorted(set(unsupported)))
            return Error(
                TranslationError.of_unsupported(
                    f"fields not yet supported by translation v2: {fields}"
                )
            )
        return Error(
            TranslationError.of_boundary(BoundaryError.of(Block.of_seq(failures)))
        )


def as_plain_json(value: object) -> Result[PlainJson, str]:
    """Admit an opaque JSON sub-tree: checks every leaf and returns a deep copy.

    A C-speed ``dumps``/``loads`` round-trip both proves the value is plain
    JSON and produces a copy that shares no structure with caller-owned data,
    so later (immutable) use inside the package cannot observe caller mutation
    and emitting it cannot leak package state back to the caller. Key order is
    preserved, matching what v1 sends on the wire.
    """
    try:
        return Ok(json.loads(json.dumps(value)))
    except (TypeError, ValueError) as exc:
        return Error(f"not plain JSON: {exc}")
