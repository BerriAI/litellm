"""The typed boundary.

Untyped data enters the package here and nowhere else: the FastAPI body dict
on the way in, provider response JSON on the way out. ``freeze`` lifts plain
JSON into the immutable ``Block``/``Map`` representation the IR is built from,
``thaw`` lowers it back to the plain ``list``/``dict`` a serializer emits, and
the ``as_*`` accessors narrow an ``object`` to a concrete shape as an ``Option``
so callers branch on presence instead of guarding with ``isinstance`` inline.
"""

from __future__ import annotations

from typing import Dict, List, cast

from expression import Nothing, Option, Some
from expression.collections import Block, Map

from .ir import Json, PlainJson


def freeze(value: object) -> Json:
    if isinstance(value, dict):
        return Map.of_seq((str(key), freeze(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return Block.of_seq(freeze(item) for item in value)
    return cast(Json, value)


def thaw(value: Json) -> PlainJson:
    if isinstance(value, Map):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, Block):
        return [thaw(item) for item in value]
    return value


def as_str(value: object) -> Option[str]:
    return Some(value) if isinstance(value, str) else Nothing


def as_mapping(value: object) -> Option[Dict[str, object]]:
    return Some(value) if isinstance(value, dict) else Nothing


def as_sequence(value: object) -> Option[List[object]]:
    return Some(value) if isinstance(value, list) else Nothing
