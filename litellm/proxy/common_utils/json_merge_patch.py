"""RFC 7386 JSON Merge Patch (https://www.rfc-editor.org/rfc/rfc7386)."""

from collections.abc import Mapping
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping

# A merge patch recurses as deep as the client's JSON nests. Cap it far above any
# realistic team-metadata shape but well below Python's stack limit, so a
# pathologically deep patch is rejected instead of overflowing the stack.
_MAX_MERGE_DEPTH: Final = 64


def _json_object(value: object) -> Mapping[str, object] | None:
    mapping: Final = as_str_mapping(value)
    return mapping if isinstance(value, dict) else None


def apply_json_merge_patch(target: object, patch: object, _depth: int = 0) -> object:
    """Apply an RFC 7386 JSON Merge Patch to ``target`` and return the result.

    - a key absent from ``patch`` keeps its value in ``target``
    - a key mapped to ``null`` in ``patch`` is removed from the result
    - any other value overwrites, recursing into nested objects

    ``target`` is never mutated; a new value is returned. Raises ``ValueError``
    if ``patch`` nests deeper than ``_MAX_MERGE_DEPTH``.
    """
    patch_object: Final = _json_object(patch)
    if patch_object is None:
        return patch
    if _depth >= _MAX_MERGE_DEPTH:
        raise ValueError(f"JSON merge patch nesting exceeds the maximum depth of {_MAX_MERGE_DEPTH}")

    base: Final = _json_object(target) or {}
    preserved: Final = {key: value for key, value in base.items() if key not in patch_object}
    applied: Final = {
        key: apply_json_merge_patch(base.get(key), value, _depth + 1)
        for key, value in patch_object.items()
        if value is not None
    }
    return {**preserved, **applied}
