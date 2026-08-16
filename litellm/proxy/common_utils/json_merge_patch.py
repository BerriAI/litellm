"""RFC 7386 JSON Merge Patch (https://www.rfc-editor.org/rfc/rfc7386)."""

from pydantic import JsonValue

# A merge patch recurses as deep as the client's JSON nests. Cap it far above any
# realistic team-metadata shape but well below Python's stack limit, so a
# pathologically deep patch is rejected instead of overflowing the stack.
_MAX_MERGE_DEPTH = 64


def apply_json_merge_patch(target: JsonValue, patch: JsonValue, _depth: int = 0) -> JsonValue:
    """Apply an RFC 7386 JSON Merge Patch to ``target`` and return the result.

    - a key absent from ``patch`` keeps its value in ``target``
    - a key mapped to ``null`` in ``patch`` is removed from the result
    - any other value overwrites, recursing into nested objects

    ``target`` is never mutated; a new value is returned. Raises ``ValueError``
    if ``patch`` nests deeper than ``_MAX_MERGE_DEPTH``.
    """
    if not isinstance(patch, dict):
        return patch
    if _depth >= _MAX_MERGE_DEPTH:
        raise ValueError(f"JSON merge patch nesting exceeds the maximum depth of {_MAX_MERGE_DEPTH}")

    base = target if isinstance(target, dict) else {}
    preserved = {key: value for key, value in base.items() if key not in patch}
    applied = {
        key: apply_json_merge_patch(base.get(key), value, _depth + 1)
        for key, value in patch.items()
        if value is not None
    }
    return {**preserved, **applied}
