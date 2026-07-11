"""RFC 7386 JSON Merge Patch (https://www.rfc-editor.org/rfc/rfc7386)."""

from pydantic import JsonValue


def apply_json_merge_patch(target: JsonValue, patch: JsonValue) -> JsonValue:
    """Apply an RFC 7386 JSON Merge Patch to ``target`` and return the result.

    - a key absent from ``patch`` keeps its value in ``target``
    - a key mapped to ``null`` in ``patch`` is removed from the result
    - any other value overwrites, recursing into nested objects

    ``target`` is never mutated; a new value is returned.
    """
    if not isinstance(patch, dict):
        return patch

    base = target if isinstance(target, dict) else {}
    preserved = {key: value for key, value in base.items() if key not in patch}
    applied = {key: apply_json_merge_patch(base.get(key), value) for key, value in patch.items() if value is not None}
    return {**preserved, **applied}
