import copy

import pytest

from litellm.proxy.common_utils.json_merge_patch import _MAX_MERGE_DEPTH, apply_json_merge_patch

# RFC 7386 Appendix A — the normative test suite for JSON Merge Patch.
# https://www.rfc-editor.org/rfc/rfc7386#appendix-A
RFC_7386_APPENDIX_A = [
    ({"a": "b"}, {"a": "c"}, {"a": "c"}),
    ({"a": "b"}, {"b": "c"}, {"a": "b", "b": "c"}),
    ({"a": "b"}, {"a": None}, {}),
    ({"a": "b", "b": "c"}, {"a": None}, {"b": "c"}),
    ({"a": ["b"]}, {"a": "c"}, {"a": "c"}),
    ({"a": "c"}, {"a": ["b"]}, {"a": ["b"]}),
    ({"a": {"b": "c"}}, {"a": {"b": "d", "c": None}}, {"a": {"b": "d"}}),
    ({"a": [{"b": "c"}]}, {"a": [1]}, {"a": [1]}),
    (["a", "b"], ["c", "d"], ["c", "d"]),
    ({"a": "b"}, ["c"], ["c"]),
    ({"a": "foo"}, None, None),
    ({"a": "foo"}, "bar", "bar"),
    ({"e": None}, {"a": 1}, {"e": None, "a": 1}),
    ([1, 2], {"a": "b", "c": None}, {"a": "b"}),
    ({}, {"a": {"bb": {"ccc": None}}}, {"a": {"bb": {}}}),
]


@pytest.mark.parametrize("target, patch, expected", RFC_7386_APPENDIX_A)
def test_rfc_7386_appendix_a(target, patch, expected):
    assert apply_json_merge_patch(target, patch) == expected


def test_does_not_mutate_target():
    """The target must be treated as immutable — a fresh value is returned."""
    target = {"keep": "me", "nested": {"a": 1, "b": 2}, "drop": "later"}
    target_snapshot = copy.deepcopy(target)

    result = apply_json_merge_patch(target, {"nested": {"b": None, "c": 3}, "drop": None})

    assert target == target_snapshot, "apply_json_merge_patch mutated its target argument"
    assert result == {"keep": "me", "nested": {"a": 1, "c": 3}}
    assert result["nested"] is not target["nested"]


def test_absent_key_is_preserved_but_null_key_is_deleted():
    """The core distinction PATCH relies on: omission preserves, explicit null deletes."""
    target = {"cost_center": "1234", "team": "core"}

    # Omitting cost_center preserves it; only the explicitly-null key is removed.
    assert apply_json_merge_patch(target, {"team": "platform"}) == {
        "cost_center": "1234",
        "team": "platform",
    }
    assert apply_json_merge_patch(target, {"cost_center": None}) == {"team": "core"}


def test_deep_nested_merge_and_delete():
    target = {"limits": {"gpt-4": {"rpm": 100, "tpm": 1000}, "gpt-3.5": {"rpm": 200}}}
    patch = {"limits": {"gpt-4": {"tpm": 2000}, "gpt-3.5": None, "claude": {"rpm": 50}}}

    assert apply_json_merge_patch(target, patch) == {
        "limits": {"gpt-4": {"rpm": 100, "tpm": 2000}, "claude": {"rpm": 50}}
    }


def test_scalar_patch_replaces_object_wholesale():
    assert apply_json_merge_patch({"a": {"b": 1}}, 5) == 5


def test_object_patch_over_non_object_target_starts_from_empty():
    assert apply_json_merge_patch("not-an-object", {"a": 1, "b": None}) == {"a": 1}


def _nest(levels: int) -> dict:
    """A patch nested ``levels`` dicts deep with a scalar leaf at the bottom."""
    value: object = "leaf"
    for _ in range(levels):
        value = {"a": value}
    return value  # type: ignore[return-value]


def test_merge_within_max_depth_is_allowed():
    """A deeply-but-not-pathologically nested patch merges without raising."""
    result = apply_json_merge_patch({}, _nest(_MAX_MERGE_DEPTH - 1))
    for _ in range(_MAX_MERGE_DEPTH - 1):
        result = result["a"]
    assert result == "leaf"


def test_merge_beyond_max_depth_raises():
    """A patch nested past the cap fails closed (ValueError) rather than
    overflowing the Python stack — the guard the recursion detector requires."""
    with pytest.raises(ValueError, match="maximum depth"):
        apply_json_merge_patch({}, _nest(_MAX_MERGE_DEPTH + 5))
