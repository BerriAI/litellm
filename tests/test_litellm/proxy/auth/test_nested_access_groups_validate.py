"""
Validate + classify coverage for nested access groups (#28032).

Split out of test_nested_access_groups.py to keep file size manageable. The
sister file covers resolve_nested_groups / _get_models_from_access_groups.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
    _classify_member_names,
    validate_models_exist,
)


# ---------------------------------------------------------------------------
# _classify_member_names
# ---------------------------------------------------------------------------


def test_classify_router_model_takes_precedence_over_group():
    """A name registered as both a router model and a group is classified as a model."""
    real, child, unknown = _classify_member_names(
        names=["shared-name"],
        router_model_names={"shared-name"},
        known_access_groups={"shared-name"},
    )
    assert real == ["shared-name"]
    assert child == []
    assert unknown == []


def test_classify_splits_mixed_input():
    """Mixed input is split across the three buckets in order."""
    real, child, unknown = _classify_member_names(
        names=["gpt-4", "image-group", "mystery"],
        router_model_names={"gpt-4"},
        known_access_groups={"image-group"},
    )
    assert real == ["gpt-4"]
    assert child == ["image-group"]
    assert unknown == ["mystery"]


def test_classify_empty_input_returns_three_empty_lists():
    """Edge case: empty names input must not error."""
    real, child, unknown = _classify_member_names(
        names=[],
        router_model_names={"gpt-4"},
        known_access_groups={"g"},
    )
    assert real == [] and child == [] and unknown == []


def test_classify_preserves_input_order():
    """Output preserves caller-provided order within each bucket - matters for error messages."""
    real, child, unknown = _classify_member_names(
        names=["c-unknown", "a-model", "b-group", "d-model"],
        router_model_names={"a-model", "d-model"},
        known_access_groups={"b-group"},
    )
    assert real == ["a-model", "d-model"]
    assert child == ["b-group"]
    assert unknown == ["c-unknown"]


# ---------------------------------------------------------------------------
# validate_models_exist
# ---------------------------------------------------------------------------


def test_validate_models_exist_accepts_known_group_names():
    """Group names are valid members when passed via known_access_groups."""

    class FakeRouter:
        def get_model_names(self):
            return ["gpt-4"]

    all_valid, missing = validate_models_exist(
        model_names=["gpt-4", "image-group"],
        llm_router=FakeRouter(),
        known_access_groups={"image-group"},
    )
    assert all_valid is True
    assert missing == []


def test_validate_models_exist_reports_unknown_names():
    """Names that match neither a router model nor a known group are reported."""

    class FakeRouter:
        def get_model_names(self):
            return ["gpt-4"]

    all_valid, missing = validate_models_exist(
        model_names=["gpt-4", "image-group", "ghost"],
        llm_router=FakeRouter(),
        known_access_groups={"image-group"},
    )
    assert all_valid is False
    assert missing == ["ghost"]


def test_validate_models_exist_backwards_compatible_without_groups():
    """Without known_access_groups, behavior is identical to today's pure-model check."""

    class FakeRouter:
        def get_model_names(self):
            return ["gpt-4"]

    all_valid, missing = validate_models_exist(
        model_names=["gpt-4", "anything-else"],
        llm_router=FakeRouter(),
    )
    assert all_valid is False
    assert missing == ["anything-else"]


def test_validate_models_exist_reports_missing_in_input_order():
    """Missing names are reported in the order they appeared, for human-readable errors."""

    class FakeRouter:
        def get_model_names(self):
            return ["a"]

    all_valid, missing = validate_models_exist(
        model_names=["z-missing", "a", "y-missing"],
        llm_router=FakeRouter(),
        known_access_groups=set(),
    )
    assert all_valid is False
    assert missing == ["z-missing", "y-missing"]


def test_validate_models_exist_with_null_router_still_accepts_known_groups():
    """DB-only deployment: llm_router is None but known_access_groups is still authoritative
    for nested-group composition - only names not in known_groups are reported missing.
    """
    all_valid, missing = validate_models_exist(
        model_names=["image", "reasoning"],
        llm_router=None,
        known_access_groups={"image", "reasoning"},
    )
    assert all_valid is True
    assert missing == []


def test_validate_models_exist_with_null_router_rejects_unknown_real_models():
    """Without a router we can't validate real model names, so anything not in
    known_access_groups is fail-closed reported as missing."""
    all_valid, missing = validate_models_exist(
        model_names=["gpt-4", "image"],
        llm_router=None,
        known_access_groups={"image"},
    )
    assert all_valid is False
    assert missing == ["gpt-4"]
