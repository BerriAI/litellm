"""
Unit tests for nested access group resolution (#28032).

Covers:
- resolve_nested_groups (DFS + cycle detection)
- _get_models_from_access_groups with the new group_memberships kwarg
- _classify_member_names precedence
- validate_models_exist with known_access_groups
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

from litellm.proxy.auth.model_checks import (
    _get_models_from_access_groups,
    resolve_nested_groups,
)
from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
    _classify_member_names,
    validate_models_exist,
)


# ---------------------------------------------------------------------------
# resolve_nested_groups + _get_models_from_access_groups
# ---------------------------------------------------------------------------


def test_resolve_flat_group_unchanged():
    """When group_memberships is empty, behavior must match today's flat path."""
    model_access_groups = {"prod": ["gpt-4", "claude-3-opus"]}
    all_models = ["prod", "gemini-pro"]

    result = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=list(all_models),
    )

    assert sorted(result) == sorted(["gpt-4", "claude-3-opus", "gemini-pro"])


def test_resolve_single_level_nested_group():
    """A parent group whose child has direct models expands to those models."""
    model_access_groups = {"image": ["dall-e-3", "stable-diffusion-xl"]}
    group_memberships = {"project-x": ["image"]}
    all_models = ["project-x"]

    result = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=list(all_models),
        group_memberships=group_memberships,
    )

    assert sorted(result) == sorted(["dall-e-3", "stable-diffusion-xl"])


def test_resolve_three_level_deep_nested_group():
    """A -> B -> C -> [models] resolves through three hops."""
    model_access_groups = {"C": ["gpt-4"]}
    group_memberships = {"A": ["B"], "B": ["C"]}

    result = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=["A"],
        group_memberships=group_memberships,
    )

    assert result == ["gpt-4"]


def test_detect_and_skip_self_referential_cycle(caplog):
    """A -> A logs a warning and returns A's direct models without exploding."""
    model_access_groups = {"A": ["gpt-4"]}
    group_memberships = {"A": ["A"]}

    with caplog.at_level("WARNING"):
        result = resolve_nested_groups(
            group_name="A",
            model_access_groups=model_access_groups,
            group_memberships=group_memberships,
            visited=set(),
        )

    assert result == ["gpt-4"]
    assert any("cycle detected" in m.lower() for m in caplog.messages)


def test_detect_and_skip_indirect_cycle(caplog):
    """A -> B -> A logs a warning and returns the union of both direct sets."""
    model_access_groups = {"A": ["gpt-4"], "B": ["claude-3"]}
    group_memberships = {"A": ["B"], "B": ["A"]}

    with caplog.at_level("WARNING"):
        result = resolve_nested_groups(
            group_name="A",
            model_access_groups=model_access_groups,
            group_memberships=group_memberships,
            visited=set(),
        )

    assert sorted(set(result)) == sorted(["gpt-4", "claude-3"])
    assert any("cycle detected" in m.lower() for m in caplog.messages)


def test_propagation_to_parent():
    """Adding a model to a child group surfaces on the parent's resolved list."""
    # Initial: parent resolves to whatever the child has at the moment of call
    model_access_groups = {"image": ["dall-e-3"]}
    group_memberships = {"project-x": ["image"]}

    before = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=["project-x"],
        group_memberships=group_memberships,
    )
    assert before == ["dall-e-3"]

    # Add a new model to the child group; parent must see it on the next resolve
    model_access_groups["image"].append("stable-diffusion-xl")
    after = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=["project-x"],
        group_memberships=group_memberships,
    )
    assert sorted(after) == sorted(["dall-e-3", "stable-diffusion-xl"])


def test_pure_composition_group_resolves():
    """A group present only in memberships (no deployment tags) still resolves."""
    model_access_groups = {"image": ["dall-e-3"], "reasoning": ["o1"]}
    group_memberships = {"project-x": ["image", "reasoning"]}

    result = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=["project-x"],
        group_memberships=group_memberships,
    )

    assert sorted(result) == sorted(["dall-e-3", "o1"])


def test_include_model_access_groups_keeps_parent_name():
    """When include_model_access_groups=True, the group name is preserved alongside expansion."""
    model_access_groups = {"image": ["dall-e-3"]}
    group_memberships = {"project-x": ["image"]}

    result = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=["project-x"],
        include_model_access_groups=True,
        group_memberships=group_memberships,
    )

    assert "project-x" in result
    assert "dall-e-3" in result


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


# ---------------------------------------------------------------------------
# Smoke: chained scenarios
# ---------------------------------------------------------------------------


def test_dag_shared_group_subtree_no_false_cycle_warning(caplog):
    """
    DAG with a shared group subtree:
        A -> [B, C]
        B -> [D]
        C -> [D]
        D -> [model-1]

    On-path cycle tracking lets D be re-traversed via both B and C without
    a spurious 'cycle detected' warning. Caller dedups.
    """
    model_access_groups = {"D": ["model-1"]}
    group_memberships = {"A": ["B", "C"], "B": ["D"], "C": ["D"]}

    with caplog.at_level("WARNING"):
        result = resolve_nested_groups(
            group_name="A",
            model_access_groups=model_access_groups,
            group_memberships=group_memberships,
            visited=set(),
        )

    assert sorted(set(result)) == ["model-1"]
    assert not any(
        "cycle detected" in m.lower() for m in caplog.messages
    ), "DAG-shared subtree must not trigger cycle warnings"


def test_diamond_inheritance_resolves_once():
    """
    Diamond shape:
        project-x -> [image, reasoning]
        image     -> [dall-e-3]
        reasoning -> [dall-e-3, o1]   (dall-e-3 deliberately shared)

    Resolution must include all models. Duplicates are caller's job to dedup.
    """
    model_access_groups = {"image": ["dall-e-3"], "reasoning": ["dall-e-3", "o1"]}
    group_memberships = {"project-x": ["image", "reasoning"]}

    result = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=["project-x"],
        group_memberships=group_memberships,
    )

    # set() removes the legitimate duplicate, all 2 unique models should be present
    assert sorted(set(result)) == sorted(["dall-e-3", "o1"])
