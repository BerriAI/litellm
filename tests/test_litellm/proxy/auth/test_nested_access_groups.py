"""
Unit tests for nested access group resolution (#28032).

Covers:
- resolve_nested_groups (DFS + cycle detection)
- _get_models_from_access_groups with the new group_memberships kwarg

Classify + validate coverage lives in test_nested_access_groups_validate.py.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

from litellm.proxy.auth.model_checks import (
    _get_models_from_access_groups,
    resolve_nested_groups,
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


# ---------------------------------------------------------------------------
# Brutal edge cases
# ---------------------------------------------------------------------------


def test_all_models_empty_returns_empty():
    """Empty all_models input produces empty output."""
    result = _get_models_from_access_groups(
        model_access_groups={"g": ["gpt-4"]},
        all_models=[],
        group_memberships={"g": []},
    )
    assert result == []


def test_no_groups_no_memberships_passes_through():
    """When neither map matches anything, all_models is returned untouched."""
    result = _get_models_from_access_groups(
        model_access_groups={},
        all_models=["gpt-4", "claude-3"],
        group_memberships={},
    )
    assert result == ["gpt-4", "claude-3"]


def test_resolve_group_with_no_direct_models_and_no_children():
    """A dangling group key with empty children expands to empty list."""
    result = resolve_nested_groups(
        group_name="empty-group",
        model_access_groups={"empty-group": []},
        group_memberships={"empty-group": []},
        visited=set(),
    )
    assert result == []


def test_resolve_unknown_group_returns_empty():
    """Resolving a name that's in neither map (orphan) returns empty, no crash."""
    result = resolve_nested_groups(
        group_name="orphan",
        model_access_groups={},
        group_memberships={},
        visited=set(),
    )
    assert result == []


def test_membership_pointing_to_missing_child_is_safe():
    """A parent referencing a child that has no direct models and no further children expands to empty."""
    result = resolve_nested_groups(
        group_name="parent",
        model_access_groups={},
        group_memberships={"parent": ["ghost-child"]},
        visited=set(),
    )
    assert result == []


def test_three_hop_cycle_logs_once_per_back_edge(caplog):
    """A -> B -> C -> A is detected when the back-edge to A is attempted."""
    model_access_groups = {"A": ["m-a"], "B": ["m-b"], "C": ["m-c"]}
    group_memberships = {"A": ["B"], "B": ["C"], "C": ["A"]}

    with caplog.at_level("WARNING"):
        result = resolve_nested_groups(
            group_name="A",
            model_access_groups=model_access_groups,
            group_memberships=group_memberships,
            visited=set(),
        )

    assert sorted(set(result)) == sorted(["m-a", "m-b", "m-c"])
    assert sum("cycle detected" in m.lower() for m in caplog.messages) >= 1


def test_self_reference_in_deep_chain_only_warns_for_self_edge(caplog):
    """A -> B -> C -> C: the leaf self-loop fires the warning; A/B traversal completes."""
    model_access_groups = {"C": ["m-c"]}
    group_memberships = {"A": ["B"], "B": ["C"], "C": ["C"]}

    with caplog.at_level("WARNING"):
        result = resolve_nested_groups(
            group_name="A",
            model_access_groups=model_access_groups,
            group_memberships=group_memberships,
            visited=set(),
        )

    assert result == ["m-c"]
    cycle_msgs = [m for m in caplog.messages if "cycle detected" in m.lower()]
    assert len(cycle_msgs) == 1
    assert "'C'" in cycle_msgs[0]


def test_deep_linear_chain_50_levels():
    """A 50-level linear chain resolves without stack overflow and returns the leaf model."""
    model_access_groups = {"g49": ["leaf-model"]}
    group_memberships = {f"g{i}": [f"g{i+1}"] for i in range(49)}

    result = resolve_nested_groups(
        group_name="g0",
        model_access_groups=model_access_groups,
        group_memberships=group_memberships,
        visited=set(),
    )
    assert result == ["leaf-model"]


def test_wide_fanout_one_parent_100_children():
    """A single parent with 100 children, each holding one unique model, resolves all 100."""
    children = [f"child-{i}" for i in range(100)]
    model_access_groups = {c: [f"model-{i}"] for i, c in enumerate(children)}
    group_memberships = {"root": children}

    result = resolve_nested_groups(
        group_name="root",
        model_access_groups=model_access_groups,
        group_memberships=group_memberships,
        visited=set(),
    )
    assert sorted(result) == sorted([f"model-{i}" for i in range(100)])


def test_duplicate_top_level_groups_dedup_after_caller_pass():
    """Same group passed twice in all_models. _get_models_from_access_groups expands both;
    the caller's dedup step (dict.fromkeys at get_key_models) handles the duplicates."""
    model_access_groups = {"g": ["gpt-4", "claude-3"]}
    result = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=["g", "g"],
    )
    # Function returns duplicates - that's contract; dedup is caller's job
    assert result.count("gpt-4") == 2
    assert result.count("claude-3") == 2


def test_disconnected_components_only_seed_subtree_resolves():
    """Resolution starting at A reaches A's subtree only, not unrelated components."""
    model_access_groups = {"A": ["m-a"], "X": ["m-x"]}
    group_memberships = {"X": ["Y"], "Y": []}

    result = resolve_nested_groups(
        group_name="A",
        model_access_groups=model_access_groups,
        group_memberships=group_memberships,
        visited=set(),
    )
    assert result == ["m-a"]
    assert "m-x" not in result


def test_unicode_and_special_chars_in_group_names():
    """Group names with unicode, hyphens, dots, slashes pass through verbatim."""
    model_access_groups = {"groupe-prod/équipe.1": ["gpt-4"]}
    group_memberships = {"meta⚡": ["groupe-prod/équipe.1"]}

    result = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=["meta⚡"],
        group_memberships=group_memberships,
    )
    assert result == ["gpt-4"]


def test_visited_set_isolation_between_top_level_calls():
    """Two top-level groups in all_models must each fully expand - fresh visited per call."""
    model_access_groups = {"shared": ["common-model"], "A": ["m-a"], "B": ["m-b"]}
    group_memberships = {"A": ["shared"], "B": ["shared"]}

    result = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=["A", "B"],
        group_memberships=group_memberships,
    )
    # Both A and B must contribute their direct models AND the shared subtree
    expanded = set(result)
    assert {"m-a", "m-b", "common-model"} <= expanded


def test_empty_group_memberships_dict_behaves_like_none():
    """Passing {} explicitly is identical to passing None (default)."""
    model_access_groups = {"g": ["gpt-4"]}
    via_none = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=["g"],
        group_memberships=None,
    )
    via_empty = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=["g"],
        group_memberships={},
    )
    assert via_none == via_empty == ["gpt-4"]


def test_membership_with_empty_child_list_does_not_explode():
    """A parent_group mapped to an empty list is a valid no-op edge set."""
    model_access_groups = {"orphan": ["gpt-4"]}
    group_memberships = {"orphan": []}

    result = _get_models_from_access_groups(
        model_access_groups=model_access_groups,
        all_models=["orphan"],
        group_memberships=group_memberships,
    )
    assert result == ["gpt-4"]


def test_resolve_with_empty_models_and_empty_memberships_returns_empty():
    """Defensive: both maps empty + unknown seed = empty result, no crash."""
    result = resolve_nested_groups(
        group_name="nothing",
        model_access_groups={},
        group_memberships={},
        visited=set(),
    )
    assert result == []
