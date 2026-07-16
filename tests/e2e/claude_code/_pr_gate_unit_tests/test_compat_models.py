"""Unit tests for the compat matrix deployment loader and yaml drift checks."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from claude_code._compat_models import (
    all_expected_model_names,
    load_all_deployments,
)

CLAUDE_CODE_DIR = Path(__file__).resolve().parents[1]


def _cell_declared_model_names() -> frozenset[str]:
    pattern = re.compile(r'"(claude-[a-zA-Z0-9._-]+)"')
    found: set[str] = set()
    for path in CLAUDE_CODE_DIR.glob("*/test_*.py"):
        if path.parent.name.startswith("_"):
            continue
        for match in pattern.finditer(path.read_text()):
            name = match.group(1)
            if "/" in name or "@" in name:
                continue
            found.add(name)
    return frozenset(found)


def test_yaml_covers_every_cell_declared_model_name() -> None:
    yaml_names = all_expected_model_names()
    cell_names = _cell_declared_model_names()
    missing = cell_names - yaml_names
    assert not missing, (
        f"compat cells reference model names not declared in "
        f"test_config.yaml: {sorted(missing)}. Add a matching "
        f"model_list entry so the session fixture can register them."
    )


def test_yaml_has_no_unused_declarations() -> None:
    yaml_names = all_expected_model_names()
    cell_names = _cell_declared_model_names()
    unused = yaml_names - cell_names
    assert not unused, (
        f"test_config.yaml declares model names no cell references: "
        f"{sorted(unused)}. Delete them or add the cell."
    )


def test_load_returns_fifteen_deployments() -> None:
    assert len(load_all_deployments()) == 15


def test_deployments_are_hashable_and_frozen() -> None:
    d = load_all_deployments()[0]
    with pytest.raises((AttributeError, TypeError)):
        d.model_name = "mutated"  # type: ignore[misc]


def test_vertex_yaml_keys_populate_pydantic_body() -> None:
    vertex = tuple(d for d in load_all_deployments() if d.model_name.endswith("-vertex"))
    assert vertex, "no vertex deployments found in yaml"
    for d in vertex:
        assert d.litellm_params.vertex_project, (
            f"{d.model_name} lost its vertex_project after normalization"
        )
        assert d.litellm_params.vertex_location, (
            f"{d.model_name} lost its vertex_location after normalization"
        )
        assert d.litellm_params.use_in_pass_through is True, (
            f"{d.model_name} must set use_in_pass_through for the passthrough row"
        )
