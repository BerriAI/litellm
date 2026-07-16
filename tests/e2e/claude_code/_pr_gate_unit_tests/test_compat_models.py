"""Regression tests for the compat-model registration loader.

The compat cells hardcode virtual model names like ``claude-sonnet-4-5``
and expect them to be registered on the proxy before the cell runs. The
session fixture in ``conftest.py`` reads ``test_config.yaml`` and POSTs
those deployments via ``/model/new``. These tests pin the invariants
that make that safe:

- The yaml declares an entry for every virtual name a cell references -
  otherwise a cell probes a name the fixture never registered, and the
  cell hits an ``Invalid model name`` 400 that is much harder to trace.

- The ``vertex_ai_*`` yaml keys get normalized to the ``vertex_*``
  pydantic-body names before ``LiteLLMParamsBody(**)`` sees them, so
  the vertex project/location aren't silently dropped by pydantic's
  ``extra="ignore"`` default.
"""

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
    """Every ``"claude-*"`` or ``"gpt-*"`` model name a compat cell
    hardcodes in a ``*_MODELS`` list. Uses a simple regex rather than
    importing every cell because the cells depend on the harness which
    depends on env the unit-test run does not have."""
    pattern = re.compile(r'"((?:claude|gpt)-[a-zA-Z0-9._-]+)"')
    found: set[str] = set()
    for path in CLAUDE_CODE_DIR.glob("*/test_*.py"):
        if path.parent.name.startswith("_"):
            continue
        for match in pattern.finditer(path.read_text()):
            name = match.group(1)
            # Skip upstream model references (they carry a version
            # suffix or the ``anthropic/`` provider prefix - we only
            # want proxy-side virtual names here).
            if "/" in name or "@" in name:
                continue
            found.add(name)
    return frozenset(found)


def test_yaml_covers_every_cell_declared_model_name() -> None:
    """Every ``"claude-..."`` string a cell probes must have a
    corresponding ``model_list`` entry in ``test_config.yaml``. A new
    cell that adds a probe for a name the yaml doesn't know fails this
    test - the alternative is a 400 at runtime that is much harder to
    diagnose."""
    yaml_names = all_expected_model_names()
    cell_names = _cell_declared_model_names()
    missing = cell_names - yaml_names
    assert not missing, (
        f"compat cells reference model names not declared in "
        f"test_config.yaml: {sorted(missing)}. Add a matching "
        f"model_list entry so the session fixture can register them."
    )


def test_yaml_has_no_unused_declarations() -> None:
    """Every declaration in ``test_config.yaml`` is referenced by at
    least one cell. A yaml entry no test exercises is dead
    configuration and drift-prone; delete it or add the cell."""
    yaml_names = all_expected_model_names()
    cell_names = _cell_declared_model_names()
    unused = yaml_names - cell_names
    assert not unused, (
        f"test_config.yaml declares model names no cell references: "
        f"{sorted(unused)}. Delete them or add the cell."
    )


def test_load_returns_twenty_four_deployments() -> None:
    """The compat matrix is 3 Claude tiers x 5 provider surfaces = 15,
    plus 3 GPT-5.6 tiers x 3 live provider columns = 9. Pin the count
    so a future edit to the yaml can't silently drop a tier."""
    assert len(load_all_deployments()) == 24


def test_deployments_are_hashable_and_frozen() -> None:
    """``CompatDeployment`` is frozen so tests cannot accidentally
    mutate the shared list mid-session."""
    d = load_all_deployments()[0]
    with pytest.raises((AttributeError, TypeError)):
        d.model_name = "mutated"  # type: ignore[misc]


def test_vertex_yaml_keys_populate_pydantic_body() -> None:
    """The yaml spells vertex fields ``vertex_ai_project`` /
    ``vertex_ai_location`` but ``LiteLLMParamsBody`` names them
    ``vertex_project`` / ``vertex_location``. Without the alias
    normalization the pydantic body silently drops the yaml keys, and
    the deployment gets registered with no vertex project - a real
    incident the drift regressed twice historically."""
    all_deployments = load_all_deployments()
    vertex = [
        d for d in all_deployments if d.model_name.endswith("-vertex")
    ]
    assert vertex, "no vertex deployments found in yaml"
    for d in vertex:
        assert d.litellm_params.vertex_project, (
            f"{d.model_name} lost its vertex_project after normalization"
        )
        assert d.litellm_params.vertex_location, (
            f"{d.model_name} lost its vertex_location after normalization"
        )


def test_vertex_deployments_keep_use_in_pass_through() -> None:
    """Vertex passthrough cells need the deployment registered with
    ``use_in_pass_through: true`` so the proxy wires project/location
    credentials into the /vertex_ai passthrough router. ``LiteLLMParamsBody``
    defaults to ``extra="ignore"``, so a missing field on the body silently
    strips the yaml flag and every vertex passthrough cell fails at runtime
    with "No credentials found on proxy for project_name=..."."""
    vertex = [
        d
        for d in load_all_deployments()
        if d.model_name.endswith("-vertex")
    ]
    assert vertex, "no vertex deployments found in yaml"
    for d in vertex:
        assert d.litellm_params.use_in_pass_through is True, (
            f"{d.model_name} lost use_in_pass_through after load; "
            f"serialized body would be "
            f"{d.litellm_params.model_dump(exclude_none=True)}"
        )


def test_yaml_litellm_params_are_all_known_body_fields() -> None:
    """Every key under ``litellm_params`` in ``test_config.yaml`` must map
    to a ``LiteLLMParamsBody`` field (after the vertex alias rewrite).
    Without this pin, a new yaml flag can land in the fixture config and
    be silently dropped by pydantic before ``/model/new`` ever sees it."""
    import yaml
    from models import LiteLLMParamsBody

    from claude_code._compat_models import (
        CONFIG_PATH,
        _YAML_TO_PYDANTIC_ALIASES,
    )

    known = frozenset(LiteLLMParamsBody.model_fields)
    doc = yaml.safe_load(CONFIG_PATH.read_text())
    model_list = doc.get("model_list") or []
    unknown = tuple(
        (entry["model_name"], key)
        for entry in model_list
        for key in entry["litellm_params"]
        if _YAML_TO_PYDANTIC_ALIASES.get(key, key) not in known
    )
    assert not unknown, (
        f"test_config.yaml litellm_params keys not on LiteLLMParamsBody "
        f"(will be silently dropped at register time): {unknown}"
    )
