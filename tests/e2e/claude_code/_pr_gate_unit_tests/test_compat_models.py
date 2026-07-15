"""Regression tests for the compat-model registration loader.

The compat cells hardcode virtual model names like ``claude-sonnet-4-6``
and expect them to be registered on the proxy before the cell runs. The
session fixture in ``conftest.py`` reads ``test_config.yaml`` and POSTs
those deployments via ``/model/new``. These tests pin the invariants
that make that safe:

- The yaml declares an entry for every virtual name a cell references —
  otherwise a cell probes a name the fixture never registered, and a
  developer running locally sees the same ``Invalid model name`` 400
  the refactor was meant to eliminate.

- The ``vertex_ai_*`` yaml keys get normalized to the ``vertex_*``
  pydantic-body names before ``LiteLLMParamsBody(**)`` sees them, so
  the vertex project/location aren't silently dropped by pydantic's
  ``extra="ignore"`` default.

- The env-based availability check gates each provider on the exact
  credentials the yaml references, so a laptop with only
  ``ANTHROPIC_API_KEY`` registers only Anthropic tiers and skips the
  rest cleanly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from claude_code._compat_models import (
    CompatDeployment,
    all_expected_model_names,
    deployments_for_available_providers,
    load_all_deployments,
)


CLAUDE_CODE_DIR = Path(__file__).resolve().parents[1]


def _cell_declared_model_names() -> frozenset[str]:
    """Every ``"claude-*"`` model name a compat cell hardcodes in a
    ``*_MODELS`` list. Uses a simple regex rather than importing every
    cell because the cells depend on the harness which depends on env
    the unit-test run does not have."""
    pattern = re.compile(r'"(claude-[a-zA-Z0-9._-]+)"')
    found: set[str] = set()
    for path in CLAUDE_CODE_DIR.glob("*/test_*.py"):
        if path.parent.name.startswith("_"):
            continue
        for match in pattern.finditer(path.read_text()):
            name = match.group(1)
            # Skip upstream model references (they carry a version
            # suffix or the ``anthropic/`` provider prefix — we only
            # want proxy-side virtual names here).
            if "/" in name or "@" in name:
                continue
            found.add(name)
    return frozenset(found)


def test_yaml_covers_every_cell_declared_model_name() -> None:
    """Every ``"claude-..."`` string a cell probes must have a
    corresponding ``model_list`` entry in ``test_config.yaml``. A new
    cell that adds a probe for a name the yaml doesn't know fails this
    test — the alternative is a 400 at runtime that is much harder to
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


def test_load_returns_fifteen_deployments() -> None:
    """The compat matrix is 3 tiers x 5 provider surfaces = 15. Pin the
    count so a future edit to the yaml can't silently drop a tier."""
    assert len(load_all_deployments()) == 15


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
    the deployment gets registered with no vertex project — a real
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


def test_availability_skips_when_no_creds_present() -> None:
    """Empty env selects zero deployments — the fixture must not try
    to register anything on a laptop with no provider creds."""
    assert deployments_for_available_providers({}) == ()


def test_availability_selects_only_anthropic_when_only_anthropic_key_set() -> None:
    subset = deployments_for_available_providers(
        {"ANTHROPIC_API_KEY": "sk-test"}
    )
    names = {d.model_name for d in subset}
    assert names == {
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
    }


def test_availability_adds_bedrock_when_aws_ambient_creds_set() -> None:
    """Bedrock authenticates through the ambient AWS chain rather than
    an ``os.environ/FOO`` reference. Setting ``AWS_ACCESS_KEY_ID``
    alone (in addition to Anthropic) must pick up both bedrock
    families (converse + invoke)."""
    subset = deployments_for_available_providers(
        {
            "ANTHROPIC_API_KEY": "sk-test",
            "AWS_ACCESS_KEY_ID": "AKIA-fake",
        }
    )
    names = {d.model_name for d in subset}
    for tier in ("haiku-4-5", "sonnet-4-6", "opus-4-7"):
        assert f"claude-{tier}-bedrock-converse" in names
        assert f"claude-{tier}-bedrock-invoke" in names


def test_availability_gates_azure_on_both_foundry_creds() -> None:
    """Azure needs both AZURE_AI_API_BASE and
    AZURE_AI_API_KEY — half-set env must NOT register the azure
    deployments (they would 401 at call time)."""
    subset = deployments_for_available_providers(
        {
            "ANTHROPIC_API_KEY": "sk-test",
            "AZURE_AI_API_BASE": "https://foo.openai.azure.com",
            # AZURE_AI_API_KEY intentionally missing
        }
    )
    names = {d.model_name for d in subset}
    assert not any(n.endswith("-azure") for n in names)


def test_availability_selects_azure_when_both_foundry_creds_set() -> None:
    subset = deployments_for_available_providers(
        {
            "ANTHROPIC_API_KEY": "sk-test",
            "AZURE_AI_API_BASE": "https://foo.openai.azure.com",
            "AZURE_AI_API_KEY": "azure-key",
        }
    )
    names = {d.model_name for d in subset}
    for tier in ("haiku-4-5", "sonnet-4-6", "opus-4-7"):
        assert f"claude-{tier}-azure" in names


def test_deployments_for_available_providers_reads_yaml_once() -> None:
    """The availability check and the returned deployment list must
    both come from the same parsed view of the file. An earlier shape
    parsed the yaml twice (once for the raw params, once for the built
    deployments) which opened a window where a config swap between
    those reads could produce a mismatched result. A counting reader
    injected via DI pins the invariant — no filesystem, no patching."""
    fake_yaml = """
model_list:
  - model_name: claude-haiku-4-5
    litellm_params:
      model: anthropic/claude-haiku-4-5
      api_key: os.environ/ANTHROPIC_API_KEY
"""
    call_count = 0

    def counting_reader(_path: Path) -> str:
        nonlocal call_count
        call_count += 1
        return fake_yaml

    result = deployments_for_available_providers(
        {"ANTHROPIC_API_KEY": "sk-test"},
        config_path=Path("in-memory"),
        reader=counting_reader,
    )

    assert call_count == 1, (
        f"deployments_for_available_providers read the config yaml "
        f"{call_count} times; must be exactly 1 to avoid a mid-setup "
        f"config-swap inconsistency window"
    )
    assert [d.model_name for d in result] == ["claude-haiku-4-5"]


def test_availability_gates_vertex_on_ambient_gcp_creds() -> None:
    """Vertex needs the yaml refs AND one of the ambient GCP env vars
    (ADC location) — an env that only sets the ``VERTEXAI_*`` project/
    location refs without the credentials file still gets skipped."""
    subset = deployments_for_available_providers(
        {
            "ANTHROPIC_API_KEY": "sk-test",
            "VERTEXAI_PROJECT": "my-proj",
            "VERTEXAI_LOCATION": "us-central1",
            # GOOGLE_APPLICATION_CREDENTIALS missing
        }
    )
    names = {d.model_name for d in subset}
    assert not any(n.endswith("-vertex") for n in names)
