"""
Unit tests for GET /key/{id}/models payload helpers (litellm.proxy.auth.model_checks).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._types import SpecialModelNames
from litellm.proxy.auth.model_checks import (
    KEY_RESOLVED_MODELS_DISPLAY_LIMIT,
    prepare_key_models_response_payload,
    resolve_key_models_for_display,
    _filter_key_models_by_search,
)


def test_filter_models_by_search():
    models = ["GPT-4", "claude-3", "embed-small"]
    assert _filter_key_models_by_search(models, None) == models
    assert _filter_key_models_by_search(models, "   ") == models
    assert _filter_key_models_by_search(models, "gpt") == ["GPT-4"]
    assert _filter_key_models_by_search(models, "CLAUDE") == ["claude-3"]


@pytest.mark.asyncio
async def test_resolve_all_team_models_without_team():
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)
    mock_router = MagicMock()
    mock_router.get_model_names.return_value = ["a", "b"]

    resolved, source, no_team = await resolve_key_models_for_display(
        key_models=["all-team-models"],
        team_id=None,
        prisma_client=mock_prisma,
        llm_router=mock_router,
    )
    assert source == SpecialModelNames.all_team_models.value
    assert resolved == ["a", "b"]
    assert no_team is True


@pytest.mark.asyncio
async def test_resolve_all_team_models_with_team():
    mock_prisma = MagicMock()
    team_row = MagicMock()
    team_row.models = ["team-m1"]
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
    mock_router = MagicMock()
    mock_router.get_model_names.return_value = ["proxy-wide"]

    resolved, source, no_team = await resolve_key_models_for_display(
        key_models=["all-team-models"],
        team_id="team-1",
        prisma_client=mock_prisma,
        llm_router=mock_router,
    )
    assert source == SpecialModelNames.all_team_models.value
    assert resolved == ["team-m1"]
    assert no_team is False


@pytest.mark.asyncio
async def test_resolve_all_proxy_models():
    mock_prisma = MagicMock()
    mock_router = MagicMock()
    mock_router.get_model_names.return_value = ["m1", "m2"]

    resolved, source, no_team = await resolve_key_models_for_display(
        key_models=["all-proxy-models"],
        team_id=None,
        prisma_client=mock_prisma,
        llm_router=mock_router,
    )
    assert source == SpecialModelNames.all_proxy_models.value
    assert resolved == ["m1", "m2"]
    assert no_team is False


def test_prepare_payload_search_and_truncation():
    big = [f"m{i}" for i in range(KEY_RESOLVED_MODELS_DISPLAY_LIMIT + 50)]
    out = prepare_key_models_response_payload(
        resolved=big,
        source=SpecialModelNames.no_default_models.value,
        all_team_models_without_team=False,
        model_access_groups={},
        search=None,
        compact=False,
        all_router_model_names=big,
    )
    assert out["resolved_total_count"] == len(big)
    assert out["matched_count"] == len(big)
    assert out["models_truncated"] is True
    ung = out["model_display_sections"][0]
    assert ung["section_kind"] == "ungrouped"
    assert len(ung["models"]) == KEY_RESOLVED_MODELS_DISPLAY_LIMIT


def test_prepare_payload_compact():
    out = prepare_key_models_response_payload(
        resolved=["x", "y"],
        source=SpecialModelNames.no_default_models.value,
        all_team_models_without_team=False,
        model_access_groups={},
        search=None,
        compact=True,
        all_router_model_names=["x", "y"],
    )
    assert out["matched_count"] == 2
    for sec in out["model_display_sections"]:
        assert sec["models"] == []


def test_prepare_payload_explicit_models_with_access_groups():
    out = prepare_key_models_response_payload(
        resolved=["a", "b", "z"],
        source=SpecialModelNames.no_default_models.value,
        all_team_models_without_team=False,
        model_access_groups={"G": ["a", "b"], "H": ["b"]},
        search=None,
        compact=False,
        all_router_model_names=["a", "b", "z"],
    )
    kinds = [s["section_kind"] for s in out["model_display_sections"]]
    assert "access_group" in kinds
    assert "ungrouped" in kinds
    ung = next(s for s in out["model_display_sections"] if s["section_kind"] == "ungrouped")
    assert ung["models"] == ["z"]


def test_all_proxy_models_includes_access_group_sections():
    """Sentinel 'all proxy' section is first; access groups are not skipped."""
    out = prepare_key_models_response_payload(
        resolved=["m1", "m2", "m3"],
        source=SpecialModelNames.all_proxy_models.value,
        all_team_models_without_team=False,
        model_access_groups={
            "grp-a": ["m1", "not-in-resolved"],
            "grp-b": ["m2", "m1"],
        },
        search=None,
        compact=False,
        all_router_model_names=["m1", "m2", "m3"],
    )
    sections = out["model_display_sections"]
    kinds = [s["section_kind"] for s in sections]
    assert kinds[0] == "all_proxy_models"
    assert sections[0]["models"] == ["m1", "m2", "m3"]
    assert kinds.count("access_group") == 2
    by_title = {s["title"]: s["models"] for s in sections}
    assert by_title["grp-a"] == ["m1"]
    assert by_title["grp-b"] == ["m2", "m1"]
    assert "ungrouped" not in kinds


def test_all_team_models_includes_access_group_sections():
    out = prepare_key_models_response_payload(
        resolved=["x", "y"],
        source=SpecialModelNames.all_team_models.value,
        all_team_models_without_team=False,
        model_access_groups={"T1": ["x"], "T2": ["x", "y"]},
        search=None,
        compact=False,
        all_router_model_names=["x", "y"],
    )
    sections = out["model_display_sections"]
    kinds = [s["section_kind"] for s in sections]
    assert kinds[0] == "all_team_models"
    assert sections[0]["models"] == ["x", "y"]
    assert kinds.count("access_group") == 2
    assert "ungrouped" not in kinds


def test_same_model_listed_under_multiple_access_groups():
    """Overlap across groups is allowed; model 'b' appears in G and H."""
    out = prepare_key_models_response_payload(
        resolved=["a", "b", "c"],
        source=SpecialModelNames.no_default_models.value,
        all_team_models_without_team=False,
        model_access_groups={"G": ["a", "b"], "H": ["b", "c"]},
        search=None,
        compact=False,
        all_router_model_names=["a", "b", "c"],
    )
    g = next(s for s in out["model_display_sections"] if s["title"] == "G")
    h = next(s for s in out["model_display_sections"] if s["title"] == "H")
    assert "b" in g["models"] and "b" in h["models"]


def test_access_group_wildcard_expands_to_concrete_models():
    router_models = ["openai/gpt-4o", "openai/gpt-3.5-turbo", "anthropic/claude-3"]
    out = prepare_key_models_response_payload(
        resolved=["openai/*"],
        source=SpecialModelNames.no_default_models.value,
        all_team_models_without_team=False,
        model_access_groups={"openai_group": ["openai/*"]},
        search=None,
        compact=False,
        all_router_model_names=router_models,
    )
    sec = next(s for s in out["model_display_sections"] if s["title"] == "openai_group")
    assert "openai/*" not in sec["models"]
    assert "openai/gpt-4o" in sec["models"]
    assert "openai/gpt-3.5-turbo" in sec["models"]
    assert "anthropic/claude-3" not in sec["models"]
