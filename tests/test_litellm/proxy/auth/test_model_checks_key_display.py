"""
Unit tests for GET /key/{id}/models helpers in litellm.proxy.auth.model_checks.

These live under tests/test_litellm/proxy/auth so they run in the same CI job
as other model_checks tests (coverage for litellm/proxy/auth/model_checks.py).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._types import SpecialModelNames
from litellm.proxy.auth.model_checks import (
    KEY_RESOLVED_MODELS_DISPLAY_LIMIT,
    build_key_resolved_model_display_sections,
    prepare_key_models_response_payload,
    resolve_key_models_for_display,
    _concrete_models_allowed_by_resolved,
    _expand_group_models_for_key_display,
    _filter_key_models_by_search,
    _models_in_any_access_group_for_key_display,
)


def test_filter_models_by_search():
    models = ["GPT-4", "claude-3", "embed-small"]
    assert _filter_key_models_by_search(models, None) == models
    assert _filter_key_models_by_search(models, "   ") == models
    assert _filter_key_models_by_search(models, "gpt") == ["GPT-4"]
    assert _filter_key_models_by_search(models, "CLAUDE") == ["claude-3"]


@pytest.mark.asyncio
async def test_resolve_plain_key_models_no_sentinels():
    mock_prisma = MagicMock()
    mock_router = MagicMock()
    mock_router.get_model_names.return_value = ["router-a"]

    resolved, source, no_team = await resolve_key_models_for_display(
        key_models=["gpt-4", "claude-3"],
        team_id="team-1",
        prisma_client=mock_prisma,
        llm_router=mock_router,
    )
    assert source == SpecialModelNames.no_default_models.value
    assert resolved == ["gpt-4", "claude-3"]
    assert no_team is False


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


@pytest.mark.asyncio
async def test_resolve_all_team_models_llm_router_none():
    """When router is absent, all_models stays empty; no-team sentinel still resolves."""
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    resolved, source, no_team = await resolve_key_models_for_display(
        key_models=["all-team-models"],
        team_id=None,
        prisma_client=mock_prisma,
        llm_router=None,
    )
    assert source == SpecialModelNames.all_team_models.value
    assert resolved == []
    assert no_team is True


@pytest.mark.asyncio
async def test_resolve_all_team_models_team_row_models_none():
    """Team exists but has no model list; key_models unchanged before all-proxy expansion."""
    mock_prisma = MagicMock()
    team_row = MagicMock()
    team_row.models = None
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
    mock_router = MagicMock()
    mock_router.get_model_names.return_value = ["r1"]

    resolved, source, no_team = await resolve_key_models_for_display(
        key_models=["all-team-models"],
        team_id="team-1",
        prisma_client=mock_prisma,
        llm_router=mock_router,
    )
    assert source == SpecialModelNames.all_team_models.value
    assert resolved == ["all-team-models"]
    assert no_team is False


@pytest.mark.asyncio
async def test_resolve_all_proxy_models_from_resolved_after_team_not_in_key_models():
    """Second branch uses `all-proxy-models in resolved` when team list contains the sentinel."""
    mock_prisma = MagicMock()
    team_row = MagicMock()
    team_row.models = ["all-proxy-models"]
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
    mock_router = MagicMock()
    mock_router.get_model_names.return_value = ["p1", "p2"]

    resolved, source, no_team = await resolve_key_models_for_display(
        key_models=["all-team-models"],
        team_id="team-1",
        prisma_client=mock_prisma,
        llm_router=mock_router,
    )
    assert source == SpecialModelNames.all_proxy_models.value
    assert resolved == ["p1", "p2"]
    assert no_team is False


def test_concrete_models_allowed_skips_duplicate_router_names():
    """Router may list the same deployment twice; output is deduped. Only resolved names count."""
    assert _concrete_models_allowed_by_resolved(["a", "b"], ["a", "a", "b"]) == ["a", "b"]


def test_concrete_models_wildcard_in_resolved_list():
    router = ["openai/gpt-4o", "anthropic/claude-3"]
    out = _concrete_models_allowed_by_resolved(["openai/*"], router)
    assert "openai/gpt-4o" in out
    assert "anthropic/claude-3" not in out


def test_expand_group_models_skips_duplicate_non_wildcard_entries():
    assert _expand_group_models_for_key_display(["x", "x"], ["x"]) == ["x"]


def test_expand_group_models_non_wildcard_not_in_pool_skipped():
    assert _expand_group_models_for_key_display(["missing", "a"], ["a"]) == ["a"]


def test_expand_group_models_wildcard_skips_duplicate_pool_rows():
    """Second pass over same concrete name hits `m in seen` in wildcard inner loop."""
    pool = ["openai/gpt-4o", "openai/gpt-4o"]
    out = _expand_group_models_for_key_display(["openai/*"], pool)
    assert out.count("openai/gpt-4o") == 1


def test_models_in_any_access_group_for_key_display():
    got = _models_in_any_access_group_for_key_display(
        {"g1": ["a", "b"], "g2": ["z"]},
        ["a", "z"],
    )
    assert got == {"a", "z"}


def test_build_display_sections_skips_access_group_with_no_display_intersection():
    sections = build_key_resolved_model_display_sections(
        display_models=["a"],
        source=SpecialModelNames.no_default_models.value,
        model_access_groups={"empty-intersection": ["z"]},
        compact=False,
    )
    assert not any(s["title"] == "empty-intersection" for s in sections)


def test_build_display_sections_compact_sentinels_empty_model_lists():
    for source in (
        SpecialModelNames.all_proxy_models.value,
        SpecialModelNames.all_team_models.value,
    ):
        sections = build_key_resolved_model_display_sections(
            display_models=["m1", "m2"],
            source=source,
            model_access_groups={"G": ["m1"]},
            compact=True,
        )
        assert sections[0]["models"] == []
        gsec = next(s for s in sections if s["section_kind"] == "access_group")
        assert gsec["models"] == []


def test_build_display_sections_ungrouped_with_compact():
    sections = build_key_resolved_model_display_sections(
        display_models=["only-me"],
        source=SpecialModelNames.no_default_models.value,
        model_access_groups={},
        compact=True,
    )
    ung = next(s for s in sections if s["section_kind"] == "ungrouped")
    assert ung["models"] == []


def test_build_display_sections_no_ungrouped_when_all_models_in_access_groups():
    sections = build_key_resolved_model_display_sections(
        display_models=["a", "b"],
        source=SpecialModelNames.no_default_models.value,
        model_access_groups={"G": ["a", "b"]},
        compact=False,
    )
    assert not any(s["section_kind"] == "ungrouped" for s in sections)


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


def test_prepare_payload_search_filters_matched_count():
    out = prepare_key_models_response_payload(
        resolved=["alpha-model", "beta-model"],
        source=SpecialModelNames.no_default_models.value,
        all_team_models_without_team=False,
        model_access_groups={},
        search="beta",
        compact=False,
        all_router_model_names=["alpha-model", "beta-model"],
    )
    assert out["matched_count"] == 1
    ung = next(s for s in out["model_display_sections"] if s["section_kind"] == "ungrouped")
    assert ung["models"] == ["beta-model"]


def test_prepare_payload_search_no_matches():
    out = prepare_key_models_response_payload(
        resolved=["alpha-model"],
        source=SpecialModelNames.no_default_models.value,
        all_team_models_without_team=False,
        model_access_groups={},
        search="zzz",
        compact=False,
        all_router_model_names=["alpha-model"],
    )
    assert out["matched_count"] == 0
    assert out["model_display_sections"] == []


def test_prepare_payload_all_team_models_without_team_flag_passthrough():
    out = prepare_key_models_response_payload(
        resolved=["x"],
        source=SpecialModelNames.all_team_models.value,
        all_team_models_without_team=True,
        model_access_groups={},
        search=None,
        compact=False,
        all_router_model_names=["x"],
    )
    assert out["all_team_models_without_team"] is True


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


def test_prepare_payload_group_expansion_empty_omits_group():
    out = prepare_key_models_response_payload(
        resolved=["a"],
        source=SpecialModelNames.no_default_models.value,
        all_team_models_without_team=False,
        model_access_groups={"never-matches": ["nonexistent-model"]},
        search=None,
        compact=False,
        all_router_model_names=["a"],
    )
    assert not any(s["title"] == "never-matches" for s in out["model_display_sections"])


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
