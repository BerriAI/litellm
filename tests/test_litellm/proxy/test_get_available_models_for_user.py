"""Tests for get_available_models_for_user UAG resolution and sentinel strip."""

import os
import sys
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy._types import (
    LiteLLM_TeamTableCachedObj,
    SpecialModelNames,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.model_checks import _resolve_team_access_group_model_names
from litellm.proxy.utils import get_available_models_for_user


def _make_team_cached_obj(
    *,
    team_id: str = "team_alpha",
    models: Optional[List[str]] = None,
    access_group_ids: Optional[List[str]] = None,
) -> LiteLLM_TeamTableCachedObj:
    return LiteLLM_TeamTableCachedObj(
        team_id=team_id,
        models=models if models is not None else [],
        access_group_ids=access_group_ids,
    )


def _make_access_group_row(group_id: str, model_names: List[str]):
    row = MagicMock()
    row.access_group_id = group_id
    row.access_model_names = model_names
    return row


def _make_prisma_client_with_ag_rows(rows: List):
    prisma = MagicMock()
    prisma.db.litellm_accessgrouptable.find_many = AsyncMock(return_value=rows)
    return prisma


def _make_user_api_key_dict(
    *,
    team_id: Optional[str] = None,
    team_models: Optional[List[str]] = None,
) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        models=[],
        team_id=team_id,
        team_models=team_models or [],
    )


def _patched_get_team_object(team_obj):
    """Returns an AsyncMock that ignores extra kwargs and returns ``team_obj``."""

    async def _impl(*_args, **_kwargs):
        return team_obj

    return _impl


# ---------------------------------------------------------------------------
# _resolve_team_access_group_model_names — pure helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolver_returns_access_group_model_names():
    team = _make_team_cached_obj(
        models=["no-default-models"], access_group_ids=["ag-1"]
    )
    prisma = _make_prisma_client_with_ag_rows(
        [_make_access_group_row("ag-1", ["gpt-4o", "claude-opus-4-5"])]
    )

    result = await _resolve_team_access_group_model_names(
        team_objects=[team], prisma_client=prisma
    )

    assert result == ["gpt-4o", "claude-opus-4-5"]


@pytest.mark.asyncio
async def test_resolver_skips_teams_with_empty_models():
    team = _make_team_cached_obj(models=[], access_group_ids=["ag-1"])
    prisma = _make_prisma_client_with_ag_rows(
        [_make_access_group_row("ag-1", ["gpt-4o"])]
    )

    result = await _resolve_team_access_group_model_names(
        team_objects=[team], prisma_client=prisma
    )

    assert result == []
    prisma.db.litellm_accessgrouptable.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_resolver_skips_teams_with_all_proxy_models_sentinel():
    team = _make_team_cached_obj(
        models=[SpecialModelNames.all_proxy_models.value],
        access_group_ids=["ag-1"],
    )
    prisma = _make_prisma_client_with_ag_rows(
        [_make_access_group_row("ag-1", ["gpt-4o"])]
    )

    result = await _resolve_team_access_group_model_names(
        team_objects=[team], prisma_client=prisma
    )

    assert result == []
    prisma.db.litellm_accessgrouptable.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_resolver_deduplicates_across_access_groups():
    team = _make_team_cached_obj(
        models=["no-default-models"], access_group_ids=["ag-1", "ag-2"]
    )
    prisma = _make_prisma_client_with_ag_rows(
        [
            _make_access_group_row("ag-1", ["gpt-4o", "claude-opus-4-5"]),
            _make_access_group_row("ag-2", ["claude-opus-4-5", "mistral-large"]),
        ]
    )

    result = await _resolve_team_access_group_model_names(
        team_objects=[team], prisma_client=prisma
    )

    assert result == ["gpt-4o", "claude-opus-4-5", "mistral-large"]


@pytest.mark.asyncio
async def test_resolver_handles_deleted_access_group():
    team = _make_team_cached_obj(
        models=["no-default-models"], access_group_ids=["ag-1", "ag-deleted"]
    )
    prisma = _make_prisma_client_with_ag_rows(
        [_make_access_group_row("ag-1", ["gpt-4o"])]
    )

    result = await _resolve_team_access_group_model_names(
        team_objects=[team], prisma_client=prisma
    )

    assert result == ["gpt-4o"]


# ---------------------------------------------------------------------------
# get_available_models_for_user — end-to-end behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_default_models_team_resolves_access_group_models():
    team = _make_team_cached_obj(
        models=["no-default-models"], access_group_ids=["ag-1"]
    )
    prisma = _make_prisma_client_with_ag_rows(
        [_make_access_group_row("ag-1", ["gpt-4o", "claude-opus-4-5"])]
    )

    user_api_key_dict = _make_user_api_key_dict(
        team_id=team.team_id, team_models=["no-default-models"]
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_team_object",
        new=_patched_get_team_object(team),
    ):
        result = await get_available_models_for_user(
            user_api_key_dict=user_api_key_dict,
            llm_router=None,
            general_settings={},
            user_model=None,
            prisma_client=prisma,
            proxy_logging_obj=MagicMock(),
            team_id=None,
            user_api_key_cache=MagicMock(),
        )

    assert SpecialModelNames.no_default_models.value not in result
    assert set(result) == {"gpt-4o", "claude-opus-4-5"}


@pytest.mark.asyncio
async def test_direct_and_access_group_models_are_merged_and_deduped():
    team = _make_team_cached_obj(models=["gpt-4o"], access_group_ids=["ag-1"])
    prisma = _make_prisma_client_with_ag_rows(
        [_make_access_group_row("ag-1", ["claude-opus-4-5", "gpt-4o"])]
    )

    user_api_key_dict = _make_user_api_key_dict(
        team_id=team.team_id, team_models=["gpt-4o"]
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_team_object",
        new=_patched_get_team_object(team),
    ):
        result = await get_available_models_for_user(
            user_api_key_dict=user_api_key_dict,
            llm_router=None,
            general_settings={},
            user_model=None,
            prisma_client=prisma,
            proxy_logging_obj=MagicMock(),
            team_id=None,
            user_api_key_cache=MagicMock(),
        )

    assert result.count("gpt-4o") == 1
    assert set(result) == {"gpt-4o", "claude-opus-4-5"}


@pytest.mark.asyncio
async def test_sentinel_only_team_without_access_groups_returns_empty():
    # Privilege-escalation guard: configures a populated llm_router so that
    # stripping the sentinel without force_empty would leak every proxy model.
    team = _make_team_cached_obj(models=["no-default-models"], access_group_ids=None)
    prisma = _make_prisma_client_with_ag_rows([])

    user_api_key_dict = _make_user_api_key_dict(
        team_id=team.team_id, team_models=["no-default-models"]
    )

    mock_router = MagicMock()
    mock_router.get_model_names.return_value = [
        "gpt-4o",
        "claude-opus-4-5",
        "mistral-large",
    ]
    mock_router.get_model_access_groups.return_value = {}

    with patch(
        "litellm.proxy.auth.auth_checks.get_team_object",
        new=_patched_get_team_object(team),
    ):
        result = await get_available_models_for_user(
            user_api_key_dict=user_api_key_dict,
            llm_router=mock_router,
            general_settings={},
            user_model=None,
            prisma_client=prisma,
            proxy_logging_obj=MagicMock(),
            team_id=None,
            user_api_key_cache=MagicMock(),
        )

    assert SpecialModelNames.no_default_models.value not in result
    assert result == []
    assert (
        "gpt-4o" not in result
    ), "A no-default-models team must not silently see every proxy model"


@pytest.mark.asyncio
async def test_db_failure_in_access_group_resolution_does_not_crash():
    # On transient Prisma failure, degrade to "no access-group contribution"
    # and still honor the sentinel short-circuit.
    team = _make_team_cached_obj(
        models=["no-default-models"], access_group_ids=["ag-1"]
    )

    prisma = MagicMock()
    prisma.db.litellm_accessgrouptable.find_many = AsyncMock(
        side_effect=RuntimeError("simulated transient DB error")
    )

    user_api_key_dict = _make_user_api_key_dict(
        team_id=team.team_id, team_models=["no-default-models"]
    )

    mock_router = MagicMock()
    mock_router.get_model_names.return_value = ["gpt-4o", "claude-opus-4-5"]
    mock_router.get_model_access_groups.return_value = {}

    with patch(
        "litellm.proxy.auth.auth_checks.get_team_object",
        new=_patched_get_team_object(team),
    ):
        result = await get_available_models_for_user(
            user_api_key_dict=user_api_key_dict,
            llm_router=mock_router,
            general_settings={},
            user_model=None,
            prisma_client=prisma,
            proxy_logging_obj=MagicMock(),
            team_id=None,
            user_api_key_cache=MagicMock(),
        )

    assert result == []
    assert "gpt-4o" not in result


@pytest.mark.asyncio
async def test_team_with_direct_models_unaffected_by_resolver_failure():
    team = _make_team_cached_obj(models=["gpt-4o"], access_group_ids=["ag-1"])

    prisma = MagicMock()
    prisma.db.litellm_accessgrouptable.find_many = AsyncMock(
        side_effect=RuntimeError("simulated transient DB error")
    )

    user_api_key_dict = _make_user_api_key_dict(
        team_id=team.team_id, team_models=["gpt-4o"]
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_team_object",
        new=_patched_get_team_object(team),
    ):
        result = await get_available_models_for_user(
            user_api_key_dict=user_api_key_dict,
            llm_router=None,
            general_settings={},
            user_model=None,
            prisma_client=prisma,
            proxy_logging_obj=MagicMock(),
            team_id=None,
            user_api_key_cache=MagicMock(),
        )

    assert result == ["gpt-4o"]


@pytest.mark.asyncio
async def test_explicit_team_id_param_resolves_access_groups():
    team = _make_team_cached_obj(
        models=["no-default-models"], access_group_ids=["ag-1"]
    )
    prisma = _make_prisma_client_with_ag_rows(
        [_make_access_group_row("ag-1", ["gpt-4o"])]
    )

    user_api_key_dict = _make_user_api_key_dict(team_id=None, team_models=[])

    async def _noop_validate_membership(*_args, **_kwargs):
        return True

    with (
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new=_patched_get_team_object(team),
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.validate_membership",
            new=_noop_validate_membership,
        ),
    ):
        result = await get_available_models_for_user(
            user_api_key_dict=user_api_key_dict,
            llm_router=None,
            general_settings={},
            user_model=None,
            prisma_client=prisma,
            proxy_logging_obj=MagicMock(),
            team_id=team.team_id,
            user_api_key_cache=MagicMock(),
        )

    assert set(result) == {"gpt-4o"}
    assert SpecialModelNames.no_default_models.value not in result


@pytest.mark.asyncio
async def test_deleted_access_group_does_not_leak_sentinel():
    team = _make_team_cached_obj(
        models=["no-default-models"], access_group_ids=["ag-deleted"]
    )
    prisma = _make_prisma_client_with_ag_rows([])

    user_api_key_dict = _make_user_api_key_dict(
        team_id=team.team_id, team_models=["no-default-models"]
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_team_object",
        new=_patched_get_team_object(team),
    ):
        result = await get_available_models_for_user(
            user_api_key_dict=user_api_key_dict,
            llm_router=None,
            general_settings={},
            user_model=None,
            prisma_client=prisma,
            proxy_logging_obj=MagicMock(),
            team_id=None,
            user_api_key_cache=MagicMock(),
        )

    assert result == []


@pytest.mark.asyncio
async def test_user_with_no_team_skips_access_group_resolution():
    user_api_key_dict = _make_user_api_key_dict(team_id=None, team_models=[])
    prisma = _make_prisma_client_with_ag_rows([])

    result = await get_available_models_for_user(
        user_api_key_dict=user_api_key_dict,
        llm_router=None,
        general_settings={},
        user_model=None,
        prisma_client=prisma,
        proxy_logging_obj=MagicMock(),
        team_id=None,
        user_api_key_cache=MagicMock(),
    )

    assert result == []
    prisma.db.litellm_accessgrouptable.find_many.assert_not_called()
