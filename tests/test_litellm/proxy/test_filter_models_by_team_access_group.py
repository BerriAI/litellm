"""
Tests for _filter_models_by_team_id resolving access group names.

Verifies that when a team's `models` field contains an access group name
(e.g., "Group-A"), the filter resolves it to the member model names before
looking up deployments — matching the behavior of the auth path in
auth_checks.py:model_in_access_group().
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


from litellm.proxy.proxy_server import _filter_models_by_team_id


def _make_model(model_name: str, model_id: str, access_groups: list[str] = None):
    """Helper to build a model dict matching the router's format."""
    return {
        "model_name": model_name,
        "litellm_params": {"model": model_name},
        "model_info": {
            "id": model_id,
            "access_groups": access_groups or [],
        },
    }


def _make_team(models: list[str], team_id: str = "team_alpha"):
    """Helper to build a mock team DB object."""
    mock = MagicMock()
    mock.model_dump.return_value = {
        "team_id": team_id,
        "team_alias": "Team Alpha",
        "models": models,
        "max_budget": None,
        "spend": 0.0,
        "blocked": False,
        "members_with_roles": [],
        "metadata": {},
    }
    return mock


@pytest.mark.asyncio
async def test_filter_resolves_access_group_names():
    """
    When team.models contains an access group name, _filter_models_by_team_id
    should resolve it to the member models and return only those deployments.
    """
    # Models on the proxy
    gpt4o = _make_model("gpt-4o", "id-1", ["Group-A"])
    gpt5 = _make_model("gpt-5", "id-2", ["Group-A"])
    claude = _make_model("claude-3", "id-3", ["Group-B"])

    all_models = [gpt4o, gpt5, claude]

    # Router mock
    mock_router = MagicMock()
    # get_model_access_groups returns {group_name: [model_names]}
    mock_router.get_model_access_groups.return_value = {
        "Group-A": ["gpt-4o", "gpt-5"],
        "Group-B": ["claude-3"],
    }

    # get_model_list returns deployments matching a model_name
    def fake_get_model_list(model_name=None, team_id=None):
        return [m for m in all_models if m["model_name"] == model_name]

    mock_router.get_model_list = MagicMock(side_effect=fake_get_model_list)

    # Team has models: ["Group-A"] — an access group name, not a literal model
    team_db = _make_team(models=["Group-A"])

    # Prisma mock
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_db)
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

    result = await _filter_models_by_team_id(
        all_models=all_models,
        team_id="team_alpha",
        prisma_client=mock_prisma,
        llm_router=mock_router,
    )

    result_ids = {m["model_info"]["id"] for m in result}
    # Should include gpt-4o and gpt-5 (Group-A), but NOT claude-3 (Group-B)
    assert result_ids == {
        "id-1",
        "id-2",
    }, f"Expected Group-A models only, got {result_ids}"

    # Verify DB fallback query received resolved model names, not access group name
    call_kwargs = mock_prisma.db.litellm_proxymodeltable.find_many.call_args[1]
    assert set(call_kwargs["where"]["model_name"]["in"]) == {
        "gpt-4o",
        "gpt-5",
    }, "find_many should receive resolved model names, not the access group name"


@pytest.mark.asyncio
async def test_filter_resolves_mix_of_access_groups_and_literal_names():
    """
    When team.models contains both an access group name and a literal model name,
    both should be resolved correctly.
    """
    gpt4o = _make_model("gpt-4o", "id-1", ["Group-A"])
    gpt5 = _make_model("gpt-5", "id-2", ["Group-A"])
    claude = _make_model("claude-3", "id-3", ["Group-B"])
    mistral = _make_model("mistral-large", "id-4", [])  # no access group

    all_models = [gpt4o, gpt5, claude, mistral]

    mock_router = MagicMock()
    mock_router.get_model_access_groups.return_value = {
        "Group-A": ["gpt-4o", "gpt-5"],
        "Group-B": ["claude-3"],
    }

    def fake_get_model_list(model_name=None, team_id=None):
        return [m for m in all_models if m["model_name"] == model_name]

    mock_router.get_model_list = MagicMock(side_effect=fake_get_model_list)

    # Team has access to Group-A (access group) + mistral-large (literal name)
    team_db = _make_team(models=["Group-A", "mistral-large"])

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_db)
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

    result = await _filter_models_by_team_id(
        all_models=all_models,
        team_id="team_alpha",
        prisma_client=mock_prisma,
        llm_router=mock_router,
    )

    result_ids = {m["model_info"]["id"] for m in result}
    # Group-A models + mistral-large, but NOT claude-3
    assert result_ids == {
        "id-1",
        "id-2",
        "id-4",
    }, f"Expected Group-A + mistral-large, got {result_ids}"


@pytest.mark.asyncio
async def test_filter_excludes_models_from_other_access_group():
    """
    Models belonging only to a different access group must not appear in results.
    """
    gpt4o = _make_model("gpt-4o", "id-1", ["Group-A"])
    claude = _make_model("claude-3", "id-3", ["Group-B"])
    llama = _make_model("llama-4", "id-4", ["Group-B"])

    all_models = [gpt4o, claude, llama]

    mock_router = MagicMock()
    mock_router.get_model_access_groups.return_value = {
        "Group-A": ["gpt-4o"],
        "Group-B": ["claude-3", "llama-4"],
    }

    def fake_get_model_list(model_name=None, team_id=None):
        return [m for m in all_models if m["model_name"] == model_name]

    mock_router.get_model_list = MagicMock(side_effect=fake_get_model_list)

    team_db = _make_team(models=["Group-A"])

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_db)
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

    result = await _filter_models_by_team_id(
        all_models=all_models,
        team_id="team_alpha",
        prisma_client=mock_prisma,
        llm_router=mock_router,
    )

    result_names = {m["model_name"] for m in result}
    assert "claude-3" not in result_names, "Group-B model should not be accessible"
    assert "llama-4" not in result_names, "Group-B model should not be accessible"
    assert "gpt-4o" in result_names, "Group-A model should be accessible"


@pytest.mark.asyncio
async def test_filter_db_fallback_receives_resolved_model_names():
    """
    When get_model_list returns no results (forcing the DB fallback path),
    the DB query should receive resolved model names, not the raw access group name.
    """
    gpt4o = _make_model("gpt-4o", "id-1", ["Group-A"])
    all_models = [gpt4o]

    mock_router = MagicMock()
    mock_router.get_model_access_groups.return_value = {
        "Group-A": ["gpt-4o", "gpt-5"],
    }
    # get_model_list returns nothing — forces reliance on the DB fallback
    mock_router.get_model_list = MagicMock(return_value=[])

    team_db = _make_team(models=["Group-A"])

    # DB returns a model that the router didn't find
    mock_db_model = MagicMock()
    mock_db_model.model_id = "id-db-1"

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_db)
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[mock_db_model])

    result = await _filter_models_by_team_id(
        all_models=all_models,
        team_id="team_alpha",
        prisma_client=mock_prisma,
        llm_router=mock_router,
    )

    # Verify DB query received resolved names, not "Group-A"
    call_kwargs = mock_prisma.db.litellm_proxymodeltable.find_many.call_args[1]
    queried_names = set(call_kwargs["where"]["model_name"]["in"])
    assert queried_names == {
        "gpt-4o",
        "gpt-5",
    }, f"DB query should receive resolved model names, got {queried_names}"
    assert "Group-A" not in queried_names, "Raw access group name should not be in DB query"


@pytest.mark.asyncio
async def test_filter_excludes_blocked_public_model_with_no_team_assignment():
    """
    GH#38949: a disabled (blocked) public deployment with no team assignment in
    the DB must not appear in a team's model list, even when a stale
    team.models entry still resolves to its public model name via the router.

    Reproduction from the issue: team "Team-Alpha" has its own aliased
    deployment; a chat call through the team key writes the public model name
    into team.models; the team filter then resolves that name to the disabled
    public deployment and listed it as a second, team-attached model.
    """
    # Disabled public deployment: no team_id, paused by the admin.
    public_blocked = {
        "model_name": "gpt-5.1",
        "litellm_params": {"model": "gpt-5.1"},
        "model_info": {"id": "id-public-blocked", "blocked": True, "team_id": None},
    }
    # The team's own deployment (unique alias name), active.
    team_alias = {
        "model_name": "gpt-5.1-team-alpha",
        "litellm_params": {"model": "gpt-5.1"},
        "model_info": {"id": "id-team-alias", "blocked": False, "team_id": "team_alpha"},
    }
    all_models = [public_blocked, team_alias]

    mock_router = MagicMock()
    mock_router.get_model_access_groups.return_value = {}
    # Public reachability: resolving the (stale) team.models name returns the
    # disabled public deployment. This mirrors should_include_deployment's
    # semantics, which this fix deliberately does not touch.
    mock_router.get_model_list = MagicMock(
        side_effect=lambda model_name=None, team_id=None: [public_blocked] if model_name == "gpt-5.1" else []
    )

    team_db = _make_team(models=["gpt-5.1"])

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_db)
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

    result = await _filter_models_by_team_id(
        all_models=all_models,
        team_id="team_alpha",
        prisma_client=mock_prisma,
        llm_router=mock_router,
    )

    result_ids = {m["model_info"]["id"] for m in result}
    assert result_ids == {
        "id-team-alias",
    }, f"Disabled public model must not be listed for the team, got {result_ids}"


@pytest.mark.asyncio
async def test_filter_excludes_blocked_models_not_owned_by_team():
    """
    A blocked deployment the team does not own (explicit access_via_team_ids
    grant or name resolution) must not surface in the team's list, while a
    blocked deployment owned by the team stays visible so admins can re-enable
    it from the team view.
    """
    blocked_granted = {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "gpt-4o"},
        "model_info": {
            "id": "id-granted-blocked",
            "blocked": True,
            "access_via_team_ids": ["team_alpha"],
        },
    }
    blocked_owned = {
        "model_name": "claude-3",
        "litellm_params": {"model": "claude-3"},
        "model_info": {"id": "id-owned-blocked", "blocked": True, "team_id": "team_alpha"},
    }
    active_granted = {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "gpt-4o"},
        "model_info": {
            "id": "id-granted-active",
            "blocked": False,
            "access_via_team_ids": ["team_alpha"],
        },
    }
    all_models = [blocked_granted, blocked_owned, active_granted]

    mock_router = MagicMock()
    mock_router.get_model_access_groups.return_value = {}
    mock_router.get_model_list = MagicMock(return_value=[])

    team_db = _make_team(models=["gpt-4o"])

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_db)
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

    result = await _filter_models_by_team_id(
        all_models=all_models,
        team_id="team_alpha",
        prisma_client=mock_prisma,
        llm_router=mock_router,
    )

    result_ids = {m["model_info"]["id"] for m in result}
    assert result_ids == {
        "id-owned-blocked",
        "id-granted-active",
    }, f"Blocked non-owned deployments must be hidden, team-owned kept; got {result_ids}"
