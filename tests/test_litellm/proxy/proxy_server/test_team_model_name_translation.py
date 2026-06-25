"""Coverage for team-scoped model-name translation in /model/info responses.

These live in tests/test_litellm/proxy/proxy_server/ (not the top-level
test_proxy_server.py) because the CI coverage job collects this directory.
They exercise the read-path fix for issue #28382: `/v1`, `/v2`, and
`/model/info` must surface `model_info.team_public_model_name` for team-scoped
rows instead of the internal routing key `model_name_{team_id}_{uuid}`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import (
    LiteLLM_UserTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.model_listing_utils import TeamModelNameTranslator
from litellm.proxy.proxy_server import (
    _get_proxy_model_info,
    _translate_model_name_for_response,
)


def _team_row() -> dict:
    return {
        "model_name": "model_name_team-abc-123_4a6b8",
        "litellm_params": {"model": "azure/gpt-5.2-low-rpm-testing"},
        "model_info": {
            "id": "byok-id-1",
            "team_id": "team-abc-123",
            "team_public_model_name": "team-claude-sonnet",
            "db_model": True,
        },
    }


def test_translate_swaps_internal_name_for_public():
    """Team-scoped row: model_name is swapped to the public name."""
    result = _translate_model_name_for_response(_team_row())
    assert result["model_name"] == "team-claude-sonnet"


def test_translate_leaves_global_row_untouched():
    """No team_id / team_public_model_name -> pass through unchanged."""
    model = {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "gpt-4o"},
        "model_info": {"id": "normal-id-1", "db_model": False},
    }
    assert _translate_model_name_for_response(model)["model_name"] == "gpt-4o"


def test_translate_leaves_non_internal_shape_untouched():
    """Team row whose model_name is not the internal routing key is not rewritten."""
    model = _team_row()
    model["model_name"] = "already-public-name"
    assert (
        _translate_model_name_for_response(model)["model_name"] == "already-public-name"
    )


def test_translate_handles_missing_or_non_dict_model_info():
    """Missing / None / non-dict model_info, and a non-dict model, must not raise."""
    # missing model_info
    assert _translate_model_name_for_response({"model_name": "x"})["model_name"] == "x"
    # model_info is None -> coerced to {} -> no team fields
    assert (
        _translate_model_name_for_response({"model_name": "x", "model_info": None})[
            "model_name"
        ]
        == "x"
    )
    # model_info is a truthy non-dict (e.g. a stray string) -> early return
    assert (
        _translate_model_name_for_response(
            {"model_name": "x", "model_info": "garbage"}
        )["model_name"]
        == "x"
    )
    # model itself is not a dict
    assert _translate_model_name_for_response("not-a-dict") == "not-a-dict"  # type: ignore[arg-type]


def test_translate_does_not_mutate_input():
    """Returns a shallow copy; the router's in-memory list keeps the routing key."""
    model = _team_row()
    result = _translate_model_name_for_response(model)
    assert result is not model
    assert model["model_name"] == "model_name_team-abc-123_4a6b8"


def test_get_proxy_model_info_returns_public_name_for_team_row():
    """`_get_proxy_model_info` must return the public name for a team-scoped
    row. Because _translate_model_name_for_response returns a shallow copy
    (it does not mutate), callers MUST use the return value -- the
    `/v1/model/info` list path historically discarded it, leaking the internal
    routing key (#28382)."""
    # Mirror the (fixed) /v1/model/info list path: assign the return back.
    all_models = [_get_proxy_model_info(model=m) for m in [_team_row()]]
    assert all_models[0]["model_name"] == "team-claude-sonnet"


@pytest.mark.asyncio
async def test_model_info_v2_translates_team_model_name(monkeypatch):
    """/v2/model/info must surface the public name for team-scoped rows.
    Covers the translation step in model_info_v2 (the read-path call site)."""
    router = MagicMock()
    router.model_list = [_team_row()]

    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "prisma_client", MagicMock())
    monkeypatch.setattr(ps.proxy_config, "get_config", AsyncMock(return_value={}))
    monkeypatch.setattr(
        ps,
        "_apply_search_filter_to_models",
        AsyncMock(side_effect=lambda all_models, **kw: (all_models, len(all_models))),
    )
    monkeypatch.setattr(
        ps, "_enrich_model_info_with_litellm_data", lambda model, **kw: model
    )
    import litellm.proxy.agent_endpoints.model_list_helpers as mlh

    monkeypatch.setattr(
        mlh,
        "append_agents_to_model_info",
        AsyncMock(side_effect=lambda models, **kw: models),
    )

    admin = UserAPIKeyAuth(user_id="u", user_role=LitellmUserRoles.PROXY_ADMIN)
    # Pass every query param explicitly: called directly (not through FastAPI),
    # the fastapi.Query(...) defaults are Query objects, not their values.
    resp = await ps.model_info_v2(
        user_api_key_dict=admin,
        model=None,
        user_models_only=False,
        include_team_models=False,
        debug=False,
        page=1,
        size=50,
        search=None,
        modelId=None,
        teamId=None,
        sortBy=None,
        sortOrder="asc",
    )

    names = [m["model_name"] for m in resp["data"]]
    assert "team-claude-sonnet" in names
    assert "model_name_team-abc-123_4a6b8" not in names


@pytest.mark.asyncio
async def test_model_info_v1_list_path_translates_team_model_name(monkeypatch):
    """/v1/model/info list path (no litellm_model_id) must include team-scoped
    deployments from the router model list and surface the public name (#28382)."""
    team_row = _team_row()
    global_row = {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "gpt-4o"},
        "model_info": {"id": "normal-id-1", "db_model": False},
    }
    router = MagicMock()
    router.model_list = [team_row, global_row]
    router.get_model_names.return_value = ["gpt-4o"]
    router.get_model_access_groups.return_value = {}

    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "llm_model_list", router.model_list)
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(
        ps, "_enrich_model_info_with_litellm_data", lambda model, **kw: model
    )

    admin = UserAPIKeyAuth(
        user_id="u", user_role=LitellmUserRoles.PROXY_ADMIN, team_models=[]
    )
    resp = await ps.model_info_v1(user_api_key_dict=admin, litellm_model_id=None)

    names = [m["model_name"] for m in resp["data"]]
    assert "team-claude-sonnet" in names
    assert "model_name_team-abc-123_4a6b8" not in names


@pytest.mark.asyncio
async def test_model_info_v1_unrestricted_key_returns_all_deployments(monkeypatch):
    """Unrestricted keys must see all router deployments (legacy v1 access logic)."""
    deployment = {
        "model_name": "gpt-4",
        "litellm_params": {"model": "gpt-4"},
        "model_info": {"id": "global-id-1", "db_model": False},
    }
    router = MagicMock()
    router.model_list = [deployment]
    router.get_model_names.return_value = ["gpt-4"]
    router.get_model_access_groups.return_value = {}

    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "llm_model_list", router.model_list)
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(
        ps, "_enrich_model_info_with_litellm_data", lambda model, **kw: model
    )

    caller = UserAPIKeyAuth(
        user_id="user-1",
        user_role=LitellmUserRoles.INTERNAL_USER,
        models=[],
        team_models=[],
    )
    resp = await ps.model_info_v1(user_api_key_dict=caller, litellm_model_id=None)

    assert [m["model_name"] for m in resp["data"]] == ["gpt-4"]


@pytest.mark.asyncio
async def test_model_info_v1_restricted_key_filters_deployments(monkeypatch):
    """Key-level model allowlists must filter router deployments."""
    team_row = _team_row()
    global_row = {
        "model_name": "gpt-4",
        "litellm_params": {"model": "gpt-4"},
        "model_info": {"id": "global-id-1", "db_model": False},
    }
    router = MagicMock()
    router.model_list = [team_row, global_row]
    router.get_model_names.return_value = ["gpt-4", "team-claude-sonnet"]
    router.get_model_access_groups.return_value = {}

    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "llm_model_list", router.model_list)
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(
        ps, "_enrich_model_info_with_litellm_data", lambda model, **kw: model
    )

    caller = UserAPIKeyAuth(
        user_id="user-1",
        user_role=LitellmUserRoles.INTERNAL_USER,
        models=["gpt-4"],
        team_models=[],
    )
    resp = await ps.model_info_v1(user_api_key_dict=caller, litellm_model_id=None)

    assert [m["model_name"] for m in resp["data"]] == ["gpt-4"]


def _other_team_row() -> dict:
    return {
        "model_name": "model_name_team-other_9f2c1",
        "litellm_params": {
            "model": "azure/gpt-5.2-low-rpm-testing",
            "api_base": "https://team-other-private.example.com",
        },
        "model_info": {
            "id": "byok-id-other",
            "team_id": "team-other",
            "team_public_model_name": "team-claude-sonnet",
            "db_model": True,
        },
    }


@pytest.mark.asyncio
async def test_model_info_v1_unrestricted_key_hides_other_team_byok(monkeypatch):
    """Unrestricted non-admin keys must not enumerate other teams' BYOK
    deployments, but must still see global models and their own team's."""
    team_row = _team_row()
    other_team_row = _other_team_row()
    global_row = {
        "model_name": "gpt-4",
        "litellm_params": {"model": "gpt-4"},
        "model_info": {"id": "global-id-1", "db_model": False},
    }
    router = MagicMock()
    router.model_list = [team_row, other_team_row, global_row]
    router.get_model_names.return_value = ["gpt-4"]
    router.get_model_access_groups.return_value = {}

    prisma_client = MagicMock()
    caller_user_row = MagicMock()
    caller_user_row.teams = ["team-abc-123"]
    caller_user_row.model_dump.return_value = {
        "user_id": "user-1",
        "teams": ["team-abc-123"],
        "models": [],
    }
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(
        return_value=caller_user_row
    )

    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "llm_model_list", router.model_list)
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "prisma_client", prisma_client)
    monkeypatch.setattr(ps, "get_all_team_models", AsyncMock(return_value={}))
    monkeypatch.setattr(
        ps, "_enrich_model_info_with_litellm_data", lambda model, **kw: model
    )

    caller = UserAPIKeyAuth(
        user_id="user-1",
        user_role=LitellmUserRoles.INTERNAL_USER,
        models=[],
        team_models=[],
    )
    resp = await ps.model_info_v1(user_api_key_dict=caller, litellm_model_id=None)

    returned_ids = {m["model_info"]["id"] for m in resp["data"]}
    assert returned_ids == {"global-id-1", "byok-id-1"}
    assert "byok-id-other" not in returned_ids
    names = [m["model_name"] for m in resp["data"]]
    assert "team-claude-sonnet" in names
    assert "gpt-4" in names


@pytest.mark.asyncio
async def test_model_info_v1_service_key_hides_all_team_byok(monkeypatch):
    """A key with no resolvable user and no team (e.g. a CI/service token
    created outside any team) sees only global deployments, never team-scoped
    BYOK rows. A team-scoped key does see its own team's rows (issue #30983),
    pinned by the /model/info route tests."""
    team_row = _team_row()
    other_team_row = _other_team_row()
    global_row = {
        "model_name": "gpt-4",
        "litellm_params": {"model": "gpt-4"},
        "model_info": {"id": "global-id-1", "db_model": False},
    }
    router = MagicMock()
    router.model_list = [team_row, other_team_row, global_row]
    router.get_model_names.return_value = ["gpt-4"]
    router.get_model_access_groups.return_value = {}

    prisma_client = MagicMock()

    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "llm_model_list", router.model_list)
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "prisma_client", prisma_client)
    monkeypatch.setattr(
        ps, "_enrich_model_info_with_litellm_data", lambda model, **kw: model
    )

    caller = UserAPIKeyAuth(
        user_id=None,
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id=None,
        models=[],
        team_models=[],
    )
    resp = await ps.model_info_v1(user_api_key_dict=caller, litellm_model_id=None)

    assert [m["model_info"]["id"] for m in resp["data"]] == ["global-id-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "find_unique",
    [
        AsyncMock(return_value=MagicMock(teams=[])),
        AsyncMock(return_value=None),
        AsyncMock(side_effect=RuntimeError("db down")),
    ],
    ids=["user-not-in-team", "user-row-missing", "user-lookup-error"],
)
async def test_model_info_v1_team_key_sees_own_byok_regardless_of_user_lookup(
    monkeypatch, find_unique
):
    """A team-scoped key sees its own team's BYOK rows even when the bound user
    is not a member of that team, has no DB row, or the lookup errors; the
    key's team_id is authoritative (issue #30983). Other teams' rows stay
    hidden."""
    global_row = {
        "model_name": "gpt-4",
        "litellm_params": {"model": "gpt-4"},
        "model_info": {"id": "global-id-1", "db_model": False},
    }
    router = MagicMock()
    router.model_list = [_team_row(), _other_team_row(), global_row]
    router.get_model_names.return_value = ["gpt-4"]
    router.get_model_access_groups.return_value = {}

    prisma_client = MagicMock()
    prisma_client.db.litellm_usertable.find_unique = find_unique

    async def _populate(**kwargs):
        return kwargs["all_models"]

    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "llm_model_list", router.model_list)
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "prisma_client", prisma_client)
    monkeypatch.setattr(ps, "_populate_team_access_on_models", _populate)
    monkeypatch.setattr(
        ps, "_enrich_model_info_with_litellm_data", lambda model, **kw: model
    )

    caller = UserAPIKeyAuth(
        user_id="user-1",
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id="team-abc-123",
        models=[],
        team_models=[],
    )
    resp = await ps.model_info_v1(user_api_key_dict=caller, litellm_model_id=None)

    assert [m["model_info"]["id"] for m in resp["data"]] == ["byok-id-1", "global-id-1"]


@pytest.mark.asyncio
async def test_model_info_v1_user_team_membership_grants_byok(monkeypatch):
    """A user's own team memberships still grant that team's BYOK rows, unioned
    with any team the key itself is scoped to."""
    global_row = {
        "model_name": "gpt-4",
        "litellm_params": {"model": "gpt-4"},
        "model_info": {"id": "global-id-1", "db_model": False},
    }
    router = MagicMock()
    router.model_list = [_team_row(), _other_team_row(), global_row]
    router.get_model_names.return_value = ["gpt-4"]
    router.get_model_access_groups.return_value = {}

    prisma_client = MagicMock()
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(
        return_value=MagicMock(teams=["team-other"])
    )

    async def _populate(**kwargs):
        return kwargs["all_models"]

    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "llm_model_list", router.model_list)
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "prisma_client", prisma_client)
    monkeypatch.setattr(ps, "_populate_team_access_on_models", _populate)
    monkeypatch.setattr(
        ps, "_enrich_model_info_with_litellm_data", lambda model, **kw: model
    )

    caller = UserAPIKeyAuth(
        user_id="user-2",
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id=None,
        models=[],
        team_models=[],
    )
    resp = await ps.model_info_v1(user_api_key_dict=caller, litellm_model_id=None)

    assert [m["model_info"]["id"] for m in resp["data"]] == [
        "byok-id-other",
        "global-id-1",
    ]


@pytest.mark.asyncio
async def test_model_info_v1_populates_access_via_team_ids(monkeypatch):
    """`/v1/model/info` must populate access_via_team_ids when the DB is connected."""
    team_id = "team-abc-123"
    team_row = _team_row()
    global_row = {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "gpt-4o"},
        "model_info": {"id": "global-id-1", "db_model": False},
    }
    router = MagicMock()
    router.model_list = [team_row, global_row]
    router.get_model_names.return_value = ["gpt-4o", "team-claude-sonnet"]
    router.get_model_access_groups.return_value = {}
    router.get_model_ids.return_value = ["global-id-1"]

    prisma_client = MagicMock()

    async def _fake_populate(**kwargs):
        for model in kwargs["all_models"]:
            model_id = model["model_info"]["id"]
            if model_id == "byok-id-1":
                model["model_info"]["access_via_team_ids"] = [team_id]
                model["model_info"]["direct_access"] = False
            elif model_id == "global-id-1":
                model["model_info"]["direct_access"] = True
        return kwargs["all_models"]

    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "llm_model_list", router.model_list)
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "prisma_client", prisma_client)
    monkeypatch.setattr(ps, "_populate_team_access_on_models", _fake_populate)
    monkeypatch.setattr(
        ps, "_enrich_model_info_with_litellm_data", lambda model, **kw: model
    )

    admin = UserAPIKeyAuth(
        user_id="u", user_role=LitellmUserRoles.PROXY_ADMIN, team_models=[]
    )
    resp = await ps.model_info_v1(user_api_key_dict=admin, litellm_model_id=None)

    by_id = {m["model_info"]["id"]: m for m in resp["data"]}
    assert by_id["byok-id-1"]["model_info"]["access_via_team_ids"] == [team_id]
    assert by_id["byok-id-1"]["model_info"]["direct_access"] is False
    assert by_id["global-id-1"]["model_info"]["direct_access"] is True


@pytest.mark.asyncio
async def test_populate_team_access_sets_direct_access_false_by_default(monkeypatch):
    """Team-accessible models without direct access must return direct_access=false."""
    team_row = _team_row()
    global_row = {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "gpt-4o"},
        "model_info": {"id": "global-id-1", "db_model": False},
    }
    router = MagicMock()
    router.get_model_ids.return_value = ["global-id-1"]
    monkeypatch.setattr(
        ps,
        "get_all_team_models",
        AsyncMock(return_value={"byok-id-1": ["team-abc-123"]}),
    )

    admin = UserAPIKeyAuth(
        user_id="u", user_role=LitellmUserRoles.PROXY_ADMIN, team_models=[]
    )
    result = await ps._populate_team_access_on_models(
        user_api_key_dict=admin,
        prisma_client=MagicMock(),
        llm_router=router,
        all_models=[team_row, global_row],
    )

    by_id = {m["model_info"]["id"]: m for m in result}
    assert by_id["byok-id-1"]["model_info"]["direct_access"] is False
    assert by_id["global-id-1"]["model_info"]["direct_access"] is True


@pytest.mark.asyncio
async def test_model_info_v1_team_id_without_db_fails_fast(monkeypatch):
    """`teamId` without a connected DB raises 500 before any enrichment work runs."""
    router = MagicMock()
    router.model_list = [_team_row()]

    enrich_spy = MagicMock(side_effect=lambda model, **kw: model)

    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "llm_model_list", router.model_list)
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps, "_enrich_model_info_with_litellm_data", enrich_spy)

    admin = UserAPIKeyAuth(
        user_id="u", user_role=LitellmUserRoles.PROXY_ADMIN, team_models=[]
    )

    with pytest.raises(ps.HTTPException) as exc_info:
        await ps.model_info_v1(
            user_api_key_dict=admin, litellm_model_id=None, teamId="team-abc-123"
        )

    assert exc_info.value.status_code == 500
    assert "DB not connected" in exc_info.value.detail["error"]
    enrich_spy.assert_not_called()


@pytest.mark.asyncio
async def test_model_info_v1_include_team_models_without_db_fails_fast(monkeypatch):
    """`include_team_models` without a connected DB raises 500 instead of silently
    returning an empty list (the access fields can only be populated from the DB)."""
    router = MagicMock()
    router.model_list = [_team_row()]

    enrich_spy = MagicMock(side_effect=lambda model, **kw: model)

    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "llm_model_list", router.model_list)
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps, "_enrich_model_info_with_litellm_data", enrich_spy)

    admin = UserAPIKeyAuth(
        user_id="u", user_role=LitellmUserRoles.PROXY_ADMIN, team_models=[]
    )

    with pytest.raises(ps.HTTPException) as exc_info:
        await ps.model_info_v1(
            user_api_key_dict=admin, litellm_model_id=None, include_team_models=True
        )

    assert exc_info.value.status_code == 500
    assert "DB not connected" in exc_info.value.detail["error"]
    enrich_spy.assert_not_called()


@pytest.mark.asyncio
async def test_model_info_v1_litellm_model_id_team_id_without_db_fails_fast(
    monkeypatch,
):
    """`litellm_model_id` + `teamId` without a connected DB must raise 500 too, not
    return 200 with a model dict missing direct_access/access_via_team_ids."""
    router = MagicMock()
    router.model_list = [_team_row()]

    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "llm_model_list", router.model_list)
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "prisma_client", None)

    admin = UserAPIKeyAuth(
        user_id="u", user_role=LitellmUserRoles.PROXY_ADMIN, team_models=[]
    )

    with pytest.raises(ps.HTTPException) as exc_info:
        await ps.model_info_v1(
            user_api_key_dict=admin,
            litellm_model_id="byok-id-1",
            teamId="team-abc-123",
        )

    assert exc_info.value.status_code == 500
    assert "DB not connected" in exc_info.value.detail["error"]
    router.get_deployment.assert_not_called()


@pytest.mark.asyncio
async def test_model_info_v1_litellm_model_id_include_team_models_filters_inaccessible(
    monkeypatch,
):
    """`litellm_model_id` + `include_team_models` must drop a model the caller cannot
    use instead of returning it unconditionally from the single-model lookup."""
    team_row = _team_row()

    router = MagicMock()
    deployment = MagicMock()
    deployment.model_dump.return_value = team_row
    router.get_deployment.return_value = deployment

    async def _fake_populate(**kwargs):
        for model in kwargs["all_models"]:
            model["model_info"]["direct_access"] = False
            model["model_info"]["access_via_team_ids"] = []
        return kwargs["all_models"]

    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "llm_model_list", [team_row])
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "prisma_client", MagicMock())
    monkeypatch.setattr(ps, "_get_proxy_model_info", lambda model: team_row)
    monkeypatch.setattr(ps, "_populate_team_access_on_models", _fake_populate)

    caller = UserAPIKeyAuth(
        user_id="u", user_role=LitellmUserRoles.INTERNAL_USER, team_models=[]
    )
    resp = await ps.model_info_v1(
        user_api_key_dict=caller,
        litellm_model_id="byok-id-1",
        include_team_models=True,
    )

    assert resp["data"] == []


@pytest.mark.asyncio
async def test_model_info_v1_litellm_model_id_team_id_applies_team_filter(monkeypatch):
    """`litellm_model_id` + `teamId` must run the teamId filter on the single model
    rather than returning it regardless of the team's access."""
    team_row = _team_row()

    router = MagicMock()
    deployment = MagicMock()
    deployment.model_dump.return_value = team_row
    router.get_deployment.return_value = deployment

    async def _fake_populate(**kwargs):
        return kwargs["all_models"]

    team_filter = AsyncMock(return_value=[])

    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "llm_model_list", [team_row])
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "prisma_client", MagicMock())
    monkeypatch.setattr(ps, "_get_proxy_model_info", lambda model: team_row)
    monkeypatch.setattr(ps, "_populate_team_access_on_models", _fake_populate)
    monkeypatch.setattr(ps, "_filter_models_by_team_id", team_filter)

    admin = UserAPIKeyAuth(
        user_id="u", user_role=LitellmUserRoles.PROXY_ADMIN, team_models=[]
    )
    resp = await ps.model_info_v1(
        user_api_key_dict=admin,
        litellm_model_id="byok-id-1",
        teamId="other-team",
    )

    assert resp["data"] == []
    team_filter.assert_awaited_once()
    assert team_filter.await_args.kwargs["team_id"] == "other-team"
    assert team_filter.await_args.kwargs["all_models"] == [team_row]


@pytest.mark.asyncio
async def test_v1_models_translates_team_model_for_access_group_key(monkeypatch):
    """Regression (#28382 sibling leak): a virtual key whose model access group
    resolves to a team BYOK deployment must list the PUBLIC name in /v1/models,
    not the internal routing key model_name_{team_id}_{uuid}.

    The /model/info read-path fix did not cover /v1/models, which builds from
    bare model-name strings via access-group expansion.
    """
    team_dep = {
        "model_name": "model_name_teamX_uuid9",
        "litellm_params": {"model": "azure/gpt-4.1"},
        "model_info": {
            "id": "id1",
            "team_id": "teamX",
            "team_public_model_name": "tushar-gpt-4.1",
            "access_groups": ["grp-a"],
        },
    }
    router = MagicMock()
    router.get_model_names.return_value = ["model_name_teamX_uuid9"]
    router.get_model_access_groups.return_value = {"grp-a": ["model_name_teamX_uuid9"]}
    router.get_fully_blocked_model_names.return_value = set()
    router.model_list = [team_dep]
    router.get_model_list.return_value = [team_dep]

    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "user_model", None)
    # Default behavior: listing surfaces public names.
    monkeypatch.setattr(ps, "general_settings", {})

    # virtual key granted access via the access group (no team membership)
    key = UserAPIKeyAuth(
        user_id="u", api_key="sk-test", models=["grp-a"], team_models=[]
    )
    resp = await ps.model_list(user_api_key_dict=key)

    ids = [d["id"] for d in resp["data"]]
    assert "tushar-gpt-4.1" in ids
    assert "model_name_teamX_uuid9" not in ids


@pytest.mark.asyncio
async def test_v1_models_keeps_internal_names_when_public_name_flag_disabled(
    monkeypatch,
):
    """Compatibility override: /v1/models can still list the internal routing
    name for consumers that scripted against those ids. Translation is enabled
    by default and disabled via general_settings['use_team_public_model_name'].
    """
    team_dep = {
        "model_name": "model_name_teamX_uuid9",
        "litellm_params": {"model": "azure/gpt-4.1"},
        "model_info": {
            "id": "id1",
            "team_id": "teamX",
            "team_public_model_name": "tushar-gpt-4.1",
            "access_groups": ["grp-a"],
        },
    }
    router = MagicMock()
    router.get_model_names.return_value = ["model_name_teamX_uuid9"]
    router.get_model_access_groups.return_value = {"grp-a": ["model_name_teamX_uuid9"]}
    router.get_fully_blocked_model_names.return_value = set()
    router.model_list = [team_dep]
    router.get_model_list.return_value = [team_dep]

    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "general_settings", {"use_team_public_model_name": False})

    key = UserAPIKeyAuth(
        user_id="u", api_key="sk-test", models=["grp-a"], team_models=[]
    )
    resp = await ps.model_list(user_api_key_dict=key)

    ids = [d["id"] for d in resp["data"]]
    assert "model_name_teamX_uuid9" in ids  # internal id preserved (backward-compat)
    assert "tushar-gpt-4.1" not in ids


@pytest.mark.asyncio
async def test_v1_models_translates_team_model_with_metadata(monkeypatch):
    """include_metadata=true must build metadata for the public model id."""
    team_dep = {
        "model_name": "model_name_teamX_uuid9",
        "litellm_params": {"model": "azure/gpt-4.1"},
        "model_info": {
            "id": "id1",
            "team_id": "teamX",
            "team_public_model_name": "tushar-gpt-4.1",
            "access_groups": ["grp-a"],
        },
    }
    router = MagicMock()
    router.get_model_names.return_value = ["model_name_teamX_uuid9"]
    router.get_model_access_groups.return_value = {"grp-a": ["model_name_teamX_uuid9"]}
    router.get_fully_blocked_model_names.return_value = set()
    router.model_list = [team_dep]
    router.get_model_list.return_value = [team_dep]
    router.get_model_group_info.return_value = None

    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "general_settings", {})

    key = UserAPIKeyAuth(
        user_id="u", api_key="sk-test", models=["grp-a"], team_models=[]
    )
    resp = await ps.model_list(user_api_key_dict=key, include_metadata=True)

    assert resp["data"] == [
        {
            "id": "tushar-gpt-4.1",
            "object": "model",
            "created": 1677610602,
            "owned_by": "openai",
            "metadata": {"fallbacks": []},
        }
    ]


@pytest.mark.asyncio
async def test_v1_models_metadata_fallbacks_use_internal_routing_key(monkeypatch):
    """Regression: with include_metadata=true, fallbacks configured for a team
    model under its internal routing key must still surface. The metadata lookup
    has to run against the internal name, not the translated public name (which
    the router's fallback config never keys on) -- otherwise fallbacks silently
    drop to []."""
    team_dep = {
        "model_name": "model_name_teamX_uuid9",
        "litellm_params": {"model": "azure/gpt-4.1"},
        "model_info": {
            "id": "id1",
            "team_id": "teamX",
            "team_public_model_name": "tushar-gpt-4.1",
            "access_groups": ["grp-a"],
        },
    }
    router = MagicMock()
    router.get_model_names.return_value = ["model_name_teamX_uuid9"]
    router.get_model_access_groups.return_value = {"grp-a": ["model_name_teamX_uuid9"]}
    router.get_fully_blocked_model_names.return_value = set()
    router.model_list = [team_dep]
    router.get_model_list.return_value = [team_dep]
    # Fallbacks are keyed on the internal routing name, as the router stores them.
    router.fallbacks = [{"model_name_teamX_uuid9": ["gpt-4o-backup"]}]
    router.get_model_group_info.return_value = None

    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "general_settings", {})

    key = UserAPIKeyAuth(
        user_id="u", api_key="sk-test", models=["grp-a"], team_models=[]
    )
    resp = await ps.model_list(user_api_key_dict=key, include_metadata=True)

    assert resp["data"] == [
        {
            "id": "tushar-gpt-4.1",
            "object": "model",
            "created": 1677610602,
            "owned_by": "openai",
            "metadata": {"fallbacks": ["gpt-4o-backup"]},
        }
    ]


@pytest.mark.asyncio
async def test_v1_models_metadata_does_not_leak_other_team_fallbacks(monkeypatch):
    """Regression: two teams can publish the same team_public_model_name. With
    include_metadata=true a caller scoped to teamX must see teamX's fallbacks for
    the shared public name, never teamY's. The metadata lookup has to stay within
    the caller's accessible models; resolving the public name through a router-wide
    reverse map could point it at another team's internal routing key."""
    team_x = {
        "model_name": "model_name_teamX_uuid9",
        "litellm_params": {"model": "azure/gpt-4.1"},
        "model_info": {
            "id": "idX",
            "team_id": "teamX",
            "team_public_model_name": "tushar-gpt-4.1",
            "access_groups": ["grp-a"],
        },
    }
    team_y = {
        "model_name": "model_name_teamY_uuidZ",
        "litellm_params": {"model": "azure/gpt-4.1"},
        "model_info": {
            "id": "idY",
            "team_id": "teamY",
            "team_public_model_name": "tushar-gpt-4.1",  # same public name, other team
        },
    }
    router = MagicMock()
    router.get_model_names.return_value = ["model_name_teamX_uuid9"]
    router.get_model_access_groups.return_value = {"grp-a": ["model_name_teamX_uuid9"]}
    router.get_fully_blocked_model_names.return_value = set()
    router.model_list = [team_x, team_y]
    router.get_model_list.return_value = [team_x, team_y]
    router.fallbacks = [
        {"model_name_teamX_uuid9": ["teamX-backup"]},
        {"model_name_teamY_uuidZ": ["teamY-backup"]},
    ]
    router.get_model_group_info.return_value = None

    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "general_settings", {})

    key = UserAPIKeyAuth(
        user_id="u", api_key="sk-test", models=["grp-a"], team_models=[]
    )
    resp = await ps.model_list(user_api_key_dict=key, include_metadata=True)

    assert resp["data"] == [
        {
            "id": "tushar-gpt-4.1",
            "object": "model",
            "created": 1677610602,
            "owned_by": "openai",
            "metadata": {"fallbacks": ["teamX-backup"]},
        }
    ]


def test_translate_team_model_names_for_listing_swaps_and_dedupes():
    """Internal team routing keys -> public name; sibling deployments sharing a
    public name collapse to one entry (order preserved); globals untouched."""
    router = MagicMock()
    router.get_model_list.return_value = [
        {
            "model_name": "model_name_teamX_uuidA",
            "model_info": {
                "team_id": "teamX",
                "team_public_model_name": "tushar-gpt-4.1",
            },
        },
        {
            "model_name": "model_name_teamX_uuidB",  # sibling: same public name
            "model_info": {
                "team_id": "teamX",
                "team_public_model_name": "tushar-gpt-4.1",
            },
        },
        {"model_name": "gpt-4o", "model_info": {"db_model": False}},
    ]

    out = TeamModelNameTranslator.translate_listing(
        ["model_name_teamX_uuidA", "model_name_teamX_uuidB", "gpt-4o"],
        router,
        {},
    )
    assert out == ["tushar-gpt-4.1", "gpt-4o"]


def test_listing_entries_keep_internal_lookup_id_for_team_rows():
    """`listing_entries` returns (public response id, internal lookup id) so the
    response shows the public name while metadata lookups keep the routing key.
    Sibling deployments collapse to one entry; globals map to themselves."""
    router = MagicMock()
    router.get_model_list.return_value = [
        {
            "model_name": "model_name_teamX_uuidA",
            "model_info": {
                "team_id": "teamX",
                "team_public_model_name": "tushar-gpt-4.1",
            },
        },
        {
            "model_name": "model_name_teamX_uuidB",  # sibling: same public name
            "model_info": {
                "team_id": "teamX",
                "team_public_model_name": "tushar-gpt-4.1",
            },
        },
        {"model_name": "gpt-4o", "model_info": {"db_model": False}},
    ]

    entries = TeamModelNameTranslator.listing_entries(
        ["model_name_teamX_uuidA", "model_name_teamX_uuidB", "gpt-4o"],
        router,
        {},
    )
    # public id for the client; an internal routing key for the metadata lookup
    assert entries[0][0] == "tushar-gpt-4.1"
    assert entries[0][1].startswith("model_name_teamX_uuid")
    assert entries[1] == ("gpt-4o", "gpt-4o")
    assert len(entries) == 2


def test_listing_entries_lookup_id_never_crosses_team_boundary():
    """Regression: when two teams share a team_public_model_name, the lookup id for
    the shared public name must stay within the caller's accessible model_names and
    never resolve to the other team's internal routing key (which would leak that
    team's fallback metadata under include_metadata=true)."""
    router = MagicMock()
    router.get_model_list.return_value = [
        {
            "model_name": "model_name_teamX_uuidA",
            "model_info": {
                "team_id": "teamX",
                "team_public_model_name": "shared-name",
            },
        },
        {
            "model_name": "model_name_teamY_uuidB",  # different team, same public name
            "model_info": {
                "team_id": "teamY",
                "team_public_model_name": "shared-name",
            },
        },
    ]

    # caller can only access teamX's internal key
    entries = TeamModelNameTranslator.listing_entries(
        ["model_name_teamX_uuidA"], router, {}
    )

    assert entries == [("shared-name", "model_name_teamX_uuidA")]


def test_listing_entries_global_wins_when_team_alias_collides_with_global():
    """Regression: when an accessible global model shares its name with a team
    deployment's `team_public_model_name`, the listing must keep the global
    entry rather than overwriting its lookup id with the colliding team's
    internal routing key (which would surface the team's metadata under the
    global id)."""
    router = MagicMock()
    router.get_model_list.return_value = [
        {
            "model_name": "model_name_teamX_uuidA",
            "model_info": {
                "team_id": "teamX",
                "team_public_model_name": "gpt-4o",
            },
        },
        {"model_name": "gpt-4o", "model_info": {"db_model": False}},
    ]

    entries = TeamModelNameTranslator.listing_entries(
        ["gpt-4o", "model_name_teamX_uuidA"], router, {}
    )

    assert entries == [("gpt-4o", "gpt-4o")]


def test_listing_and_resolve_agree_on_sibling_internal_key():
    """Regression: when two team deployments share a public name, listing and
    retrieve must pick the same internal routing key, otherwise `/v1/models/{id}`
    describes a different deployment than what the listing's metadata was built
    from."""
    router = MagicMock()
    router.get_model_list.return_value = [
        {
            "model_name": "model_name_teamX_uuidA",
            "model_info": {
                "team_id": "teamX",
                "team_public_model_name": "tushar-gpt-4.1",
            },
        },
        {
            "model_name": "model_name_teamX_uuidB",
            "model_info": {
                "team_id": "teamX",
                "team_public_model_name": "tushar-gpt-4.1",
            },
        },
    ]
    available = ["model_name_teamX_uuidA", "model_name_teamX_uuidB"]

    [(_, listing_lookup)] = TeamModelNameTranslator.listing_entries(
        available, router, {}
    )
    resolve_lookup = TeamModelNameTranslator.resolve_public_name(
        model_id="tushar-gpt-4.1",
        available_models=available,
        llm_router=router,
        general_settings={},
    )

    assert listing_lookup == resolve_lookup


def test_listing_entries_skips_empty_team_public_model_name():
    """Regression: a misconfigured row with `team_public_model_name: ""` must not
    produce a listing entry with an empty `id`; the internal routing key should
    pass through unchanged, matching `/v1/model/info`'s falsy-check behavior."""
    router = MagicMock()
    router.get_model_list.return_value = [
        {
            "model_name": "model_name_teamX_uuidA",
            "model_info": {
                "team_id": "teamX",
                "team_public_model_name": "",
            },
        },
    ]

    entries = TeamModelNameTranslator.listing_entries(
        ["model_name_teamX_uuidA"], router, {}
    )

    assert entries == [("model_name_teamX_uuidA", "model_name_teamX_uuidA")]


def test_listing_entries_passthrough_when_disabled():
    """Legacy flag / no router -> response id equals lookup id (no translation)."""
    assert TeamModelNameTranslator.listing_entries(["a", "b"], None, {}) == [
        ("a", "a"),
        ("b", "b"),
    ]


def test_translate_team_model_names_for_listing_leaves_unmapped_names():
    """Names with no team mapping (globals, access-group keys) pass through."""
    router = MagicMock()
    router.get_model_list.return_value = [
        {"model_name": "gpt-4o", "model_info": {"db_model": False}}
    ]

    assert TeamModelNameTranslator.translate_listing(
        ["gpt-4o", "beta-group"], router, {}
    ) == ["gpt-4o", "beta-group"]


def test_translate_team_model_names_for_listing_none_router():
    """No router -> return the input list unchanged."""
    assert TeamModelNameTranslator.translate_listing(["a", "b"], None, {}) == ["a", "b"]


def test_translate_team_model_names_for_listing_respects_legacy_flag():
    """Operators can keep returning the legacy internal routing key."""
    router = MagicMock()
    router.get_model_list.return_value = [
        {
            "model_name": "model_name_teamX_uuidA",
            "model_info": {
                "team_id": "teamX",
                "team_public_model_name": "tushar-gpt-4.1",
            },
        }
    ]

    assert TeamModelNameTranslator.translate_listing(
        ["model_name_teamX_uuidA"], router, {"use_team_public_model_name": False}
    ) == ["model_name_teamX_uuidA"]


def _public_named_router(*team_rows: dict) -> MagicMock:
    router = MagicMock()
    router.get_model_list.return_value = list(team_rows)
    return router


def test_resolve_public_name_to_internal_routing_key():
    """A public team name resolves back to the internal routing key the router
    indexes by, so `GET /v1/models/{public_name}` can find the deployment."""
    router = _public_named_router(_team_row())

    assert (
        TeamModelNameTranslator.resolve_public_name(
            model_id="team-claude-sonnet",
            available_models=["model_name_team-abc-123_4a6b8"],
            llm_router=router,
            general_settings={},
        )
        == "model_name_team-abc-123_4a6b8"
    )


def test_resolve_public_name_is_access_scoped_across_teams():
    """Two teams can publish the SAME public name. A caller's query must resolve
    to the internal key they can actually access, never another team's."""
    # both rows share public name "team-claude-sonnet"
    router = _public_named_router(_team_row(), _other_team_row())

    # caller only has access to their own team's internal key
    resolved = TeamModelNameTranslator.resolve_public_name(
        model_id="team-claude-sonnet",
        available_models=["model_name_team-abc-123_4a6b8"],
        llm_router=router,
        general_settings={},
    )
    assert resolved == "model_name_team-abc-123_4a6b8"
    assert resolved != "model_name_team-other_9f2c1"


def test_resolve_public_name_unmapped_passes_through():
    """A public name with no accessible internal mapping is returned unchanged so
    the caller hits the normal 404/access path; internal names pass through too."""
    router = _public_named_router(_team_row())

    # not accessible -> unchanged (downstream validate_model_access will 404)
    assert (
        TeamModelNameTranslator.resolve_public_name(
            model_id="team-claude-sonnet",
            available_models=[],
            llm_router=router,
            general_settings={},
        )
        == "team-claude-sonnet"
    )
    # already an internal routing key -> unchanged
    assert (
        TeamModelNameTranslator.resolve_public_name(
            model_id="model_name_team-abc-123_4a6b8",
            available_models=["model_name_team-abc-123_4a6b8"],
            llm_router=router,
            general_settings={},
        )
        == "model_name_team-abc-123_4a6b8"
    )


def test_resolve_public_name_respects_legacy_flag():
    """With the legacy flag set, no public-name resolution happens."""
    router = _public_named_router(_team_row())

    assert (
        TeamModelNameTranslator.resolve_public_name(
            model_id="team-claude-sonnet",
            available_models=["model_name_team-abc-123_4a6b8"],
            llm_router=router,
            general_settings={"use_team_public_model_name": False},
        )
        == "team-claude-sonnet"
    )


@pytest.mark.asyncio
async def test_retrieve_model_by_public_name_returns_200(monkeypatch):
    """Regression: `GET /v1/models/{public_name}` must NOT 404. The listing
    advertises the public team name, so retrieve must accept the same name,
    resolve it to the internal routing key for lookup, and echo the public name
    back as the model id."""
    import litellm
    import litellm.proxy.utils as proxy_utils

    team_row = _team_row()
    router = _public_named_router(team_row)
    deployment = MagicMock()
    deployment.litellm_params.model = "azure/gpt-5.2-low-rpm-testing"
    router.get_deployment_by_model_group_name.return_value = deployment

    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "general_settings", {})
    monkeypatch.setattr(
        proxy_utils,
        "get_available_models_for_user",
        AsyncMock(return_value=["model_name_team-abc-123_4a6b8"]),
    )
    monkeypatch.setattr(
        litellm, "get_llm_provider", lambda model: (model, "openai", None, None)
    )

    key = UserAPIKeyAuth(user_id="u", api_key="sk-test", team_models=[])
    resp = await ps.model_info(model_id="team-claude-sonnet", user_api_key_dict=key)

    assert resp["id"] == "team-claude-sonnet"
    # lookup happened by the internal routing key, not the public name
    router.get_deployment_by_model_group_name.assert_called_once_with(
        "model_name_team-abc-123_4a6b8"
    )


@pytest.mark.asyncio
async def test_retrieve_model_by_internal_name_returns_public_id(monkeypatch):
    """Regression: retrieving by the internal routing key must echo the SAME
    public id `/v1/models` advertises for that deployment, not the path. Otherwise
    a client iterating the listing's id and then retrieving each one would observe
    a different id depending on which alias they queried by."""
    import litellm
    import litellm.proxy.utils as proxy_utils

    router = _public_named_router(_team_row())
    deployment = MagicMock()
    deployment.litellm_params.model = "azure/gpt-5.2-low-rpm-testing"
    router.get_deployment_by_model_group_name.return_value = deployment

    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "general_settings", {})
    monkeypatch.setattr(
        proxy_utils,
        "get_available_models_for_user",
        AsyncMock(return_value=["model_name_team-abc-123_4a6b8"]),
    )
    monkeypatch.setattr(
        litellm, "get_llm_provider", lambda model: (model, "openai", None, None)
    )

    key = UserAPIKeyAuth(user_id="u", api_key="sk-test", team_models=[])
    resp = await ps.model_info(
        model_id="model_name_team-abc-123_4a6b8", user_api_key_dict=key
    )

    assert resp["id"] == "team-claude-sonnet"


@pytest.mark.asyncio
async def test_retrieve_model_by_internal_name_keeps_internal_id_when_flag_disabled(
    monkeypatch,
):
    """With `use_team_public_model_name=false`, retrieve must keep the internal
    routing key as the response id, mirroring `/v1/models`' legacy output."""
    import litellm
    import litellm.proxy.utils as proxy_utils

    router = _public_named_router(_team_row())
    deployment = MagicMock()
    deployment.litellm_params.model = "azure/gpt-5.2-low-rpm-testing"
    router.get_deployment_by_model_group_name.return_value = deployment

    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "general_settings", {"use_team_public_model_name": False})
    monkeypatch.setattr(
        proxy_utils,
        "get_available_models_for_user",
        AsyncMock(return_value=["model_name_team-abc-123_4a6b8"]),
    )
    monkeypatch.setattr(
        litellm, "get_llm_provider", lambda model: (model, "openai", None, None)
    )

    key = UserAPIKeyAuth(user_id="u", api_key="sk-test", team_models=[])
    resp = await ps.model_info(
        model_id="model_name_team-abc-123_4a6b8", user_api_key_dict=key
    )

    assert resp["id"] == "model_name_team-abc-123_4a6b8"


@pytest.mark.asyncio
async def test_retrieve_model_by_inaccessible_public_name_404s(monkeypatch):
    """A caller without access to a team model still gets 404 when retrieving by
    its public name; resolution never crosses the access boundary."""
    import litellm
    import litellm.proxy.utils as proxy_utils

    router = _public_named_router(_team_row())
    deployment = MagicMock()
    deployment.litellm_params.model = "azure/gpt-5.2-low-rpm-testing"
    router.get_deployment_by_model_group_name.return_value = deployment

    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "general_settings", {})
    monkeypatch.setattr(
        proxy_utils,
        "get_available_models_for_user",
        AsyncMock(return_value=[]),  # caller has no access
    )
    monkeypatch.setattr(
        litellm, "get_llm_provider", lambda model: (model, "openai", None, None)
    )

    key = UserAPIKeyAuth(user_id="u", api_key="sk-test", team_models=[])
    with pytest.raises(ps.HTTPException) as exc_info:
        await ps.model_info(model_id="team-claude-sonnet", user_api_key_dict=key)

    assert exc_info.value.status_code == 404
    router.get_deployment_by_model_group_name.assert_not_called()


def test_get_direct_access_models_expands_all_proxy_models_sentinel():
    """A user provisioned with 'all-proxy-models' has direct access to every non-team
    deployment. The sentinel must resolve via get_model_ids, not be looked up as a
    literal model_name (which matches nothing). Regression for GH#22791."""
    router = MagicMock()
    router.get_model_ids.return_value = ["global-id-1", "global-id-2"]
    router.get_model_list.return_value = []

    user = LiteLLM_UserTable(
        user_id="u",
        models=[ps.SpecialModelNames.all_proxy_models.value],
        teams=[],
    )

    result = ps.get_direct_access_models(user_db_object=user, llm_router=router)

    assert result == ["global-id-1", "global-id-2"]
    router.get_model_ids.assert_called_once_with(exclude_team_models=True)
    router.get_model_list.assert_not_called()


def test_get_direct_access_models_resolves_explicit_model_names():
    """Without the sentinel, only the user's explicitly listed models resolve to ids;
    the all-proxy-models shortcut must not fire."""
    router = MagicMock()
    router.get_model_list.return_value = [{"model_info": {"id": "gpt4o-id"}}]

    user = LiteLLM_UserTable(user_id="u", models=["gpt-4o"], teams=[])

    result = ps.get_direct_access_models(user_db_object=user, llm_router=router)

    assert result == ["gpt4o-id"]
    router.get_model_ids.assert_not_called()
    router.get_model_list.assert_called_once_with(model_name="gpt-4o")


@pytest.mark.asyncio
async def test_populate_team_access_grants_all_proxy_models_user_direct_access(
    monkeypatch,
):
    """An internal user provisioned with 'all-proxy-models' and no teams must see proxy
    models on the Models+Endpoints page. Before the fix _populate_team_access_on_models
    marked direct_access=False, so _filter_models_to_user_accessible dropped every model
    and the page was empty. Regression for GH#22791."""
    global_row = {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "gpt-4o"},
        "model_info": {"id": "global-id-1", "db_model": False},
    }

    router = MagicMock()
    router.get_model_ids.return_value = ["global-id-1"]

    user_row = LiteLLM_UserTable(
        user_id="u",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        models=[ps.SpecialModelNames.all_proxy_models.value],
        teams=[],
    )
    prisma_client = MagicMock()
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=user_row)

    monkeypatch.setattr(ps, "get_all_team_models", AsyncMock(return_value={}))

    caller = UserAPIKeyAuth(
        user_id="u", user_role=LitellmUserRoles.INTERNAL_USER, team_models=[]
    )

    populated = await ps._populate_team_access_on_models(
        user_api_key_dict=caller,
        prisma_client=prisma_client,
        llm_router=router,
        all_models=[global_row],
    )
    visible = ps._filter_models_to_user_accessible(populated)

    assert [m["model_info"]["id"] for m in visible] == ["global-id-1"]
    assert visible[0]["model_info"]["direct_access"] is True
