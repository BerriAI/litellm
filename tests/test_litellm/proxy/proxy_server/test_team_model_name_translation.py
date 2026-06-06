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
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
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
    """/v1/model/info list path (no litellm_model_id) must surface the public
    name. Covers the list comprehension that assigns _get_proxy_model_info's
    return back into all_models (#28382 review)."""
    router = MagicMock()
    router.get_model_names.return_value = ["team-claude-sonnet"]
    router.get_model_access_groups.return_value = {}
    router.get_model_list.return_value = [_team_row()]

    monkeypatch.setattr(ps, "user_model", None)
    monkeypatch.setattr(ps, "llm_model_list", [_team_row()])
    monkeypatch.setattr(ps, "llm_router", router)
    monkeypatch.setattr(ps, "get_key_models", lambda **kw: [])
    monkeypatch.setattr(ps, "get_team_models", lambda **kw: [])
    monkeypatch.setattr(
        ps, "get_complete_model_list", lambda **kw: ["team-claude-sonnet"]
    )

    admin = UserAPIKeyAuth(
        user_id="u", user_role=LitellmUserRoles.PROXY_ADMIN, team_models=[]
    )
    resp = await ps.model_info_v1(user_api_key_dict=admin, litellm_model_id=None)

    names = [m["model_name"] for m in resp["data"]]
    assert "team-claude-sonnet" in names
    assert "model_name_team-abc-123_4a6b8" not in names
