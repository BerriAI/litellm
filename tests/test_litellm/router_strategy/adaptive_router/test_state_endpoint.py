"""Tests for the GET /adaptive_router/state introspection endpoint and the
underlying `AdaptiveRouter.get_state_snapshot()` helper."""

import time
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.router_strategy.adaptive_router.adaptive_router import AdaptiveRouter
from litellm.router_strategy.adaptive_router.bandit import BanditCell, apply_delta
from litellm.types.router import (
    AdaptiveRouterConfig,
    AdaptiveRouterPreferences,
    RequestType,
)


def _make_router(name: str = "r1") -> AdaptiveRouter:
    cfg = AdaptiveRouterConfig(available_models=["fast", "smart"])
    prefs = {
        "fast": AdaptiveRouterPreferences(quality_tier=1, strengths=[]),
        "smart": AdaptiveRouterPreferences(
            quality_tier=3, strengths=[RequestType.CODE_GENERATION]
        ),
    }
    costs = {"fast": 0.0001, "smart": 0.001}
    return AdaptiveRouter(
        router_name=name,
        config=cfg,
        model_to_prefs=prefs,
        model_to_cost=costs,
    )


# ---- snapshot helper ---------------------------------------------------


@pytest.mark.asyncio
async def test_get_state_snapshot_returns_cell_per_request_type_per_model():
    r = _make_router()
    snap = await r.get_state_snapshot()

    # Top-level shape
    assert snap["router_name"] == "r1"
    assert snap["available_models"] == ["fast", "smart"]
    assert snap["weights"] == {"quality": 0.7, "cost": 0.3}
    assert snap["model_costs"] == {"fast": 0.0001, "smart": 0.001}
    assert snap["owner_cache_live"] == 0
    assert snap["skipped_updates_total"] == 0
    assert set(snap["queue"].keys()) == {
        "state_pending",
        "session_pending",
        "max_state_seen",
        "max_session_seen",
    }

    # 7 request types x 2 models = 14 cells
    assert len(snap["cells"]) == len(list(RequestType)) * 2
    for cell in snap["cells"]:
        assert set(cell.keys()) == {
            "request_type",
            "model",
            "alpha",
            "beta",
            "samples",
            "quality_mean",
        }
        assert cell["model"] in {"fast", "smart"}
        assert cell["request_type"] in {rt.value for rt in RequestType}


@pytest.mark.asyncio
async def test_get_state_snapshot_quality_mean_matches_alpha_over_total():
    r = _make_router()

    # Manually mutate one cell to a known state so the math is verifiable.
    key = (RequestType.CODE_GENERATION, "smart")
    r._cells[key] = apply_delta(r._cells[key], delta_alpha=10.0, delta_beta=0.0)
    expected = r._cells[key]
    expected_mean = expected.alpha / (expected.alpha + expected.beta)

    snap = await r.get_state_snapshot()
    cell = next(
        c
        for c in snap["cells"]
        if c["request_type"] == "code_generation" and c["model"] == "smart"
    )
    assert cell["alpha"] == expected.alpha
    assert cell["beta"] == expected.beta
    # `samples` reports net observations after subtracting the cold-start
    # prior mass, so operators aren't misled by the initial value.
    assert cell["samples"] == expected.total_samples
    assert cell["quality_mean"] == pytest.approx(expected_mean)


@pytest.mark.asyncio
async def test_get_state_snapshot_counts_only_live_owner_cache_entries():
    r = _make_router()
    now = time.time()
    r._owner_cache["live-1"] = ("fast", now + 3600)
    r._owner_cache["live-2"] = ("smart", now + 3600)
    r._owner_cache["expired-1"] = ("fast", now - 1)

    snap = await r.get_state_snapshot()
    assert snap["owner_cache_live"] == 2


@pytest.mark.asyncio
async def test_get_state_snapshot_exposes_skipped_updates_total():
    r = _make_router()
    r._skipped_updates_total = 7
    snap = await r.get_state_snapshot()
    assert snap["skipped_updates_total"] == 7


# ---- endpoint --------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoint_returns_404_when_no_adaptive_router(monkeypatch):
    """When llm_router is set but has no adaptive routers configured, return 404."""
    from litellm.proxy import proxy_server

    fake_router = MagicMock()
    fake_router.adaptive_routers = {}
    monkeypatch.setattr(proxy_server, "llm_router", fake_router)

    admin = UserAPIKeyAuth(api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN)
    with pytest.raises(HTTPException) as exc:
        await proxy_server.get_adaptive_router_state(user_api_key_dict=admin)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_endpoint_returns_404_when_llm_router_is_none(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", None)

    admin = UserAPIKeyAuth(api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN)
    with pytest.raises(HTTPException) as exc:
        await proxy_server.get_adaptive_router_state(user_api_key_dict=admin)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_endpoint_rejects_non_admin_role(monkeypatch):
    from litellm.proxy import proxy_server

    fake_router = MagicMock()
    fake_router.adaptive_routers = {"r1": _make_router()}
    monkeypatch.setattr(proxy_server, "llm_router", fake_router)

    non_admin = UserAPIKeyAuth(
        api_key="sk-user", user_role=LitellmUserRoles.INTERNAL_USER
    )
    with pytest.raises(HTTPException) as exc:
        await proxy_server.get_adaptive_router_state(user_api_key_dict=non_admin)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_endpoint_returns_snapshot_list_for_admin(monkeypatch):
    """Single configured router still returns the {"routers": [...]} list shape."""
    from litellm.proxy import proxy_server

    fake_router = MagicMock()
    fake_router.adaptive_routers = {"r1": _make_router("r1")}
    monkeypatch.setattr(proxy_server, "llm_router", fake_router)

    admin = UserAPIKeyAuth(api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN)
    result = await proxy_server.get_adaptive_router_state(user_api_key_dict=admin)
    assert list(result.keys()) == ["routers"]
    assert len(result["routers"]) == 1
    snap = result["routers"][0]
    assert snap["router_name"] == "r1"
    assert snap["available_models"] == ["fast", "smart"]
    assert len(snap["cells"]) == len(list(RequestType)) * 2


@pytest.mark.asyncio
async def test_endpoint_returns_one_snapshot_per_router(monkeypatch):
    """With multiple adaptive routers configured, return one snapshot per router."""
    from litellm.proxy import proxy_server

    fake_router = MagicMock()
    fake_router.adaptive_routers = {
        "r1": _make_router("r1"),
        "r2": _make_router("r2"),
    }
    monkeypatch.setattr(proxy_server, "llm_router", fake_router)

    admin = UserAPIKeyAuth(api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN)
    result = await proxy_server.get_adaptive_router_state(user_api_key_dict=admin)
    names = sorted(s["router_name"] for s in result["routers"])
    assert names == ["r1", "r2"]
