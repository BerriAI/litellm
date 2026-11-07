"""
Tests for health-check-driven routing filter in the Router.
"""

import time

import pytest

from litellm.router import Router


def _make_deployment(model_id: str, model_name: str = "gpt-4") -> dict:
    """Helper to create a deployment dict for testing."""
    return {
        "model_name": model_name,
        "litellm_params": {"model": "openai/gpt-4", "api_key": "fake"},
        "model_info": {"id": model_id},
    }


def _health_routed_router(enable: bool, unhealthy_ids: set[str], model_groups: list[str] | None = None) -> Router:
    router = Router(
        model_list=[
            _make_deployment("deploy-1", model_name="prod"),
            _make_deployment("deploy-2", model_name="prod"),
            _make_deployment("deploy-3", model_name="prod"),
            _make_deployment("other-1", model_name="other"),
            _make_deployment("other-2", model_name="other"),
        ],
        enable_health_check_routing=enable,
        background_health_check_model_groups=model_groups,
    )
    now = time.time()
    router.health().health_checks.set_deployment_health_states(
        {uid: {"is_healthy": False, "timestamp": now, "reason": "test_unhealthy"} for uid in unhealthy_ids}
    )
    return router


@pytest.mark.parametrize(
    ("enable", "unhealthy_ids", "model_groups", "routable"),
    [
        (True, {"deploy-2"}, None, {"prod": ["deploy-1", "deploy-3"], "other": ["other-1", "other-2"]}),
        (False, {"deploy-2"}, None, {"prod": ["deploy-1", "deploy-2", "deploy-3"], "other": ["other-1", "other-2"]}),
        (True, set(), None, {"prod": ["deploy-1", "deploy-2", "deploy-3"], "other": ["other-1", "other-2"]}),
        (
            True,
            {"deploy-1", "deploy-2", "deploy-3"},
            None,
            {"prod": ["deploy-1", "deploy-2", "deploy-3"], "other": ["other-1", "other-2"]},
        ),
        (True, {"deploy-2", "other-1"}, ["prod"], {"prod": ["deploy-1", "deploy-3"], "other": ["other-1", "other-2"]}),
        (True, {"deploy-2", "other-1"}, None, {"prod": ["deploy-1", "deploy-3"], "other": ["other-2"]}),
    ],
    ids=[
        "removes-unhealthy",
        "noop-when-disabled",
        "all-pass-when-cache-empty",
        "safety-net-when-all-unhealthy",
        "scoped-to-listed-model-groups",
        "unscoped-when-model-groups-unset",
    ],
)
@pytest.mark.asyncio
async def test_health_check_routing_filters_unhealthy_deployments(
    enable: bool, unhealthy_ids: set[str], model_groups: list[str] | None, routable: dict[str, list[str]]
) -> None:
    router = _health_routed_router(enable, unhealthy_ids, model_groups)

    for model_group, expected in routable.items():
        healthy = await router.select(model_group, {}, kind="healthy")
        assert sorted(d["model_info"]["id"] for d in healthy) == expected, model_group


class TestBuildDeploymentHealthStates:
    """Test the build_deployment_health_states function."""

    def test_builds_states_from_endpoints(self):
        from litellm.proxy.health_check import build_deployment_health_states

        healthy = [{"model": "gpt-4", "model_id": "deploy-1"}]
        unhealthy = [{"model": "gpt-4", "model_id": "deploy-2", "error": "timeout"}]

        states = build_deployment_health_states(healthy, unhealthy)
        assert states["deploy-1"]["is_healthy"] is True
        assert states["deploy-2"]["is_healthy"] is False

    def test_no_model_id_skipped(self):
        from litellm.proxy.health_check import build_deployment_health_states

        healthy = [{"model": "gpt-4"}]  # no model_id
        unhealthy = [{"model": "gpt-4", "model_id": "deploy-2"}]

        states = build_deployment_health_states(healthy, unhealthy)
        assert "deploy-1" not in states
        assert states["deploy-2"]["is_healthy"] is False

    def test_empty_endpoints(self):
        from litellm.proxy.health_check import build_deployment_health_states

        states = build_deployment_health_states([], [])
        assert states == {}
