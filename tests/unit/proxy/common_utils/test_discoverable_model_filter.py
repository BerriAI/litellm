"""
Tests for the operator-declared discoverability filter shared by the model
listing endpoints: a deployment marked `model_info: {discoverable: false}` is
hidden from listings for callers without the admin view while it still routes.
"""

import pytest

from litellm import Router
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.discoverable_model_filter import (
    discoverable_rows,
    undiscoverable_model_names,
)


def _deployment(model_name: str, model: str = "openai/gpt-4o", **model_info):
    return {
        "model_name": model_name,
        "litellm_params": {"model": model, "api_key": "sk-fake"},
        "model_info": {"id": f"{model_name}-id", **model_info},
    }


def _router(*deployments, **router_kwargs) -> Router:
    return Router(model_list=list(deployments), **router_kwargs)


def _non_admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_role=LitellmUserRoles.INTERNAL_USER)


def _admin(role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_role=role)


def test_flagged_model_is_undiscoverable_for_non_admin():
    router = _router(_deployment("gpt-4"), _deployment("internal-evaluator", discoverable=False))

    assert undiscoverable_model_names(["gpt-4", "internal-evaluator"], router, _non_admin(), None) == {
        "internal-evaluator"
    }


def test_missing_flag_and_explicit_true_are_discoverable():
    router = _router(_deployment("gpt-4"), _deployment("public-eval", discoverable=True))

    assert undiscoverable_model_names(["gpt-4", "public-eval"], router, _non_admin(), None) == frozenset()


@pytest.mark.parametrize("role", [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
def test_admin_view_sees_flagged_models(role):
    router = _router(_deployment("internal-evaluator", discoverable=False))

    assert undiscoverable_model_names(["internal-evaluator"], router, _admin(role), None) == frozenset()


def test_group_with_one_discoverable_deployment_stays_listed():
    router = _router(
        _deployment("shared", discoverable=False),
        {
            "model_name": "shared",
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-fake"},
            "model_info": {"id": "shared-public"},
        },
    )

    assert undiscoverable_model_names(["shared"], router, _non_admin(), None) == frozenset()


def test_unknown_name_and_missing_router_fail_open():
    router = _router(_deployment("internal-evaluator", discoverable=False))

    assert undiscoverable_model_names(["not-configured"], router, _non_admin(), None) == frozenset()
    assert undiscoverable_model_names(["internal-evaluator"], None, _non_admin(), None) == frozenset()


def test_alias_follows_its_target_deployments():
    router = _router(
        _deployment("gpt-4"),
        _deployment("internal-evaluator", discoverable=False),
        model_group_alias={"eval": "internal-evaluator", "chat": "gpt-4"},
    )

    assert undiscoverable_model_names(["eval", "chat"], router, _non_admin(), None) == {"eval"}


def test_wildcard_expansions_follow_the_wildcard_entry():
    router = _router(_deployment("gpt-4"), _deployment("anthropic/*", model="anthropic/*", discoverable=False))

    hidden = undiscoverable_model_names(
        ["gpt-4", "anthropic/*", "anthropic/claude-opus-5"], router, _non_admin(), None
    )

    assert hidden == {"anthropic/*", "anthropic/claude-opus-5"}


def test_flagged_team_model_is_undiscoverable_for_its_team_member():
    router = _router(
        _deployment("gpt-4"),
        _deployment(
            "model_name_team1_abc", team_id="team1", team_public_model_name="team-gpt", discoverable=False
        ),
    )
    member = UserAPIKeyAuth(
        api_key="sk-test", user_role=LitellmUserRoles.INTERNAL_USER, team_id="team1", team_models=["team-gpt"]
    )

    assert undiscoverable_model_names(["gpt-4", "team-gpt"], router, member, "team1") == {"team-gpt"}


def test_hidden_model_still_routes_for_direct_requests():
    router = _router(_deployment("gpt-4"), _deployment("internal-evaluator", discoverable=False))

    assert "internal-evaluator" in undiscoverable_model_names(["internal-evaluator"], router, _non_admin(), None)
    deployment = router.get_available_deployment(
        model="internal-evaluator", messages=[{"role": "user", "content": "hi"}]
    )
    assert deployment["model_name"] == "internal-evaluator"


def test_discoverable_rows_drops_flagged_rows_only_for_non_admin():
    rows = [
        {"model_name": "gpt-4", "model_info": {"id": "a"}},
        {"model_name": "internal-evaluator", "model_info": {"id": "b", "discoverable": False}},
        {"model_name": "no-model-info"},
    ]

    assert [row["model_name"] for row in discoverable_rows(rows, _non_admin())] == ["gpt-4", "no-model-info"]
    assert [row["model_name"] for row in discoverable_rows(rows, _admin())] == [
        "gpt-4",
        "internal-evaluator",
        "no-model-info",
    ]


def test_expanded_name_served_by_a_discoverable_wildcard_too_stays_listed():
    router = _router(
        _deployment("anthropic/*", model="anthropic/*", discoverable=False),
        _deployment("anthropic/claude-*", model="anthropic/claude-*"),
    )

    hidden = undiscoverable_model_names(
        ["anthropic/claude-opus-5", "anthropic/other-model"], router, _non_admin(), None
    )

    assert hidden == {"anthropic/other-model"}


def test_hidden_alias_of_a_flagged_model_is_undiscoverable():
    router = _router(
        _deployment("internal-evaluator", discoverable=False),
        model_group_alias={"eval": {"model": "internal-evaluator", "hidden": True}},
    )

    assert undiscoverable_model_names(["eval"], router, _non_admin(), None) == {"eval"}
