from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.team import LiteLLM_TeamTable, Member
from litellm.proxy.auth.host.facts import (
    TeamFacts,
    budget_params,
    end_user_params,
    key_facts,
    team_facts,
    with_end_user,
)
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.auth.user_roles import LitellmUserRoles

MIDNIGHT_UTC: Final = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()


@pytest.mark.parametrize(
    "expires",
    [
        datetime(2024, 1, 1),
        datetime(2024, 1, 1, 1, tzinfo=timezone(timedelta(hours=1))),
        "2024-01-01T00:00:00",
        "2024-01-01T01:00:00+01:00",
    ],
)
def test_expiry_reaches_rust_as_utc_epoch_seconds(expires: datetime | str) -> None:
    assert key_facts(UserAPIKeyAuth(token="hashed", expires=expires)).expires == MIDNIGHT_UTC


def test_key_without_expiry_never_expires() -> None:
    assert key_facts(UserAPIKeyAuth(token="hashed")).expires is None


@pytest.mark.parametrize(
    ("role", "is_proxy_admin"),
    [
        (LitellmUserRoles.PROXY_ADMIN, True),
        (LitellmUserRoles.INTERNAL_USER, False),
        (None, False),
    ],
)
def test_only_a_proxy_admin_role_is_an_admin_key(role: LitellmUserRoles | None, is_proxy_admin: bool) -> None:
    assert key_facts(UserAPIKeyAuth(token="hashed", user_role=role)).is_proxy_admin is is_proxy_admin


@pytest.mark.parametrize(("blocked", "expected"), [(True, True), (False, False), (None, False)])
def test_unset_blocked_column_is_not_blocked(blocked: bool | None, expected: bool) -> None:
    assert key_facts(UserAPIKeyAuth(token="hashed", blocked=blocked)).blocked is expected


def test_ids_and_refresh_time_come_from_the_token() -> None:
    facts: Final = key_facts(
        UserAPIKeyAuth(
            token="hashed",
            user_id="u",
            team_id="t",
            team_object_permission_id="op",
            project_id="p",
            last_refreshed_at=12.5,
        )
    )

    assert (facts.token, facts.user_id, facts.team_id, facts.team_object_permission_id, facts.project_id) == (
        "hashed",
        "u",
        "t",
        "op",
        "p",
    )
    assert facts.last_refreshed_at == 12.5


@pytest.mark.parametrize(
    ("metadata", "allowed", "denied"),
    [
        ({"allowed_passthrough_routes": ["/a"], "denied_passthrough_routes": ["/b"]}, ("/a",), ("/b",)),
        ({"allowed_passthrough_routes": [], "denied_passthrough_routes": None}, (), ()),
        ({}, None, ()),
        (None, None, ()),
    ],
)
def test_a_team_carries_its_pass_through_grants_and_row(
    metadata: dict[str, object] | None, allowed: tuple[str, ...] | None, denied: tuple[str, ...]
) -> None:
    team: Final = LiteLLM_TeamTable(team_id="t", models=["gpt"], metadata=metadata)
    found: Final = team_facts(team)
    assert found.row is team
    assert found.facts == TeamFacts(
        team_id="t",
        models=("gpt",),
        allowed_passthrough_routes=allowed,
        denied_passthrough_routes=denied,
        members=(),
    )


def test_a_team_lists_the_user_ids_of_its_members() -> None:
    team: Final = LiteLLM_TeamTable(
        team_id="t",
        members_with_roles=[Member(user_id="u1", role="admin"), Member(user_email="e@x.com", role="user")],
    )

    assert team_facts(team).facts.members == ("u1",)


def test_a_budget_sets_only_the_limits_it_has() -> None:
    budget: Final = LiteLLM_BudgetTable(tpm_limit=7, max_budget=1.5, model_rpm_limit={"gpt-a": 2})

    assert budget_params(budget) == {
        "end_user_tpm_limit": 7,
        "end_user_max_budget": 1.5,
        "end_user_model_rpm_limit": {"gpt-a": 2},
    }


def test_a_known_end_user_sets_its_region_even_when_it_has_none() -> None:
    end_user: Final = LiteLLM_EndUserTable(user_id="eu", blocked=False)

    assert end_user_params("eu", end_user, None) == {"end_user_id": "eu", "allowed_model_region": None}


def test_an_unknown_end_user_takes_the_default_budget_and_sets_no_region() -> None:
    assert end_user_params("eu", None, LiteLLM_BudgetTable(rpm_limit=3)) == {
        "end_user_id": "eu",
        "end_user_rpm_limit": 3,
    }


def test_end_user_limits_replace_the_keys_while_an_unset_limit_and_the_max_budget_do_not() -> None:
    key: Final = UserAPIKeyAuth(
        token="hashed", end_user_id="stale", end_user_tpm_limit=1, end_user_rpm_limit=3, allowed_model_region="eu"
    )

    stamped: Final = with_end_user(
        key, {"end_user_tpm_limit": 7, "end_user_max_budget": 1.5, "allowed_model_region": None}
    )

    assert (
        stamped.end_user_id,
        stamped.end_user_tpm_limit,
        stamped.end_user_rpm_limit,
        stamped.end_user_max_budget,
        stamped.allowed_model_region,
    ) == (None, 7, 3, None, "eu")
    assert (key.end_user_id, key.end_user_tpm_limit) == ("stale", 1)
