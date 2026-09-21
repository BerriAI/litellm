from typing import Final

import pytest

from litellm.proxy._types import (
    LiteLLM_JWTAuth,
    LiteLLM_TeamTable,
    LiteLLMRoutes,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.pass_through_access import (
    Denied,
    Granted,
    NotAuthEnforced,
    PassThroughGrants,
    authorize_pass_through,
)
from litellm.proxy.auth.route_checks import RouteChecks

ENDPOINT: Final = "/model-host/v1/demographics-extractor"
SUBPATH: Final = f"{ENDPOINT}/predict"
JWT_CLAIMS: Final = {"sub": "user-1"}


def _always_enforced(*, route: str, method: str | None) -> bool:
    return True


def _never_enforced(*, route: str, method: str | None) -> bool:
    return False


def _jwt_auth(*team_allowed_routes: str) -> LiteLLM_JWTAuth:
    return LiteLLM_JWTAuth(team_allowed_routes=list(team_allowed_routes))


@pytest.mark.parametrize(
    "team_allowed_routes, route, expected",
    [
        (("/model-host/*",), SUBPATH, Granted()),
        (("/model-host/*",), ENDPOINT, Granted()),
        (("/model-host/*",), "/model-host", Denied()),
        (("/model-host/*",), "/model-hosting/v1", Denied()),
        ((ENDPOINT,), ENDPOINT, Granted()),
        ((ENDPOINT,), SUBPATH, Denied()),
        (tuple(LiteLLMRoutes.__members__), SUBPATH, Denied()),
        (tuple(LiteLLMRoutes.__members__), "/v1/chat/completions", Denied()),
        (("openai_routes", "/model-host/*"), SUBPATH, Granted()),
        (("*",), SUBPATH, Denied()),
        (("/*",), SUBPATH, Denied()),
        (("/",), "/", Denied()),
        (("/*", "/model-host/*"), SUBPATH, Granted()),
        ((), SUBPATH, Denied()),
    ],
)
def test_jwt_team_is_granted_only_by_team_allowed_routes_entries_that_name_a_path(team_allowed_routes, route, expected):
    grants = PassThroughGrants.for_jwt_team(
        team=LiteLLM_TeamTable(team_id="team-a", metadata={}),
        jwt_auth=_jwt_auth(*team_allowed_routes),
    )

    assert (
        authorize_pass_through(route=route, method="POST", grants=grants, is_auth_enforced=_always_enforced) == expected
    )


@pytest.mark.parametrize(
    "allowlist, route, expected",
    [
        (["/model-host"], SUBPATH, Granted()),
        (["/model-host"], "/model-host", Granted()),
        (["/model-host"], "/model-hosting/v1", Denied()),
        (["/model-host/*"], SUBPATH, Denied()),
        (["/other", ENDPOINT], SUBPATH, Granted()),
        ([], SUBPATH, Denied()),
        (None, SUBPATH, Denied()),
        ("/model-host", SUBPATH, Denied()),
        ([7, "/model-host"], SUBPATH, Granted()),
    ],
)
def test_team_metadata_allowlist_covers_a_path_and_its_subpaths_and_treats_star_literally(allowlist, route, expected):
    grants = PassThroughGrants.for_jwt_team(
        team=LiteLLM_TeamTable(team_id="team-a", metadata={"allowed_passthrough_routes": allowlist}),
        jwt_auth=_jwt_auth("openai_routes"),
    )

    assert (
        authorize_pass_through(route=route, method="POST", grants=grants, is_auth_enforced=_always_enforced) == expected
    )


def test_jwt_team_without_a_team_object_or_metadata_has_only_the_config_grants():
    without_team_object = PassThroughGrants.for_jwt_team(team=None, jwt_auth=_jwt_auth("/model-host/*"))
    without_metadata_or_config = PassThroughGrants.for_jwt_team(
        team=LiteLLM_TeamTable(team_id="team-a", metadata=None), jwt_auth=None
    )

    assert without_team_object.covers(SUBPATH) is True
    assert without_team_object.covers("/elsewhere") is False
    assert without_metadata_or_config.covers(SUBPATH) is False


@pytest.mark.parametrize(
    "key_allowlist, team_allowlist, route, allowed",
    [
        (["/key-route"], ["/team-route"], "/key-route/v1", True),
        (["/key-route"], ["/team-route"], "/team-route/v1", False),
        ([], ["/team-route"], "/team-route/v1", True),
        (None, ["/team-route"], "/team-route/v1", True),
        (None, None, "/team-route/v1", False),
    ],
)
def test_a_non_empty_key_allowlist_shadows_the_team_allowlist(key_allowlist, team_allowlist, route, allowed):
    token = UserAPIKeyAuth(
        metadata={"allowed_passthrough_routes": key_allowlist},
        team_metadata={"allowed_passthrough_routes": team_allowlist},
    )

    access = authorize_pass_through(
        route=route,
        method="POST",
        grants=PassThroughGrants.for_token(token=token),
        is_auth_enforced=_always_enforced,
    )

    assert access == (Granted() if allowed else Denied())
    assert RouteChecks.check_passthrough_route_access(route=route, user_api_key_dict=token) is allowed


@pytest.mark.parametrize(
    "token, expected",
    [
        (UserAPIKeyAuth(team_id="team-a", jwt_claims=JWT_CLAIMS), Granted()),
        (UserAPIKeyAuth(token="sk-virtual-key", team_id="team-a"), Denied()),
        (UserAPIKeyAuth(token="sk-jwt-mapped-key", team_id="team-a", jwt_claims=JWT_CLAIMS), Denied()),
        (UserAPIKeyAuth(team_id="team-a"), Denied()),
        (UserAPIKeyAuth(jwt_claims=JWT_CLAIMS), Denied()),
    ],
    ids=["jwt-team-caller", "virtual-key", "jwt-mapped-virtual-key", "keyless-non-jwt-caller", "jwt-without-team"],
)
def test_only_a_keyless_jwt_token_with_a_team_receives_the_jwt_config_grants(token, expected):
    grants = PassThroughGrants.for_token(token=token, jwt_auth=_jwt_auth("/model-host/*"))

    assert (
        authorize_pass_through(route=SUBPATH, method="POST", grants=grants, is_auth_enforced=_always_enforced)
        == expected
    )


def test_a_jwt_token_without_a_jwt_config_has_no_config_grants():
    jwt_caller = UserAPIKeyAuth(team_id="team-a", jwt_claims=JWT_CLAIMS)

    assert PassThroughGrants.for_token(token=jwt_caller, jwt_auth=None).covers(SUBPATH) is False


def test_jwt_caller_metadata_and_config_grants_are_both_honored():
    token = UserAPIKeyAuth(
        team_id="team-a",
        jwt_claims=JWT_CLAIMS,
        team_metadata={"allowed_passthrough_routes": ["/team-route"]},
    )
    grants = PassThroughGrants.for_token(token=token, jwt_auth=_jwt_auth("/model-host/*"))

    assert grants.covers("/team-route/v1") is True
    assert grants.covers(SUBPATH) is True
    assert grants.covers("/elsewhere") is False


def test_routes_that_are_not_auth_enforced_are_left_to_the_callers_own_rules():
    seen = []

    def _record(*, route: str, method: str | None) -> bool:
        seen.append((route, method))
        return False

    access = authorize_pass_through(
        route=SUBPATH,
        method="GET",
        grants=PassThroughGrants(),
        is_auth_enforced=_record,
    )

    assert access == NotAuthEnforced()
    assert seen == [(SUBPATH, "GET")]
    assert (
        authorize_pass_through(
            route=SUBPATH,
            method="GET",
            grants=PassThroughGrants(prefix_routes=("/model-host",)),
            is_auth_enforced=_never_enforced,
        )
        == NotAuthEnforced()
    )
