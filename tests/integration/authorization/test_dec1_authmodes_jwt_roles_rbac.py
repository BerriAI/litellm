"""JWT auth: admin scopes, route allowances per role, `enforce_rbac` with `role_permissions`, and role mappings
that turn an identity provider role into a LiteLLM team or user.

Contract (litellm/proxy/auth/handle_jwt.py `get_rbac_role`, `is_admin`, `check_admin_access`,
`check_rbac_role`, `can_rbac_role_call_model`, `can_rbac_role_call_route`; litellm/proxy/auth/auth_checks.py
`allowed_routes_check`, `get_role_based_routes`):

* A token whose `scope` claim (space separated string or list) contains `admin_jwt_scope` (default
  `litellm_proxy_admin`) is a proxy admin, limited to `admin_allowed_routes` (default: management, spend
  tracking, global spend tracking and info routes, so not inference). A route outside them is a 401 naming
  the route and the expanded allowed list.
* `enforce_rbac: true` refuses a token that maps to no role with a 403, and applies `general_settings.
  role_permissions` (models and routes per role) with a 403 per violation.
* A role is INTERNAL_USER when the token has a user id claim or a role from `user_roles_jwt_field` that is in
  `user_allowed_roles`. Outside `enforce_rbac`, `user_allowed_roles` refuses nothing.
* `roles_jwt_field` plus `role_mappings` map an IdP role to `team` or `internal_user`, and
  `object_id_jwt_field` then names that team or user.

These checks run before any database lookup, which makes them a natural first port to Rust and an easy
place to drift.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.client import string_value
from tests.integration.authorization.dec1_authmodes_support import (
    MODEL_DENIED_TEMPLATE,
    JwtRig,
    chat_as,
    jwt_rig,
    message_of,
    served_body,
    spend_row,
    unique,
    user_row,
)
from tests.integration.authorization.dec1_keys_support import assert_denied

ADMIN_ROUTE_REFUSAL: Final = (
    "Authentication Error, Admin not allowed to access this route. Route={route}, Allowed Routes=["
)
RBAC_MODEL: Final = unique("rbac-model")
RBAC_OTHER_MODEL: Final = unique("rbac-other-model")
UNMATCHED: Final = (
    "Unmatched token passed in. enforce_rbac is set to True. Token must belong to a proxy admin, team, or user."
)


@pytest.fixture(scope="module")
def defaults(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-admin-defaults"),
        user_id_jwt_field="sub",
        user_id_upsert=True,
        user_roles_jwt_field="roles",
        user_allowed_roles=["dec1-user"],
    ) as built:
        yield built


@pytest.fixture(scope="module")
def custom_admin(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-admin-custom"),
        user_id_jwt_field="sub",
        user_id_upsert=True,
        admin_jwt_scope="dec1.admin",
        admin_allowed_routes=["openai_routes", "/key/generate"],
    ) as built:
        yield built


@pytest.fixture(scope="module")
def rbac(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with (
        jwt_rig(
            tmp_path_factory.mktemp("dec1-authmodes-rbac"),
            general_settings={
                "role_permissions": [
                    {"role": "internal_user", "models": [RBAC_MODEL], "routes": ["/v1/chat/completions"]},
                ]
            },
            enforce_rbac=True,
            user_roles_jwt_field="roles",
            user_allowed_roles=["dec1-user"],
        ) as built,
        built.proxy.scenario() as scenario,
    ):
        scenario.model(name=RBAC_MODEL)
        scenario.model(name=RBAC_OTHER_MODEL)
        yield built


@pytest.fixture(scope="module")
def mapped(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-role-mappings"),
        roles_jwt_field="app.roles",
        role_mappings=[
            {"role": "dec1-team-app", "internal_role": "team"},
            {"role": "dec1-person", "internal_role": "internal_user"},
        ],
        object_id_jwt_field="oid",
        user_id_upsert=True,
    ) as built:
        yield built


def _new_key(rig: JwtRig, token: str) -> str:
    response: Final = rig.proxy.request("POST", "/key/generate", {"key_alias": unique("admin-made")}, key=token)
    assert response.status_code == 200, response.text
    key: Final = string_value(response.json()["key"])
    rig.proxy.post("/key/delete", {"keys": [key]})
    return key


def test_a_token_with_the_admin_scope_can_call_a_management_route(defaults: JwtRig) -> None:
    """A token's space separated `scope` claim contains the default admin scope `litellm_proxy_admin`.

    `POST /key/generate` is a management route inside the default `admin_allowed_routes`, so it must answer
    200 with a new `sk-` key.
    """
    token: Final = defaults.idp.token(sub=unique("admin"), scope="openid litellm_proxy_admin")
    assert _new_key(defaults, token).startswith("sk-")


def test_an_admin_scope_given_as_a_list_claim_is_honoured(defaults: JwtRig) -> None:
    """A token's `scope` claim is a JSON list `["openid", "litellm_proxy_admin"]` instead of a string.

    `get_scopes` accepts both shapes, so `POST /key/generate` must answer 200.
    """
    token: Final = defaults.idp.token(sub=unique("admin-list"), scope=["openid", "litellm_proxy_admin"])
    assert _new_key(defaults, token).startswith("sk-")


def test_a_default_admin_token_is_refused_on_inference_routes(defaults: JwtRig) -> None:
    """A token with the default admin scope sends a chat completion.

    Pinned current behaviour: the default `admin_allowed_routes` hold no inference routes, so the proxy must
    answer 401 `auth_error` whose message starts "Authentication Error, Admin not allowed to access this route.
    Route=/v1/chat/completions, Allowed Routes=[" (the expanded route list follows).
    """
    with defaults.proxy.scenario() as scenario:
        model: Final = scenario.model()
        refused: Final = chat_as(
            defaults.proxy, defaults.idp.token(sub=unique("admin-chat"), scope="litellm_proxy_admin"), model
        )
        assert refused.status_code == 401, refused.text
        assert message_of(refused).startswith(ADMIN_ROUTE_REFUSAL.format(route="/v1/chat/completions")), refused.text


def test_without_enforce_rbac_a_role_outside_user_allowed_roles_is_still_served(defaults: JwtRig) -> None:
    """`user_roles_jwt_field: roles` with `user_allowed_roles: ["dec1-user"]`, no `enforce_rbac`. A token with
    a `sub` and `roles: ["intruder"]` sends a chat completion.

    Pinned current behaviour: `user_allowed_roles` only feeds the RBAC role decision, so without
    `enforce_rbac` nothing refuses the token and the proxy must answer 200. Operators who read the setting
    as an allow list are not protected by it.
    """
    with defaults.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("intruder")
        assert served_body(chat_as(defaults.proxy, defaults.idp.token(sub=subject, roles=["intruder"]), model))
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_a_custom_admin_scope_grants_admin(custom_admin: JwtRig) -> None:
    """`admin_jwt_scope: dec1.admin`; a token's `scope` is `dec1.admin`.

    `POST /key/generate` is in this proxy's `admin_allowed_routes`, so it must answer 200 with an `sk-` key.
    """
    token: Final = custom_admin.idp.token(sub=unique("custom-admin"), scope="dec1.admin")
    assert _new_key(custom_admin, token).startswith("sk-")


def test_admin_allowed_routes_can_open_inference_to_admin_tokens(custom_admin: JwtRig) -> None:
    """`admin_allowed_routes: ["openai_routes", "/key/generate"]`; an admin token sends a chat completion.

    The `openai_routes` group covers `/v1/chat/completions`, so the proxy must answer 200 with the provider's
    reply.
    """
    with custom_admin.proxy.scenario() as scenario:
        model: Final = scenario.model()
        assert served_body(
            chat_as(custom_admin.proxy, custom_admin.idp.token(sub=unique("admin-chat"), scope="dec1.admin"), model)
        )


def test_admin_allowed_routes_refuse_an_admin_on_a_route_outside_them(custom_admin: JwtRig) -> None:
    """Same proxy; an admin token calls `POST /user/new`, which is neither an OpenAI route nor `/key/generate`.

    The proxy must answer 401 `auth_error` whose message starts "Authentication Error, Admin not allowed to
    access this route. Route=/user/new, Allowed Routes=[", and no user may be created.
    """
    target: Final = unique("never-created")
    refused: Final = custom_admin.proxy.request(
        "POST", "/user/new", {"user_id": target}, key=custom_admin.idp.token(sub=unique("admin"), scope="dec1.admin")
    )
    assert refused.status_code == 401, refused.text
    assert message_of(refused).startswith(ADMIN_ROUTE_REFUSAL.format(route="/user/new")), refused.text
    assert user_row(custom_admin.proxy, target) is None


def test_the_default_admin_scope_is_not_admin_once_admin_jwt_scope_is_changed(custom_admin: JwtRig) -> None:
    """Same proxy (`admin_jwt_scope: dec1.admin`); a token carries the default scope `litellm_proxy_admin`.

    The token is now an ordinary user, so `POST /user/new` must be refused with the non-admin 401 whose
    message starts "Authentication Error, Only proxy admin can be used to generate, delete, update info for
    new keys/users/teams. Route=/user/new." rather than the admin route refusal.
    """
    refused: Final = custom_admin.proxy.request(
        "POST",
        "/user/new",
        {"user_id": unique("never-created")},
        key=custom_admin.idp.token(sub=unique("not-admin"), scope="litellm_proxy_admin"),
    )
    assert refused.status_code == 401, refused.text
    assert message_of(refused).startswith(
        "Authentication Error, Only proxy admin can be used to generate, delete, update info for new "
        "keys/users/teams. Route=/user/new."
    ), refused.text


def test_enforce_rbac_serves_a_token_whose_role_is_in_user_allowed_roles(rbac: JwtRig) -> None:
    """`enforce_rbac: true`, no user id field, `user_roles_jwt_field: roles`, `user_allowed_roles:
    ["dec1-user"]`. A token with `roles: ["dec1-user"]` asks for the one model `role_permissions` grants.

    The token resolves to INTERNAL_USER and the route and model are permitted, so the proxy must answer 200.
    """
    assert served_body(chat_as(rbac.proxy, rbac.idp.token(roles=["dec1-user"]), RBAC_MODEL))


def test_enforce_rbac_refuses_a_token_that_maps_to_no_role(rbac: JwtRig) -> None:
    """Same proxy; a token carries `roles: ["intruder"]`, no admin scope, no team and no user id.

    The proxy must answer 403 `auth_error` with exactly "Unmatched token passed in. enforce_rbac is set to
    True. Token must belong to a proxy admin, team, or user." (handle_jwt.py:1517).
    """
    assert_denied(
        chat_as(rbac.proxy, rbac.idp.token(roles=["intruder"]), RBAC_MODEL),
        status=403,
        error_type="auth_error",
        message=UNMATCHED,
    )


def test_role_permissions_refuse_a_model_outside_the_roles_list(rbac: JwtRig) -> None:
    """Same proxy; a `dec1-user` token asks for a served model that `role_permissions` does not list for
    `internal_user`.

    The proxy must answer 403 `auth_error`, `param: null`, with the generic client message "The requested model
    '<model>' is not available for this API key, or the model name is invalid. Check the models available to
    you and try again." (the role and allowed list stay in the server log only).
    """
    assert_denied(
        chat_as(rbac.proxy, rbac.idp.token(roles=["dec1-user"]), RBAC_OTHER_MODEL),
        status=403,
        error_type="auth_error",
        message=MODEL_DENIED_TEMPLATE.format(model=RBAC_OTHER_MODEL),
    )


def test_role_permissions_refuse_a_route_outside_the_roles_list(rbac: JwtRig) -> None:
    """Same proxy; a `dec1-user` token calls `GET /v1/models`, while `internal_user` may only call
    `/v1/chat/completions`.

    The proxy must answer 403 `auth_error` with exactly "Role=internal_user not allowed to call
    route=/v1/models. Allowed routes=['/v1/chat/completions']" (handle_jwt.py:1446).
    """
    refused: Final = rbac.proxy.request("GET", "/v1/models", key=rbac.idp.token(roles=["dec1-user"]))
    assert_denied(
        refused,
        status=403,
        error_type="auth_error",
        message="Role=internal_user not allowed to call route=/v1/models. Allowed routes=['/v1/chat/completions']",
    )


def test_a_role_mapped_to_team_makes_the_object_id_claim_the_team(mapped: JwtRig) -> None:
    """`roles_jwt_field: app.roles` maps `dec1-team-app` to `team`, and `object_id_jwt_field: oid`. A token
    with `{"app": {"roles": ["dec1-team-app"]}}` and `oid` naming an existing team sends a chat completion.

    The request must be served and attributed to that team: the spend row's `team_id` must be the `oid`.
    """
    with mapped.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        token: Final = mapped.idp.token(app={"roles": ["dec1-team-app"]}, oid=team)
        assert spend_row(mapped.proxy, chat_as(mapped.proxy, token, model))["team_id"] == team


def test_a_role_mapped_to_internal_user_makes_the_object_id_claim_the_user(mapped: JwtRig) -> None:
    """Same proxy; a token with `{"app": {"roles": ["dec1-person"]}}` and an `oid` for a new person.

    The role maps to `internal_user`, so `oid` becomes the user id: the spend row's `user` must be the `oid`
    and, with `user_id_upsert`, a user row with that id must exist.
    """
    with mapped.proxy.scenario() as scenario:
        model: Final = scenario.model()
        person: Final = unique("oid-person")
        token: Final = mapped.idp.token(app={"roles": ["dec1-person"]}, oid=person)
        assert spend_row(mapped.proxy, chat_as(mapped.proxy, token, model))["user"] == person
        assert user_row(mapped.proxy, person) is not None
        scenario.cleanups.callback(scenario.delete_user, person)
