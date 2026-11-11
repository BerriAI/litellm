"""JWT auth: keeping a user's LiteLLM role and team memberships in step with the identity provider, refusing
users from other email domains, and limiting models by OAuth scope.

Contract (litellm/proxy/auth/handle_jwt.py `map_jwt_role_to_litellm_role`, `sync_user_role_and_teams`,
`get_user_info`, `is_allowed_domain`, `check_scope_based_access`, `user_api_key_auth_from_result`):

* `jwt_litellm_role_map` maps an IdP role (read from `roles_jwt_field`, matched with shell style wildcards,
  first mapping wins) to a LiteLLM user role. With `sync_user_role_and_teams: true` the user's stored
  `user_role` is rewritten to it on every request, and a user synced to `proxy_admin` is treated as a proxy
  admin from then on. The same flag reconciles the user's team memberships with `team_ids_jwt_field`: teams
  in the claim are joined, teams missing from it are left.
* `user_allowed_email_domain` refuses a token whose email claim has another domain (exact match on the part
  after the last `@`, so subdomains are refused too) or no email at all, with 403 `auth_error` param
  `user_email`.
* `enforce_scope_based_access` with `scope_mappings` allows only the models mapped to the token's scopes.
  Any other model is 403 `auth_error`, and the message is the Python repr of a dict.

The role sync is the one place a token can raise a user's privileges in the database, so the Rust port has
to reproduce exactly when it fires and what it writes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.client import eventually, string_value
from tests.integration.authorization.dec1_authmodes_support import (
    MODEL_DENIED_TEMPLATE,
    JwtRig,
    chat_as,
    jwt_rig,
    served_body,
    spend_row,
    unique,
    user_row,
)
from tests.integration.authorization.dec1_keys_support import assert_denied

DOMAIN: Final = "dec1-authmodes.example"
SCOPED_MODEL: Final = unique("scope-model")
SECOND_SCOPED_MODEL: Final = unique("scope-model-two")
UNSCOPED_MODEL: Final = unique("scope-unmapped")


@pytest.fixture(scope="module")
def synced(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-role-sync"),
        user_id_jwt_field="sub",
        user_id_upsert=True,
        roles_jwt_field="roles",
        team_ids_jwt_field="groups",
        sync_user_role_and_teams=True,
        jwt_litellm_role_map=[
            {"jwt_role": "dec1-admin*", "litellm_role": "proxy_admin"},
            {"jwt_role": "dec1-viewer", "litellm_role": "internal_user_viewer"},
            {"jwt_role": "dec1-*", "litellm_role": "internal_user"},
        ],
    ) as built:
        yield built


@pytest.fixture(scope="module")
def unsynced(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-role-nosync"),
        user_id_jwt_field="sub",
        user_id_upsert=True,
        roles_jwt_field="roles",
        jwt_litellm_role_map=[{"jwt_role": "dec1-admin*", "litellm_role": "proxy_admin"}],
    ) as built:
        yield built


@pytest.fixture(scope="module")
def domains(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-email-domain"),
        user_id_jwt_field="sub",
        user_id_upsert=True,
        user_email_jwt_field="email",
        user_allowed_email_domain=DOMAIN,
    ) as built:
        yield built


@pytest.fixture(scope="module")
def scopes(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with (
        jwt_rig(
            tmp_path_factory.mktemp("dec1-authmodes-scopes"),
            user_id_jwt_field="sub",
            user_id_upsert=True,
            enforce_scope_based_access=True,
            scope_mappings=[
                {"scope": "dec1.chat", "models": [SCOPED_MODEL]},
                {"scope": "dec1.chat.extra", "models": [SECOND_SCOPED_MODEL]},
            ],
        ) as built,
        built.proxy.scenario() as scenario,
    ):
        scenario.model(name=SCOPED_MODEL)
        scenario.model(name=SECOND_SCOPED_MODEL)
        scenario.model(name=UNSCOPED_MODEL)
        yield built


def _stored_role(rig: JwtRig, user_id: str) -> str | None:
    row: Final = user_row(rig.proxy, user_id)
    assert row is not None, user_id
    role: Final = row["user_role"]
    return None if role is None else string_value(role)


def _stored_teams(rig: JwtRig, user_id: str) -> frozenset[str]:
    row: Final = user_row(rig.proxy, user_id)
    assert row is not None, user_id
    teams: Final = row["teams"]
    assert isinstance(teams, list), row
    return frozenset(str(team) for team in teams)


def test_role_sync_writes_the_mapped_role_to_the_user_row(synced: JwtRig) -> None:
    """With `sync_user_role_and_teams` on, a new user's token carries `roles: ["dec1-viewer"]`, which the
    second `jwt_litellm_role_map` entry maps to `internal_user_viewer`.

    The request must be served and the user's stored `user_role` must become `internal_user_viewer`.
    """
    with synced.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("viewer")
        scenario.cleanups.callback(scenario.delete_user, subject)
        assert served_body(chat_as(synced.proxy, synced.idp.token(sub=subject, roles=["dec1-viewer"]), model))
        assert _stored_role(synced, subject) == "internal_user_viewer"


def test_role_map_entries_match_with_wildcards_and_the_first_match_wins(synced: JwtRig) -> None:
    """A token's `roles` is `["dec1-admin-east"]`. Both `dec1-admin*` (proxy_admin) and the later `dec1-*`
    (internal_user) match it.

    The first listed mapping wins, so the user's stored `user_role` must be `proxy_admin`.
    """
    with synced.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("wild-admin")
        scenario.cleanups.callback(scenario.delete_user, subject)
        chat_as(synced.proxy, synced.idp.token(sub=subject, roles=["dec1-admin-east"]), model)
        assert _stored_role(synced, subject) == "proxy_admin"


def test_a_user_synced_to_proxy_admin_can_call_admin_only_management_routes(synced: JwtRig) -> None:
    """A token whose `roles` maps to `proxy_admin` (and carries no admin scope) calls `POST /user/new`.

    The sync runs before the route check reads the user's role, so the proxy must answer 200 and create the
    user, the same as for a master key caller.
    """
    with synced.proxy.scenario() as scenario:
        subject: Final = unique("synced-admin")
        created: Final = unique("made-by-synced-admin")
        scenario.cleanups.callback(scenario.delete_user, subject)
        token: Final = synced.idp.token(sub=subject, roles=["dec1-admin"])
        response: Final = synced.proxy.request("POST", "/user/new", {"user_id": created}, key=token)
        assert response.status_code == 200, response.text
        scenario.cleanups.callback(scenario.delete_user, created)
        assert user_row(synced.proxy, created) is not None


def test_role_sync_demotes_a_user_when_the_idp_role_changes(synced: JwtRig) -> None:
    """A user first calls with `roles: ["dec1-admin"]` (stored as `proxy_admin`), then with `roles:
    ["dec1-viewer"]`.

    After the second request the stored `user_role` must be `internal_user_viewer`, so losing the role at the
    identity provider removes admin rights on the next call.
    """
    with synced.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("demoted")
        scenario.cleanups.callback(scenario.delete_user, subject)
        chat_as(synced.proxy, synced.idp.token(sub=subject, roles=["dec1-admin"]), model)
        assert _stored_role(synced, subject) == "proxy_admin"
        chat_as(synced.proxy, synced.idp.token(sub=subject, roles=["dec1-viewer"]), model)
        assert _stored_role(synced, subject) == "internal_user_viewer"


def test_a_role_that_matches_no_mapping_leaves_the_stored_role_alone(synced: JwtRig) -> None:
    """A new user's token carries `roles: ["unrelated"]`, which no `jwt_litellm_role_map` entry matches.

    The request must be served and the stored role must stay what the upsert wrote: NULL, since
    `get_user_object` sets no role and the `user_role` column has no default (schema.prisma:311).
    """
    with synced.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("unmapped-role")
        scenario.cleanups.callback(scenario.delete_user, subject)
        assert served_body(chat_as(synced.proxy, synced.idp.token(sub=subject, roles=["unrelated"]), model))
        assert _stored_role(synced, subject) is None


def test_without_sync_the_role_map_does_not_change_the_stored_role(unsynced: JwtRig) -> None:
    """`jwt_litellm_role_map` is set but `sync_user_role_and_teams` is not. A new user's token carries
    `roles: ["dec1-admin"]`.

    The stored role must stay NULL (nothing wrote it), and `POST /user/new` with that token must not succeed.
    """
    with unsynced.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("nosync")
        scenario.cleanups.callback(scenario.delete_user, subject)
        token: Final = unsynced.idp.token(sub=subject, roles=["dec1-admin"])
        assert served_body(chat_as(unsynced.proxy, token, model))
        assert _stored_role(unsynced, subject) is None
        refused: Final = unsynced.proxy.request("POST", "/user/new", {"user_id": unique("never-made")}, key=token)
        assert refused.status_code in {401, 403}, refused.text


def test_team_sync_joins_claimed_teams_and_leaves_unclaimed_ones(synced: JwtRig) -> None:
    """A user first calls with `groups: [A]`, then with `groups: [B]`; both teams can call the model.

    After the second call the user's stored `teams` must be exactly {B}: B joined, A left. A is no longer in
    the claim, so membership the IdP revoked must not linger.
    """
    with synced.proxy.scenario() as scenario:
        model: Final = scenario.model()
        first: Final = scenario.team(models=[model])
        second: Final = scenario.team(models=[model])
        subject: Final = unique("team-sync")
        scenario.cleanups.callback(scenario.delete_user, subject)
        chat_as(synced.proxy, synced.idp.token(sub=subject, roles=["dec1-user"], groups=[first]), model)
        assert first in _stored_teams(synced, subject)
        row: Final = spend_row(
            synced.proxy,
            chat_as(synced.proxy, synced.idp.token(sub=subject, roles=["dec1-user"], groups=[second]), model),
        )
        assert row["team_id"] == second, row
        assert eventually(lambda: _stored_teams(synced, subject), lambda teams: teams == {second}) == {second}


def test_an_email_in_the_allowed_domain_is_served(domains: JwtRig) -> None:
    """`user_allowed_email_domain` is set; a token's `email` is `<name>@<that domain>`.

    The proxy must answer 200 and store the email on the upserted user.
    """
    with domains.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("domain-ok")
        email: Final = f"{subject}@{DOMAIN}"
        scenario.cleanups.callback(scenario.delete_user, subject)
        assert served_body(chat_as(domains.proxy, domains.idp.token(sub=subject, email=email), model))
        row: Final = user_row(domains.proxy, subject)
        assert row is not None and row["user_email"] == email, row


@pytest.mark.parametrize(
    "email",
    [
        "someone@elsewhere.example",
        f"someone@sub.{DOMAIN}",
        f"someone@{DOMAIN}.attacker.example",
    ],
    ids=["other-domain", "subdomain", "suffix-trick"],
)
def test_an_email_outside_the_allowed_domain_is_refused(domains: JwtRig, email: str) -> None:
    """A token's `email` is in another domain, a subdomain of the allowed one, or a domain that only starts
    with the allowed one.

    The domain check is an exact match, so the proxy must answer 403 `auth_error` param `user_email` with
    exactly "Email domain not allowed. User email: <email>. Allowed domain: <domain>".
    """
    with domains.proxy.scenario() as scenario:
        model: Final = scenario.model()
        assert_denied(
            chat_as(domains.proxy, domains.idp.token(sub=unique("domain-no"), email=email), model),
            status=403,
            error_type="auth_error",
            message=f"Email domain not allowed. User email: {email}. Allowed domain: {DOMAIN}",
            param="user_email",
        )


def test_a_token_without_an_email_is_refused_when_a_domain_is_enforced(domains: JwtRig) -> None:
    """`user_allowed_email_domain` is set and a token carries a `sub` but no `email` claim.

    A missing email counts as not allowed, so the proxy must answer 403 `auth_error` param `user_email` with
    "Email domain not allowed. User email: None. Allowed domain: <domain>".
    """
    with domains.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("no-email")
        assert_denied(
            chat_as(domains.proxy, domains.idp.token(sub=subject), model),
            status=403,
            error_type="auth_error",
            message=f"Email domain not allowed. User email: None. Allowed domain: {DOMAIN}",
            param="user_email",
        )
        assert user_row(domains.proxy, subject) is None


def test_a_scope_mapped_to_the_model_is_served(scopes: JwtRig) -> None:
    """`enforce_scope_based_access` maps scope `dec1.chat` to one model; a token's `scope` is
    `openid dec1.chat`.

    The proxy must answer 200 for that model.
    """
    subject: Final = unique("scope-ok")
    with scopes.proxy.scenario() as scenario:
        scenario.cleanups.callback(scenario.delete_user, subject)
        assert served_body(chat_as(scopes.proxy, scopes.idp.token(sub=subject, scope="openid dec1.chat"), SCOPED_MODEL))


def test_a_model_outside_the_tokens_scopes_is_refused_with_a_dict_message(scopes: JwtRig) -> None:
    """A token with scope `dec1.chat` asks for a model that is deployed but mapped to no scope.

    The proxy must answer 403 `auth_error` whose message is the Python repr `{'error': "<model denied
    text>"}`, because the refusal's detail is a dict that the error handler stringifies.
    """
    denied: Final = MODEL_DENIED_TEMPLATE.format(model=UNSCOPED_MODEL)
    assert_denied(
        chat_as(scopes.proxy, scopes.idp.token(sub=unique("scope-no"), scope="dec1.chat"), UNSCOPED_MODEL),
        status=403,
        error_type="auth_error",
        message=str({"error": denied}),
    )


def test_a_token_without_a_scope_claim_can_call_no_model(scopes: JwtRig) -> None:
    """With scope based access on, a token carries no `scope` claim and asks for a mapped model.

    No scope grants it, so the proxy must answer 403 `auth_error` with the same dict repr message.
    """
    denied: Final = MODEL_DENIED_TEMPLATE.format(model=SCOPED_MODEL)
    assert_denied(
        chat_as(scopes.proxy, scopes.idp.token(sub=unique("scope-none")), SCOPED_MODEL),
        status=403,
        error_type="auth_error",
        message=str({"error": denied}),
    )


def test_models_from_several_scopes_add_up(scopes: JwtRig) -> None:
    """A token's `scope` is the list `["dec1.chat", "dec1.chat.extra"]`; each scope maps a different model.

    Both models must be served, and a token holding only `dec1.chat` must be refused the second model.
    """
    subject: Final = unique("scope-both")
    with scopes.proxy.scenario() as scenario:
        scenario.cleanups.callback(scenario.delete_user, subject)
        both: Final = scopes.idp.token(sub=subject, scope=["dec1.chat", "dec1.chat.extra"])
        assert served_body(chat_as(scopes.proxy, both, SCOPED_MODEL))
        assert served_body(chat_as(scopes.proxy, both, SECOND_SCOPED_MODEL))
        only_first: Final = scopes.idp.token(sub=subject, scope="dec1.chat")
        assert chat_as(scopes.proxy, only_first, SECOND_SCOPED_MODEL).status_code == 403
