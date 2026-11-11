"""OAuth2 authentication: token introspection against an identity provider (`enable_oauth2_auth`) and identity
headers set by a trusted reverse proxy such as oauth2-proxy (`enable_oauth2_proxy_auth`).

Contract, introspection (litellm/proxy/auth/oauth2_check.py `Oauth2Handler.check_oauth2_token`;
litellm/proxy/auth/user_api_key_auth.py `oauth2_introspection_applies`):

* Only LLM API, info and `/lens` routes are introspected, and there every bearer value that is not a JWT is
  sent to `OAUTH_TOKEN_INFO_ENDPOINT`: opaque tokens, but also `sk-` virtual keys and the master key.
  Management routes keep the normal key path.
* The call is a `POST` of the form `token=<value>` with HTTP Basic client credentials only when the URL
  contains `introspect` and both `OAUTH_CLIENT_ID` and `OAUTH_CLIENT_SECRET` are set. Otherwise it is a `GET`
  with `Authorization: Bearer <value>`. Only the POST form honours `"active": false`.
* The caller's user, role and team come from the response fields named by `OAUTH_USER_ID_FIELD_NAME`
  (`sub`), `OAUTH_USER_ROLE_FIELD_NAME` (`role`) and `OAUTH_USER_TEAM_ID_FIELD_NAME` (`team_id`).
* Any failure is 401 `auth_error` "Authentication Error, Oauth 2.0 Token validation failed: <httpx status
  error>" for an HTTP error status, or "Authentication Error, An error occurred during token validation:
  <reason>" for anything else. Without a license it is 403 param `premium_user`.

Contract, header trust (litellm/proxy/auth/oauth2_proxy_hook.py, trusted_proxy_utils.py):

* The direct TCP peer must be inside `general_settings.trusted_proxy_ranges`; with none configured every
  request is refused. `oauth2_config_mappings` maps identity fields only (user_id, user_email, team_id,
  team_alias, org_id, models); any other field refuses every request. The bearer value is not checked.
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs

import httpx
import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import Gateway, gateway_from_environment
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server
from tests.integration.authorization.dec1_authmodes_support import (
    NOT_PREMIUM,
    authmodes_proxy,
    chat_as,
    config_file,
    message_of,
    served_body,
    spend_row,
    unique,
)
from tests.integration.authorization.dec1_keys_support import assert_denied

CLIENT_ID: Final = "dec1-authmodes-client"
CLIENT_SECRET: Final = "dec1-authmodes-client-secret"
HEADER_MODEL: Final = unique("header-auth-model")
USER_HEADER: Final = "X-Dec1-User"
EMAIL_HEADER: Final = "X-Dec1-Email"
MODELS_HEADER: Final = "X-Dec1-Models"
STATUS_FAILURE: Final = "Authentication Error, Oauth 2.0 Token validation failed: "
OTHER_FAILURE: Final = "Authentication Error, An error occurred during token validation: "
MASTER_IDENTITY: Final = "dec1-oauth-master"


@dataclass
class TokenInfo:
    """A token-info / introspection double. `answers` maps a token to its JSON reply; an unknown token is a
    401, and a token mapped to `None` drops the connection without a response."""

    answers: dict[str, JsonValue | None] = field(default_factory=dict)
    seen: list[Request] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    url: str = ""

    def grant(self, token: str, answer: JsonValue | None) -> str:
        with self.lock:
            self.answers[token] = answer
        return token

    def requests_for(self, token: str) -> list[Request]:
        with self.lock:
            return [request for request in self.seen if token in _presented(request)]

    def respond(self, request: Request) -> Reply:
        with self.lock:
            self.seen.append(request)
            token: Final = _presented(request)
            if token not in self.answers:
                return Reply(status=401, body=b'{"error": "invalid_token"}')
            answer: Final = self.answers[token]
        if answer is None:
            return Reply(drop_connection=True)
        return Reply(body=json.dumps(answer).encode())


def _presented(request: Request) -> str:
    """The token a request carries: the form field for a POST, the bearer value for a GET."""
    if request.method == "POST":
        return parse_qs(request.body.decode()).get("token", [""])[0]
    authorization: Final = {name.lower(): value for name, value in request.headers.items()}.get("authorization", "")
    return authorization.removeprefix("Bearer ")


@dataclass(frozen=True, slots=True)
class IntrospectionRig:
    info: TokenInfo
    proxy: Gateway


@contextmanager
def _introspection_rig(
    directory: Path,
    *,
    path: str,
    environment: Mapping[str, str] | None = None,
    remove_environment: tuple[str, ...] = (),
) -> Iterator[IntrospectionRig]:
    info: Final = TokenInfo()
    with wire_server(info.respond) as wire:
        info.url = f"{wire.url}{path}"
        with authmodes_proxy(
            directory,
            environment={"OAUTH_TOKEN_INFO_ENDPOINT": info.url, **(environment or {})},
            general_settings={"enable_oauth2_auth": True},
            remove_environment=remove_environment,
        ) as proxy:
            # Info routes such as /model/info and /key/info are introspected too, and the harness reads them
            # with the master key while creating and cleaning up, so the double vouches for it as an admin.
            info.grant(
                proxy.key, {"active": True, "sub": MASTER_IDENTITY, "uid": MASTER_IDENTITY, "role": "proxy_admin"}
            )
            yield IntrospectionRig(info, proxy)


@contextmanager
def _header_proxy(directory: Path, general_settings: Mapping[str, JsonValue]) -> Iterator[Gateway]:
    """An owned proxy with header trust on and one model in its config file.

    Under header trust even the master key is replaced by header identity, so models cannot be added over
    HTTP; the model points at the suite's mock upstream from the config file instead.
    """
    with gateway_from_environment() as shared:
        config: Final = config_file(directory, general_settings={"enable_oauth2_proxy_auth": True, **general_settings})
        loaded: Final = yaml.safe_load(config.read_text())
        loaded["model_list"] = [
            *loaded.get("model_list", []),
            {
                "model_name": HEADER_MODEL,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "integration-provider-key",
                    "api_base": f"{shared.upstream_url}/v1",
                },
            },
        ]
        config.write_text(yaml.safe_dump(loaded))
        with owned_proxy(shared, directory / "proxy", {}, config=config) as owned:
            yield owned


IDENTITY_MAPPINGS: Final = {"user_id": USER_HEADER, "user_email": EMAIL_HEADER, "models": MODELS_HEADER}


@pytest.fixture(scope="module")
def tokeninfo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[IntrospectionRig]:
    with _introspection_rig(tmp_path_factory.mktemp("dec1-authmodes-tokeninfo"), path="/tokeninfo") as built:
        yield built


@pytest.fixture(scope="module")
def introspect(tmp_path_factory: pytest.TempPathFactory) -> Iterator[IntrospectionRig]:
    with _introspection_rig(
        tmp_path_factory.mktemp("dec1-authmodes-introspect"),
        path="/oauth2/introspect",
        environment={
            "OAUTH_CLIENT_ID": CLIENT_ID,
            "OAUTH_CLIENT_SECRET": CLIENT_SECRET,
            "OAUTH_USER_ID_FIELD_NAME": "uid",
            "OAUTH_USER_TEAM_ID_FIELD_NAME": "group",
        },
    ) as built:
        yield built


@pytest.fixture(scope="module")
def public_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[IntrospectionRig]:
    with _introspection_rig(
        tmp_path_factory.mktemp("dec1-authmodes-introspect-public"),
        path="/oauth2/introspect",
        environment={"OAUTH_CLIENT_ID": CLIENT_ID},
        remove_environment=("OAUTH_CLIENT_SECRET",),
    ) as built:
        yield built


@pytest.fixture(scope="module")
def headers_trusted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with _header_proxy(
        tmp_path_factory.mktemp("dec1-authmodes-headers"),
        {"trusted_proxy_ranges": ["127.0.0.1/32", "::1/128"], "oauth2_config_mappings": IDENTITY_MAPPINGS},
    ) as built:
        yield built


def _header_chat(proxy: Gateway, headers: Mapping[str, str], *, bearer: str = "dec1-not-checked") -> httpx.Response:
    return chat_as(proxy, bearer, HEADER_MODEL, headers=headers)


def test_an_active_token_is_served_and_attributed_to_the_token_info_user_and_team(tokeninfo: IntrospectionRig) -> None:
    """A GET token-info endpoint answers `{"sub": <user>, "team_id": <team>}` for an opaque token.

    The chat must be served and its spend row's `user` and `team_id` must be the endpoint's values.
    UNVERIFIED: team attribution of an introspected caller is read from the returned `UserAPIKeyAuth`, not run.
    """
    with tokeninfo.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        user: Final = unique("oauth-user")
        token: Final = tokeninfo.info.grant(unique("opaque"), {"sub": user, "team_id": team})
        row: Final = spend_row(tokeninfo.proxy, chat_as(tokeninfo.proxy, token, model))
        assert (row["user"], row["team_id"]) == (user, team), row


def test_a_token_info_endpoint_is_called_with_get_and_the_bearer_token(tokeninfo: IntrospectionRig) -> None:
    """The endpoint URL has no `introspect` in it.

    After one served chat the double must have seen exactly one GET for the token, carrying
    `Authorization: Bearer <token>` and no body.
    """
    with tokeninfo.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = tokeninfo.info.grant(unique("opaque-get"), {"sub": unique("oauth-user")})
        assert served_body(chat_as(tokeninfo.proxy, token, model))
        seen: Final = tokeninfo.info.requests_for(token)
        assert [request.method for request in seen] == ["GET"], seen
        assert not seen[0].body, seen


def test_a_token_the_endpoint_rejects_is_a_401_naming_the_status(tokeninfo: IntrospectionRig) -> None:
    """The token-info endpoint answers 401 for a token it does not know.

    The proxy must answer 401 `auth_error` whose message starts "Authentication Error, Oauth 2.0 Token
    validation failed: Client error '401 Unauthorized' for url '<endpoint>'" (httpx's own status error text).
    """
    with tokeninfo.proxy.scenario() as scenario:
        model: Final = scenario.model()
        response: Final = chat_as(tokeninfo.proxy, unique("unknown-opaque"), model)
        assert_denied(response, status=401, error_type="auth_error")
        assert message_of(response).startswith(
            f"{STATUS_FAILURE}Client error '401 Unauthorized' for url '{tokeninfo.info.url}'"
        ), response.text


def test_an_unreachable_endpoint_is_a_401_not_a_503(tokeninfo: IntrospectionRig) -> None:
    """The token-info endpoint drops the connection for this token without answering.

    The proxy must answer 401 `auth_error` starting "Authentication Error, An error occurred during token
    validation: ". Unlike a JWKS outage (503), an introspection outage looks like a bad token to the client.
    """
    with tokeninfo.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = tokeninfo.info.grant(unique("dropped"), None)
        response: Final = chat_as(tokeninfo.proxy, token, model)
        assert_denied(response, status=401, error_type="auth_error")
        assert message_of(response).startswith(OTHER_FAILURE), response.text


def test_a_token_info_reply_that_is_not_a_json_object_is_refused(tokeninfo: IntrospectionRig) -> None:
    """The endpoint answers 200 with the JSON list `["not", "an", "object"]`.

    The strict response check rejects it, so the proxy must answer 401 `auth_error` starting "Authentication
    Error, An error occurred during token validation: ".
    """
    with tokeninfo.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = tokeninfo.info.grant(unique("listy"), ["not", "an", "object"])
        response: Final = chat_as(tokeninfo.proxy, token, model)
        assert_denied(response, status=401, error_type="auth_error")
        assert message_of(response).startswith(OTHER_FAILURE), response.text


def test_a_get_token_info_reply_saying_inactive_is_still_accepted(tokeninfo: IntrospectionRig) -> None:
    """The GET endpoint answers 200 `{"sub": <user>, "active": false}`.

    `active` is only read for the POST introspection form, so the proxy must serve the request. Pinned as
    current behaviour; a port that starts honouring `active` here changes who is let in.
    """
    with tokeninfo.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = tokeninfo.info.grant(unique("inactive-get"), {"sub": unique("oauth-user"), "active": False})
        assert served_body(chat_as(tokeninfo.proxy, token, model))


def test_spend_logs_keep_a_hash_of_an_opaque_token_not_the_token(tokeninfo: IntrospectionRig) -> None:
    """An opaque introspected token serves a chat.

    The spend row's `api_key` must be the SHA-256 hex digest of the token, never the token itself.
    UNVERIFIED: `UserAPIKeyAuth` keeps a non-`sk-`, non-JWT value unhashed; the digest is read from the
    `user_api_key_hash` the batch logging contract pins for the same path.
    """
    with tokeninfo.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = tokeninfo.info.grant(unique("hashed"), {"sub": unique("oauth-user")})
        row: Final = spend_row(tokeninfo.proxy, chat_as(tokeninfo.proxy, token, model))
        assert row["api_key"] != token, row
        assert row["api_key"] == sha256(token.encode()).hexdigest(), row


def test_a_virtual_key_on_an_llm_route_is_sent_to_the_token_info_endpoint(tokeninfo: IntrospectionRig) -> None:
    """With OAuth2 auth on, a valid `sk-` virtual key calls a chat completion.

    The key is not a JWT, so it is introspected instead of looked up: the double must receive it and,
    since it does not know it, the proxy must answer 401 with the token validation failure. Pinned because
    it means virtual keys stop working on LLM routes and are disclosed to the identity provider.
    """
    with tokeninfo.proxy.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        response: Final = chat_as(tokeninfo.proxy, key, model)
        assert_denied(response, status=401, error_type="auth_error")
        assert message_of(response).startswith(STATUS_FAILURE), response.text
        assert len(tokeninfo.info.requests_for(key)) == 1


def test_the_master_key_is_introspected_on_llm_and_info_routes_but_not_on_management_routes(
    tokeninfo: IntrospectionRig,
) -> None:
    """The double vouches for the master key as user `dec1-oauth-master` with role `proxy_admin`. The master
    key calls `POST /key/generate`, then a chat completion.

    The management call must not reach the double. The chat must reach it once and be attributed to the
    double's user, not to the master key's usual admin identity, so the identity provider decides who the
    master key is on LLM routes.
    """
    master: Final = tokeninfo.proxy.key
    with tokeninfo.proxy.scenario() as scenario:
        model: Final = scenario.model()
        before: Final = len(tokeninfo.info.requests_for(master))
        scenario.key(models=[model])
        assert len(tokeninfo.info.requests_for(master)) == before
        row: Final = spend_row(tokeninfo.proxy, chat_as(tokeninfo.proxy, master, model))
        assert len(tokeninfo.info.requests_for(master)) == before + 1
        assert row["user"] == MASTER_IDENTITY, row


def test_the_model_list_info_route_is_introspected(tokeninfo: IntrospectionRig) -> None:
    """A granted opaque token calls `GET /v1/models`.

    Info routes are covered, so the double must receive the token and the proxy must answer 200.
    """
    token: Final = tokeninfo.info.grant(unique("models-route"), {"sub": unique("oauth-user")})
    response: Final = tokeninfo.proxy.request("GET", "/v1/models", key=token)
    assert response.status_code == 200, response.text
    assert len(tokeninfo.info.requests_for(token)) == 1


def test_an_introspection_endpoint_with_client_credentials_gets_a_post_with_basic_auth(
    introspect: IntrospectionRig,
) -> None:
    """The URL contains `introspect` and both client id and secret are set.

    The double must receive a POST whose form body is `token=<token>` and whose `Authorization` is `Basic
    base64(<id>:<secret>)`, and the chat must be served.
    """
    with introspect.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = introspect.info.grant(unique("posted"), {"active": True, "uid": unique("oauth-user")})
        assert served_body(chat_as(introspect.proxy, token, model))
        seen: Final = introspect.info.requests_for(token)
        assert [request.method for request in seen] == ["POST"], seen
        headers: Final = {name.lower(): value for name, value in seen[0].headers.items()}
        expected: Final = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
        assert headers["authorization"] == f"Basic {expected}", headers
        assert parse_qs(seen[0].body.decode()) == {"token": [token]}


def test_an_inactive_introspection_result_is_refused(introspect: IntrospectionRig) -> None:
    """The POST introspection endpoint answers `{"active": false}`.

    The proxy must answer 401 `auth_error` with exactly "Authentication Error, An error occurred during token
    validation: Token is not active".
    """
    with introspect.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = introspect.info.grant(unique("inactive"), {"active": False, "uid": unique("oauth-user")})
        assert_denied(
            chat_as(introspect.proxy, token, model),
            status=401,
            error_type="auth_error",
            message=f"{OTHER_FAILURE}Token is not active",
        )


def test_custom_field_names_choose_which_reply_fields_name_the_user_and_team(introspect: IntrospectionRig) -> None:
    """`OAUTH_USER_ID_FIELD_NAME=uid` and `OAUTH_USER_TEAM_ID_FIELD_NAME=group`; the reply carries `uid`,
    `group`, and decoy `sub` and `team_id` fields.

    The spend row's `user` must be the `uid` value and its `team_id` the `group` value.
    """
    with introspect.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        user: Final = unique("uid-user")
        token: Final = introspect.info.grant(
            unique("fields"),
            {"active": True, "uid": user, "group": team, "sub": "decoy-sub", "team_id": "decoy-team"},
        )
        row: Final = spend_row(introspect.proxy, chat_as(introspect.proxy, token, model))
        assert (row["user"], row["team_id"]) == (user, team), row


def test_an_introspect_url_without_a_client_secret_falls_back_to_get(public_client: IntrospectionRig) -> None:
    """The URL contains `introspect`, `OAUTH_CLIENT_ID` is set, `OAUTH_CLIENT_SECRET` is not.

    Both credentials are needed for the POST form, so the double must receive a GET with the bearer token.
    """
    with public_client.proxy.scenario() as scenario:
        model: Final = scenario.model()
        token: Final = public_client.info.grant(unique("public"), {"sub": unique("oauth-user")})
        assert served_body(chat_as(public_client.proxy, token, model))
        assert [request.method for request in public_client.info.requests_for(token)] == ["GET"]


def test_oauth2_introspection_without_a_license_is_refused(tmp_path_factory: pytest.TempPathFactory) -> None:
    """`enable_oauth2_auth` is on and `LITELLM_LICENSE` is removed; an opaque token calls a chat completion.

    The proxy must answer 403 `auth_error` param `premium_user` with "Oauth2 token validation is only
    available for premium users. " followed by the enterprise notice, without calling the endpoint.
    """
    with _introspection_rig(
        tmp_path_factory.mktemp("dec1-authmodes-oauth2-unlicensed"),
        path="/tokeninfo",
        remove_environment=("LITELLM_LICENSE",),
    ) as rig:
        token: Final = rig.info.grant(unique("unlicensed"), {"sub": unique("oauth-user")})
        assert_denied(
            chat_as(rig.proxy, token, "any-model"),
            status=403,
            error_type="auth_error",
            message=f"Oauth2 token validation is only available for premium users. {NOT_PREMIUM}",
            param="premium_user",
        )
        assert rig.info.requests_for(token) == []


def test_identity_headers_from_a_trusted_peer_name_the_caller(headers_trusted: Gateway) -> None:
    """The proxy trusts 127.0.0.1 and maps `X-Dec1-User` to `user_id`; a chat sends that header and an
    arbitrary bearer value that is no key at all.

    The chat must be served and its spend row's `user` must be the header value.
    """
    user: Final = unique("header-user")
    response: Final = _header_chat(headers_trusted, {USER_HEADER: user, EMAIL_HEADER: f"{user}@example.test"})
    assert served_body(response)
    assert spend_row(headers_trusted, response)["user"] == user


def test_the_models_header_is_split_on_commas_but_not_enforced(headers_trusted: Gateway) -> None:
    """`models` is mapped from `X-Dec1-Models`; a chat sends `X-Dec1-Models: some-other-model, another` and
    asks for the configured model.

    UNVERIFIED: the header fills `UserAPIKeyAuth.models`, but nothing on this path asks for the key model
    check, so the request is served. Pinned as a suspected gap: the header looks like a restriction and is not.
    """
    response: Final = _header_chat(
        headers_trusted, {USER_HEADER: unique("header-models"), MODELS_HEADER: "some-other-model, another"}
    )
    assert served_body(response)


def test_header_trust_replaces_the_master_key_on_management_routes(headers_trusted: Gateway) -> None:
    """With header trust on, the master key calls `POST /user/new` with no identity headers.

    Header auth runs before the master key comparison, so the caller is an anonymous header identity and
    must not be allowed to create a user. UNVERIFIED: the exact refusal status is not read from code.
    """
    response: Final = headers_trusted.request("POST", "/user/new", {"user_id": unique("never-made")})
    assert response.status_code in {401, 403}, response.text


def test_header_trust_without_trusted_proxy_ranges_refuses_everything(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """`enable_oauth2_proxy_auth` is on with mappings but no `trusted_proxy_ranges`.

    Every request must be 401 `auth_error` with exactly "Authentication Error, OAuth2 proxy auth requires
    general_settings.trusted_proxy_ranges before trusting identity headers from an upstream proxy."
    """
    with _header_proxy(
        tmp_path_factory.mktemp("dec1-authmodes-headers-unranged"), {"oauth2_config_mappings": IDENTITY_MAPPINGS}
    ) as proxy:
        assert_denied(
            _header_chat(proxy, {USER_HEADER: unique("header-user")}),
            status=401,
            error_type="auth_error",
            message=(
                "Authentication Error, OAuth2 proxy auth requires general_settings.trusted_proxy_ranges before "
                "trusting identity headers from an upstream proxy."
            ),
        )


def test_identity_headers_from_an_untrusted_peer_are_refused(tmp_path_factory: pytest.TempPathFactory) -> None:
    """`trusted_proxy_ranges` is `["10.0.0.0/8"]` and the test connects from 127.0.0.1.

    The proxy must answer 401 `auth_error` with exactly "Authentication Error, OAuth2 proxy auth only accepts
    identity headers from configured trusted proxy ranges. Direct client IP '127.0.0.1' is not trusted."
    """
    with _header_proxy(
        tmp_path_factory.mktemp("dec1-authmodes-headers-untrusted"),
        {"trusted_proxy_ranges": ["10.0.0.0/8"], "oauth2_config_mappings": IDENTITY_MAPPINGS},
    ) as proxy:
        assert_denied(
            _header_chat(proxy, {USER_HEADER: unique("header-user"), "X-Forwarded-For": "10.1.2.3"}),
            status=401,
            error_type="auth_error",
            message=(
                "Authentication Error, OAuth2 proxy auth only accepts identity headers from configured trusted "
                "proxy ranges. Direct client IP '127.0.0.1' is not trusted."
            ),
        )


def test_mapping_a_privilege_field_from_a_header_refuses_every_request(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """`oauth2_config_mappings` maps `user_role` (and `max_budget`) from headers next to `user_id`.

    The proxy must answer 401 `auth_error` whose message starts "Authentication Error, Oauth2 proxy auth
    refuses to map non-identity UserAPIKeyAuth fields from request headers: ['max_budget', 'user_role'].",
    so a caller can never send `X-Role: proxy_admin`.
    """
    with _header_proxy(
        tmp_path_factory.mktemp("dec1-authmodes-headers-privilege"),
        {
            "trusted_proxy_ranges": ["127.0.0.1/32", "::1/128"],
            "oauth2_config_mappings": {"user_id": USER_HEADER, "user_role": "X-Role", "max_budget": "X-Budget"},
        },
    ) as proxy:
        response: Final = _header_chat(proxy, {USER_HEADER: unique("header-user"), "X-Role": "proxy_admin"})
        assert_denied(response, status=401, error_type="auth_error")
        assert message_of(response).startswith(
            "Authentication Error, Oauth2 proxy auth refuses to map non-identity UserAPIKeyAuth fields from "
            "request headers: ['max_budget', 'user_role']."
        ), response.text
