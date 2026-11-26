import base64
import contextlib
import json
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import jwt
import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, object_value, string_value
from tests.integration._support.database import read_rows, write_rows
from tests.integration._support.process import OWNED_STATE_VARIABLE, OwnedState, owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

CLIENT_ID: Final = "integration-sso-client"
CLIENT_SECRET: Final = "integration-sso-secret"
IDP_PROVIDER: Final = "integration-idp"


def _idp_reply(request: Request, claims: Mapping[str, Mapping[str, JsonValue]] | None = None) -> Reply:
    target: Final = urlparse(request.target)
    query: Final = parse_qs(target.query)
    if target.path == "/authorize":
        code: Final = f"code-{query['login_hint'][0]}"
        location: Final = f"{query['redirect_uri'][0]}?{urlencode({'code': code, 'state': query['state'][0]})}"
        return Reply(status=302, body=b"", headers={"location": location})
    if target.path == "/token":
        expected: Final = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
        form: Final = parse_qs(request.body.decode())
        if request.headers.get("authorization") != f"Basic {expected}" or form.get("grant_type") != [
            "authorization_code"
        ]:
            return Reply(status=401, body=b'{"error": "invalid_client"}')
        subject: Final = form["code"][0].removeprefix("code-")
        return Reply(body=json.dumps({"access_token": f"access-{subject}", "token_type": "Bearer"}).encode())
    if target.path == "/userinfo":
        signed_in: Final = request.headers.get("authorization", "").removeprefix("Bearer access-")
        return Reply(
            body=json.dumps(
                {
                    "sub": signed_in,
                    "preferred_username": signed_in,
                    "email": f"{signed_in}@example.com",
                    "provider": IDP_PROVIDER,
                    "dept": _dept(signed_in),
                    "groups": [_group(signed_in)],
                    **(claims.get(signed_in, {}) if claims is not None else {}),
                }
            ).encode()
        )
    return Reply(status=404, body=b'{"error": "not_found"}')


def _dept(subject: str) -> str:
    return f"dept-{subject}"


def _group(subject: str) -> str:
    return f"group-{subject}"


@contextmanager
def _sso_gateway(
    directory: Path,
    litellm_settings: Mapping[str, JsonValue],
    general_settings: Mapping[str, JsonValue] | None = None,
    db_sso_settings: Mapping[str, JsonValue] | None = None,
    *,
    claims: Mapping[str, Mapping[str, JsonValue]] | None = None,
) -> Iterator[Gateway]:
    config: Final = directory / "sso_proxy_config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_list": [],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    **(general_settings or {}),
                },
                "litellm_settings": dict(litellm_settings),
            }
        )
    )
    with wire_server(partial(_idp_reply, claims=claims)) as idp, gateway_from_environment() as rig:
        environment: Final = {
            "GENERIC_CLIENT_ID": CLIENT_ID,
            "GENERIC_CLIENT_SECRET": CLIENT_SECRET,
            "GENERIC_AUTHORIZATION_ENDPOINT": f"{idp.url}/authorize",
            "GENERIC_TOKEN_ENDPOINT": f"{idp.url}/token",
            "GENERIC_USERINFO_ENDPOINT": f"{idp.url}/userinfo",
            "OAUTHLIB_INSECURE_TRANSPORT": "1",
        }
        with owned_proxy(rig, directory, environment, config=config, remove_environment=("PROXY_BASE_URL",)) as gw:
            if db_sso_settings is None:
                yield gw
                return
            with contextlib.ExitStack() as cleanups:
                cleanups.callback(
                    write_rows,
                    'DELETE FROM "LiteLLM_SSOConfig" WHERE id = %s',
                    ("sso_config",),
                    database_url=gw.database_url,
                )
                stored: Final = _store_sso_settings(gw, {**_generic_sso_settings(environment), **db_sso_settings})
                assert stored.status_code == 200, stored.text
                yield gw


def _generic_sso_settings(environment: Mapping[str, str]) -> Mapping[str, JsonValue]:
    return {
        "generic_client_id": environment["GENERIC_CLIENT_ID"],
        "generic_client_secret": environment["GENERIC_CLIENT_SECRET"],
        "generic_authorization_endpoint": environment["GENERIC_AUTHORIZATION_ENDPOINT"],
        "generic_token_endpoint": environment["GENERIC_TOKEN_ENDPOINT"],
        "generic_userinfo_endpoint": environment["GENERIC_USERINFO_ENDPOINT"],
    }


def _store_sso_settings(gateway: Gateway, settings: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request("PATCH", "/update/sso_settings", settings)


def _sign_in(gateway: Gateway, subject: str) -> httpx.Response:
    proxy_url: Final = str(gateway.client.base_url).rstrip("/")
    with httpx.Client(follow_redirects=False, trust_env=False, timeout=30) as browser:
        start: Final = browser.get(f"{proxy_url}/sso/key/generate")
        assert start.is_redirect, f"{start.status_code} {start.text}"
        at_idp: Final = browser.get(f"{start.headers['location']}&{urlencode({'login_hint': subject})}")
        assert at_idp.status_code == 302, f"{at_idp.status_code} {at_idp.text}"
        return browser.get(at_idp.headers["location"])


def _dashboard_session(gateway: Gateway, subject: str) -> dict[str, JsonValue]:
    callback: Final = _sign_in(gateway, subject)
    assert callback.status_code == 303, callback.text
    assert callback.headers["location"].startswith(f"{str(gateway.client.base_url).rstrip('/')}/ui/?login=success")
    return JSON_OBJECT.validate_python(jwt.decode(callback.cookies["token"], gateway.key, algorithms=["HS256"]))


def _delete_user(gateway: Gateway, subject: str) -> None:
    response: Final = gateway.request("POST", "/user/delete", {"user_ids": [subject]})
    assert response.status_code == 200, response.text


def _signed_in_user(subject: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT user_email, user_role, metadata FROM "LiteLLM_UserTable" WHERE user_id = %s', (subject,))


@pytest.mark.parametrize(
    ("litellm_settings", "expected_role"),
    [
        ({}, "internal_user_viewer"),
        ({"default_internal_user_params": {"user_role": "internal_user"}}, "internal_user"),
    ],
    ids=["no-default", "default-internal-user"],
)
def test_a_first_sso_sign_in_creates_the_user_with_the_default_role(
    tmp_path: Path, litellm_settings: Mapping[str, JsonValue], expected_role: str
) -> None:
    subject: Final = f"sso-new-user-{uuid.uuid4().hex[:12]}"
    with _sso_gateway(tmp_path, litellm_settings) as gateway, contextlib.ExitStack() as cleanups:
        cleanups.callback(_delete_user, gateway, subject)
        callback: Final = _sign_in(gateway, subject)

        assert callback.status_code == 303, f"{callback.status_code} {callback.text}"
        assert callback.headers["location"].startswith(f"{str(gateway.client.base_url).rstrip('/')}/ui/?login=success")
        assert _signed_in_user(subject) == [
            {
                "user_email": f"{subject}@example.com",
                "user_role": expected_role,
                "metadata": {"auth_provider": IDP_PROVIDER},
            }
        ]


def test_generic_sso_links_a_verified_private_email_to_the_existing_local_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject: Final = f"directory-{uuid.uuid4().hex}"
    email: Final = f"Dan-{uuid.uuid4().hex}@Corp.LOCAL"
    claims: Final[Mapping[str, Mapping[str, JsonValue]]] = {subject: {"email": email, "email_verified": True}}
    monkeypatch.setenv(OWNED_STATE_VARIABLE, OwnedState.PRIVATE.value)
    monkeypatch.setenv("GENERIC_USER_ID_ATTRIBUTE", "sub")
    monkeypatch.setenv("ALLOWED_EMAIL_DOMAINS", "corp.local")
    with _sso_gateway(tmp_path, {}, claims=claims) as gateway, gateway.scenario() as scenario:
        account: Final = scenario.user(user_email=email.lower(), user_role="internal_user")
        for _ in range(2):
            session: Final = _dashboard_session(gateway, subject)
            assert session["user_id"] == account
            own: Final = gateway.request("GET", "/user/info", key=string_value(session["key"]))
            assert own.status_code == 200, own.text
            assert object_value(JSON_OBJECT.validate_json(own.content)["user_info"])["user_id"] == account
            assert gateway.rows(
                'SELECT user_id, user_role, sso_user_id FROM "LiteLLM_UserTable" '
                "WHERE lower(user_email) = %s OR user_id IN (%s, %s)",
                (email.lower(), account, subject),
            ) == [{"user_id": account, "user_role": "internal_user", "sso_user_id": subject}]


def test_generic_sso_rejects_a_private_domain_outside_the_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subject: Final = f"directory-{uuid.uuid4().hex}"
    email: Final = f"{subject}@corp.local"
    monkeypatch.setenv(OWNED_STATE_VARIABLE, OwnedState.PRIVATE.value)
    monkeypatch.setenv("ALLOWED_EMAIL_DOMAINS", "example.com")
    with _sso_gateway(tmp_path, {}, claims={subject: {"email": email, "email_verified": True}}) as gateway:
        denied: Final = _sign_in(gateway, subject)
        assert denied.status_code == 401, denied.text
        assert "email domain=corp.local" in denied.text
        assert "token" not in denied.cookies
        assert (
            gateway.rows(
                'SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s OR lower(user_email) = %s',
                (subject, email),
            )
            == []
        )


@pytest.mark.parametrize("domain", ["example.com", "corp.local"])
@pytest.mark.parametrize("email_verified,existing_link", [(False, None), (True, "previous-directory-id")])
def test_generic_sso_preserves_the_email_owner_when_the_directory_identity_cannot_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, domain: str, email_verified: bool, existing_link: str | None
) -> None:
    subject: Final = f"directory-{uuid.uuid4().hex}"
    email: Final = f"member@{domain}"
    claims: Final[Mapping[str, Mapping[str, JsonValue]]] = {subject: {"email": email, "email_verified": email_verified}}
    monkeypatch.setenv(OWNED_STATE_VARIABLE, OwnedState.PRIVATE.value)
    monkeypatch.setenv("LITELLM_EMAIL_FALLBACK_LINKS_ANY_USER", "false")
    with _sso_gateway(tmp_path, {}, claims=claims) as gateway, gateway.scenario() as scenario:
        owner: Final = scenario.user(user_email=email, sso_user_id=existing_link)
        session: Final = _dashboard_session(gateway, subject)
        assert session["user_id"] == subject
        assert gateway.rows(
            'SELECT user_id, sso_user_id FROM "LiteLLM_UserTable" WHERE lower(user_email) = %s', (email,)
        ) == [{"user_id": owner, "sso_user_id": existing_link}]
        assert gateway.rows('SELECT user_email FROM "LiteLLM_UserTable" WHERE user_id = %s', (subject,)) == [
            {"user_email": None}
        ]


def test_free_sso_admission_preserves_existing_users_and_serializes_the_last_seat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(OWNED_STATE_VARIABLE, OwnedState.PRIVATE.value)
    monkeypatch.delenv("LITELLM_LICENSE", raising=False)
    with _sso_gateway(tmp_path, {}) as gateway:
        for index in range(4):
            signed_in: Final = _sign_in(gateway, f"free-sso-{index}")
            assert signed_in.status_code == 303, signed_in.text

        contenders: Final = tuple(f"free-sso-contender-{index}" for index in range(8))
        with ThreadPoolExecutor(max_workers=len(contenders)) as pool:
            responses: Final = tuple(pool.map(partial(_sign_in, gateway), contenders))

        assert sorted(response.status_code for response in responses) == [303, 403, 403, 403, 403, 403, 403, 403]
        assert gateway.rows('SELECT COUNT(*) AS count FROM "LiteLLM_UserTable"', ()) == [{"count": 5}]
        for subject, response in zip(contenders, responses):
            if response.status_code == 403:
                assert "requires a LiteLLM Enterprise license" in response.text
                assert not response.cookies
                assert gateway.rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (subject,)) == []

        existing: Final = _sign_in(gateway, "free-sso-0")
        assert existing.status_code == 303, existing.text

        admin_created: Final = gateway.request(
            "POST", "/user/new", {"user_id": "api-created-user", "auto_create_key": False}
        )
        assert admin_created.status_code == 200, admin_created.text
        above_capacity: Final = _sign_in(gateway, "free-sso-0")
        refused: Final = _sign_in(gateway, "free-sso-extra")
        assert (above_capacity.status_code, refused.status_code) == (303, 403)
        assert gateway.rows('SELECT COUNT(*) AS count FROM "LiteLLM_UserTable"', ()) == [{"count": 6}]


@pytest.mark.parametrize(
    ("enable_jwt_auth", "team_id_upsert", "team_exists_before", "expect_member"),
    [(True, True, False, True), (False, True, False, False), (True, False, False, False), (True, False, True, True)],
    ids=[
        "upsert-creates-missing-team",
        "upsert-without-jwt-auth-skips-missing-team",
        "no-upsert-skips-missing-team",
        "no-upsert-joins-existing-team",
    ],
)
def test_sso_sign_in_upserts_the_team_id_jwt_field_team_only_when_team_id_upsert_is_on(
    tmp_path: Path, enable_jwt_auth: bool, team_id_upsert: bool, team_exists_before: bool, expect_member: bool
) -> None:
    subject: Final = f"sso-dept-user-{uuid.uuid4().hex[:12]}"
    team_id: Final = _dept(subject)
    jwtauth: Final = {
        "enable_jwt_auth": enable_jwt_auth,
        "litellm_jwtauth": {"team_id_jwt_field": "dept", "team_id_upsert": team_id_upsert},
    }
    with _sso_gateway(tmp_path, {}, jwtauth) as gateway, contextlib.ExitStack() as cleanups:
        cleanups.callback(gateway.request, "POST", "/team/delete", {"team_ids": [team_id]})
        cleanups.callback(_delete_user, gateway, subject)
        if team_exists_before:
            created: Final = gateway.request("POST", "/team/new", {"team_id": team_id})
            assert created.status_code == 200, created.text

        callback: Final = _sign_in(gateway, subject)

        assert callback.status_code == 303, f"{callback.status_code} {callback.text}"
        teams: Final = gateway.rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))
        user: Final = gateway.rows('SELECT teams FROM "LiteLLM_UserTable" WHERE user_id = %s', (subject,))
        assert (len(teams), user) == (int(expect_member), [{"teams": [team_id] if expect_member else []}])


def test_sso_sign_in_with_db_team_mappings_creates_the_team_id_jwt_field_team_and_only_joins_groups(
    tmp_path: Path,
) -> None:
    subject: Final = f"sso-mapped-user-{uuid.uuid4().hex[:12]}"
    team_id: Final = _dept(subject)
    group_team_id: Final = _group(subject)
    jwtauth: Final = {
        "enable_jwt_auth": True,
        "litellm_jwtauth": {"team_id_jwt_field": "dept", "team_id_upsert": True},
    }
    team_mappings: Final = {"team_mappings": {"team_ids_jwt_field": "groups"}}
    with (
        _sso_gateway(tmp_path, {}, jwtauth, db_sso_settings=team_mappings) as gateway,
        contextlib.ExitStack() as cleanups,
    ):
        cleanups.callback(gateway.request, "POST", "/team/delete", {"team_ids": [team_id, group_team_id]})
        cleanups.callback(_delete_user, gateway, subject)

        callback: Final = _sign_in(gateway, subject)

        assert callback.status_code == 303, f"{callback.status_code} {callback.text}"
        teams: Final = gateway.rows(
            'SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id IN (%s, %s)', (team_id, group_team_id)
        )
        user: Final = gateway.rows('SELECT teams FROM "LiteLLM_UserTable" WHERE user_id = %s', (subject,))
        assert (teams, user) == ([{"team_id": team_id}], [{"teams": [team_id]}])
