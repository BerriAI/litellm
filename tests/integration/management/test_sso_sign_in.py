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
import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import Gateway, gateway_from_environment
from tests.integration._support.database import read_rows, write_rows
from tests.integration._support.process import OWNED_STATE_VARIABLE, OwnedState, owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

CLIENT_ID: Final = "integration-sso-client"
CLIENT_SECRET: Final = "integration-sso-secret"
IDP_PROVIDER: Final = "integration-idp"


def _idp_reply(request: Request) -> Reply:
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
    with wire_server(_idp_reply) as idp, gateway_from_environment() as rig:
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
