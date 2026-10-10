import base64
import json
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import Gateway, gateway_from_environment
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

pytestmark: Final = pytest.mark.timeout(300)

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
                }
            ).encode()
        )
    return Reply(status=404, body=b'{"error": "not_found"}')


@contextmanager
def _sso_gateway(directory: Path, litellm_settings: Mapping[str, JsonValue]) -> Iterator[Gateway]:
    config: Final = directory / "sso_proxy_config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_list": [],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
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
            yield gw


def _sign_in(gateway: Gateway, subject: str) -> httpx.Response:
    proxy_url: Final = str(gateway.client.base_url).rstrip("/")
    with httpx.Client(follow_redirects=False, trust_env=False, timeout=30) as browser:
        start: Final = browser.get(f"{proxy_url}/sso/key/generate")
        assert start.is_redirect, f"{start.status_code} {start.text}"
        at_idp: Final = browser.get(f"{start.headers['location']}&{urlencode({'login_hint': subject})}")
        assert at_idp.status_code == 302, f"{at_idp.status_code} {at_idp.text}"
        return browser.get(at_idp.headers["location"])


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
    with _sso_gateway(tmp_path, litellm_settings) as gateway:
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
