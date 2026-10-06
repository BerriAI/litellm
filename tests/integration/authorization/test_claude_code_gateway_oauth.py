import base64
import hashlib
import html
import json
import os
import re
import uuid
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Final, Literal
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
import yaml
from anthropic.types import Message
from integration._support import claude_code as cc
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, JsonValue, TypeAdapter

_CLIENT_ID: Final = "scripted-cli-client"
_CLIENT_SECRET: Final = "scripted-cli-secret"
_AUTHORIZATION_CODE: Final = "scripted-auth-code"
_IDP_ACCESS_TOKEN: Final = "scripted-idp-access-token"
_USER_ID: Final = "gateway-sso-user"
_USER_EMAIL: Final = "gateway-sso-user@example.com"
_BACKEND: Final = "claude-sonnet-4-6"
_API_KEY: Final = "synthetic-anthropic-key"
_EXPECTED_BETA: Final = ",".join(
    sorted(set((*cc.CLI_BETA.split(","), "context-management-2025-06-27", "prompt-caching-scope-2026-01-05")))
)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGE: Final = TypeAdapter(Message)
_BROWSER_TOKEN: Final = re.compile(r'name="browser_complete_token" value="([^"]+)"')


class _DeviceAuthorization(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str | None = None
    expires_in: int
    interval: int


class _AccessToken(BaseModel):
    access_token: str
    expires_in: int
    token_type: Literal["Bearer"]


def _owned_config(path: Path) -> Path:
    config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    general_settings: Final = _JSON_OBJECT.validate_python(config.get("general_settings", {}))
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "general_settings": {**general_settings, "enable_claude_code_gateway": True},
            }
        )
    )
    return path


def _identity_provider(request: Request) -> Reply:
    parsed_target: Final = urlsplit(request.target)
    if request.method == "GET" and parsed_target.path == "/authorize":
        query: Final = parse_qs(parsed_target.query)
        assert query["client_id"] == [_CLIENT_ID], query
        assert query["response_type"] == ["code"], query
        assert query["scope"] == ["openid email profile"], query
        assert query["code_challenge_method"] == ["S256"], query
        assert len(query["code_challenge"][0]) == 43, query
        callback: Final = query["redirect_uri"][0]
        assert urlsplit(callback).path == "/sso/callback", query
        state: Final = query["state"][0]
        assert state.startswith("litellm-session-token:"), query
        location: Final = f"{callback}?{urlencode({'code': _AUTHORIZATION_CODE, 'state': state})}"
        return Reply(status=302, headers={"location": location})

    if request.method == "POST" and parsed_target.path == "/token":
        form: Final = parse_qs(request.body.decode())
        encoded_credentials: Final = base64.b64encode(f"{_CLIENT_ID}:{_CLIENT_SECRET}".encode()).decode()
        assert request.headers["authorization"] == f"Basic {encoded_credentials}", request.headers
        assert request.headers["content-type"] == "application/x-www-form-urlencoded", request.headers
        assert set(form) == {"code", "code_verifier", "grant_type", "redirect_uri"}, form
        assert form["grant_type"] == ["authorization_code"], form
        assert form["code"] == [_AUTHORIZATION_CODE], form
        assert len(form["code_verifier"][0]) == 43, form
        assert urlsplit(form["redirect_uri"][0]).path == "/sso/callback", form
        return Reply(body=json.dumps({"access_token": _IDP_ACCESS_TOKEN, "token_type": "Bearer"}).encode())

    if request.method == "GET" and parsed_target.path == "/userinfo":
        assert request.headers["authorization"] == f"Bearer {_IDP_ACCESS_TOKEN}", request.headers
        return Reply(body=json.dumps({"sub": _USER_ID, "email": _USER_EMAIL}).encode())

    raise AssertionError(f"unexpected scripted IdP request: {request.method} {request.target}")


_REPLY_CONTENT: Final = ({"type": "text", "text": "authorized"},)
_REPLY_USAGE: Final = {"input_tokens": 5, "output_tokens": 1}


def _anthropic_reply(request: Request, expected_body: dict[str, JsonValue], message_id: str) -> Reply:
    assert request.method == "POST" and request.target == "/v1/messages", request.target
    assert request.headers["x-api-key"] == _API_KEY, request.headers
    assert request.headers["anthropic-version"] == "2023-06-01", request.headers
    assert request.headers["anthropic-beta"] == _EXPECTED_BETA, request.headers
    assert "authorization" not in request.headers, request.headers
    body: Final = _JSON_OBJECT.validate_json(request.body)
    streamed: Final = body.get("stream") is True
    assert body == {**expected_body, "stream": streamed}, request.body
    if streamed:
        return Reply(
            chunks=cc.message_stream(message_id, _BACKEND, _REPLY_CONTENT, _REPLY_USAGE),
            content_type="text/event-stream",
        )
    return Reply(body=cc.message_reply(message_id, _BACKEND, _REPLY_CONTENT, _REPLY_USAGE))


@pytest.fixture(scope="module")
def identity_provider() -> Iterator[Wire]:
    with wire_server(_identity_provider) as wire:
        yield wire


@pytest.fixture(scope="module")
def enabled_gateway(identity_provider: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    assert os.environ.get("LITELLM_LICENSE"), "LITELLM_LICENSE is required for the generic SSO integration test"
    with gateway_from_environment() as base_gateway:
        directory: Final = tmp_path_factory.mktemp("claude-code-gateway-oauth")
        config: Final = _owned_config(directory / "gateway-oauth.yaml")
        overrides: Final = {
            "GENERIC_CLIENT_ID": _CLIENT_ID,
            "GENERIC_CLIENT_SECRET": _CLIENT_SECRET,
            "GENERIC_AUTHORIZATION_ENDPOINT": f"{identity_provider.url}/authorize",
            "GENERIC_TOKEN_ENDPOINT": f"{identity_provider.url}/token",
            "GENERIC_USERINFO_ENDPOINT": f"{identity_provider.url}/userinfo",
            "GENERIC_USER_ID_ATTRIBUTE": "sub",
            "GENERIC_USER_EMAIL_ATTRIBUTE": "email",
            "GENERIC_CLIENT_USE_PKCE": "true",
            "GENERIC_INCLUDE_CLIENT_ID": "false",
        }
        with owned_proxy_process(
            base_gateway,
            directory,
            overrides,
            config=config,
            workers=2,
            remove_environment=("GOOGLE_CLIENT_ID", "MICROSOFT_CLIENT_ID", "PROXY_BASE_URL"),
        ) as owned:
            yield owned.gateway


@pytest.mark.timeout(300)
def test_device_authorization_polls_complete_sso_and_mints_a_model_restricted_bearer(
    enabled_gateway: Gateway, identity_provider: Wire
) -> None:
    request_template: Final = {
        **cc.claude_code_request("oauth-model-restriction"),
        "model": "",
        "stream": False,
    }
    expected_body: Final = {**request_template, "model": _BACKEND}
    run_id: Final = uuid.uuid4().hex
    message_ids: Final = (f"msg_oauth_{run_id}_a", f"msg_oauth_{run_id}_a_stream", f"msg_oauth_{run_id}_c")
    pending_message_ids: Final = deque(message_ids)

    with enabled_gateway.scenario() as scenario:
        with wire_server(
            lambda request: _anthropic_reply(request, expected_body, pending_message_ids.popleft())
        ) as upstream:
            model_a: Final = scenario.model(model=f"anthropic/{_BACKEND}", api_base=upstream.url, api_key=_API_KEY)
            model_b: Final = scenario.model(model=f"anthropic/{_BACKEND}", api_base=upstream.url, api_key=_API_KEY)
            model_c: Final = scenario.model(model=f"anthropic/{_BACKEND}", api_base=upstream.url, api_key=_API_KEY)
            user_id: Final = scenario.user(
                user_id=_USER_ID,
                user_email=_USER_EMAIL,
                user_role="internal_user",
                models=[model_a, model_b],
            )
            team_id: Final = scenario.team(models=[model_a, model_c])
            enabled_gateway.post(
                "/team/member_add", {"team_id": team_id, "member": {"role": "user", "user_id": user_id}}
            )

            authorization_response: Final = enabled_gateway.client.post(
                "/claude_code_gateway/oauth/device_authorization",
                data={"client_id": _CLIENT_ID},
            )
            assert authorization_response.status_code == 200, authorization_response.text
            authorization: Final = _DeviceAuthorization.model_validate_json(authorization_response.content)
            authorization_body: Final = _JSON_OBJECT.validate_json(authorization_response.content)
            assert authorization_body == authorization.model_dump(exclude_none=True), authorization_response.text
            assert authorization.expires_in > 0 and authorization.interval > 0, authorization_response.text
            login_id, separator, poll_secret = authorization.device_code.partition(".")
            assert separator == "." and login_id.startswith("cli-") and poll_secret, authorization_response.text

            verification_parts: Final = urlsplit(authorization.verification_uri)
            assert verification_parts.path == "/sso/key/generate", authorization.verification_uri
            verification_query: Final = parse_qs(verification_parts.query)
            assert verification_query == {"source": ["litellm-cli"], "key": [login_id]}, authorization.verification_uri

            pending: Final = enabled_gateway.client.post(
                "/claude_code_gateway/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": authorization.device_code,
                    "client_id": _CLIENT_ID,
                },
            )
            assert pending.status_code == 400, pending.text
            pending_body: Final = _JSON_OBJECT.validate_json(pending.content)
            assert pending_body == {"error": "authorization_pending"}, pending.text

            wrong_secret: Final = enabled_gateway.client.post(
                "/claude_code_gateway/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": f"{login_id}.wrong-poll-secret",
                    "client_id": _CLIENT_ID,
                },
            )
            assert wrong_secret.status_code == 400, wrong_secret.text
            assert _JSON_OBJECT.validate_json(wrong_secret.content) == {"error": "expired_token"}, wrong_secret.text
            assert "authorization_pending" not in wrong_secret.text, wrong_secret.text

            verification_response: Final = enabled_gateway.client.get(
                authorization.verification_uri,
                follow_redirects=False,
            )
            assert verification_response.status_code == 303, verification_response.text
            assert "litellm_oauth_state" in verification_response.headers.get("set-cookie", ""), (
                verification_response.headers,
            )
            browser_state: Final = enabled_gateway.client.cookies.get("litellm_oauth_state")
            assert browser_state, verification_response.headers

            identity_provider_response: Final = enabled_gateway.client.get(
                verification_response.headers["location"],
                follow_redirects=False,
            )
            assert identity_provider_response.status_code == 302, identity_provider_response.text
            callback_location: Final = identity_provider_response.headers["location"]
            callback_parts: Final = urlsplit(callback_location)
            assert callback_parts.path == "/sso/callback", callback_location
            callback_query: Final = parse_qs(callback_parts.query)
            assert callback_query["state"] == [browser_state], callback_location

            callback_response: Final = enabled_gateway.client.get(callback_location, follow_redirects=False)
            assert callback_response.status_code == 200, callback_response.text
            browser_token_match: Final = _BROWSER_TOKEN.search(callback_response.text)
            assert browser_token_match is not None, callback_response.text
            browser_complete_token: Final = html.unescape(browser_token_match.group(1))

            still_pending: Final = enabled_gateway.client.post(
                "/claude_code_gateway/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": authorization.device_code,
                    "client_id": _CLIENT_ID,
                },
            )
            assert still_pending.status_code == 400, still_pending.text
            assert _JSON_OBJECT.validate_json(still_pending.content) == {"error": "authorization_pending"}, (
                still_pending.text,
            )

            wrong_code_completion: Final = enabled_gateway.client.post(
                f"/sso/cli/complete/{login_id}",
                data={"user_code": "WRONG-CODE", "browser_complete_token": browser_complete_token},
            )
            assert wrong_code_completion.status_code == 400, wrong_code_completion.text
            assert _JSON_OBJECT.validate_json(wrong_code_completion.content) == {
                "detail": "Invalid verification code"
            }, wrong_code_completion.text

            pending_after_wrong_code: Final = enabled_gateway.client.post(
                "/claude_code_gateway/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": authorization.device_code,
                    "client_id": _CLIENT_ID,
                },
            )
            assert pending_after_wrong_code.status_code == 400, pending_after_wrong_code.text
            assert _JSON_OBJECT.validate_json(pending_after_wrong_code.content) == {"error": "authorization_pending"}, (
                pending_after_wrong_code.text
            )

            verification_complete: Final = enabled_gateway.client.post(
                f"/sso/cli/complete/{login_id}",
                data={"user_code": authorization.user_code, "browser_complete_token": browser_complete_token},
            )
            assert verification_complete.status_code == 200, verification_complete.text

            token_response: Final = enabled_gateway.client.post(
                "/claude_code_gateway/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": authorization.device_code,
                    "client_id": _CLIENT_ID,
                },
            )
            assert token_response.status_code == 200, token_response.text
            access_token: Final = _AccessToken.model_validate_json(token_response.content)
            access_token_body: Final = _JSON_OBJECT.validate_json(token_response.content)
            assert access_token_body == access_token.model_dump(), token_response.text
            assert set(access_token_body) == {"access_token", "expires_in", "token_type"}, token_response.text
            assert access_token.token_type == "Bearer" and access_token.expires_in > 0, token_response.text

            second_poll: Final = enabled_gateway.client.post(
                "/claude_code_gateway/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": authorization.device_code,
                    "client_id": _CLIENT_ID,
                },
            )
            assert second_poll.status_code == 400, second_poll.text
            assert _JSON_OBJECT.validate_json(second_poll.content) == {"error": "expired_token"}, second_poll.text

            unsupported: Final = enabled_gateway.client.post(
                "/claude_code_gateway/oauth/token",
                data={"grant_type": "urn:unsupported"},
            )
            assert unsupported.status_code == 400, unsupported.text
            assert _JSON_OBJECT.validate_json(unsupported.content) == {
                "error": "unsupported_grant_type",
                "error_description": "Unsupported grant_type: urn:unsupported",
            }, unsupported.text

            refresh: Final = enabled_gateway.client.post(
                "/claude_code_gateway/oauth/token",
                data={"grant_type": "refresh_token"},
            )
            assert refresh.status_code == 401, refresh.text
            assert _JSON_OBJECT.validate_json(refresh.content) == {
                "error": "invalid_grant",
                "error_description": "This gateway does not issue refresh tokens; sign in again",
            }, refresh.text

            bearer_headers: Final = {
                "authorization": f"Bearer {access_token.access_token}",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": cc.CLI_BETA,
                "x-claude-code-session-id": f"oauth-{user_id}",
            }
            request_body: Final = {**request_template, "model": model_a}
            expected_message: Final = _MESSAGE.validate_python(
                {
                    **_JSON_OBJECT.validate_json(
                        cc.message_reply(message_ids[0], _BACKEND, _REPLY_CONTENT, _REPLY_USAGE)
                    ),
                    "model": model_a,
                }
            )
            model_a_response: Final = enabled_gateway.client.post(
                "/claude_code_gateway/v1/messages", json=request_body, headers=bearer_headers
            )
            assert model_a_response.status_code == 200, model_a_response.text
            parsed_message: Final = _MESSAGE.validate_json(model_a_response.content)
            assert parsed_message.model_dump(mode="json") == expected_message.model_dump(mode="json"), (
                model_a_response.text,
            )

            model_a_stream: Final = enabled_gateway.client.post(
                "/claude_code_gateway/v1/messages?beta=true",
                json={**request_body, "stream": True},
                headers=bearer_headers,
            )
            assert model_a_stream.status_code == 200, model_a_stream.text
            expected_stream_events: Final = tuple(
                (
                    event,
                    {**data, "message": {**_JSON_OBJECT.validate_python(data["message"]), "model": model_a}}
                    if event == "message_start"
                    else data,
                )
                for event, data in cc.sse_events(
                    b"".join(cc.message_stream(message_ids[1], _BACKEND, _REPLY_CONTENT, _REPLY_USAGE)).decode()
                )
            )
            assert cc.sse_events(model_a_stream.text) == expected_stream_events, model_a_stream.text

            upstream_requests: Final = upstream.drain()
            assert len(upstream_requests) == 2, upstream_requests
            assert tuple(
                (request.method, request.target, _JSON_OBJECT.validate_json(request.body))
                for request in upstream_requests
            ) == (
                ("POST", "/v1/messages", expected_body),
                ("POST", "/v1/messages", {**expected_body, "stream": True}),
            ), upstream_requests

            model_b_response: Final = enabled_gateway.client.post(
                "/claude_code_gateway/v1/messages", json={**request_body, "model": model_b}, headers=bearer_headers
            )
            assert model_b_response.status_code == 403, model_b_response.text
            assert _JSON_OBJECT.validate_json(model_b_response.content) == {
                "error": {
                    "message": (
                        f"The requested model '{model_b}' is not available for this API key, or the model name is "
                        "invalid. Check the models available to you and try again."
                    ),
                    "type": "team_model_access_denied",
                    "param": "model",
                    "code": "403",
                }
            }, model_b_response.text
            assert upstream.drain() == ()

            model_c_response: Final = enabled_gateway.client.post(
                "/claude_code_gateway/v1/messages", json={**request_body, "model": model_c}, headers=bearer_headers
            )
            assert model_c_response.status_code == 200, model_c_response.text
            assert _MESSAGE.validate_json(model_c_response.content).model_dump(mode="json") == (
                expected_message.model_copy(update={"id": message_ids[2], "model": model_c}).model_dump(mode="json")
            ), model_c_response.text
            assert tuple(
                (request.method, request.target, _JSON_OBJECT.validate_json(request.body))
                for request in upstream.drain()
            ) == (("POST", "/v1/messages", expected_body),)

            spend_rows: Final = eventually(
                lambda: read_rows(
                    'SELECT request_id, "user", team_id, model_group, call_type FROM "LiteLLM_SpendLogs" '
                    "WHERE team_id=%s AND status='success' ORDER BY \"startTime\"",
                    (team_id,),
                ),
                lambda rows: len(rows) == 3,
                seconds=70,
                return_last_on_timeout=True,
            )
            assert spend_rows == [
                {
                    "request_id": message_id,
                    "user": _USER_ID,
                    "team_id": team_id,
                    "model_group": model,
                    "call_type": "anthropic_messages",
                }
                for message_id, model in zip(message_ids, (model_a, model_a, model_c), strict=True)
            ], spend_rows
            idp_requests: Final = identity_provider.drain()
            assert len(idp_requests) == 3, idp_requests
            (authorization_request, token_request, userinfo_request) = idp_requests
            authorization_query: Final = parse_qs(urlsplit(authorization_request.target).query)
            assert authorization_request.method == "GET" and urlsplit(authorization_request.target).path == "/authorize"
            assert set(authorization_query) == {
                "client_id",
                "code_challenge",
                "code_challenge_method",
                "redirect_uri",
                "response_type",
                "scope",
                "state",
            }, authorization_query
            assert authorization_query["client_id"] == [_CLIENT_ID], authorization_query
            assert authorization_query["response_type"] == ["code"], authorization_query
            assert authorization_query["scope"] == ["openid email profile"], authorization_query
            assert authorization_query["code_challenge_method"] == ["S256"], authorization_query
            token_form: Final = parse_qs(token_request.body.decode())
            verifier: Final = token_form["code_verifier"][0]
            expected_challenge: Final = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            assert authorization_query["code_challenge"] == [expected_challenge.rstrip(b"=").decode()], token_request
            assert token_request.method == "POST" and urlsplit(token_request.target).path == "/token", token_request
            assert token_request.headers["content-type"] == "application/x-www-form-urlencoded", token_request.headers
            assert set(token_form) == {"code", "code_verifier", "grant_type", "redirect_uri"}, token_form
            assert userinfo_request.method == "GET" and urlsplit(userinfo_request.target).path == "/userinfo"
            assert user_id == _USER_ID
