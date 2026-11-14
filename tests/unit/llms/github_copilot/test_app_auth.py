import asyncio
import ipaddress
import json
import ssl
import threading
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Final

import httpx
import jwt
import pytest
import respx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

import litellm
from litellm.exceptions import AuthenticationError, BadRequestError, RateLimitError, ServiceUnavailableError
from litellm.litellm_core_utils.clientside_credential_handler import get_dynamic_litellm_params
from litellm.llms.github_copilot import app_auth
from litellm.llms.github_copilot.app_auth import APP_FIELDS, AppConfig, AppTokenManager, app_config
from litellm.llms.github_copilot.per_user_auth import (
    GithubCopilotUserSession,
    is_github_copilot_per_user_request,
    require_github_copilot_user_session,
)
from litellm.types.router import holds_secret_pointer, reject_server_owned_wif_fields
from litellm.types.utils import CredentialItem

_NOW: Final = 1_800_000_000.0
_CONFIG: Final = AppConfig("test-client", 123, "os.environ/TEST_APP_KEY")


@dataclass
class Clock:
    now: float = _NOW
    elapsed: float = 0

    def wall(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.elapsed

    def advance(self, seconds: float) -> None:
        self.now += seconds
        self.elapsed += seconds


@pytest.fixture
def private_key() -> str:
    key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()


def token_payload(token: str = "installation-token", expires: float = _NOW + 3600) -> Mapping[str, object]:
    return {
        "token": token,
        "expires_at": datetime.fromtimestamp(expires, timezone.utc).isoformat(),
        "permissions": {"organization_copilot_requests": "read"},
    }


def manager(
    private_key: str,
    respond: Callable[[httpx.Request], httpx.Response],
    clock: Clock,
    secret_reader: Callable[[str], str | None] | None = None,
) -> AppTokenManager:
    return AppTokenManager(
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(respond)),
        secret_reader=secret_reader or (lambda _: private_key),
        clock=clock.wall,
        monotonic=clock.monotonic,
    )


def test_jwt_signature_and_minimum_permissions(private_key: str) -> None:
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=token_payload())

    session: Final = manager(private_key, respond, Clock()).resolve(_CONFIG)
    request: Final = requests[0]
    signing_key: Final = serialization.load_pem_private_key(private_key.encode(), password=None)
    assert isinstance(signing_key, rsa.RSAPrivateKey)
    claims: Final = jwt.decode(
        request.headers["Authorization"].removeprefix("Bearer "),
        signing_key.public_key(),
        algorithms=["RS256"],
        options={"verify_exp": False, "verify_iat": False},
    )
    assert claims == {"iss": _CONFIG.app_id, "iat": int(_NOW) - 60, "exp": int(_NOW) + 540}
    assert str(request.url) == _CONFIG.token_url
    assert json.loads(request.content) == {"permissions": {"organization_copilot_requests": "read"}}
    assert session.token == "installation-token"
    assert session.api_base == _CONFIG.api_base
    assert "installation-token" not in repr(session)


def test_refreshes_before_expiry_and_reuses_new_token(private_key: str) -> None:
    clock: Final = Clock()
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=token_payload(f"token-{len(requests)}", clock.now + 3600))

    auth: Final = manager(private_key, respond, clock)
    assert auth.resolve(_CONFIG).token == "token-1"
    clock.advance(3479)
    assert auth.resolve(_CONFIG).token == "token-1"
    clock.advance(1)
    assert auth.resolve(_CONFIG).token == "token-2"
    assert auth.resolve(_CONFIG).token == "token-2"
    assert len(requests) == 2


def test_monotonic_expiry_survives_wall_clock_rollback(private_key: str) -> None:
    clock: Final = Clock()
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=token_payload(f"token-{len(requests)}", clock.now + 3600))

    auth: Final = manager(private_key, respond, clock)
    assert auth.resolve(_CONFIG).token == "token-1"
    clock.elapsed = 3480
    clock.now -= 600
    assert auth.resolve(_CONFIG).token == "token-2"


@pytest.mark.parametrize("async_calls", [False, True])
async def test_concurrent_callers_mint_one_token(private_key: str, async_calls: bool) -> None:
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=token_payload())

    auth: Final = manager(private_key, respond, Clock())
    if async_calls:
        sessions: Final = await asyncio.gather(*(auth.aresolve(_CONFIG) for _ in range(20)))
        assert {session.token for session in sessions} == {"installation-token"}
    else:
        with ThreadPoolExecutor(max_workers=20) as pool:
            tokens: Final = tuple(pool.map(lambda _: auth.resolve(_CONFIG).token, range(20)))
        assert set(tokens) == {"installation-token"}
    assert len(requests) == 1


def test_app_installation_host_and_rotated_key_never_share_tokens(
    private_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=token_payload(f"token-{len(requests)}"))

    def upload(key: str) -> None:
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="uploaded",
                    credential_values={
                        "github_copilot_auth_type": "github_app",
                        "github_copilot_app_id": _CONFIG.app_id,
                        "github_copilot_app_installation_id": "123",
                        "github_copilot_app_private_key": key,
                    },
                    credential_info={"custom_llm_provider": "github_copilot"},
                )
            ],
        )

    auth: Final = manager(private_key, respond, Clock())
    configs: Final = (
        _CONFIG,
        AppConfig("other-app", 123, _CONFIG.private_key_ref),
        AppConfig(_CONFIG.app_id, 456, _CONFIG.private_key_ref),
        AppConfig(_CONFIG.app_id, 123, _CONFIG.private_key_ref, "example.ghe.com"),
    )
    assert tuple(auth.resolve(config).token for config in configs) == ("token-1", "token-2", "token-3", "token-4")
    upload(private_key)
    uploaded: Final = app_config({"litellm_credential_name": "uploaded"})
    assert uploaded is not None
    assert auth.resolve(uploaded).token == "token-5"
    upload(private_key + "\n")
    assert auth.resolve(uploaded).token == "token-6"
    assert auth.resolve(configs[1]).token == "token-2"


async def test_cached_token_is_served_without_reading_the_key_secret_again(private_key: str) -> None:
    reads: Final[list[str]] = []
    mints: Final[list[httpx.Request]] = []

    def read_secret(ref: str) -> str:
        reads.append(ref)
        if len(reads) > 1:
            raise RuntimeError("secret backend is down")
        return private_key

    def respond(request: httpx.Request) -> httpx.Response:
        mints.append(request)
        return httpx.Response(201, json=token_payload())

    auth: Final = manager(private_key, respond, Clock(), read_secret)
    assert auth.resolve(_CONFIG).token == "installation-token"
    assert {(await auth.aresolve(_CONFIG)).token for _ in range(3)} == {"installation-token"}
    assert auth.resolve(_CONFIG).token == "installation-token"
    assert (len(reads), len(mints)) == (1, 1)


async def test_callers_waiting_on_a_mint_leave_the_default_thread_pool_free(private_key: str) -> None:
    started: Final = threading.Event()
    release: Final = threading.Event()
    mints: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        mints.append(request)
        started.set()
        _ = release.wait(10)
        return httpx.Response(201, json=token_payload())

    auth: Final = manager(private_key, respond, Clock())
    loop: Final = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=2))
    try:
        callers: Final = tuple(asyncio.create_task(auth.aresolve(_CONFIG)) for _ in range(10))
        while not started.is_set():
            await asyncio.sleep(0.01)
        assert await asyncio.wait_for(loop.run_in_executor(None, lambda: "unrelated job ran"), 2) == (
            "unrelated job ran"
        )
    finally:
        release.set()
    assert {session.token for session in await asyncio.gather(*callers)} == {"installation-token"}
    assert len(mints) == 1


async def test_cancelled_caller_still_finishes_the_refresh_for_later_callers(private_key: str) -> None:
    pool_busy: Final = threading.Event()
    mints: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        mints.append(request)
        return httpx.Response(201, json=token_payload())

    auth: Final = manager(private_key, respond, Clock())
    loop: Final = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    blocker: Final = loop.run_in_executor(None, pool_busy.wait, 10)
    first_caller: Final = asyncio.create_task(auth.aresolve(_CONFIG))
    await asyncio.sleep(0.05)
    _ = first_caller.cancel()
    pool_busy.set()
    _ = await blocker
    with pytest.raises(asyncio.CancelledError):
        await first_caller
    assert (await asyncio.wait_for(auth.aresolve(_CONFIG), 2)).token == "installation-token"
    assert len(mints) == 1


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, AuthenticationError),
        (403, AuthenticationError),
        (404, AuthenticationError),
        (422, BadRequestError),
        (429, RateLimitError),
        (500, ServiceUnavailableError),
    ],
)
def test_failed_refresh_never_returns_old_token_or_leaks_response(
    private_key: str, status: int, error_type: type[Exception]
) -> None:
    clock: Final = Clock()
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(201, json=token_payload())
        return httpx.Response(status, json={"message": private_key + "installation-token"})

    auth: Final = manager(private_key, respond, clock)
    assert auth.resolve(_CONFIG).token == "installation-token"
    clock.advance(3480)
    for _ in range(2):
        with pytest.raises(error_type) as error:
            auth.resolve(_CONFIG)
        assert "installation-token" not in str(error.value)
        assert "PRIVATE KEY" not in str(error.value)
        assert error.value.__cause__ is None
    assert len(requests) == 2


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {**token_payload(), "token": ""},
        {**token_payload(), "token": "secret\r\ninjected-header"},
        {**token_payload(), "expires_at": "not-a-date"},
        {**token_payload(), "expires_at": "2027-01-15T08:00:00"},
        token_payload(expires=_NOW),
        {**token_payload(), "permissions": {"contents": "write"}},
    ],
)
def test_rejects_invalid_tokens_without_logging_payload(private_key: str, payload: Mapping[str, object]) -> None:
    auth: Final = manager(private_key, lambda _: httpx.Response(201, json=payload), Clock())
    with pytest.raises((AuthenticationError, ServiceUnavailableError)) as error:
        auth.resolve(_CONFIG)
    assert "installation-token" not in str(error.value)


def test_redirect_does_not_forward_app_jwt(private_key: str) -> None:
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(307, headers={"Location": "https://attacker.example/token"})

    auth: Final = manager(private_key, respond, Clock())
    with pytest.raises(ServiceUnavailableError, match="HTTP 307"):
        auth.resolve(_CONFIG)
    assert tuple(str(request.url) for request in requests) == (_CONFIG.token_url,)


@pytest.mark.parametrize("secret", [None, "", "invalid-private-key"])
def test_missing_or_invalid_key_never_contacts_github(secret: str | None) -> None:
    def unexpected_request(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"Unexpected request to {request.url}")

    auth: Final = manager("", unexpected_request, Clock(), lambda _: secret)
    with pytest.raises(AuthenticationError):
        auth.resolve(_CONFIG)


def test_named_credential_owns_auth_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    values: Final = {
        "github_copilot_auth_type": "github_app",
        "github_copilot_app_id": "stored-app",
        "github_copilot_app_installation_id": "123",
        "github_copilot_app_private_key_ref": "os.environ/STORED_KEY",
    }
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [CredentialItem(credential_name="app", credential_values=values, credential_info={})],
    )
    assert app_config({"litellm_credential_name": "app", **values, "github_copilot_app_id": "attacker"}) == AppConfig(
        "stored-app", 123, "os.environ/STORED_KEY"
    )


@pytest.mark.parametrize("field", APP_FIELDS)
def test_app_configuration_is_server_owned(field: str) -> None:
    with pytest.raises(ValueError, match="server-owned GitHub Copilot authentication parameter"):
        reject_server_owned_wif_fields({field: "attacker"})
    assert holds_secret_pointer("github_copilot_app_private_key_ref")


@pytest.mark.parametrize("override", [{"api_base": "https://attacker.example"}, {"api_key": "attacker"}])
def test_clientside_override_cannot_disable_app_auth(override: Mapping[str, str]) -> None:
    with pytest.raises(BadRequestError, match="do not accept"):
        get_dynamic_litellm_params(
            {
                "github_copilot_auth_type": "github_app",
                "github_copilot_app_id": "app",
                "github_copilot_app_installation_id": 123,
                "github_copilot_app_private_key_ref": "os.environ/KEY",
            },
            dict(override),
        )


@pytest.mark.parametrize(
    "surface,async_call",
    [
        ("chat", False),
        ("chat", True),
        ("responses", False),
        ("responses", True),
        ("messages", True),
        ("embeddings", False),
        ("embeddings", True),
    ],
)
@pytest.mark.parametrize("uploaded", [False, True])
async def test_public_calls_pin_app_identity_and_keep_credentials_out_of_body(
    private_key: str,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    surface: str,
    async_call: bool,
    uploaded: bool,
) -> None:
    clock: Final = Clock()
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(app_auth, "APP_TOKEN_MANAGER", AppTokenManager(clock=clock.wall, monotonic=clock.monotonic))
    monkeypatch.setenv("TEST_APP_KEY", private_key)
    token_route: Final = respx_mock.post(_CONFIG.token_url).respond(201, json=token_payload())
    upstream: Final = respx_mock.post(host="api.githubcopilot.com").respond(200, json=_model_payload(surface))
    auth_fields: Final = {
        "github_copilot_auth_type": "github_app",
        "github_copilot_app_id": _CONFIG.app_id,
        "github_copilot_app_installation_id": str(_CONFIG.installation_id),
        **(
            {"github_copilot_app_private_key": private_key}
            if uploaded
            else {"github_copilot_app_private_key_ref": _CONFIG.private_key_ref}
        ),
    }
    if uploaded:
        monkeypatch.setattr(
            litellm,
            "credential_list",
            [
                CredentialItem(
                    credential_name="uploaded",
                    credential_values=auth_fields,
                    credential_info={"custom_llm_provider": "github_copilot"},
                )
            ],
        )
    kwargs: Final = {
        **({"litellm_credential_name": "uploaded"} if uploaded else auth_fields),
        "api_base": "https://attacker.example",
        "api_key": "attacker-key",
        "extra_headers": {"aUtHoRiZaTiOn": "Bearer attacker", "Copilot-Integration-ID": "attacker", "x-keep": "ok"},
    }
    if surface == "chat":
        chat_args: Final = {
            "model": "github_copilot/claude-sonnet-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            **kwargs,
        }
        chat: Final = await litellm.acompletion(**chat_args) if async_call else litellm.completion(**chat_args)
        assert chat.choices[0].message.content == "OK"
    elif surface == "responses":
        response_args: Final = {"model": "github_copilot/gpt-6.1-sol", "input": "hi", **kwargs}
        response: Final = (
            await litellm.aresponses(**response_args) if async_call else litellm.responses(**response_args)
        )
        assert response.status == "completed"
    elif surface == "messages":
        message_args: Final = {
            "model": "github_copilot/claude-sonnet-5.5",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            **kwargs,
        }
        message: Final = (
            await litellm.anthropic.messages.acreate(**message_args)
            if async_call
            else litellm.anthropic.messages.create(**message_args)
        )
        assert message["content"] == [{"type": "text", "text": "OK"}]
    else:
        embedding_args: Final = {"model": "github_copilot/text-embedding-3-small", "input": ["hi"], **kwargs}
        embedding: Final = (
            await litellm.aembedding(**embedding_args) if async_call else litellm.embedding(**embedding_args)
        )
        assert embedding.data[0]["embedding"] == [0.1, 0.2]
    assert token_route.call_count == 1
    assert upstream.call_count == 1
    request: Final = upstream.calls.last.request
    assert request.headers["authorization"] == "Bearer installation-token"
    assert request.headers["copilot-integration-id"] == "copilot-developer-cli"
    if surface != "embeddings":
        assert request.headers["x-keep"] == "ok"
    assert not any(field in json.loads(request.content) for field in APP_FIELDS)
    assert private_key.encode() not in request.content
    assert kwargs["extra_headers"]["aUtHoRiZaTiOn"] == "Bearer attacker"


def _model_payload(surface: str) -> Mapping[str, object]:
    if surface == "chat":
        return {
            "id": "chatcmpl-app",
            "object": "chat.completion",
            "created": 1,
            "model": "claude-sonnet-5.5",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }
    if surface == "responses":
        return {
            "id": "response-app",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-6.1-sol",
            "output": [],
            "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
        }
    if surface == "messages":
        return {
            "id": "msg-app",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5.5",
            "content": [{"type": "text", "text": "OK"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 4, "output_tokens": 2},
        }
    return {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }


def test_app_session_is_deployment_owned_and_shares_the_response_cache() -> None:
    from litellm.caching.caching_handler import _is_response_cache_excluded
    from litellm.router_utils.cooldown_handlers import _is_per_user_provider_request

    params: Final = {
        "github_copilot_user_session": GithubCopilotUserSession("app-token", _CONFIG.api_base, is_app=True)
    }
    per_user: Final = {"github_copilot_user_session": GithubCopilotUserSession("user-token", _CONFIG.api_base)}
    assert not is_github_copilot_per_user_request(params)
    assert not _is_per_user_provider_request({"litellm_params": params})
    assert not _is_response_cache_excluded("github_copilot/claude-sonnet-5.5", params)
    assert _is_response_cache_excluded("github_copilot/claude-sonnet-5.5", per_user)


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("app", [False, True])
async def test_caller_headers_keep_device_login_wire_headers_and_cannot_unpin_app_identity(
    private_key: str,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    tmp_path: Path,
    app: bool,
    async_call: bool,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(app_auth, "APP_TOKEN_MANAGER", AppTokenManager())
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("TEST_APP_KEY", private_key)
    _ = (tmp_path / "access-token").write_text("gho_device_token")
    _ = respx_mock.post(_CONFIG.token_url).respond(201, json=token_payload())
    upstream: Final = respx_mock.post(host="api.githubcopilot.com").respond(200, json=_model_payload("chat"))
    app_fields: Final = {
        "github_copilot_auth_type": "github_app",
        "github_copilot_app_id": _CONFIG.app_id,
        "github_copilot_app_installation_id": str(_CONFIG.installation_id),
        "github_copilot_app_private_key_ref": _CONFIG.private_key_ref,
    }
    args: Final = {
        "model": "github_copilot/claude-sonnet-5.5",
        "messages": [{"role": "user", "content": "hi"}],
        "headers": {"x-forwarded": "client", "Copilot-Integration-ID": "client-integration"},
        **(app_fields if app else {}),
    }
    _ = await litellm.acompletion(**args) if async_call else litellm.completion(**args)
    sent: Final = upstream.calls.last.request.headers
    assert sent["x-forwarded"] == "client"
    if app:
        assert sent["authorization"] == "Bearer installation-token"
        assert sent.get_list("copilot-integration-id") == ["copilot-developer-cli"]
        assert "editor-version" in sent
    else:
        assert sent.get_list("copilot-integration-id") == ["client-integration"]
        assert "editor-version" not in sent


@pytest.mark.parametrize("async_call", [False, True])
async def test_app_deployment_whose_credential_is_not_loaded_fails_closed_before_any_github_call(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, async_call: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "credential_list", [])
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "copilot",
                "litellm_params": {
                    "model": "github_copilot/claude-sonnet-5.5",
                    "litellm_credential_name": "copilot-service-a",
                    "github_copilot_auth_type": "github_app",
                },
            }
        ],
        num_retries=0,
    )
    messages: Final = [{"role": "user", "content": "hi"}]

    async def call() -> object:
        if async_call:
            return await router.acompletion(model="copilot", messages=messages)
        return router.completion(model="copilot", messages=messages)

    with pytest.raises(BadRequestError, match="GitHub App authentication requires"):
        await call()
    assert respx_mock.calls.call_count == 0


async def test_unloaded_credential_name_keeps_inline_app_auth_and_other_providers_working(
    private_key: str, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    clock: Final = Clock()
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "credential_list", [])
    monkeypatch.setattr(app_auth, "APP_TOKEN_MANAGER", AppTokenManager(clock=clock.wall, monotonic=clock.monotonic))
    monkeypatch.setenv("TEST_APP_KEY", private_key)
    respx_mock.post(_CONFIG.token_url).respond(201, json=token_payload())
    copilot: Final = respx_mock.post(host="api.githubcopilot.com").respond(200, json=_model_payload("chat"))
    openai: Final = respx_mock.post(host="api.openai.com").respond(200, json=_model_payload("chat"))
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "copilot",
                "litellm_params": {
                    "model": "github_copilot/claude-sonnet-5.5",
                    "litellm_credential_name": "not-loaded",
                    "github_copilot_auth_type": "github_app",
                    "github_copilot_app_id": _CONFIG.app_id,
                    "github_copilot_app_installation_id": _CONFIG.installation_id,
                    "github_copilot_app_private_key_ref": _CONFIG.private_key_ref,
                },
            },
            {
                "model_name": "openai",
                "litellm_params": {
                    "model": "openai/gpt-5.5",
                    "litellm_credential_name": "not-loaded",
                    "api_key": "sk-test",
                },
            },
        ],
        num_retries=0,
    )
    messages: Final = [{"role": "user", "content": "hi"}]

    await router.acompletion(model="copilot", messages=messages)
    await router.acompletion(model="openai", messages=messages)

    assert copilot.calls.last.request.headers["authorization"] == "Bearer installation-token"
    assert openai.calls.last.request.headers["authorization"] == "Bearer sk-test"


async def test_numeric_app_id_from_yaml_boots_and_signs_as_that_app(
    private_key: str, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    clock: Final = Clock()
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(app_auth, "APP_TOKEN_MANAGER", AppTokenManager(clock=clock.wall, monotonic=clock.monotonic))
    monkeypatch.setenv("TEST_APP_KEY", private_key)
    mint: Final = respx_mock.post(_CONFIG.token_url).respond(201, json=token_payload())
    upstream: Final = respx_mock.post(host="api.githubcopilot.com").respond(200, json=_model_payload("chat"))
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "copilot",
                "litellm_params": {
                    "model": "github_copilot/claude-sonnet-5.5",
                    "github_copilot_auth_type": "github_app",
                    "github_copilot_app_id": 424242,
                    "github_copilot_app_installation_id": _CONFIG.installation_id,
                    "github_copilot_app_private_key_ref": _CONFIG.private_key_ref,
                },
            }
        ],
        num_retries=0,
    )

    await router.acompletion(model="copilot", messages=[{"role": "user", "content": "hi"}])

    app_jwt: Final = mint.calls.last.request.headers["authorization"].removeprefix("Bearer ")
    assert jwt.decode(app_jwt, options={"verify_signature": False})["iss"] == "424242"
    assert upstream.calls.last.request.headers["authorization"] == "Bearer installation-token"


def test_app_mode_without_session_cannot_fall_back_to_user_login() -> None:
    with pytest.raises(BadRequestError, match="resolved installation session"):
        require_github_copilot_user_session(
            {
                "github_copilot_auth_type": "github_app",
                "github_copilot_app_id": _CONFIG.app_id,
                "github_copilot_app_installation_id": _CONFIG.installation_id,
                "github_copilot_app_private_key_ref": _CONFIG.private_key_ref,
            }
        )


async def test_router_isolates_two_apps_and_rejects_credential_switching(
    private_key: str, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    clock: Final = Clock()
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(app_auth, "APP_TOKEN_MANAGER", AppTokenManager(clock=clock.wall, monotonic=clock.monotonic))
    monkeypatch.setenv("TEST_APP_KEY", private_key)
    credentials: Final = [
        CredentialItem(
            credential_name=name,
            credential_values={
                "github_copilot_auth_type": "github_app",
                "github_copilot_app_id": name,
                "github_copilot_app_installation_id": installation,
                "github_copilot_app_private_key_ref": "os.environ/TEST_APP_KEY",
            },
            credential_info={},
        )
        for name, installation in (("service-a", 123), ("service-b", 456))
    ]
    monkeypatch.setattr(litellm, "credential_list", credentials)
    app_a: Final = respx_mock.post(_CONFIG.token_url).respond(201, json=token_payload("token-a"))
    app_b: Final = respx_mock.post("https://api.github.com/app/installations/456/access_tokens").respond(
        201, json=token_payload("token-b")
    )
    upstream: Final = respx_mock.post(host="api.githubcopilot.com").respond(200, json=_model_payload("chat"))
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": "github_copilot/claude-sonnet-5.5",
                    "litellm_credential_name": name,
                },
                "model_info": {"id": name},
            }
            for name in ("service-a", "service-b")
        ],
        num_retries=0,
    )
    for name, token in (("service-a", "token-a"), ("service-b", "token-b"), ("service-a", "token-a")):
        response: Final = await router.acompletion(model=name, messages=[{"role": "user", "content": "hi"}])
        assert response.choices[0].message.content == "OK"
        assert response._hidden_params["model_id"] == name
        assert upstream.calls.last.request.headers["authorization"] == f"Bearer {token}"
    assert app_a.call_count == app_b.call_count == 1
    for override in ("service-b", None):
        with pytest.raises(BadRequestError, match="litellm_credential_name cannot be overridden"):
            await router.acompletion(
                model="service-a",
                messages=[{"role": "user", "content": "hi"}],
                litellm_credential_name=override,
            )
    assert upstream.call_count == 3


@pytest.mark.parametrize(
    "status,headers,cooldown",
    [
        (429, {"retry-after": "90"}, 90),
        (403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(_NOW + 120))}, 120),
        (429, {"retry-after": "invalid"}, 60),
        (429, {"retry-after": "9" * 400}, 3600),
    ],
)
def test_mint_rate_limits_are_cached_and_expose_retry_after(
    private_key: str, status: int, headers: Mapping[str, str], cooldown: int
) -> None:
    clock: Final = Clock()
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(status, headers=headers)
        return httpx.Response(201, json=token_payload(expires=clock.now + 3600))

    auth: Final = manager(private_key, respond, clock)
    with pytest.raises(RateLimitError) as error:
        auth.resolve(_CONFIG)
    assert error.value.headers == {"retry-after": str(cooldown)}
    clock.advance(cooldown - 1)
    with pytest.raises(RateLimitError) as cached:
        auth.resolve(_CONFIG)
    assert cached.value.headers == {"retry-after": "1"}
    assert len(requests) == 1
    clock.advance(1)
    assert auth.resolve(_CONFIG).token == "installation-token"
    assert len(requests) == 2


def test_secret_backend_failure_is_sanitized() -> None:
    def failing_secret_reader(ref: str) -> str:
        raise ValueError("sensitive-backend-details")

    auth: Final = AppTokenManager(secret_reader=failing_secret_reader)
    with pytest.raises(AuthenticationError) as error:
        auth.resolve(_CONFIG)
    assert "sensitive-backend-details" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ or error.value.__context__ is None


def test_token_cache_is_bounded(private_key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_auth, "_CACHE_SIZE", 2)
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=token_payload(f"token-{len(requests)}"))

    auth: Final = manager(private_key, respond, Clock())
    for installation in (123, 456, 789):
        auth.resolve(AppConfig(_CONFIG.app_id, installation, _CONFIG.private_key_ref))
    assert auth.resolve(_CONFIG).token == "token-4"
    assert len(requests) == 4


@pytest.mark.parametrize(
    "override",
    [
        {"github_copilot_auth_type": "per_user"},
        {"github_copilot_app_id": ""},
        {"github_copilot_app_installation_id": "../123"},
        {"github_copilot_app_installation_id": 0},
        {"github_copilot_app_installation_id": True},
        {"github_copilot_app_private_key_ref": "-----BEGIN PRIVATE KEY-----"},
        {"github_copilot_app_private_key_ref": "oidc/https://attacker.example"},
        {"github_copilot_host": "attacker.example"},
    ],
)
def test_invalid_app_configuration_fails_closed(override: Mapping[str, object]) -> None:
    with pytest.raises(BadRequestError, match="GitHub App authentication requires"):
        app_config(
            {
                "github_copilot_auth_type": "github_app",
                "github_copilot_app_id": _CONFIG.app_id,
                "github_copilot_app_installation_id": _CONFIG.installation_id,
                "github_copilot_app_private_key_ref": _CONFIG.private_key_ref,
                **override,
            }
        )


def test_network_failure_does_not_expose_jwt(private_key: str) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(request.headers["authorization"], request=request)

    auth: Final = manager(private_key, fail, Clock())
    with pytest.raises(ServiceUnavailableError) as error:
        auth.resolve(_CONFIG)
    assert "Bearer" not in str(error.value)
    assert error.value.__context__ is None


class _CreatedHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.send_response(201)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def private_ca_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now: Final = datetime.now(timezone.utc)
    cert: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    ca_file: Final = tmp_path / "ca.pem"
    ca_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file: Final = tmp_path / "key.pem"
    key_file.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(ca_file, key_file)
    server: Final = HTTPServer(("127.0.0.1", 0), _CreatedHandler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"https://127.0.0.1:{server.server_port}/app/installations/123/access_tokens", ca_file
    server.shutdown()
    server.server_close()


def test_installation_token_requests_trust_the_configured_ca_bundle(
    private_ca_server: tuple[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, ca_file = private_ca_server
    monkeypatch.setenv("SSL_VERIFY", str(ca_file))
    with app_auth._new_client() as client:
        assert client.post(url).status_code == 201


@pytest.mark.parametrize("is_app", [False, True])
def test_session_serialization_redacts_bearer_token(is_app: bool) -> None:
    from litellm.types.router import GenericLiteLLMParams

    session: Final = GithubCopilotUserSession("sensitive-bearer", _CONFIG.api_base, is_app=is_app)
    params: Final = GenericLiteLLMParams(github_copilot_user_session=session)
    assert session.token == "sensitive-bearer"
    assert "sensitive-bearer" not in repr(session)
    assert "sensitive-bearer" not in str(params.model_dump())
    assert "sensitive-bearer" not in params.model_dump_json()


def test_uploaded_key_is_kept_out_of_model_kwargs_and_used_for_signing(
    private_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
    from litellm.utils import with_named_credential

    values: Final = {
        "github_copilot_auth_type": "github_app",
        "github_copilot_app_id": "test-client",
        "github_copilot_app_installation_id": "123",
        "github_copilot_app_private_key": private_key,
    }
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="uploaded",
                credential_values=values,
                credential_info={"custom_llm_provider": "github_copilot"},
            )
        ],
    )
    assert "github_copilot_app_private_key" not in CredentialAccessor.get_credential_values("uploaded")
    assert "github_copilot_app_private_key" not in with_named_credential({"litellm_credential_name": "uploaded"})
    config: Final = app_config({"litellm_credential_name": "uploaded"})
    assert config is not None
    assert private_key not in repr(config)
    assert config.private_key_ref == ""
    with pytest.raises(BadRequestError):
        app_config(values)
    app_config(values, validate_stored_key=True)

    def respond(request: httpx.Request) -> httpx.Response:
        jwt.decode(
            request.headers["Authorization"].split()[1],
            serialization.load_pem_private_key(private_key.encode(), None).public_key(),
            algorithms=["RS256"],
            options={"verify_exp": False, "verify_iat": False},
        )
        return httpx.Response(201, json=token_payload())

    assert manager("unused", respond, Clock()).resolve(config).token == "installation-token"


@pytest.mark.parametrize("kind", ["weak-rsa", "ec", "encrypted"])
def test_uploaded_key_rejects_unsupported_key_types(kind: str) -> None:
    from cryptography.hazmat.primitives.asymmetric import ec

    key: Final = (
        ec.generate_private_key(ec.SECP256R1())
        if kind == "ec"
        else rsa.generate_private_key(public_exponent=65537, key_size=1024 if kind == "weak-rsa" else 2048)
    )
    pem: Final = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(b"test-password")
        if kind == "encrypted"
        else serialization.NoEncryption(),
    ).decode()
    with pytest.raises(BadRequestError, match="unencrypted RSA PEM") as error:
        app_auth.validate_private_key(pem)
    assert pem not in str(error.value)
