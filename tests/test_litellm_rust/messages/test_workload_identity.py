import re
from collections.abc import Iterator, Mapping
from typing import Final

import pytest

import litellm
from litellm.llms.base_llm.auth.token_exchange import default_token_exchange_engine
from tests.test_litellm_rust.messages.data import MESSAGES_EVENTS, MESSAGES_RESPONSE, call
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec, relaxed
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

IDENTITY_TOKEN: Final = "eyJ.native-identity.jwt"
MINTED_TOKEN: Final = "sk-ant-oat01-native-minted"
FEDERATION_IDS: Final = {"anthropic_federation_rule_id": "fdrl_native", "anthropic_organization_id": "org-native"}
MINTED: Final = ResponseSpec(body={"access_token": MINTED_TOKEN, "token_type": "Bearer", "expires_in": 3600})
EXCHANGE: Final = {
    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
    "federation_rule_id": "fdrl_native",
    "organization_id": "org-native",
    "assertion": IDENTITY_TOKEN,
}
FEDERATION_ENV_NAMES: Final = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_IDENTITY_SOURCE",
    "LITELLM_ANTHROPIC_WIF_ALLOWED_HOSTS",
)


@pytest.fixture
def server(server: RecordingServer) -> Iterator[RecordingServer]:
    yield from relaxed(server)


@pytest.fixture
def federation(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in FEDERATION_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", IDENTITY_TOKEN)
    monkeypatch.setenv("LITELLM_ANTHROPIC_WIF_ALLOWED_HOSTS", "127.0.0.1")
    with (
        rebound(default_token_exchange_engine, "_entries", {}),
        rebound(default_token_exchange_engine, "_shared_store", None),
    ):
        yield


def wire(server: RecordingServer) -> tuple[tuple[str, object, Mapping[str, str]], ...]:
    return tuple((sent.path, sent.body, sent.headers) for sent in server.requests)


def assert_minted_bearer(headers: Mapping[str, str]) -> None:
    assert headers.get("authorization") == f"Bearer {MINTED_TOKEN}"
    assert "x-api-key" not in headers
    assert "oauth-2025-04-20" in headers.get("anthropic-beta", "").split(",")


@pytest.mark.asyncio
@pytest.mark.usefixtures("federation")
@pytest.mark.parametrize(
    ("kwargs", "env", "exchange"),
    (
        pytest.param(FEDERATION_IDS, {}, EXCHANGE, id="ids-from-params"),
        pytest.param(
            {},
            {"ANTHROPIC_FEDERATION_RULE_ID": "fdrl_env", "ANTHROPIC_ORGANIZATION_ID": "org-env"},
            {**EXCHANGE, "federation_rule_id": "fdrl_env", "organization_id": "org-env"},
            id="ids-from-environment",
        ),
        pytest.param(
            {
                **FEDERATION_IDS,
                "anthropic_federation_workspace_id": "wrkspc_native",
                "anthropic_service_account_id": "svac_native",
            },
            {},
            {**EXCHANGE, "workspace_id": "wrkspc_native", "service_account_id": "svac_native"},
            id="workspace-and-service-account",
        ),
    ),
)
async def test_workload_identity_mints_a_bearer_for_a_call_without_a_key(
    server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: Mapping[str, object],
    env: Mapping[str, str],
    exchange: Mapping[str, str],
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    server.enqueue(MINTED)

    response: Final = await call(server, api_key=None, **kwargs)

    assert_served_natively(server, count=2)
    (token_path, token_body, _), (messages_path, _, headers) = wire(server)
    assert (token_path, token_body) == ("/v1/oauth/token", exchange)
    assert messages_path == "/v1/messages"
    assert_minted_bearer(headers)
    assert isinstance(response, dict)
    assert response["id"] == MESSAGES_RESPONSE["id"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("federation")
async def test_workload_identity_keeps_the_caller_betas_and_headers(server: RecordingServer) -> None:
    server.enqueue(MINTED)

    await call(
        server,
        api_key=None,
        extra_headers={"anthropic-beta": "files-api-2025-04-14", "x-trace-id": "trace-1"},
        **FEDERATION_IDS,
    )

    assert_served_natively(server, count=2)
    _, (_, _, headers) = wire(server)
    assert_minted_bearer(headers)
    assert "files-api-2025-04-14" in headers["anthropic-beta"].split(",")
    assert headers["x-trace-id"] == "trace-1"


@pytest.mark.asyncio
@pytest.mark.usefixtures("federation")
async def test_workload_identity_streams_with_the_minted_bearer(server: RecordingServer) -> None:
    server.enqueue(MINTED)
    server.enqueue(ResponseSpec(body={}, events=MESSAGES_EVENTS))

    stream: Final = await call(server, api_key=None, stream=True, **FEDERATION_IDS)
    chunks: Final = [chunk async for chunk in stream]  # pyright: ignore[reportGeneralTypeIssues]  # the public call returns an async stream

    assert_served_natively(server, count=2)
    _, (_, _, headers) = wire(server)
    assert_minted_bearer(headers)
    assert b"message_stop" in b"".join(chunk if isinstance(chunk, bytes) else str(chunk).encode() for chunk in chunks)


@pytest.mark.asyncio
@pytest.mark.usefixtures("federation")
async def test_a_static_key_shadows_workload_identity(server: RecordingServer) -> None:
    await call(server, api_key="sk-ant-static", **FEDERATION_IDS)

    assert_served_natively(server)
    ((path, _, headers),) = wire(server)
    assert path == "/v1/messages"
    assert headers.get("x-api-key") == "sk-ant-static"
    assert "authorization" not in headers


@pytest.mark.asyncio
@pytest.mark.usefixtures("federation")
@pytest.mark.parametrize(
    ("kwargs", "env", "upstream", "message", "sent"),
    (
        pytest.param(
            FEDERATION_IDS,
            {},
            (ResponseSpec(body={"error": "authentication_error"}, status=401),),
            "Anthropic workload identity federation failed. The token endpoint returned HTTP 401",
            ("/v1/oauth/token",),
            id="exchange-denied",
        ),
        pytest.param(
            FEDERATION_IDS,
            {"LITELLM_ANTHROPIC_WIF_ALLOWED_HOSTS": None},
            (),
            "Anthropic workload identity federation refused to use host",
            (),
            id="untrusted-exchange-host",
        ),
        pytest.param(
            {"anthropic_identity_token": "oidc/env/ANTHROPIC_IDENTITY_TOKEN"},
            {},
            (),
            "anthropic_identity_token is set, but anthropic_federation_rule_id and anthropic_organization_id",
            (),
            id="identity-token-without-ids",
        ),
        pytest.param(
            {**FEDERATION_IDS, "anthropic_disable_workload_identity_federation": True},
            {},
            (),
            "Missing Anthropic API Key",
            (),
            id="federation-disabled",
        ),
    ),
)
async def test_workload_identity_failures_raise_before_the_messages_call(
    server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: Mapping[str, object],
    env: Mapping[str, str | None],
    upstream: tuple[ResponseSpec, ...],
    message: str,
    sent: tuple[str, ...],
) -> None:
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, value)
    for spec in upstream:
        server.enqueue(spec)

    with pytest.raises(litellm.AuthenticationError, match=re.escape(message)) as raised:
        await call(server, api_key=None, **kwargs)

    assert raised.value.llm_provider == "anthropic"
    assert_served_natively(server, count=len(sent))
    assert tuple(path for path, _, _ in wire(server)) == sent
