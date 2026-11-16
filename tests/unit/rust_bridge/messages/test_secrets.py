from __future__ import annotations

from collections.abc import Awaitable, Mapping
from dataclasses import replace
from typing import Final, Protocol, cast  # noqa: TID251  # narrows the parametrized path to its protocol

import httpx
import pytest

import litellm
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.llms.anthropic.messages.native.handler import anthropic_messages
from litellm.rust_bridge.host import secret_manager, settings
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem
from tests.test_litellm_rust.messages.data import MESSAGES, MESSAGES_MODEL, MESSAGES_RESPONSE
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service

pytestmark = pytest.mark.usefixtures("local_model_cost_map")

BEDROCK_MODEL: Final = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
SECRET_NAMES: Final = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_BASE",
    "ANTHROPIC_BASE_URL",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_BEDROCK_RUNTIME_ENDPOINT",
    "AWS_REGION_NAME",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
)
CREDENTIALS: Final = {
    MESSAGES_MODEL: ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "x-api-key", "{}"),
    BEDROCK_MODEL: ("AWS_BEARER_TOKEN_BEDROCK", "AWS_BEDROCK_RUNTIME_ENDPOINT", "authorization", "Bearer {}"),
}


class Messages(Protocol):
    def __call__(self) -> Awaitable[object]: ...


class _ManagedSecrets(CustomSecretManager):
    def __init__(self, values: Mapping[str, str]) -> None:
        super().__init__(secret_manager_name="rust_bridge_messages_test")
        self.values: Final = values

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        raise AssertionError("get_secret reads custom managers synchronously")

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        return self.values.get(secret_name)


def _native_request(model: str = MESSAGES_MODEL) -> NativeCall:
    supplied: Final = _public_kwargs(model)
    return NativeCall(
        args=(),
        kwargs=supplied,
        base={
            "model": model,
            "messages": MESSAGES,
            "max_tokens": 8,
            "stream": None,
            "api_key": None,
            "api_base": None,
            "custom_llm_provider": None,
            **supplied,
        },
    )


def _public_kwargs(model: str = MESSAGES_MODEL) -> dict[str, object]:
    return {"model": model, "messages": [dict(message) for message in MESSAGES], "max_tokens": 8}


async def _python_messages() -> object:
    return await anthropic_messages(**_public_kwargs())


async def _rust_amessages(model: str = MESSAGES_MODEL) -> object:
    route: Final = NATIVE_AMESSAGES.load()
    assert route is not None
    return await route(_native_request(model))


@pytest.fixture(params=(_python_messages, _rust_amessages), ids=("python-async", "rust-async"))
def messages(request: pytest.FixtureRequest) -> Messages:
    return cast(Messages, request.param)


async def test_secret_manager_supplies_the_anthropic_key_and_base(
    monkeypatch: pytest.MonkeyPatch, messages: Messages
) -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    with recording_service() as server:
        server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
        monkeypatch.setattr(
            litellm,
            "secret_manager_client",
            _ManagedSecrets({"ANTHROPIC_API_KEY": "vault-key", "ANTHROPIC_BASE_URL": server.base_url}),
        )
        monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.CUSTOM)
        monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="read_only"))
        configured: Final = settings.secret_manager
        monkeypatch.setattr(settings, "secret_manager", lambda: replace(configured(), native=True))

        await messages()

    assert len(server.requests) == 1
    assert server.requests[0].headers["x-api-key"] == "vault-key"


def _recorded_secret_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    reads: Final[list[str]] = []
    read_secret_str: Final = secret_manager.read_secret_str

    def recording(name: str) -> str | None:
        reads.append(name)
        return read_secret_str(name)

    monkeypatch.setattr(secret_manager, "read_secret_str", recording)
    return reads


@pytest.mark.parametrize("model", tuple(CREDENTIALS), ids=("anthropic", "bedrock"))
async def test_environment_secrets_are_read_without_calling_python_until_a_manager_is_configured(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    for name in SECRET_NAMES:
        monkeypatch.delenv(name, raising=False)
    key_name, base_name, header, shown = CREDENTIALS[model]
    monkeypatch.setenv(key_name, "env-key")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_system", None)
    monkeypatch.setattr(litellm, "_key_management_settings", None)
    reads: Final = _recorded_secret_reads(monkeypatch)
    with recording_service() as server:
        server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
        server.expected_requests = 2
        monkeypatch.setenv(base_name, server.base_url)

        await _rust_amessages(model)
        unmanaged_reads: Final = tuple(reads)

        monkeypatch.setattr(litellm, "secret_manager_client", _ManagedSecrets({key_name: "vault-key"}))
        monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.CUSTOM)
        monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="read_only"))
        configured: Final = settings.secret_manager
        monkeypatch.setattr(settings, "secret_manager", lambda: replace(configured(), native=False))

        await _rust_amessages(model)

    assert unmanaged_reads == ()
    assert key_name in reads
    assert [request.headers[header] for request in server.requests] == [
        shown.format("env-key"),
        shown.format("vault-key"),
    ]
