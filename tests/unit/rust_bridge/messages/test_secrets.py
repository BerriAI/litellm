from __future__ import annotations

from collections.abc import Awaitable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Final, Protocol, cast  # noqa: TID251  # narrows the parametrized path to its protocol

import httpx
import pytest

import litellm
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.llms.anthropic.experimental_pass_through.messages.handler import anthropic_messages
from litellm.rust_bridge import settings
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES, NATIVE_MESSAGES, LiteLLMMessagesRequest
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_MODEL, MESSAGES_RESPONSE

pytest.importorskip("litellm.rust_bridge._native")

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


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


def _native_request() -> LiteLLMMessagesRequest:
    return LiteLLMMessagesRequest(
        model=MESSAGES_MODEL,
        messages=MESSAGES,
        max_tokens=8,
        stream=None,
        api_key=None,
        api_base=None,
        custom_llm_provider=None,
        kwargs=MappingProxyType({}),
    )


def _public_kwargs() -> dict[str, object]:
    return {"model": MESSAGES_MODEL, "messages": [dict(message) for message in MESSAGES], "max_tokens": 8}


async def _python_messages() -> object:
    return await anthropic_messages(**_public_kwargs())


async def _rust_messages() -> object:
    route: Final = NATIVE_MESSAGES.load()
    assert route is not None
    return route(_native_request(), (), _public_kwargs())


async def _rust_amessages() -> object:
    route: Final = NATIVE_AMESSAGES.load()
    assert route is not None
    return await route(_native_request(), (), _public_kwargs())


@pytest.fixture(
    params=(_python_messages, _rust_messages, _rust_amessages), ids=("python-async", "rust-sync", "rust-async")
)
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
