import json
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.models.credentials import CredentialItem
from litellm.types.utils import CallTypes, ModelResponse
from tests.test_litellm_rust.chat_completions.data import call
from tests.test_litellm_rust.messages.data import MESSAGES_RESPONSE
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests._support.provider import SharedProvider
from tests._support.wire import Reply
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

_OBJECT: Final = TypeAdapter(dict[str, object])


@pytest.mark.asyncio
@pytest.mark.parametrize("from_credentials", (False, True))
async def test_resource_setup_uses_deployment_hook_arguments(
    server: SharedProvider, monkeypatch: pytest.MonkeyPatch, from_credentials: bool
) -> None:
    invalid_settings: Final = {"ssl_verify": object()}
    credential: Final = CredentialItem(
        credential_name="resource-settings", credential_info={}, credential_values=invalid_settings
    )
    monkeypatch.setattr(litellm, "credential_list", [credential])

    class Prepare(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: CallTypes | None
        ) -> dict[str, object]:
            return {
                **kwargs,
                **({"litellm_credential_name": credential.credential_name} if from_credentials else invalid_settings),
            }

    litellm.callbacks.append(Prepare())
    recorder: Final = RecordingLogger()
    with pytest.raises(ValueError, match=r"request\.ssl_verify") as caught:
        await call(server, callbacks=[recorder])

    failure: Final = await recorder.wait_for_async("async_log_failure_event")
    assert len(failure) == 1
    assert _OBJECT.validate_python(failure[0].kwargs)["exception"] is caught.value
    requests: Final = server.received()
    assert requests == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("source", ("explicit", "base_url", "global", "provider", "environment", "empty"))
async def test_connection_settings_reach_the_provider(
    server: SharedProvider, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, source: str
) -> None:
    key: Final = "selected-key"
    explicit: Final = source in ("explicit", "base_url")
    monkeypatch.setattr(litellm, "api_key", key if source in ("global", "empty") else ("unused" if explicit else None))
    monkeypatch.setattr(litellm, "anthropic_key", key if source == "provider" else ("unused" if explicit else None))
    monkeypatch.setattr(
        litellm,
        "api_base",
        None if source == "environment" else ("http://127.0.0.1:1" if explicit else server.wire.url),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", key if source == "environment" else "unused")
    for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE"):
        monkeypatch.setenv(name, server.wire.url if source == "environment" else "http://127.0.0.1:1")

    server.expect(Reply(body=json.dumps(MESSAGES_RESPONSE).encode()))
    result: Final = await call(
        server,
        asynchronous=asynchronous,
        api_key=key if explicit else ("" if source == "empty" else None),
        api_base=server.wire.url if source == "explicit" else ("" if source == "empty" else None),
        **({"base_url": server.wire.url} if source == "base_url" else {}),
    )

    assert isinstance(result, ModelResponse)
    requests: Final = assert_served_natively(server, result)
    assert requests[0].headers["x-api-key"] == key
