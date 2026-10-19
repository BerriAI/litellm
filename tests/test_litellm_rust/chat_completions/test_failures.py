import json
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm import RateLimitError
from tests.test_litellm_rust.chat_completions.data import RATE_LIMITED, call
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests._support.provider import SharedProvider
from tests._support.wire import Reply

pytestmark = pytest.mark.requires_rust_extension

_OBJECT: Final = TypeAdapter(dict[str, object])


def failure_signal(asynchronous: bool) -> str:
    return "async_log_failure_event" if asynchronous else "log_failure_event"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_sdk_policy_rejection_precedes_resource_setup_and_is_logged_once(
    server: SharedProvider, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    monkeypatch.setattr(litellm, "max_budget", 1.0)
    monkeypatch.setattr(litellm, "_current_cost", 2.0)
    monkeypatch.setattr(litellm, "ssl_verify", object())
    recorder: Final = RecordingLogger()

    with pytest.raises(litellm.BudgetExceededError) as caught:
        await call(server, asynchronous=asynchronous, callbacks=[recorder])

    failure: Final = await recorder.wait_for_async(failure_signal(asynchronous))
    assert len(failure) == 1
    assert _OBJECT.validate_python(failure[0].kwargs)["exception"] is caught.value
    requests: Final = server.received()
    assert requests == ()
    assert not any("success" in name for name in recorder.names)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_provider_failure_is_terminal_and_shared_with_callbacks(
    server: SharedProvider, asynchronous: bool
) -> None:
    recorder: Final = RecordingLogger()
    server.expect(Reply(body=json.dumps(RATE_LIMITED).encode(), status=429))

    with pytest.raises(RateLimitError) as caught:
        await call(server, asynchronous=asynchronous, callbacks=[recorder])

    assert caught.value.status_code == 429
    requests: Final = server.received()
    assert len(requests) == 1
    failure: Final = await recorder.wait_for_async(failure_signal(asynchronous))
    assert len(failure) == 1
    assert _OBJECT.validate_python(failure[0].kwargs)["exception"] is caught.value
    assert not any("success" in name for name in recorder.names)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_empty_messages_fail_before_provider_io(server: SharedProvider, asynchronous: bool) -> None:
    with pytest.raises(Exception, match="chat completions requires at least one message"):
        await call(server, asynchronous=asynchronous, messages=[])

    requests: Final = server.received()
    assert requests == ()
