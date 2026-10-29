from typing import Final

import pytest

import litellm
from tests.test_litellm_rust.messages.data import MESSAGES, MESSAGES_RESPONSE, call
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

CAPABILITY_MODEL: Final = "claude-test-runtime-capabilities"


def sent(server: RecordingServer) -> tuple[dict[str, object], dict[str, str]]:
    assert_served_natively(server)
    request: Final = server.requests[0]
    assert isinstance(request.body, dict)
    return request.body, request.headers


def capabilities(monkeypatch: pytest.MonkeyPatch, *, sampling: bool) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        CAPABILITY_MODEL,
        {"litellm_provider": "anthropic", "mode": "chat", "supports_sampling_params": sampling},
    )


@pytest.mark.asyncio
async def test_replayed_tool_ids_are_sanitized_before_they_reach_the_provider(server: RecordingServer) -> None:
    history: Final = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ""},
                {
                    "type": "tool_use",
                    "id": "functions.Bash:0",
                    "name": "Bash",
                    "input": {},
                    "provider_specific_fields": {"x": 1},
                },
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions.Bash:0", "content": "ok"}]},
    ]

    await call(server, messages=history)

    body, _ = sent(server)
    assert body["messages"] == [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "functions_Bash_0", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions_Bash_0", "content": "ok"}]},
    ]


@pytest.mark.asyncio
async def test_provider_specific_headers_scoped_to_anthropic_reach_the_wire(server: RecordingServer) -> None:
    await call(
        server,
        provider_specific_header=[
            {"custom_llm_provider": "anthropic, azure_ai", "extra_headers": {"x-scoped": "yes"}},
            {"custom_llm_provider": "openai", "extra_headers": {"x-other": "no"}},
        ],
    )

    _, headers = sent(server)
    assert headers["x-scoped"] == "yes"
    assert "x-other" not in headers


@pytest.mark.asyncio
async def test_scoped_headers_override_extra_headers_which_override_forwarded_headers(server: RecordingServer) -> None:
    await call(
        server,
        headers={"x-priority": "forwarded", "x-forwarded-only": "kept"},
        extra_headers={"x-priority": "extra", "x-extra-only": "kept"},
        provider_specific_header={"custom_llm_provider": "anthropic", "extra_headers": {"x-priority": "scoped"}},
    )

    _, headers = sent(server)
    assert {name: headers.get(name) for name in ("x-priority", "x-forwarded-only", "x-extra-only")} == {
        "x-priority": "scoped",
        "x-forwarded-only": "kept",
        "x-extra-only": "kept",
    }


@pytest.mark.asyncio
async def test_model_capabilities_are_read_from_model_cost_at_call_time(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.expected_requests = 2
    model: Final = f"anthropic/{CAPABILITY_MODEL}"

    capabilities(monkeypatch, sampling=True)
    await call(server, model=model, temperature=0.2, drop_params=True)
    capabilities(monkeypatch, sampling=False)
    await call(server, model=model, temperature=0.2, drop_params=True)

    assert_served_natively(server, count=2)
    first, second = (request.body for request in server.requests)
    assert isinstance(first, dict)
    assert first["temperature"] == 0.2
    assert second == {name: value for name, value in first.items() if name != "temperature"}


@pytest.mark.asyncio
async def test_global_drop_params_drops_unsupported_params(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    capabilities(monkeypatch, sampling=False)
    monkeypatch.setattr(litellm, "drop_params", True)

    await call(server, model=f"anthropic/{CAPABILITY_MODEL}", top_k=5)

    body, _ = sent(server)
    assert "top_k" not in body


@pytest.mark.asyncio
async def test_unsupported_params_are_rejected_before_io_without_drop_params(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    capabilities(monkeypatch, sampling=False)
    monkeypatch.setattr(litellm, "drop_params", False)
    server.expected_requests = 0

    with pytest.raises(litellm.BadRequestError, match="top_k"):
        await call(server, model=f"anthropic/{CAPABILITY_MODEL}", top_k=5)

    assert server.requests == []


@pytest.mark.asyncio
async def test_optional_positional_body_parameters_reach_the_wire(server: RecordingServer) -> None:
    metadata: Final = {"user_id": "caller"}
    args: Final = (16, list(MESSAGES), "anthropic/claude-test", metadata, None, False, "Be brief", 0.25)

    response: Final = await litellm.anthropic_messages(*args, api_key="test-key", api_base=server.base_url)

    assert isinstance(response, dict)
    assert response["id"] == MESSAGES_RESPONSE["id"]
    body, _ = sent(server)
    assert (body["metadata"], body["system"], body["temperature"]) == (args[3], args[6], args[7])


def _user_text(*texts: object) -> tuple[dict[str, object], ...]:
    return ({"role": "user", "content": [{"type": "text", "text": text} for text in texts]},)


FAILURE_EVENTS: Final = {True: "async_log_failure_event", False: "log_failure_event"}
BOTH_MODES: Final = pytest.mark.parametrize("asynchronous", (True, False), ids=("async", "sync"))


@pytest.mark.asyncio
@BOTH_MODES
@pytest.mark.parametrize("malformed", (None, 42))
async def test_non_string_user_text_answers_400_before_io_and_reaches_the_failure_callback(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, asynchronous: bool, malformed: object
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    server.expected_requests = 0
    recorder: Final = RecordingLogger()

    with pytest.raises(litellm.BadRequestError, match=r"messages\.0\.content\.1\.text: Input should be a valid string") as raised:
        await call(server, asynchronous=asynchronous, messages=_user_text("hi", malformed), callbacks=[recorder])
    events: Final = await recorder.wait_for_async(FAILURE_EVENTS[asynchronous])

    assert server.requests == []
    assert raised.value.llm_provider == "anthropic"
    assert isinstance(events[0].kwargs, dict)
    assert events[0].kwargs["exception"] is raised.value


@pytest.mark.asyncio
@BOTH_MODES
@pytest.mark.parametrize("malformed", (None, 42))
async def test_drop_params_drops_non_string_user_text(server: RecordingServer, asynchronous: bool, malformed: object) -> None:
    await call(server, asynchronous=asynchronous, messages=_user_text("hi", malformed), drop_params=True)

    body, _ = sent(server)
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


@pytest.mark.asyncio
@BOTH_MODES
async def test_messages_emptied_by_drop_params_answer_400_before_io(server: RecordingServer, asynchronous: bool) -> None:
    server.expected_requests = 0
    recorder: Final = RecordingLogger()

    with pytest.raises(litellm.BadRequestError, match="no message has content left"):
        await call(server, asynchronous=asynchronous, messages=_user_text(None), drop_params=True, callbacks=[recorder])
    await recorder.wait_for_async(FAILURE_EVENTS[asynchronous])

    assert server.requests == []


@pytest.mark.asyncio
@BOTH_MODES
async def test_drop_params_leaves_a_malformed_message_list_for_the_request_validation(
    server: RecordingServer, asynchronous: bool
) -> None:
    server.expected_requests = 0

    with pytest.raises(litellm.BadRequestError, match="invalid Anthropic messages request"):
        await call(server, asynchronous=asynchronous, messages=[*_user_text("hi", 42), "stray"], drop_params=True)

    assert server.requests == []
