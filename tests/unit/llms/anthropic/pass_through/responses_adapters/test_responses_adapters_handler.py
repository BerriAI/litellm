import datetime
import json
import os
import sys
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
import respx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

import litellm
from litellm.llms.anthropic.pass_through.responses_adapters.handler import (
    LiteLLMMessagesToResponsesAPIHandler,
    _build_responses_kwargs,
)
from litellm.types.integrations.custom_logger import CONVERTED_STREAM_KEYS, converted_stream_requested

MESSAGES = [{"role": "user", "content": "hello"}]
CLAUDE_CODE_USER_ID = json.dumps({"device_id": "d" * 64, "account_uuid": "", "session_id": "session-abc"})

RESPONSES_SSE_BODY = (
    b"event: response.created\n"
    b'data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_lit6825",'
    b'"object":"response","created_at":1,"status":"in_progress","model":"gpt-5.6-luna","output":[],'
    b'"parallel_tool_calls":true,"tool_choice":"auto","tools":[]}}\n\n'
    b"event: response.completed\n"
    b'data: {"type":"response.completed","sequence_number":1,"response":{"id":"resp_lit6825",'
    b'"object":"response","created_at":1,"status":"completed","model":"gpt-5.6-luna","output":[],'
    b'"parallel_tool_calls":true,"tool_choice":"auto","tools":[],'
    b'"usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}}}\n\n'
)


def test_build_responses_kwargs_never_asks_the_nested_call_to_replay_a_stream():
    responses_kwargs: Final = _build_responses_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model="azure/gpt-5.6-luna",
        extra_kwargs={"custom_llm_provider": "azure", **{key: True for key in CONVERTED_STREAM_KEYS}},
    )

    assert not converted_stream_requested(responses_kwargs)
    assert responses_kwargs["custom_llm_provider"] == "azure"


def test_build_responses_kwargs_derives_prompt_cache_key_from_claude_code_session_id():
    responses_kwargs = _build_responses_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model="openai/gpt-5.6-luna",
        metadata={"user_id": CLAUDE_CODE_USER_ID},
        extra_kwargs={"custom_llm_provider": "openai"},
    )
    assert responses_kwargs["user"] == CLAUDE_CODE_USER_ID[:64]
    assert responses_kwargs["prompt_cache_key"] == "session-abc"


def test_build_responses_kwargs_sets_no_prompt_cache_key_for_plain_user_id():
    responses_kwargs = _build_responses_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model="openai/gpt-5.6-luna",
        metadata={"user_id": "session-abc"},
        extra_kwargs={"custom_llm_provider": "openai"},
    )
    assert responses_kwargs["user"] == "session-abc"
    assert "prompt_cache_key" not in responses_kwargs


def test_build_responses_kwargs_prefers_explicit_prompt_cache_key_over_derived():
    responses_kwargs = _build_responses_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model="openai/gpt-5.6-luna",
        metadata={"user_id": CLAUDE_CODE_USER_ID},
        extra_kwargs={"custom_llm_provider": "openai", "prompt_cache_key": "explicit-key"},
    )
    assert responses_kwargs["user"] == CLAUDE_CODE_USER_ID[:64]
    assert responses_kwargs["prompt_cache_key"] == "explicit-key"


def test_build_responses_kwargs_asks_openai_for_encrypted_reasoning_without_thinking():
    responses_kwargs = _build_responses_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model="openai/gpt-5.6-luna",
        extra_kwargs={"custom_llm_provider": "openai"},
    )
    assert responses_kwargs["include"] == ["reasoning.encrypted_content"]
    assert "reasoning" not in responses_kwargs


def test_build_responses_kwargs_skips_include_for_a_responses_provider_that_rejects_it():
    responses_kwargs = _build_responses_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model="perplexity/sonar",
        thinking={"type": "enabled", "budget_tokens": 4096},
        extra_kwargs={"custom_llm_provider": "perplexity"},
    )
    assert "include" not in responses_kwargs
    assert "reasoning" in responses_kwargs


def test_build_responses_kwargs_keeps_the_deployment_include_next_to_encrypted_reasoning():
    responses_kwargs = _build_responses_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model="openai/gpt-5.6-luna",
        extra_kwargs={"custom_llm_provider": "openai", "include": ["file_search_call.results"]},
    )
    assert responses_kwargs["include"] == ["reasoning.encrypted_content", "file_search_call.results"]


def test_build_responses_kwargs_without_metadata_sets_no_prompt_cache_key():
    responses_kwargs = _build_responses_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model="openai/gpt-5.6-luna",
        extra_kwargs={"custom_llm_provider": "openai"},
    )
    assert "user" not in responses_kwargs
    assert "prompt_cache_key" not in responses_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested_model, expected_reported_model",
    [
        ("openai/gpt-5.6-luna", "gpt-5.6-luna"),
        ("perplexity/perplexity/kimi-k3", "perplexity/kimi-k3"),
    ],
)
async def test_streaming_message_start_reports_the_provider_local_model(requested_model, expected_reported_model):
    """
    BerriAI/litellm#37716 sends the caller's unresolved id down this bridge so the provider
    resolves it once. ``message_start`` is a reporting field rather than a wire value, so it
    keeps naming the model as the provider knows it, with only the leading provider segment gone.
    """

    async def empty_stream():
        return
        yield

    with patch.object(litellm, "aresponses", AsyncMock(return_value=empty_stream())):
        sse = await LiteLLMMessagesToResponsesAPIHandler.async_anthropic_messages_handler(
            max_tokens=1024,
            messages=MESSAGES,
            model=requested_model,
            stream=True,
            custom_llm_provider=requested_model.split("/")[0],
        )
        events = [json.loads(chunk.decode().split("data: ", 1)[1]) async for chunk in sse]

    message_start = next(e for e in events if e["type"] == "message_start")
    assert message_start["message"]["model"] == expected_reported_model


@pytest.mark.asyncio
async def test_streaming_hands_the_logging_object_the_message_id_the_caller_is_streamed(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    """
    The bridge mints the ``msg_`` id itself, and it is the only request id a streaming
    /v1/messages caller ever sees, so the spend row has to be keyed on that same value.
    """
    from litellm.litellm_core_utils.litellm_logging import Logging

    monkeypatch.setenv("OPENAI_API_KEY", "sk-lit6825-test")
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post("https://api.openai.com/v1/responses").respond(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        content=RESPONSES_SSE_BODY,
    )

    logging_obj = Logging(
        model="gpt-5.6-luna",
        messages=MESSAGES,
        stream=True,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(datetime.timezone.utc),
        litellm_call_id="6825beef-0000-4000-8000-000000000003",
        function_id="1234",
    )

    sse = await LiteLLMMessagesToResponsesAPIHandler.async_anthropic_messages_handler(
        max_tokens=1024,
        messages=MESSAGES,
        model="openai/gpt-5.6-luna",
        stream=True,
        custom_llm_provider="openai",
        litellm_logging_obj=logging_obj,
    )
    events = [json.loads(chunk.decode().split("data: ", 1)[1]) async for chunk in sse]

    message_start = next(e for e in events if e["type"] == "message_start")
    assert message_start["message"]["id"].startswith("msg_")
    assert logging_obj.streamed_anthropic_message_id == message_start["message"]["id"]


OPENAI_API: Final = {"custom_llm_provider": "openai", "api_base": "https://api.openai.com/v1"}
OTHER_HOST: Final = {"custom_llm_provider": "openai", "api_base": "https://gateway.example/v1"}
EPHEMERAL: Final = {"type": "ephemeral"}


def _request_with_client_cache_marks(cache_control: dict[str, str] | None) -> dict[str, object]:
    mark: Final = {} if cache_control is None else {"cache_control": cache_control}
    return {
        "max_tokens": 1024,
        "system": [{"type": "text", "text": "Be concise."}, {"type": "text", "text": "Be helpful.", **mark}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "look", **mark}, {"type": "text", "text": "now"}]}
        ],
    }


@pytest.mark.parametrize(
    "extra_kwargs",
    [
        pytest.param(OPENAI_API, id="api.openai.com"),
        pytest.param({**OTHER_HOST, "prompt_cache_options": {"mode": "implicit"}}, id="other-host-opted-in"),
    ],
)
def test_client_cache_control_reaches_gpt_5_6_as_a_prompt_cache_breakpoint_on_the_same_block(
    extra_kwargs: dict[str, object],
) -> None:
    responses_kwargs: Final = _build_responses_kwargs(
        model="openai/gpt-5.6",
        extra_kwargs=extra_kwargs,
        **_request_with_client_cache_marks({**EPHEMERAL, "ttl": "1h"}),
    )

    assert "instructions" not in responses_kwargs
    assert responses_kwargs["input"] == [
        {
            "type": "message",
            "role": "developer",
            "content": [
                {"type": "input_text", "text": "Be concise."},
                {"type": "input_text", "text": "Be helpful.", "prompt_cache_breakpoint": {"mode": "explicit"}},
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "look", "prompt_cache_breakpoint": {"mode": "explicit"}},
                {"type": "input_text", "text": "now"},
            ],
        },
    ]


@pytest.mark.parametrize(
    ("model", "extra_kwargs"),
    [
        pytest.param("openai/gpt-5.5", OPENAI_API, id="model-older-than-gpt-5.6"),
        pytest.param("openai/gpt-5.6", OTHER_HOST, id="other-host-not-opted-in"),
    ],
)
def test_client_cache_control_changes_nothing_for_a_deployment_that_does_not_read_prompt_cache_breakpoint(
    model: str, extra_kwargs: dict[str, object]
) -> None:
    marked: Final = _build_responses_kwargs(
        model=model, extra_kwargs=extra_kwargs, **_request_with_client_cache_marks(EPHEMERAL)
    )
    unmarked: Final = _build_responses_kwargs(
        model=model, extra_kwargs=extra_kwargs, **_request_with_client_cache_marks(None)
    )

    assert marked == unmarked


@pytest.mark.parametrize("tool_choice", [{}, {"name": "get_weather"}, {"type": "web_search"}])
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_messages_rejects_invalid_tool_choice_before_provider_call(
    tool_choice: dict[str, str], stream: bool
) -> None:
    with pytest.raises(litellm.BadRequestError) as exc_info:
        await litellm.anthropic_messages(
            model="openai/gpt-6.1-sol",
            max_tokens=8,
            messages=[{"role": "user", "content": "Hello"}],
            tool_choice=tool_choice,
            stream=stream,
        )

    assert exc_info.value.status_code == 400
    assert "tool_choice" in exc_info.value.message
