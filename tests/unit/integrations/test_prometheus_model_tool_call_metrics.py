import datetime
from collections.abc import Mapping
from typing import Final, cast

import pytest
from prometheus_client import REGISTRY
from prometheus_client.samples import Sample

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.utils import StandardLoggingPayload

NOW: Final = datetime.datetime(2026, 11, 3, 12, 0, 0)


def _clear_prometheus_registry() -> None:
    for collector in tuple(REGISTRY._collector_to_names):  # pyright: ignore[reportPrivateUsage]  # test registry reset
        REGISTRY.unregister(collector)


@pytest.fixture(autouse=True)
def isolated_registry():
    _clear_prometheus_registry()
    yield
    _clear_prometheus_registry()


def _success_kwargs() -> dict[str, object]:
    payload: Final = cast(
        StandardLoggingPayload,
        {
            "id": "t",
            "call_type": "acompletion",
            "response_cost": 0.001,
            "status": "success",
            "total_tokens": 30,
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "startTime": NOW - datetime.timedelta(seconds=3),
            "endTime": NOW,
            "completionStartTime": NOW - datetime.timedelta(seconds=1),
            "model": "gpt-4.1",
            "model_id": "model-123",
            "model_group": "gpt-4.1-group",
            "api_base": "https://api.openai.com",
            "custom_llm_provider": "openai",
            "request_tags": [],
            "stream": False,
            "metadata": {
                "user_api_key_hash": "key-hash",
                "user_api_key_alias": "key-alias",
                "user_api_key_team_id": "team-1",
                "user_api_key_team_alias": "team-alias",
                "user_api_key_user_id": "user-1",
                "user_api_key_user_email": None,
                "user_api_key_org_id": None,
                "user_api_key_org_alias": None,
                "requester_metadata": None,
                "user_api_key_end_user_id": None,
                "usage_object": None,
            },
            "hidden_params": {"litellm_overhead_time_ms": None, "additional_headers": None},
        },
    )
    return {
        "model": "gpt-4.1",
        "litellm_params": {"metadata": {}},
        "standard_logging_object": payload,
        "stream": False,
        "start_time": NOW - datetime.timedelta(seconds=3),
        "api_call_start_time": NOW - datetime.timedelta(seconds=2),
        "completion_start_time": NOW - datetime.timedelta(seconds=1),
        "end_time": NOW,
    }


def _samples(name: str) -> tuple[Sample, ...]:
    return tuple(sample for metric in REGISTRY.collect() for sample in metric.samples if sample.name == f"{name}_total")


def _tool_call_counts() -> Mapping[str, float]:
    return {sample.labels["tool_name"]: sample.value for sample in _samples("litellm_model_tool_calls")}


def _chat_tool_call(call_id: str, name: str) -> Mapping[str, object]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _chat_choice(index: int, *tool_calls: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "index": index,
        "finish_reason": "tool_calls",
        "message": {"role": "assistant", "content": None, "tool_calls": list(tool_calls)},
    }


CHAT_RESPONSE: Final = litellm.ModelResponse(
    choices=[
        _chat_choice(0, _chat_tool_call("a", "Read"), _chat_tool_call("b", "Read"), _chat_tool_call("c", "Bash")),
        _chat_choice(1, _chat_tool_call("d", "Bash")),
    ]
)
RESPONSES_API_RESPONSE: Final = {
    "id": "resp_1",
    "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "checking"}]},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
    ],
}
MESSAGES_TOOL_USE_RESPONSE: Final = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Let me look"},
        {"type": "tool_use", "id": "toolu_1", "name": "Grep", "input": {"pattern": "x"}},
        {"type": "tool_use", "id": "toolu_2", "name": "Grep", "input": {"pattern": "y"}},
    ],
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    (
        pytest.param(CHAT_RESPONSE, {"Read": 2, "Bash": 2}, id="chat-completions-every-choice"),
        pytest.param(RESPONSES_API_RESPONSE, {"get_weather": 1}, id="responses-api"),
        pytest.param(MESSAGES_TOOL_USE_RESPONSE, {"Grep": 2}, id="messages-tool-use"),
    ),
)
async def test_counts_each_tool_the_model_called(response: object, expected: Mapping[str, float]):
    logger: Final = PrometheusLogger()

    await logger.async_log_success_event(_success_kwargs(), response, NOW, NOW)

    assert _tool_call_counts() == expected


@pytest.mark.asyncio
async def test_counts_add_up_across_requests():
    logger: Final = PrometheusLogger()

    await logger.async_log_success_event(_success_kwargs(), MESSAGES_TOOL_USE_RESPONSE, NOW, NOW)
    await logger.async_log_success_event(_success_kwargs(), MESSAGES_TOOL_USE_RESPONSE, NOW, NOW)

    assert _tool_call_counts() == {"Grep": 4}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    (
        pytest.param(litellm.ModelResponse(), id="chat-without-tool-calls"),
        pytest.param({"id": "msg_2", "content": [{"type": "text", "text": "hi"}]}, id="messages-text-only"),
        pytest.param(None, id="no-response"),
    ),
)
async def test_emits_nothing_when_the_model_called_no_tool(response: object):
    logger: Final = PrometheusLogger()

    await logger.async_log_success_event(_success_kwargs(), response, NOW, NOW)

    assert _tool_call_counts() == {}


@pytest.mark.asyncio
async def test_a_failure_while_counting_tool_calls_does_not_stop_the_other_metrics(monkeypatch: pytest.MonkeyPatch):
    def explode(_response: object) -> list[str]:
        raise RuntimeError("unreadable response")

    monkeypatch.setattr("litellm.proxy.db.spend_log_tool_index.response_tool_call_names", explode)
    logger: Final = PrometheusLogger()

    await logger.async_log_success_event(_success_kwargs(), MESSAGES_TOOL_USE_RESPONSE, NOW, NOW)

    assert _tool_call_counts() == {}
    assert _samples("litellm_output_tokens_metric")


@pytest.mark.asyncio
async def test_series_carry_the_output_token_labels_plus_tool_name():
    logger: Final = PrometheusLogger()

    await logger.async_log_success_event(_success_kwargs(), MESSAGES_TOOL_USE_RESPONSE, NOW, NOW)

    token_labels: Final = {**_samples("litellm_output_tokens_metric")[0].labels}
    (tool_sample,) = _samples("litellm_model_tool_calls")
    assert tool_sample.labels == {**token_labels, "tool_name": "Grep"}
