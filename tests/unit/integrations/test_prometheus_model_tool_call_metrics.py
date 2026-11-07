import datetime
import logging
from collections.abc import Mapping
from typing import Final, cast

import pytest
from prometheus_client import REGISTRY
from prometheus_client.samples import Sample

import litellm
from litellm.constants import REDACTED_BY_LITELLM
from litellm.integrations.prometheus import PrometheusLogger
from litellm.litellm_core_utils.prompt_templates.factory import (
    get_tool_call_names_from_response,
    get_tool_calls_from_response,
)
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


SECRET: Final = "card 4111-1111-1111-1111"


def _chat_response_with_arguments(*arguments: str) -> litellm.ModelResponse:
    calls: Final = tuple(
        {"id": f"call_{n}", "type": "function", "function": {"name": "pay", "arguments": raw}}
        for n, raw in enumerate(arguments)
    )
    return litellm.ModelResponse(choices=[_chat_choice(0, *calls)])


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def litellm_log_records():
    handler: Final = _Records()
    lg: Final = logging.getLogger("LiteLLM")
    previous_level: Final = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    yield handler.messages
    lg.removeHandler(handler)
    lg.setLevel(previous_level)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    (
        pytest.param('{"note": "' + SECRET, id="truncated-repairable"),
        pytest.param('{"note": "' + SECRET + '" oops', id="unparseable"),
    ),
)
async def test_counting_never_logs_the_tool_arguments(arguments: str, litellm_log_records: list[str]):
    logger: Final = PrometheusLogger()

    await logger.async_log_success_event(_success_kwargs(), _chat_response_with_arguments(arguments), NOW, NOW)

    assert _tool_call_counts() == {"pay": 1}
    assert not [message for message in litellm_log_records if "4111" in message]


@pytest.mark.parametrize(
    "arguments",
    (
        pytest.param(('{"a": 1}',), id="valid"),
        pytest.param(('{"a": 1',), id="truncated"),
        pytest.param(('{"a": 1}{"a": 2}',), id="concatenated"),
        pytest.param(('{"a": 1}{"a": 1}',), id="concatenated-identical"),
        pytest.param(("".join(f'{{"a": {n}}}' for n in range(12)),), id="concatenated-over-cap"),
        pytest.param(("not json",), id="garbage"),
        pytest.param(("",), id="empty"),
        pytest.param((REDACTED_BY_LITELLM,), id="redacted"),
        pytest.param(('{"a": 1}', '{"b": 2}{"b": 3}'), id="two-calls"),
    ),
)
def test_name_only_extraction_counts_like_the_parse_path(arguments: tuple[str, ...]):
    response: Final = _chat_response_with_arguments(*arguments)

    parsed: Final = tuple(call["name"] for call in get_tool_calls_from_response(response, include_all_choices=True))

    assert get_tool_call_names_from_response(response, include_all_choices=True) == parsed


def _dict_chat_response(*choices: Mapping[str, object]) -> Mapping[str, object]:
    return {"id": "chat_1", "choices": list(choices)}


@pytest.mark.parametrize(
    "response",
    (
        pytest.param(
            _dict_chat_response(
                {"message": {"tool_calls": [{"id": "a", "function": {"name": "f", "arguments": '{"x": 1}{"x": 2}'}}]}}
            ),
            id="plain-dict-chat",
        ),
        pytest.param(
            _dict_chat_response({"message": {"tool_calls": [{"id": "a", "function": {"name": "f"}}]}}),
            id="missing-arguments",
        ),
        pytest.param(
            _dict_chat_response(
                {"message": {"tool_calls": [{"id": "a", "function": {"name": "f", "arguments": None}}]}}
            ),
            id="none-arguments",
        ),
        pytest.param(
            _dict_chat_response(
                {"message": {"tool_calls": [{"id": "a", "function": {"name": "f", "arguments": {"x": 1}}}]}}
            ),
            id="parsed-dict-arguments",
        ),
        pytest.param(
            _dict_chat_response(
                {"message": {"tool_calls": [{"id": "a", "function": None}, {"id": "b", "function": {"name": "g"}}]}}
            ),
            id="function-none",
        ),
        pytest.param(
            _dict_chat_response(
                {"message": {"content": "no tools"}},
                {"message": {"tool_calls": [{"id": "b", "function": {"name": "g", "arguments": '{"y": 1'}}]}},
            ),
            id="second-choice-only",
        ),
        pytest.param(
            {"output": [{"type": "function_call", "call_id": "c", "name": "h", "arguments": '{"z": 1}{"z": 2}'}]},
            id="responses-api-concatenated",
        ),
        pytest.param(
            {"output": [{"type": "function_call", "call_id": "c", "name": "h"}]}, id="responses-api-no-arguments"
        ),
        pytest.param(MESSAGES_TOOL_USE_RESPONSE, id="messages-tool-use"),
        pytest.param(RESPONSES_API_RESPONSE, id="responses-api"),
    ),
)
def test_name_only_extraction_matches_the_parse_path_on_every_surface(response: object):
    parsed: Final = tuple(call["name"] for call in get_tool_calls_from_response(response, include_all_choices=True))

    assert get_tool_call_names_from_response(response, include_all_choices=True) == parsed


@pytest.mark.asyncio
async def test_responses_api_counting_never_logs_the_tool_arguments(litellm_log_records: list[str]):
    logger: Final = PrometheusLogger()
    response: Final = {
        "id": "resp_2",
        "output": [
            {"type": "function_call", "call_id": "c", "name": "pay", "arguments": '{"note": "' + SECRET + '" x'}
        ],
    }

    await logger.async_log_success_event(_success_kwargs(), response, NOW, NOW)

    assert _tool_call_counts() == {"pay": 1}
    assert not [message for message in litellm_log_records if "4111" in message]


@pytest.mark.asyncio
async def test_a_cap_of_zero_counts_every_tool_as_other(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "prometheus_model_tool_name_max_values", 0)
    logger: Final = PrometheusLogger()

    await logger.async_log_success_event(_success_kwargs(), MESSAGES_TOOL_USE_RESPONSE, NOW, NOW)

    assert _tool_call_counts() == {"other": 2}


@pytest.mark.asyncio
async def test_tool_names_past_the_cap_count_as_other(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "prometheus_model_tool_name_max_values", 2)
    logger: Final = PrometheusLogger()
    response: Final = litellm.ModelResponse(
        choices=[_chat_choice(0, *(_chat_tool_call(name, name) for name in ("Read", "Bash", "Grep", "Glob", "Read")))]
    )

    await logger.async_log_success_event(_success_kwargs(), response, NOW, NOW)

    assert _tool_call_counts() == {"Read": 2, "Bash": 1, "other": 2}


@pytest.mark.asyncio
async def test_no_cap_when_the_setting_is_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "prometheus_model_tool_name_max_values", None)
    logger: Final = PrometheusLogger()
    names: Final = tuple(f"tool_{n}" for n in range(1200))
    response: Final = litellm.ModelResponse(choices=[_chat_choice(0, *(_chat_tool_call(n, n) for n in names))])

    await logger.async_log_success_event(_success_kwargs(), response, NOW, NOW)

    assert len(_tool_call_counts()) == len(names)
