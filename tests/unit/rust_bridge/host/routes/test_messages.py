import json
import math
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS
from litellm.rust_bridge.host.routes import messages as messages_host
from litellm.rust_bridge.host.routes.messages import response


def test_response_is_a_detached_public_messages_dict() -> None:
    native: Final = MappingProxyType(
        {
            "id": "msg_native",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "native"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
    )

    built: Final = response(native)

    assert built == dict(native)
    assert isinstance(built, dict)
    built["_hidden_params"] = {"annotated": True}
    assert "_hidden_params" not in native


def test_settings_project_caller_configuration_without_resolving_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    monkeypatch.setattr(litellm, "reasoning_auto_summary", True)
    monkeypatch.setattr(litellm, "request_timeout", 1800)
    monkeypatch.setattr(litellm, "request_timeout_explicitly_set", True)
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", "true")

    projected: Final = messages_host.settings({"drop_params": "true", "additional_drop_params": ["metadata.user_id"]})

    assert projected == {
        "drop_params": True,
        "reasoning_auto_summary": True,
        "additional_drop_params": ("metadata.user_id",),
        "request_timeout": 1800.0,
        "include_cost_in_streaming_usage": True,
    }


@pytest.mark.parametrize(
    ("timeout", "explicitly_set", "expected"),
    [
        (DEFAULT_REQUEST_TIMEOUT_SECONDS, False, None),
        (DEFAULT_REQUEST_TIMEOUT_SECONDS, True, DEFAULT_REQUEST_TIMEOUT_SECONDS),
        (300, False, 300.0),
    ],
)
def test_request_timeout_carries_only_a_configured_global_timeout(
    monkeypatch: pytest.MonkeyPatch, timeout: float, explicitly_set: bool, expected: float | None
) -> None:
    monkeypatch.setattr(litellm, "request_timeout", timeout)
    monkeypatch.setattr(litellm, "request_timeout_explicitly_set", explicitly_set)

    assert messages_host.settings({})["request_timeout"] == expected


@pytest.mark.parametrize(
    ("global_flag", "kwargs", "expected"),
    [
        (False, {}, False),
        (True, {}, True),
        (False, {"drop_params": "true"}, True),
        (False, {"drop_params": "nonsense"}, False),
        (False, {"drop_params": False}, False),
    ],
)
def test_drop_params_merges_the_global_flag_with_the_request(
    monkeypatch: pytest.MonkeyPatch, global_flag: bool, kwargs: dict[str, object], expected: bool
) -> None:
    monkeypatch.setattr(litellm, "drop_params", global_flag)

    assert messages_host.settings(kwargs)["drop_params"] is expected


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (["tools[*].input_examples", 3, "metadata.user_id"], ("tools[*].input_examples", "metadata.user_id")),
        ("tools", ()),
        (None, ()),
    ],
)
def test_additional_drop_params_keep_only_string_paths(configured: object, expected: tuple[str, ...]) -> None:
    settings: Final = messages_host.settings({"additional_drop_params": configured})

    assert settings["additional_drop_params"] == expected


def test_native_request_rejections_map_to_the_public_400() -> None:
    from types import MappingProxyType

    from litellm.rust_bridge.public_call import NativeCall

    request: Final = NativeCall(
        args=(),
        kwargs=MappingProxyType({}),
        base={
            "model": "anthropic/claude-sonnet-5",
            "messages": (),
            "max_tokens": 8,
            "stream": None,
            "api_key": None,
            "api_base": None,
            "custom_llm_provider": None,
            **MappingProxyType({}),
        },
    )
    rejected: Final = ValueError("claude-sonnet-5 does not support top_k=5")
    rejected.messages_request_error = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(rejected, request.resolved, "anthropic")

    assert isinstance(mapped, litellm.BadRequestError)
    assert mapped.status_code == 400
    assert "does not support top_k=5" in mapped.message
    assert mapped.model == "claude-sonnet-5"
    assert not isinstance(
        messages_host.map_failure(ValueError("plain"), request.resolved, "anthropic"), litellm.BadRequestError
    )


def test_stream_hidden_params_projects_upstream_headers_the_way_the_python_handler_does() -> None:
    hidden: Final = messages_host.stream_hidden_params(
        (("request-id", "req_upstream_123"), ("x-ratelimit-remaining-requests", "41"))
    )

    additional: Final = hidden["additional_headers"]
    assert isinstance(additional, dict)
    assert additional["llm_provider-request-id"] == "req_upstream_123"
    assert additional["x-ratelimit-remaining-requests"] == "41"
    assert additional["request-id"] == additional["llm_provider-request-id"]


_ADAPTIVE_OPUS: Final = "anthropic/claude-opus-4-7"


@pytest.mark.parametrize(
    ("model", "expected"),
    [("claude-sonnet-4-5", "anthropic"), ("gpt-4o", "openai"), ("not-a-known-model", None)],
)
def test_inferred_provider_is_the_one_the_python_handler_resolves(model: str, expected: str | None) -> None:
    assert messages_host.inferred_provider(model, None) == expected


@pytest.mark.parametrize(
    ("request_effort", "levels", "drop_params", "expected"),
    [
        ("max", ["low", "high"], False, "high"),
        ("high", ["low"], True, "low"),
    ],
)
def test_body_params_round_effort_to_the_levels_the_deployment_declares(
    request_effort: str, levels: list[str], drop_params: bool, expected: str
) -> None:
    request: Final[dict[str, object]] = {
        "model": _ADAPTIVE_OPUS,
        "messages": [],
        "max_tokens": 8,
        "output_config": {"effort": request_effort},
        "drop_params": drop_params,
        "model_info": {"id": "deployment-1", "reasoning_effort_levels": levels},
    }

    assert messages_host.body_params(request)["output_config"] == {"effort": expected}


def test_body_params_keep_the_requested_effort_without_deployment_levels() -> None:
    request: Final[dict[str, object]] = {
        "model": _ADAPTIVE_OPUS,
        "messages": [],
        "max_tokens": 8,
        "output_config": {"effort": "max"},
    }

    assert messages_host.body_params(request)["output_config"] == {"effort": "max"}


def _marked(phase: str | None = None, *args: object) -> ValueError:
    error: Final = ValueError(*args)
    if phase is not None:
        error.messages_timeout = phase  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets
    return error


def test_a_stalled_stream_body_reaches_the_caller_as_the_transport_timeout() -> None:
    import httpx

    mapped: Final = messages_host.map_failure(_marked("Body"), {"model": _ADAPTIVE_OPUS}, "anthropic")

    assert isinstance(mapped, httpx.ReadTimeout)
    assert str(mapped) == "Timeout on reading data from socket"


def test_a_native_read_timeout_maps_to_the_public_timeout_with_python_text() -> None:
    mapped: Final = messages_host.map_failure(_marked("Read"), {"model": _ADAPTIVE_OPUS, "timeout": 1.0}, "anthropic")

    assert isinstance(mapped, litellm.Timeout)
    assert "Read timed out. Timeout passed=1.0" in str(mapped)


def test_an_upstream_status_maps_through_the_provider_error_class() -> None:
    upstream: Final = ValueError(429, "slow down")
    upstream.headers = [("retry-after", "3")]  # pyright: ignore[reportAttributeAccessIssue]  # set by the native host

    mapped: Final = messages_host.map_failure(upstream, {"model": _ADAPTIVE_OPUS}, "anthropic")

    assert isinstance(mapped, litellm.RateLimitError)
    assert "slow down" in str(mapped)
    assert mapped.__context__ is upstream


def test_a_late_rejection_tells_the_logger_the_resolved_call_first() -> None:
    import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging

    logger: Final = Logging(
        model=_ADAPTIVE_OPUS,
        messages=[],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="late",
        function_id="late",
    )
    rejected: Final = ValueError("budget_tokens must be a number")
    rejected.messages_late_rejection = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets
    rejected.messages_request_error = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(
        rejected,
        {"model": _ADAPTIVE_OPUS, "messages": [], "max_tokens": 8, "litellm_logging_obj": logger},
        "anthropic",
    )

    assert isinstance(mapped, litellm.BadRequestError)
    assert logger.model_call_details["custom_llm_provider"] == "anthropic"
    assert logger.model_call_details["optional_params"] == {"max_tokens": 8}


def _logger(model: str) -> object:
    import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging

    logger: Final = Logging(
        model=model,
        messages=[],
        stream=True,
        call_type="anthropic_messages",
        start_time=datetime.datetime(2026, 1, 1),
        litellm_call_id="stream",
        function_id="stream",
    )
    logger.model_call_details["model"] = model
    return logger


@pytest.mark.parametrize(
    ("mock", "error_class"),
    [
        ("litellm.RateLimitError", litellm.RateLimitError),
        ("litellm.InternalServerError", litellm.InternalServerError),
        ("litellm.ContextWindowExceededError", litellm.ContextWindowExceededError),
    ],
)
def test_a_mocked_error_maps_to_the_error_python_mock_response_raises(mock: str, error_class: type[Exception]) -> None:
    native: Final = ValueError("mocked")
    native.messages_mock_error = mock  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(native, {"model": "anthropic/claude-sonnet-4-5"}, "anthropic")

    assert type(mapped) is error_class
    assert getattr(mapped, "model") == "claude-sonnet-4-5"
    assert getattr(mapped, "llm_provider") == "anthropic"


def test_a_mock_naming_no_error_maps_to_a_runtime_error_naming_the_mock() -> None:
    native: Final = ValueError("mocked")
    native.messages_mock_error = "hello"  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(native, {"model": "anthropic/claude-sonnet-4-5"}, "anthropic")

    assert isinstance(mapped, RuntimeError)
    assert "'hello'" in str(mapped)


@pytest.mark.parametrize(
    ("enabled", "model", "expected"),
    [
        (False, "claude-sonnet-4-5", None),
        (True, "claude-sonnet-4-5", "claude-sonnet-4-5"),
        (True, "", None),
    ],
)
def test_stream_cost_model_is_the_logged_model_only_when_cost_is_on(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, model: str, expected: str | None
) -> None:
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", enabled)

    assert messages_host.stream_cost_model({"litellm_logging_obj": _logger(model)}) == expected


def test_stream_cost_model_needs_a_logging_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)

    assert messages_host.stream_cost_model({"litellm_logging_obj": "not a logger"}) is None
    assert messages_host.stream_cost_model({}) is None


_EVENT: Final = TypeAdapter(dict[str, object])
_USAGE: Final = TypeAdapter(dict[str, int | float])
_USAGE_EVENT: Final = {
    "type": "message_delta",
    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
    "usage": {"input_tokens": 1000, "output_tokens": 500},
}


@pytest.mark.parametrize("with_logger", [True, False])
def test_stream_chunk_with_cost_adds_the_cost_to_the_usage_event(
    monkeypatch: pytest.MonkeyPatch, with_logger: bool
) -> None:
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
    model: Final = "claude-sonnet-4-5"
    chunk: Final = f"event: message_delta\ndata: {json.dumps(_USAGE_EVENT)}\n\n".encode()
    request: Final[dict[str, object]] = {"litellm_logging_obj": _logger(model)} if with_logger else {}

    priced: Final = messages_host.stream_chunk_with_cost(chunk, model, request)

    assert isinstance(priced, bytes)
    data_line: Final = next(line for line in priced.decode().splitlines() if line.startswith("data: "))
    event: Final = _EVENT.validate_json(data_line.removeprefix("data: "))
    usage: Final = dict(_USAGE.validate_python(event["usage"]))
    cost: Final = usage.pop("cost")
    assert usage == _USAGE_EVENT["usage"]
    assert {key: value for key, value in event.items() if key != "usage"} == {
        key: value for key, value in _USAGE_EVENT.items() if key != "usage"
    }
    expected: Final = litellm.completion_cost(
        model=model,
        completion_response=litellm.ModelResponse(
            usage=litellm.Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        ),
    )
    assert expected > 0
    assert math.isclose(float(cost), expected)


def test_stream_chunk_with_cost_leaves_a_chunk_without_usage_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
    chunk: Final = b'event: ping\ndata: {"type": "ping"}\n\n'

    assert messages_host.stream_chunk_with_cost(chunk, "claude-sonnet-4-5", {}) == chunk


_TOOLS: Final = (
    {"name": "first", "input_schema": {"type": "object"}},
    {"name": "last", "input_schema": {"type": "object"}},
)


def _prompt_request(**extra: object) -> dict[str, object]:
    return {
        "model": "anthropic/claude-sonnet-4-5",
        "custom_llm_provider": "anthropic",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "system": [{"type": "text", "text": "be brief"}],
        **extra,
    }


def test_injected_prompt_is_none_without_prompt_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", False)

    assert messages_host.injected_prompt(_prompt_request(cache_control_injection_points=[])) is None


def test_injected_prompt_marks_the_configured_message_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", False)
    request: Final = _prompt_request(cache_control_injection_points=[{"location": "message", "role": "user"}])

    injected: Final = messages_host.injected_prompt(request)

    assert injected is not None
    messages: Final = injected["messages"]
    assert isinstance(messages, list)
    assert messages[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert injected["system"] == [{"type": "text", "text": "be brief"}]
    assert injected["tools"] is None


@pytest.mark.parametrize("per_request", [True, False])
def test_injected_prompt_places_default_breakpoints_when_caching_is_enabled(
    monkeypatch: pytest.MonkeyPatch, per_request: bool
) -> None:
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", not per_request)
    request: Final = _prompt_request(**({"enable_prompt_caching": True} if per_request else {}))

    injected: Final = messages_host.injected_prompt(request)

    assert injected is not None
    assert "cache_control" in json.dumps(injected)
    assert "cache_control" not in json.dumps(request["messages"]) + json.dumps(request["system"])


def test_injected_prompt_marks_the_last_tool_and_credits_the_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.types.integrations.anthropic_cache_control_hook import GATEWAY_INJECTED_CACHE_METADATA_KEY

    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", False)
    metadata: Final[dict[str, object]] = {}
    request: Final = _prompt_request(
        tools=list(_TOOLS),
        cache_control_injection_points=[{"location": "tool_config"}],
        litellm_metadata=metadata,
        model_info={"id": "deployment-1"},
    )

    injected: Final = messages_host.injected_prompt(request)

    assert injected is not None
    assert injected["tools"] == [_TOOLS[0], {**_TOOLS[1], "cache_control": {"type": "ephemeral"}}]
    assert metadata[GATEWAY_INJECTED_CACHE_METADATA_KEY] == "deployment-1"
    assert request["tools"] == list(_TOOLS)


def test_injected_prompt_keeps_the_client_tool_breakpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.types.integrations.anthropic_cache_control_hook import GATEWAY_INJECTED_CACHE_METADATA_KEY

    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", False)
    tools: Final = [{**_TOOLS[0], "cache_control": {"type": "ephemeral"}}, _TOOLS[1]]
    metadata: Final[dict[str, object]] = {}
    request: Final = _prompt_request(
        tools=tools,
        cache_control_injection_points=[{"location": "tool_config"}],
        litellm_metadata=metadata,
        model_info={"id": "deployment-1"},
    )

    injected: Final = messages_host.injected_prompt(request)

    assert injected is not None
    assert injected["tools"] is tools
    assert GATEWAY_INJECTED_CACHE_METADATA_KEY not in metadata
