"""Native Responses calls must be indistinguishable from the Python implementation.

Each case sends one public `litellm.aresponses` call through Python and through the native route against the same
recorded OpenAI response, then compares what reached the provider, what the caller got back, and what the success
callback logged. The native leg fails instead of silently falling back to Python
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIResponse
from tests.test_litellm_rust.responses.support import PUBLIC_ARESPONSES, PYTHON_ARESPONSES, FallbackProbe
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.isolation import isolated_callback_registries
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension

MODEL: Final = "openai/gpt-6.1-sol"
_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_LOGGED: Final = TypeAdapter(dict[str, object])
_COMPARED_HEADERS: Final = ("api-key", "authorization", "content-type", "openai-organization", "openai-project")
_COMPARED_LOG_FIELDS: Final = (
    "call_type",
    "custom_llm_provider",
    "model",
    "model_group",
    "model_id",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "response_cost",
    "status",
    "api_base",
    "cache_hit",
    "saved_cache_cost",
    "response_cost_failure_debug_info",
)
_COMPARED_HIDDEN_PARAMS: Final = ("custom_llm_provider", "model_id", "api_base", "response_cost")

TEXT_MESSAGE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "id": "msg_0a1b2c3d4e5f",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "Paris"}],
    }
)

TEXT_RESPONSE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "id": "resp_0a1b2c3d4e5f",
        "object": "response",
        "created_at": 1760486400,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": "gpt-6.1-sol-2026-09-30",
        "output": [
            {
                "id": "rs_0a1b2c3d4e5f",
                "type": "reasoning",
                "summary": [],
            },
            dict(TEXT_MESSAGE),
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": "medium", "summary": None},
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 70,
            "output_tokens_details": {"reasoning_tokens": 64},
            "total_tokens": 82,
        },
        "user": None,
        "metadata": {},
    }
)

FUNCTION_CALL_RESPONSE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        **TEXT_RESPONSE,
        "id": "resp_6f5e4d3c2b1a",
        "output": [
            {
                "id": "fc_6f5e4d3c2b1a",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"city":"Paris"}',
                "call_id": "call_6f5e4d3c2b1a",
                "name": "get_weather",
            }
        ],
    }
)

WEATHER_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "type": "function",
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
)


ENCRYPTED_REASONING_RESPONSE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        **TEXT_RESPONSE,
        "output": [
            {"id": "rs_0a1b2c3d4e5f", "type": "reasoning", "summary": [], "encrypted_content": "gAAAAB0a1b2c3d"},
            {
                "id": "ci_0a1b2c3d4e5f",
                "type": "code_interpreter_call",
                "status": "completed",
                "code": "print(1)",
                "container_id": "cntr_0a1b2c3d4e5f",
                "outputs": [],
            },
            dict(TEXT_MESSAGE),
        ],
    }
)

AFFINITY_REPLAY_INPUT: Final[list[JsonValue]] = [
    {
        "type": "reasoning",
        "id": ResponsesAPIRequestUtils._build_encrypted_item_id("deploy-1", "rs_0a1b2c3d4e5f"),  # pyright: ignore[reportPrivateUsage]  # the encoder Python applies on the way out
        "summary": [],
        "encrypted_content": ResponsesAPIRequestUtils.wrap_encrypted_content_with_model_id(
            "gAAAAB0a1b2c3d", "deploy-1"
        ),
    },
    {"role": "user", "content": "And Germany?"},
]

AZURE_MODEL: Final = "azure/my-deployment"

WEATHER_TOOL_FUNCTION: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {name: value for name, value in WEATHER_TOOL.items() if name != "type"}
)

AZURE_STATUS_REPLAY_INPUT: Final[list[JsonValue]] = [
    {"type": "message", "role": "user", "content": "Weather in Paris?", "status": "completed"},
    {"type": "reasoning", "id": "rs_0a1b2c3d4e5f", "summary": [], "status": "completed"},
    {"role": "user", "content": "And Germany?"},
]

MINTED_REASONING_REPLAY_INPUT: Final[list[JsonValue]] = [
    {"type": "reasoning", "id": "rs_6f9619ff-8b86-4d01-b42d-00c04fc964ff", "summary": []},
    {
        "type": "reasoning",
        "id": "rs_0a1b2c3d4e5f",
        "summary": [],
        "encrypted_content": '[{"type":"thinking","thinking":"x","signature":"sig"}]',
    },
    {"role": "user", "content": "And Germany?"},
]

FOREIGN_TOOL_CALL_REPLAY_INPUT: Final[list[JsonValue]] = [
    {
        "type": "function_call",
        "id": "toolu_6f5e4d3c2b1a",
        "call_id": "call_6f5e4d3c2b1a",
        "name": "get_weather",
        "arguments": '{"city":"Paris"}',
    },
    {"type": "function_call_output", "call_id": "call_6f5e4d3c2b1a", "output": "18C and sunny"},
]

PATTERNED_PARAMETERS: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "object", "properties": {"city": {"type": "string", "pattern": "^\\p{L}+$"}}}
)

REASONING_REPLAY_INPUT: Final[list[JsonValue]] = [
    {"type": "reasoning", "id": "rs_0a1b2c3d4e5f", "summary": [], "status": None},
    {"role": "user", "content": "And Germany?"},
]


@dataclass(frozen=True, slots=True)
class Leg:
    request_path: str
    request_body: Mapping[str, JsonValue]
    request_headers: Mapping[str, str]
    response: Mapping[str, JsonValue]
    logged: Mapping[str, object]
    hooks: tuple[str, ...]
    hidden_params: Mapping[str, object]
    fell_back: bool


ASPECTS: Final = ("dispatch", "request", "response", "logged", "hooks")


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    kwargs: Mapping[str, object]
    upstream: Mapping[str, JsonValue] = TEXT_RESPONSE
    gaps: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    falls_back: bool = False

    def params(self) -> Iterator[object]:
        for aspect in ASPECTS:
            gap = self.gaps.get(aspect)
            marks = () if gap is None else (pytest.mark.xfail(reason=gap, strict=True),)
            yield pytest.param(self, aspect, id=f"{self.name}-{aspect}", marks=marks)


DEPLOYMENT_METADATA: Final = MappingProxyType(
    {"model_info": {"id": "deploy-1"}, "encrypted_content_affinity_enabled": True}
)


CASES: Final = (
    Case("plain_text", {"input": "What is the capital of France?"}),
    Case(
        "instructions_and_input_items",
        {
            "instructions": "Answer in one word",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Capital of France?"}]}],
        },
    ),
    Case(
        "function_tool",
        {"input": "Weather in Paris?", "tools": [dict(WEATHER_TOOL)], "tool_choice": "auto"},
        upstream=FUNCTION_CALL_RESPONSE,
    ),
    Case(
        "encoded_previous_response_id",
        {
            "input": [{"type": "function_call_output", "call_id": "call_6f5e4d3c2b1a", "output": "18C and sunny"}],
            "previous_response_id": ResponsesAPIRequestUtils._build_responses_api_response_id(  # pyright: ignore[reportPrivateUsage]  # the public encoder needs a response object
                custom_llm_provider="openai", model_id=None, response_id="resp_6f5e4d3c2b1a"
            ),
        },
    ),
    Case(
        "deployment_scoped_ids",
        {"input": "hi", "litellm_metadata": dict(DEPLOYMENT_METADATA)},
        upstream=ENCRYPTED_REASONING_RESPONSE,
    ),
    Case(
        "affinity_encoded_reasoning_replay",
        {"input": AFFINITY_REPLAY_INPUT, "litellm_metadata": dict(DEPLOYMENT_METADATA)},
        upstream=ENCRYPTED_REASONING_RESPONSE,
    ),
    Case(
        "warm_up_probe_below_minimum_max_output_tokens",
        {"input": "hi", "max_output_tokens": 1},
    ),
    Case(
        "gpt_5_family_drops_temperature_while_reasoning",
        {"input": "hi", "temperature": 0.2, "reasoning": {"effort": "high"}, "drop_params": True},
        falls_back=True,
    ),
    Case(
        "anthropic_cache_control_is_stripped",
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
                }
            ],
            "tools": [{**WEATHER_TOOL, "cache_control": {"type": "ephemeral"}}],
        },
    ),
    Case(
        "reasoning_replay_without_status",
        {
            "input": REASONING_REPLAY_INPUT,
        },
    ),
    Case(
        "tool_parameters_sent_as_json_string",
        {
            "input": "Weather in Paris?",
            "tools": [{**WEATHER_TOOL, "parameters": '{"type":"object","properties":{"city":{"type":"string"}}}'}],
        },
    ),
    Case(
        "item_level_cache_control_is_stripped",
        {"input": [{"role": "user", "content": "hi", "cache_control": {"type": "ephemeral"}}]},
    ),
    Case(
        "bridge_minted_reasoning_is_dropped",
        {"input": MINTED_REASONING_REPLAY_INPUT},
    ),
    Case(
        "foreign_tool_call_ids_are_dropped",
        {"input": FOREIGN_TOOL_CALL_REPLAY_INPUT, "tools": [dict(WEATHER_TOOL)]},
    ),
    Case("azure_default_v1_surface", {"model": AZURE_MODEL, "input": "hi"}),
    Case("azure_dated_api_version", {"model": AZURE_MODEL, "input": "hi", "api_version": "2025-03-01-preview"}),
    Case(
        "azure_replay_without_status",
        {
            "model": AZURE_MODEL,
            "input": AZURE_STATUS_REPLAY_INPUT,
            "tools": [{"type": "function", "function": dict(WEATHER_TOOL_FUNCTION)}],
        },
    ),
    Case(
        "azure_forwarded_api_key_header",
        {"model": AZURE_MODEL, "input": "hi", "api_key": None, "extra_headers": {"api-key": "caller-key"}},
    ),
    Case("azure_o_series_stays_on_python", {"model": "azure/o_series/my-o3", "input": "hi"}, falls_back=True),
    Case(
        "tool_schema_repair_stays_on_python",
        {
            "input": "Weather in Paris?",
            "tools": [{**WEATHER_TOOL, "parameters": dict(PATTERNED_PARAMETERS)}],
        },
        falls_back=True,
    ),
)


def _kwargs(case: Case, server: RecordingServer, recorder: RecordingLogger) -> dict[str, object]:
    return {"model": MODEL, "api_key": "sk-test", "api_base": server.base_url, "callbacks": [recorder], **case.kwargs}


def _response(result: object) -> Mapping[str, JsonValue]:
    assert isinstance(result, ResponsesAPIResponse)
    return _OBJECT.validate_python(result.model_dump(mode="json", exclude_none=True))


def _hidden_params(result: object) -> Mapping[str, object]:
    assert isinstance(result, ResponsesAPIResponse)
    hidden: Final = _LOGGED.validate_python(result._hidden_params)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # the hidden params callers read
    return MappingProxyType({name: hidden.get(name) for name in _COMPARED_HIDDEN_PARAMS})


async def _leg(case: Case, server: RecordingServer, probe: FallbackProbe | None) -> Leg:
    with isolated_callback_registries():
        return await _isolated_leg(case, server, probe)


async def _isolated_leg(case: Case, server: RecordingServer, probe: FallbackProbe | None) -> Leg:
    recorder: Final = RecordingLogger()
    server.enqueue(ResponseSpec(body=dict(case.upstream)))
    request_count: Final = len(server.requests)
    kwargs: Final = _kwargs(case, server, recorder)
    fallbacks_before: Final = 0 if probe is None else probe.calls
    if probe is None:
        result = await PYTHON_ARESPONSES(**kwargs)
    else:
        result = await PUBLIC_ARESPONSES(**kwargs)
    await drain_logging()
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(server.requests) == request_count + 1
    sent: Final = server.requests[-1]
    standard: Final = _LOGGED.validate_python(_LOGGED.validate_python(success[0].kwargs)["standard_logging_object"])
    return Leg(
        request_path=sent.path,
        request_body=_OBJECT.validate_python(sent.body),
        request_headers=MappingProxyType(
            {name: value for name, value in sent.headers.items() if name in _COMPARED_HEADERS}
        ),
        response=_response(result),
        logged=MappingProxyType({name: standard.get(name) for name in _COMPARED_LOG_FIELDS}),
        hooks=recorder.names,
        hidden_params=_hidden_params(result),
        fell_back=probe is not None and probe.calls > fallbacks_before,
    )


@pytest.fixture
def parity_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.expected_requests = 2
    return recording_server


@pytest.mark.asyncio
@pytest.mark.parametrize(("case", "aspect"), tuple(param for case in CASES for param in case.params()))
async def test_native_responses_match_python(
    case: Case, aspect: str, parity_server: RecordingServer, native_responses: FallbackProbe
) -> None:
    python: Final = await _leg(case, parity_server, probe=None)
    native: Final = await _leg(case, parity_server, probe=native_responses)
    match aspect:
        case "dispatch":
            assert native.fell_back == case.falls_back
        case "request":
            assert (native.request_path, native.request_body, native.request_headers) == (
                python.request_path,
                python.request_body,
                python.request_headers,
            )
        case "response":
            assert native.response == python.response
        case "logged":
            assert (native.logged, native.hidden_params) == (python.logged, python.hidden_params)
        case _:
            assert native.hooks == python.hooks
