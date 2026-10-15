"""Native Responses calls must be indistinguishable from the Python implementation.

Each case sends one public `litellm.aresponses` call through Python and through the native route against the same
recorded OpenAI response, then compares what reached the provider, what the caller got back, and what the success
callback logged. The native leg fails instead of silently falling back to Python
"""

from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, cast

import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.responses import dispatch
from litellm.responses import main as python_responses
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.types.llms.openai import ResponsesAPIResponse
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.isolation import isolated_callback_registries, rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension

MODEL: Final = "openai/gpt-6.1-sol"
_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_LOGGED: Final = TypeAdapter(dict[str, object])
_COMPARED_HEADERS: Final = ("authorization", "content-type", "openai-organization", "openai-project")
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
            {
                "id": "msg_0a1b2c3d4e5f",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "Paris"}],
            },
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


REASONING_REPLAY_INPUT: Final[list[JsonValue]] = [
    {"type": "reasoning", "id": "rs_0a1b2c3d4e5f", "summary": [], "status": None},
    {"role": "user", "content": "And Germany?"},
]

_PYTHON_ARESPONSES: Final = cast(Callable[..., Awaitable[object]], python_responses.aresponses)  # noqa: TID251  # the Python baseline under comparison
_PUBLIC_ARESPONSES: Final = cast(Callable[..., Awaitable[object]], litellm.aresponses)


@dataclass(frozen=True, slots=True)
class Leg:
    request_body: Mapping[str, JsonValue]
    request_headers: Mapping[str, str]
    response: Mapping[str, JsonValue]
    logged: Mapping[str, object]
    fell_back: bool


ASPECTS: Final = ("dispatch", "request", "response", "logged")


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    kwargs: Mapping[str, object]
    upstream: Mapping[str, JsonValue] = TEXT_RESPONSE
    gaps: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def params(self) -> Iterator[object]:
        for aspect in ASPECTS:
            gap = self.gaps.get(aspect)
            marks = () if gap is None else (pytest.mark.xfail(reason=gap, strict=True),)
            yield pytest.param(self, aspect, id=f"{self.name}-{aspect}", marks=marks)


ID_ENCODING_GAP: Final = MappingProxyType(
    {"response": "native responses return the provider's raw response id instead of LiteLLM's encoded resp_ id"}
)


def _gaps(**request_gaps: str) -> Mapping[str, str]:
    return MappingProxyType({**ID_ENCODING_GAP, **request_gaps})


CASES: Final = (
    Case("plain_text", {"input": "What is the capital of France?"}, gaps=_gaps()),
    Case(
        "instructions_and_input_items",
        {
            "instructions": "Answer in one word",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Capital of France?"}]}],
        },
        gaps=_gaps(),
    ),
    Case(
        "function_tool",
        {"input": "Weather in Paris?", "tools": [dict(WEATHER_TOOL)], "tool_choice": "auto"},
        upstream=FUNCTION_CALL_RESPONSE,
        gaps=_gaps(),
    ),
    Case(
        "encoded_previous_response_id",
        {
            "input": [{"type": "function_call_output", "call_id": "call_6f5e4d3c2b1a", "output": "18C and sunny"}],
            "previous_response_id": ResponsesAPIRequestUtils._build_responses_api_response_id(  # pyright: ignore[reportPrivateUsage]  # the public encoder needs a response object
                custom_llm_provider="openai", model_id=None, response_id="resp_6f5e4d3c2b1a"
            ),
        },
        gaps=_gaps(),
    ),
    Case(
        "warm_up_probe_below_minimum_max_output_tokens",
        {"input": "hi", "max_output_tokens": 1},
        gaps=_gaps(request="native requests send max_output_tokens below OpenAI's minimum of 16"),
    ),
    Case(
        "gpt_5_family_drops_temperature_while_reasoning",
        {"input": "hi", "temperature": 0.2, "reasoning": {"effort": "high"}, "drop_params": True},
        gaps=MappingProxyType(
            {"dispatch": "native responses decline drop_params instead of applying the GPT-5 sampling rules"}
        ),
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
        gaps=_gaps(),
    ),
    Case(
        "reasoning_replay_without_status",
        {
            "input": REASONING_REPLAY_INPUT,
        },
        gaps=_gaps(request="native requests forward reasoning items with status null"),
    ),
    Case(
        "tool_parameters_sent_as_json_string",
        {
            "input": "Weather in Paris?",
            "tools": [{**WEATHER_TOOL, "parameters": '{"type":"object","properties":{"city":{"type":"string"}}}'}],
        },
        gaps=_gaps(request="native requests forward tool parameters encoded as a JSON string"),
    ),
)


@dataclass(slots=True)
class FallbackProbe:
    calls: int = 0


@pytest.fixture
def native_responses() -> Iterator[FallbackProbe]:
    probe: Final = FallbackProbe()
    python_aresponses: Final = dispatch._PYTHON_ARESPONSES  # pyright: ignore[reportPrivateUsage]  # the fallback under observation

    async def python_fallback(*args: object, **kwargs: object) -> object:
        probe.calls += 1
        return await python_aresponses(*args, **kwargs)

    with (
        rebound(catalog, "RULES", (RouteRule(Route.RESPONSES, Rollout.RUST_OPT_IN), *catalog.RULES)),
        rebound(dispatch, "_PYTHON_ARESPONSES", python_fallback),
    ):
        yield probe


def _kwargs(case: Case, server: RecordingServer, recorder: RecordingLogger) -> dict[str, object]:
    return {"model": MODEL, "api_key": "sk-test", "api_base": server.base_url, "callbacks": [recorder], **case.kwargs}


def _response(result: object) -> Mapping[str, JsonValue]:
    assert isinstance(result, ResponsesAPIResponse)
    return _OBJECT.validate_python(result.model_dump(mode="json", exclude_none=True))


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
        result = await _PYTHON_ARESPONSES(**kwargs)
    else:
        result = await _PUBLIC_ARESPONSES(**kwargs)
    await drain_logging()
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(server.requests) == request_count + 1
    sent: Final = server.requests[-1]
    standard: Final = _LOGGED.validate_python(_LOGGED.validate_python(success[0].kwargs)["standard_logging_object"])
    return Leg(
        request_body=_OBJECT.validate_python(sent.body),
        request_headers=MappingProxyType(
            {name: value for name, value in sent.headers.items() if name in _COMPARED_HEADERS}
        ),
        response=_response(result),
        logged=MappingProxyType({name: standard.get(name) for name in _COMPARED_LOG_FIELDS}),
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
            assert not native.fell_back
        case "request":
            assert (native.request_body, native.request_headers) == (python.request_body, python.request_headers)
        case "response":
            assert native.response == python.response
        case _:
            assert native.logged == python.logged
