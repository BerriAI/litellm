"""Tests for ``LangfuseOpenTelemetryV2``: the root observation's input and output are stamped from the
request-task hooks, while the root span is still recording, so Langfuse can show them on the trace."""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Final

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

import litellm  # noqa: E402
from litellm.caching.dual_cache import DualCache  # noqa: E402
from litellm.integrations.otel.logger import build_otel_v2_logger  # noqa: E402
from litellm.integrations.otel.model.config import OpenTelemetryV2Config, is_otel_v2_enabled  # noqa: E402
from litellm.integrations.otel.model.spans import LITELLM_PROXY_REQUEST_SPAN_NAME, SpanRole  # noqa: E402
from litellm.integrations.otel.plumbing import context as otel_context  # noqa: E402
from litellm.integrations.otel.plumbing import providers  # noqa: E402
from litellm.integrations.otel.plumbing.context import set_request_root_span  # noqa: E402
from litellm.litellm_core_utils.litellm_logging import _maybe_construct_otel_v2  # noqa: E402
from litellm.proxy._types import UserAPIKeyAuth  # noqa: E402
from litellm.proxy.utils import ProxyLogging  # noqa: E402
from litellm.types.llms.openai import (  # noqa: E402
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import (  # noqa: E402
    Choices,
    Delta,
    Embedding,
    EmbeddingResponse,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

INPUT_ATTR: Final = "langfuse.observation.input"
OUTPUT_ATTR: Final = "langfuse.observation.output"
CHAT_DATA: Final = {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "ping"}]}


@pytest.fixture(autouse=True)
def _reset_request_root_span():
    otel_context._request_root_span.set(None)
    yield
    otel_context._request_root_span.set(None)


def _logger(*, capture: str = "span_only", mappers: Sequence[str] = ("genai", "langfuse")):
    cfg = OpenTelemetryV2Config(exporter="in_memory", mapper_names=list(mappers), capture_message_content=capture)
    exporter = InMemorySpanExporter()
    tracer_provider = providers.build_tracer_provider(cfg, exporter=exporter)
    return build_otel_v2_logger(config=cfg, tracer_provider=tracer_provider), exporter


def _start_root(logger):
    root = logger._emitter.start_span(SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME)
    set_request_root_span(root)
    return root


def _root_attrs(exporter):
    by_name = {span.name: span for span in exporter.get_finished_spans()}
    return dict(by_name[LITELLM_PROXY_REQUEST_SPAN_NAME].attributes or {})


def _run_request(logger, data: dict, call_type: str, response: object):
    root = _start_root(logger)
    asyncio.run(logger.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, call_type))
    asyncio.run(logger.async_post_call_success_hook(data=data, user_api_key_dict=UserAPIKeyAuth(), response=response))
    root.end()


async def _relay(logger, chunks: Sequence[object], data: dict) -> list[object]:
    async def source() -> AsyncIterator[object]:
        for chunk in chunks:
            yield chunk

    return [chunk async for chunk in logger.async_post_call_streaming_iterator_hook(UserAPIKeyAuth(), source(), data)]


def _run_stream(logger, data: dict, chunks: Sequence[object]) -> list[object]:
    root = _start_root(logger)
    asyncio.run(logger.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, "acompletion"))
    relayed = asyncio.run(_relay(logger, chunks, data))
    root.end()
    return relayed


def _chat_chunk(content: str | None, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-1",
        created=1,
        model="gpt-5.4-mini",
        choices=[StreamingChoices(index=0, delta=Delta(content=content), finish_reason=finish_reason)],
    )


def _responses_api_response() -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_1",
        created_at=1,
        output=[
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "pong", "annotations": []}],
            }
        ],
    )


def _anthropic_sse_frames() -> tuple[bytes, ...]:
    events = (
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "po"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ng"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    )
    return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def test_chat_request_stamps_root_observation_input_and_output():
    logger, exporter = _logger()
    response = ModelResponse(choices=[Choices(message=Message(role="assistant", content="pong"))])

    _run_request(logger, CHAT_DATA, "acompletion", response)

    attrs = _root_attrs(exporter)
    assert json.loads(attrs[INPUT_ATTR]) == [{"role": "user", "content": "ping"}]
    output = json.loads(attrs[OUTPUT_ATTR])
    assert [(turn["role"], turn["content"]) for turn in output] == [("assistant", "pong")]


def test_responses_request_folds_instructions_into_input_and_stamps_output_items():
    logger, exporter = _logger()
    data = {"model": "gpt-5.4-mini", "instructions": "be terse", "input": "ping"}

    _run_request(logger, data, "aresponses", _responses_api_response())

    attrs = _root_attrs(exporter)
    assert json.loads(attrs[INPUT_ATTR]) == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "ping"},
    ]
    output = json.loads(attrs[OUTPUT_ATTR])
    assert output[0]["role"] == "assistant"
    assert output[0]["content"][0]["text"] == "pong"


def test_anthropic_messages_request_folds_system_into_input_and_stamps_content_blocks():
    logger, exporter = _logger()
    data = {"model": "claude-sonnet-4-5", "system": "be terse", "messages": [{"role": "user", "content": "ping"}]}
    response = {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "pong"}]}

    _run_request(logger, data, "aanthropic_messages", response)

    attrs = _root_attrs(exporter)
    assert json.loads(attrs[INPUT_ATTR]) == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "ping"},
    ]
    assert json.loads(attrs[OUTPUT_ATTR]) == [{"role": "assistant", "content": [{"type": "text", "text": "pong"}]}]


def test_chat_stream_relays_chunks_untouched_and_stamps_assembled_output():
    logger, exporter = _logger()
    chunks = (_chat_chunk("po"), _chat_chunk("ng"), _chat_chunk(None, finish_reason="stop"))

    relayed = _run_stream(logger, CHAT_DATA, chunks)

    assert [id(chunk) for chunk in relayed] == [id(chunk) for chunk in chunks]
    output = json.loads(_root_attrs(exporter)[OUTPUT_ATTR])
    assert [(turn["role"], turn["content"]) for turn in output] == [("assistant", "pong")]


def test_responses_stream_stamps_output_from_the_completed_event():
    logger, exporter = _logger()
    completed = ResponseCompletedEvent(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=_responses_api_response()
    )
    chunks = ({"type": "response.created"}, {"type": "response.output_text.delta", "delta": "pong"}, completed)

    relayed = _run_stream(logger, {"model": "gpt-5.4-mini", "input": "ping"}, chunks)

    assert relayed == list(chunks)
    output = json.loads(_root_attrs(exporter)[OUTPUT_ATTR])
    assert output[0]["content"][0]["text"] == "pong"


def test_anthropic_sse_stream_stamps_output_from_the_assembled_frames():
    logger, exporter = _logger()
    frames = _anthropic_sse_frames()

    relayed = _run_stream(
        logger, {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "ping"}]}, frames
    )

    assert relayed == list(frames)
    output = json.loads(_root_attrs(exporter)[OUTPUT_ATTR])
    assert [(turn["role"], turn["content"]) for turn in output] == [("assistant", "pong")]


def test_root_observation_io_survives_the_root_ending_before_the_success_callback():
    logger, exporter = _logger()
    response = ModelResponse(choices=[Choices(message=Message(role="assistant", content="pong"))])
    root = _start_root(logger)
    asyncio.run(logger.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), CHAT_DATA, "acompletion"))
    logger.log_pre_api_call(
        model="gpt-5.4-mini",
        messages=[],
        kwargs={"litellm_call_id": "call_1", "litellm_params": {"metadata": {}}},
    )
    asyncio.run(
        logger.async_post_call_success_hook(data=CHAT_DATA, user_api_key_dict=UserAPIKeyAuth(), response=response)
    )
    root.end()

    payload = {
        "call_type": "acompletion",
        "custom_llm_provider": "openai",
        "model": "gpt-5.4-mini",
        "messages": CHAT_DATA["messages"],
        "response": response.model_dump(),
        "status": "success",
        "litellm_call_id": "call_1",
        "metadata": {},
        "hidden_params": {},
    }
    asyncio.run(
        logger.async_log_success_event(
            {"standard_logging_object": payload, "litellm_params": {"metadata": {}}}, response, None, None
        )
    )

    attrs = _root_attrs(exporter)
    assert INPUT_ATTR in attrs and OUTPUT_ATTR in attrs
    generation = next(span for span in exporter.get_finished_spans() if span.name != LITELLM_PROXY_REQUEST_SPAN_NAME)
    assert OUTPUT_ATTR in dict(generation.attributes or {})


def test_root_input_is_the_request_as_the_pre_call_chain_left_it():
    logger, exporter = _logger()
    raw = {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "my ssn is 123-45-6789"}]}
    masked = {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "my ssn is [REDACTED]"}]}
    response = ModelResponse(choices=[Choices(message=Message(role="assistant", content="noted"))])
    root = _start_root(logger)
    asyncio.run(logger.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), raw, "acompletion"))
    asyncio.run(logger.async_post_call_success_hook(data=masked, user_api_key_dict=UserAPIKeyAuth(), response=response))
    root.end()

    assert json.loads(_root_attrs(exporter)[INPUT_ATTR]) == masked["messages"]


def test_root_already_ended_is_left_alone():
    logger, exporter = _logger()
    response = ModelResponse(choices=[Choices(message=Message(role="assistant", content="pong"))])
    root = _start_root(logger)
    root.end()

    asyncio.run(
        logger.async_post_call_success_hook(data=CHAT_DATA, user_api_key_dict=UserAPIKeyAuth(), response=response)
    )

    attrs = _root_attrs(exporter)
    assert INPUT_ATTR not in attrs and OUTPUT_ATTR not in attrs


def test_responses_without_a_message_body_stamp_neither_input_nor_output():
    logger, exporter = _logger()
    embedding = EmbeddingResponse(model="e", data=[Embedding(embedding=[0.1], index=0, object="embedding")])

    _run_request(logger, {"model": "e", "input": "ping"}, "aembedding", embedding)

    attrs = _root_attrs(exporter)
    assert INPUT_ATTR not in attrs and OUTPUT_ATTR not in attrs


def test_unrenderable_output_never_raises_into_the_request():
    logger, exporter = _logger()

    _run_request(logger, CHAT_DATA, "acompletion", object())

    attrs = _root_attrs(exporter)
    assert INPUT_ATTR not in attrs and OUTPUT_ATTR not in attrs


@pytest.mark.parametrize(
    ("capture", "mappers"),
    [("no_content", ("genai", "langfuse")), ("span_only", ("genai",))],
)
def test_factory_keeps_the_base_logger_unless_langfuse_content_capture_is_on(capture, mappers):
    logger, exporter = _logger(capture=capture, mappers=mappers)

    _run_request(logger, CHAT_DATA, "acompletion", ModelResponse())
    attrs = _root_attrs(exporter)
    assert INPUT_ATTR not in attrs and OUTPUT_ATTR not in attrs


@pytest.mark.parametrize(
    ("capture", "mappers", "relays_streams"),
    [
        ("span_only", ("genai", "langfuse"), True),
        ("no_content", ("genai", "langfuse"), False),
        ("span_only", ("genai",), False),
    ],
)
def test_only_langfuse_content_capture_takes_proxy_streams_off_the_fast_path(
    monkeypatch, capture, mappers, relays_streams
):
    logger, _ = _logger(capture=capture, mappers=mappers)
    monkeypatch.setattr(litellm, "callbacks", [logger])

    assert ProxyLogging._callback_capabilities().has_iterator_override is relays_streams


def test_langfuse_otel_preset_builds_a_logger_that_stamps_the_root(monkeypatch):
    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "span_only")
    is_otel_v2_enabled.cache_clear()

    loggers: list = []
    try:
        built = _maybe_construct_otel_v2("langfuse_otel", loggers)
        assert built is not None
        assert _maybe_construct_otel_v2("langfuse_otel", loggers) is built
        root = _start_root(built)
        response = ModelResponse(choices=[Choices(message=Message(role="assistant", content="pong"))])
        asyncio.run(
            built.async_post_call_success_hook(data=CHAT_DATA, user_api_key_dict=UserAPIKeyAuth(), response=response)
        )
        attrs = dict(root.attributes or {})
        assert INPUT_ATTR in attrs and OUTPUT_ATTR in attrs
    finally:
        is_otel_v2_enabled.cache_clear()
