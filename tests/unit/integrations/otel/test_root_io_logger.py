"""Tests for the OpenInference OTel v2 logger: the request root span carries the request's input and output, which
Arize AX and Phoenix show as the trace's, when this logger's provider started that root."""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Final

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

import litellm  # noqa: E402
from litellm.caching.dual_cache import DualCache  # noqa: E402
from litellm.integrations.custom_logger import CustomLogger  # noqa: E402
from litellm.integrations.otel.langfuse_logger import LangfuseContentOpenTelemetryV2  # noqa: E402
from litellm.integrations.otel.logger import OpenTelemetryV2, build_otel_v2_logger  # noqa: E402
from litellm.integrations.otel.model.config import OpenTelemetryV2Config, is_otel_v2_enabled  # noqa: E402
from litellm.integrations.otel.model.destination import OtelDestination  # noqa: E402
from litellm.integrations.otel.model.spans import LITELLM_PROXY_REQUEST_SPAN_NAME, SpanRole  # noqa: E402
from litellm.integrations.otel.plumbing import context as otel_context  # noqa: E402
from litellm.integrations.otel.plumbing import providers  # noqa: E402
from litellm.integrations.otel.plumbing.context import set_request_root_span  # noqa: E402
from litellm.integrations.otel.root_io_logger import OpenInferenceRootIOOpenTelemetryV2  # noqa: E402
from litellm.litellm_core_utils.litellm_logging import Logging, _maybe_construct_otel_v2  # noqa: E402
from litellm.litellm_core_utils.logging_utils import truncate_base64_in_messages  # noqa: E402
from litellm.proxy._types import UserAPIKeyAuth  # noqa: E402
from litellm.proxy.utils import ProxyLogging  # noqa: E402
from litellm.types.llms.openai import ResponsesAPIResponse  # noqa: E402
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

OPENINFERENCE: Final = ("genai", "openinference")
ROOT_IO_ATTRS: Final = (
    "openinference.span.kind",
    "input.value",
    "input.mime_type",
    "output.value",
    "output.mime_type",
)
CHAT_DATA: Final = {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "ping"}]}
PONG: Final = ModelResponse(choices=[Choices(message=Message(role="assistant", content="pong"))])


@pytest.fixture(autouse=True)
def _reset_request_root_span():
    otel_context._request_root_span.set(None)
    yield
    otel_context._request_root_span.set(None)


def _logger(
    *,
    capture: str = "span_only",
    mappers: Sequence[str] = OPENINFERENCE,
    settings: Mapping[str, object] = MappingProxyType({}),
):
    cfg = OpenTelemetryV2Config(exporter="in_memory", mapper_names=list(mappers), capture_message_content=capture)
    exporter = InMemorySpanExporter()
    tracer_provider = providers.build_tracer_provider(cfg, exporter=exporter)
    return build_otel_v2_logger(config=cfg, tracer_provider=tracer_provider, settings=settings), exporter


def _start_root(logger):
    root = logger._emitter.start_span(SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME, None)
    set_request_root_span(root)
    return root


def _root_attrs(exporter) -> dict:
    by_name = {span.name: span for span in exporter.get_finished_spans()}
    return dict(by_name[LITELLM_PROXY_REQUEST_SPAN_NAME].attributes or {})


def _run_request(logger, data: dict, call_type: str, response: object, root_owner=None) -> None:
    root = _start_root(root_owner or logger)
    asyncio.run(logger.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, call_type))
    asyncio.run(logger.async_post_call_success_hook(data=data, user_api_key_dict=UserAPIKeyAuth(), response=response))
    root.end()


async def _relay(logger, chunks: Sequence[object], data: dict) -> list[object]:
    async def source() -> AsyncIterator[object]:
        for chunk in chunks:
            yield chunk

    return [chunk async for chunk in logger.async_post_call_streaming_iterator_hook(UserAPIKeyAuth(), source(), data)]


def _chat_chunk(content: str | None, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-1",
        created=1,
        model="gpt-5.4-mini",
        choices=[StreamingChoices(index=0, delta=Delta(content=content), finish_reason=finish_reason)],
    )


def test_chat_request_stamps_the_root_kind_input_and_output():
    logger, exporter = _logger()

    _run_request(logger, CHAT_DATA, "acompletion", PONG)

    attrs: Final = _root_attrs(exporter)
    assert attrs["openinference.span.kind"] == "CHAIN"
    assert json.loads(attrs["input.value"]) == [{"role": "user", "content": "ping"}]
    assert attrs["input.mime_type"] == "application/json"
    assert [(turn["role"], turn["content"]) for turn in json.loads(attrs["output.value"])] == [("assistant", "pong")]
    assert attrs["output.mime_type"] == "application/json"


def test_responses_request_folds_instructions_into_the_root_input():
    logger, exporter = _logger()
    data: Final = {"model": "gpt-5.4-mini", "instructions": "be terse", "input": "ping"}
    response: Final = ResponsesAPIResponse(
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

    _run_request(logger, data, "aresponses", response)

    attrs: Final = _root_attrs(exporter)
    assert json.loads(attrs["input.value"]) == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "ping"},
    ]
    assert json.loads(attrs["output.value"])[0]["content"][0]["text"] == "pong"


def test_chat_stream_relays_chunks_untouched_and_stamps_the_assembled_output():
    logger, exporter = _logger()
    chunks: Final = (_chat_chunk("po"), _chat_chunk("ng"), _chat_chunk(None, finish_reason="stop"))
    root = _start_root(logger)
    asyncio.run(logger.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), CHAT_DATA, "acompletion"))

    relayed: Final = asyncio.run(_relay(logger, chunks, CHAT_DATA))
    root.end()

    assert [id(chunk) for chunk in relayed] == [id(chunk) for chunk in chunks]
    attrs: Final = _root_attrs(exporter)
    assert json.loads(attrs["input.value"]) == [{"role": "user", "content": "ping"}]
    assert [(turn["role"], turn["content"]) for turn in json.loads(attrs["output.value"])] == [("assistant", "pong")]


def test_a_root_another_callback_started_is_left_alone():
    owner, owner_exporter = _logger(mappers=("genai",))
    logger, _ = _logger()

    _run_request(logger, CHAT_DATA, "acompletion", PONG, root_owner=owner)

    assert set(ROOT_IO_ATTRS).isdisjoint(_root_attrs(owner_exporter))


def test_root_already_ended_is_left_alone():
    logger, exporter = _logger()
    root = _start_root(logger)
    root.end()

    asyncio.run(logger.async_post_call_success_hook(data=CHAT_DATA, user_api_key_dict=UserAPIKeyAuth(), response=PONG))

    assert set(ROOT_IO_ATTRS).isdisjoint(_root_attrs(exporter))


@pytest.mark.parametrize(
    ("data", "call_type", "response"),
    [
        (
            {"model": "e", "input": "ping"},
            "aembedding",
            EmbeddingResponse(model="e", data=[Embedding(embedding=[0.1], index=0, object="embedding")]),
        ),
        (CHAT_DATA, "acompletion", object()),
    ],
    ids=["no-message-body", "unrenderable"],
)
def test_a_response_without_a_message_stamps_nothing(data: dict, call_type: str, response: object):
    logger, exporter = _logger()

    _run_request(logger, data, call_type, response)

    assert set(ROOT_IO_ATTRS).isdisjoint(_root_attrs(exporter))


def test_root_io_requires_content_capture():
    logger, exporter = _logger(capture="no_content")

    _run_request(logger, CHAT_DATA, "acompletion", PONG)

    assert set(ROOT_IO_ATTRS).isdisjoint(_root_attrs(exporter))


def test_content_on_destination_stamps_the_root_when_global_capture_is_off():
    logger, exporter = _logger(capture="no_content")
    destination: Final = OtelDestination(
        endpoint="https://team.example.com/v1/traces",
        callback_name="arize_phoenix",
        capture_message_content="span_only",
    )
    token: Final = otel_context.set_request_destinations((destination,))
    try:
        _run_request(logger, CHAT_DATA, "acompletion", PONG)
    finally:
        otel_context.reset_request_destinations(token)

    assert json.loads(_root_attrs(exporter)["input.value"]) == [{"role": "user", "content": "ping"}]


def _logging_obj(dynamic_params: dict, *, no_log: bool = False) -> Logging:
    logging_obj = Logging(
        model="gpt-5.4-mini",
        messages=CHAT_DATA["messages"],
        stream=False,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="root-io",
        function_id="fn",
    )
    logging_obj.update_environment_variables(litellm_params={"metadata": {}, "no-log": no_log}, optional_params={})
    logging_obj.model_call_details["standard_callback_dynamic_params"] = dynamic_params
    return logging_obj


def test_root_io_is_withheld_when_message_logging_is_turned_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    logger, exporter = _logger()

    _run_request(logger, CHAT_DATA, "acompletion", PONG)

    assert set(ROOT_IO_ATTRS).isdisjoint(_root_attrs(exporter))


def test_root_io_is_withheld_when_the_request_header_asks_for_redaction():
    logger, exporter = _logger()
    data: Final = {**CHAT_DATA, "metadata": {"headers": {"x-litellm-enable-message-redaction": "true"}}}
    chunks: Final = (_chat_chunk("po"), _chat_chunk("ng"), _chat_chunk(None, finish_reason="stop"))
    root = _start_root(logger)

    relayed: Final = asyncio.run(_relay(logger, chunks, data))
    root.end()

    assert [id(chunk) for chunk in relayed] == [id(chunk) for chunk in chunks]
    assert set(ROOT_IO_ATTRS).isdisjoint(_root_attrs(exporter))


@pytest.mark.parametrize(("turn_off", "stamped"), [(True, False), (False, True)], ids=["redacted", "not-redacted"])
def test_root_io_follows_the_request_redaction_setting_on_its_logging_object(turn_off: bool, stamped: bool):
    logger, exporter = _logger()
    data: Final = {**CHAT_DATA, "litellm_logging_obj": _logging_obj({"turn_off_message_logging": turn_off})}

    _run_request(logger, data, "acompletion", PONG)

    assert ("input.value" in _root_attrs(exporter)) is stamped


@pytest.mark.parametrize(
    "settings",
    [{"message_logging": False}, {"turn_off_message_logging": True}],
    ids=["message-logging-false", "turn-off-message-logging"],
)
def test_root_io_is_withheld_for_a_logger_with_message_logging_off(settings: dict):
    logger, exporter = _logger(settings=settings)

    _run_request(logger, CHAT_DATA, "acompletion", PONG)

    assert set(ROOT_IO_ATTRS).isdisjoint(_root_attrs(exporter))


def _no_log_request(on_logging_object: bool) -> dict:
    return (
        {**CHAT_DATA, "litellm_logging_obj": _logging_obj({}, no_log=True)}
        if on_logging_object
        else {**CHAT_DATA, "no-log": True}
    )


@pytest.mark.parametrize("on_logging_object", [False, True], ids=["request-body", "logging-object"])
def test_root_io_is_withheld_for_a_no_log_request(on_logging_object: bool):
    logger, exporter = _logger()

    _run_request(logger, _no_log_request(on_logging_object), "acompletion", PONG)

    assert set(ROOT_IO_ATTRS).isdisjoint(_root_attrs(exporter))


@pytest.mark.parametrize("on_logging_object", [False, True], ids=["request-body", "logging-object"])
def test_root_io_is_stamped_for_a_no_log_request_when_the_no_log_param_is_disabled(
    on_logging_object: bool, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "global_disable_no_log_param", True)
    logger, exporter = _logger()

    _run_request(logger, _no_log_request(on_logging_object), "acompletion", PONG)

    assert json.loads(_root_attrs(exporter)["input.value"]) == [{"role": "user", "content": "ping"}]


class _SkipsLoggerInit(CustomLogger):
    def __init__(self) -> None:
        pass


@pytest.mark.parametrize(
    ("sibling", "stamped"),
    [(CustomLogger(message_logging=False), False), (_SkipsLoggerInit(), True)],
    ids=["message-logging-false", "never-set-message-logging"],
)
def test_root_io_follows_another_callback_with_message_logging_off(
    sibling: CustomLogger, stamped: bool, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "callbacks", [sibling])
    logger, exporter = _logger()

    _run_request(logger, CHAT_DATA, "acompletion", PONG)

    assert ("input.value" in _root_attrs(exporter)) is stamped


def test_root_input_cuts_an_image_data_uri_as_the_logged_messages_do():
    logger, exporter = _logger()
    messages: Final = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "iVBORw0KGgo" * 100}},
            ],
        }
    ]

    _run_request(logger, {"model": "gpt-5.4-mini", "messages": messages}, "acompletion", PONG)

    root_input: Final = json.loads(_root_attrs(exporter)["input.value"])
    assert root_input == truncate_base64_in_messages(messages)
    assert root_input != messages


@pytest.mark.parametrize(
    ("mappers", "expected"),
    [
        (OPENINFERENCE, OpenInferenceRootIOOpenTelemetryV2),
        (("genai", "openinference", "weave"), OpenTelemetryV2),
        (("genai", "langfuse"), LangfuseContentOpenTelemetryV2),
        (("genai",), OpenTelemetryV2),
    ],
    ids=["arize-phoenix", "weave", "langfuse", "genai"],
)
def test_logger_class_follows_the_configured_vocabularies(mappers: tuple[str, ...], expected: type) -> None:
    logger, _ = _logger(mappers=mappers)

    assert type(logger) is expected


@pytest.mark.parametrize(
    ("mappers", "relays_streams"),
    [(OPENINFERENCE, True), (("genai", "openinference", "weave"), False)],
    ids=["arize-phoenix", "weave"],
)
def test_openinference_logger_relays_streams(
    monkeypatch: pytest.MonkeyPatch, mappers: tuple[str, ...], relays_streams: bool
) -> None:
    logger, _ = _logger(mappers=mappers)
    monkeypatch.setattr(litellm, "callbacks", [logger])

    assert ProxyLogging._callback_capabilities().has_iterator_override is relays_streams


def test_openinference_logger_keeps_the_pre_call_hook_that_seeds_metadata_baggage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger, _ = _logger()
    monkeypatch.setattr(litellm, "callbacks", [logger])

    assert ProxyLogging._callback_capabilities().has_pre_call_override is True


class _MasksReply(CustomLogger):
    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        async for chunk in response:
            choice = chunk.choices[0]
            yield _chat_chunk("<PERSON>" if choice.delta.content else None, finish_reason=choice.finish_reason)


async def _serve(chunks: Sequence[ModelResponseStream], data: dict) -> list[object]:
    async def source() -> AsyncIterator[ModelResponseStream]:
        for chunk in chunks:
            yield chunk

    return [
        chunk
        async for chunk in ProxyLogging(user_api_key_cache=DualCache()).async_post_call_streaming_iterator_hook(
            response=source(),
            user_api_key_dict=UserAPIKeyAuth(request_route="/chat/completions"),
            request_data=data,
        )
    ]


@pytest.mark.parametrize(
    ("mappers", "output_key"),
    [(OPENINFERENCE, "output.value"), (("genai", "langfuse"), "langfuse.observation.output")],
    ids=["arize-phoenix", "langfuse"],
)
def test_stream_root_output_is_what_a_later_masking_hook_served(
    monkeypatch: pytest.MonkeyPatch, mappers: tuple[str, ...], output_key: str
) -> None:
    logger, exporter = _logger(mappers=mappers)
    monkeypatch.setattr(litellm, "callbacks", [logger, _MasksReply()])
    root = _start_root(logger)

    served: Final = asyncio.run(
        _serve((_chat_chunk("John Smith"), _chat_chunk(None, finish_reason="stop")), {**CHAT_DATA})
    )
    root.end()

    assert [chunk.choices[0].delta.content for chunk in served] == ["<PERSON>", None]
    output: Final = json.loads(_root_attrs(exporter)[output_key])
    assert [(turn["role"], turn["content"]) for turn in output] == [("assistant", "<PERSON>")], output


@pytest.mark.parametrize(
    ("callback", "environment"),
    [
        ("arize_phoenix", {"PHOENIX_COLLECTOR_HTTP_ENDPOINT": "http://127.0.0.1:6006/v1/traces"}),
        ("arize", {"ARIZE_SPACE_ID": "space", "ARIZE_API_KEY": "key"}),
    ],
)
def test_arize_and_phoenix_presets_build_a_logger_that_stamps_the_root(
    monkeypatch: pytest.MonkeyPatch, callback: str, environment: dict[str, str]
) -> None:
    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "span_only")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    is_otel_v2_enabled.cache_clear()

    loggers: list = []
    try:
        built = _maybe_construct_otel_v2(callback, loggers)
        assert isinstance(built, OpenInferenceRootIOOpenTelemetryV2)
        root = _start_root(built)
        asyncio.run(
            built.async_post_call_success_hook(data=CHAT_DATA, user_api_key_dict=UserAPIKeyAuth(), response=PONG)
        )
        attrs = dict(root.attributes or {})
        assert json.loads(attrs["input.value"]) == [{"role": "user", "content": "ping"}]
        assert attrs["openinference.span.kind"] == "CHAIN"
    finally:
        is_otel_v2_enabled.cache_clear()
