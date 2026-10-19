import asyncio
import json
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final
from uuid import uuid4

import httpx
import pytest
import respx
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from pydantic import TypeAdapter

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.integrations.langfuse.langfuse import LangFuseLogger
from litellm.integrations.langfuse.langfuse_prompt_management import langfuse_client_init
from litellm.integrations.langfuse.langfuse_sdk import (
    flush_langfuse_tracing,
    resolve_observation_id,
    resolve_trace_id,
)
from litellm.litellm_core_utils import litellm_logging
from litellm.litellm_core_utils.litellm_logging import DynamicLoggingCache as LegacyDynamicLoggingCache
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.litellm_core_utils.thread_pool_executor import MAX_THREADS, executor
from litellm.llms.custom_httpx.http_handler import HTTPHandler

_EXPORT_URL: Final = "https://langfuse.unit/api/public/otel/v1/traces"
_VOLATILE_ATTRIBUTES: Final = frozenset(
    {
        "langfuse.observation.completion_start_time",
        "langfuse.observation.metadata.applied_guardrails",
        "langfuse.observation.metadata.cache_hit",
        "langfuse.observation.metadata.hidden_params",
        "langfuse.observation.metadata.litellm_call_id",
        "langfuse.observation.metadata.litellm_response_cost",
        "langfuse.observation.metadata.requester_metadata",
        "langfuse.observation.metadata.response_id",
        "langfuse.observation.metadata.usage_object",
    }
)


def _decode_attribute(value: AnyValue) -> object:
    match value.WhichOneof("value"):
        case "string_value":
            try:
                return json.loads(value.string_value)
            except json.JSONDecodeError:
                return value.string_value
        case "bool_value":
            return value.bool_value
        case "int_value":
            return value.int_value
        case "double_value":
            return value.double_value
        case "array_value":
            return [_decode_attribute(item) for item in value.array_value.values]
        case _:
            return None


def _exported_spans(route: respx.Route) -> tuple[dict[str, object], ...]:
    exported: Final = tuple(
        ExportTraceServiceRequest.FromString(call.request.content)
        for call in route.calls
    )
    return tuple(
        {
            "trace_id": span.trace_id.hex(),
            "span_id": span.span_id.hex(),
            "parent_span_id": span.parent_span_id.hex() or None,
            "name": span.name,
            "attributes": {
                attribute.key: _decode_attribute(attribute.value)
                for attribute in span.attributes
            },
        }
        for request in exported
        for resource in request.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    )


def _comparable(span: Mapping[str, object]) -> dict[str, object]:
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(span["attributes"])
    return {
        "name": span["name"],
        "parent_span_id": span["parent_span_id"],
        "attributes": {
            key: value for key, value in sorted(attributes.items()) if key not in _VOLATILE_ATTRIBUTES
        },
    }


def _expected_generation(route: respx.Route, fixture_name: str, trace_id: str) -> dict[str, object]:
    otel_trace_id: Final = resolve_trace_id(trace_id)
    generations: Final = tuple(
        span
        for span in _exported_spans(route)
        if span["trace_id"] == otel_trace_id
        and TypeAdapter(dict[str, object]).validate_python(span["attributes"]).get("langfuse.observation.type")
        == "generation"
    )
    assert len(generations) == 1
    expected_path: Final = Path(__file__).with_name("langfuse_expected_request_body") / fixture_name
    expected: Final = TypeAdapter(dict[str, object]).validate_python(json.loads(expected_path.read_text()))
    actual: Final = _comparable(generations[0])
    expected_comparable: Final = _comparable(expected)
    assert actual == expected_comparable, json.dumps(
        {"actual": actual, "expected": expected_comparable}, indent=2, sort_keys=True
    )
    return actual


def _drain_logging_executor() -> None:
    workers: Final = MAX_THREADS
    gate: Final = threading.Barrier(workers + 1)
    futures: Final = tuple(executor.submit(gate.wait) for _ in range(workers))
    gate.wait()
    for future in futures:
        future.result()


async def _flush_langfuse() -> None:
    loop: Final = asyncio.get_running_loop()
    scheduled_callbacks: Final[asyncio.Future[None]] = loop.create_future()
    loop.call_soon(scheduled_callbacks.set_result, None)
    await scheduled_callbacks
    await GLOBAL_LOGGING_WORKER.flush()
    _drain_logging_executor()
    assert flush_langfuse_tracing()


@pytest.fixture
def langfuse_export(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> Iterator[tuple[respx.Route, str]]:
    trace_id: Final = f"litellm-migration-{uuid4()}"
    route: Final = respx_mock.post(_EXPORT_URL).respond(status_code=200)
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.unit")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", f"{trace_id}-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", f"{trace_id}-secret")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    client_cache: Final = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", client_cache)
    http_client: Final = httpx.Client(transport=httpx.MockTransport(respx_mock.handler))
    client_cache.set_cache(
        key="httpx_client",
        value=HTTPHandler(client=http_client),
        litellm_owned_client=False,
    )
    monkeypatch.setattr(litellm, "success_callback", ["langfuse"])
    monkeypatch.setattr(litellm_logging, "langFuseLogger", None)
    monkeypatch.setattr(litellm_logging, "in_memory_dynamic_logger_cache", LegacyDynamicLoggingCache())
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [])
    langfuse_client_init.cache_clear()
    yield route, trace_id
    logger: Final = litellm_logging.in_memory_dynamic_logger_cache.get_cache(
        credentials={}, service_name="langfuse"
    )
    if isinstance(logger, LangFuseLogger):
        logger.stop()
    client_cache.flush_cache()
    http_client.close()


@pytest.mark.asyncio
async def test_langfuse_logging_completion(langfuse_export: tuple[respx.Route, str]) -> None:
    route, trace_id = langfuse_export
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response="Hello! How can I assist you today?",
        metadata={"trace_id": trace_id},
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_completion_with_tags(langfuse_export: tuple[respx.Route, str]) -> None:
    route, trace_id = langfuse_export
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response="Hello! How can I assist you today?",
        metadata={"trace_id": trace_id, "tags": ["test_tag", "test_tag_2"]},
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_tags.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_completion_with_tags_stream(langfuse_export: tuple[respx.Route, str]) -> None:
    route, trace_id = langfuse_export
    response: Final = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response="Hello! How can I assist you today?",
        metadata={"trace_id": trace_id, "tags": ["test_tag_stream", "test_tag_2_stream"]},
        stream=True,
    )
    chunks: Final = tuple([chunk async for chunk in response])
    assert chunks
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_tags_stream.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_generation_id_metadata_names_the_exported_observation(
    langfuse_export: tuple[respx.Route, str],
) -> None:
    route, trace_id = langfuse_export
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response="Hello! How can I assist you today?",
        metadata={"trace_id": trace_id, "generation_id": "my-generation"},
    )
    await _flush_langfuse()
    generation: Final = next(span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id))
    assert generation["span_id"] == resolve_observation_id("my-generation")
    _expected_generation(route, "completion.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_completion_with_langfuse_metadata(
    langfuse_export: tuple[respx.Route, str],
) -> None:
    route, trace_id = langfuse_export
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response="Hello! How can I assist you today?",
        metadata={
            "trace_id": trace_id,
            "tags": ["test_tag", "test_tag_2"],
            "generation_name": "test_generation_name",
            "parent_observation_id": "test_parent_observation_id",
            "version": "test_version",
            "trace_user_id": "test_user_id",
            "session_id": "test_session_id",
            "trace_name": "test_trace_name",
            "trace_metadata": {"test_key": "test_value"},
            "trace_version": "test_trace_version",
            "trace_release": "test_trace_release",
        },
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_langfuse_metadata.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_with_non_serializable_metadata(
    langfuse_export: tuple[respx.Route, str],
) -> None:
    from pydantic import BaseModel, ConfigDict

    class UserPreferences(BaseModel):
        model_config = ConfigDict(frozen=True)
        favorite_colors: frozenset[str]
        settings: dict[str, object]

    route, trace_id = langfuse_export
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response="Hello! How can I assist you today?",
        metadata={
            "user_prefs": UserPreferences(favorite_colors=frozenset({"red", "blue"}), settings={"theme": "dark"}),
            "nested_set": {"inner_set": frozenset({1, 2, 3})},
            "trace_id": trace_id,
        },
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_complex_metadata.json", trace_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "fixture_name"),
    [
        ({"a": 1, "b": 2, "c": 3}, "simple_metadata.json"),
        ({"a": {"nested_a": 1}, "b": {"nested_b": 2}}, "nested_metadata.json"),
        ({"a": [1, 2, 3], "b": {4, 5, 6}}, "simple_metadata2.json"),
        ({"a": (1, 2), "b": frozenset({3, 4}), "c": {"d": [5, 6]}}, "simple_metadata3.json"),
        ({"lock": threading.Lock()}, "metadata_with_lock.json"),
        ({"func": lambda x: x + 1}, "metadata_with_function.json"),
        (
            {
                "int": 42,
                "str": "hello",
                "list": [1, 2, 3],
                "set": {4, 5},
                "dict": {"nested": "value"},
                "non_copyable": threading.Lock(),
                "function": print,
            },
            "complex_metadata.json",
        ),
        ({"list": ["list", "not", "a", "dict"]}, "complex_metadata_2.json"),
        ({}, "empty_metadata.json"),
    ],
)
async def test_langfuse_logging_with_various_metadata_types(
    langfuse_export: tuple[respx.Route, str], metadata: dict[str, object], fixture_name: str
) -> None:
    route, trace_id = langfuse_export
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response="Hello! How can I assist you today?",
        metadata={**metadata, "trace_id": trace_id},
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, fixture_name, trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_completion_with_malformed_llm_response(
    langfuse_export: tuple[respx.Route, str],
) -> None:
    route, trace_id = langfuse_export
    response: Final = litellm.ModelResponse(
        choices=[],
        usage=litellm.Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        model="gpt-3.5-turbo",
        object="chat.completion",
        created=1723081200,
    ).model_dump()
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response=response,
        metadata={"trace_id": trace_id},
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_no_choices.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_completion_with_bedrock_llm_response(
    langfuse_export: tuple[respx.Route, str],
) -> None:
    route, trace_id = langfuse_export
    response: Final = litellm.ModelResponse(
        choices=[],
        usage=litellm.Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
        object="chat.completion",
        created=1723081200,
    ).model_dump()
    await litellm.acompletion(
        model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response=response,
        metadata={"trace_id": trace_id},
        aws_access_key_id="fake-key",
        aws_secret_access_key="fake-key",
        aws_region="us-east-1",
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_bedrock_call.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_completion_with_vertex_llm_response(
    langfuse_export: tuple[respx.Route, str],
) -> None:
    route, trace_id = langfuse_export
    response: Final = litellm.ModelResponse(
        choices=[],
        usage=litellm.Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        model="vertex/gemini-3-flash-preview",
        object="chat.completion",
        created=1723081200,
    ).model_dump()
    await litellm.acompletion(
        model="vertex_ai/gemini-3-flash-preview",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response=response,
        metadata={"trace_id": trace_id},
        vertex_credentials="my-mock-credentials",
        api_key="my-mock-credentials-2",
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_vertex_call.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_vllm_embedding(
    langfuse_export: tuple[respx.Route, str], respx_mock: respx.MockRouter
) -> None:
    route, trace_id = langfuse_export
    embedding_url: Final = "http://my-fake-vllm.com/v1/embeddings"
    upstream: Final = respx_mock.post(embedding_url).mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "BAAI/bge-small-en-v1.5",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )
    )
    await litellm.aembedding(
        model="hosted_vllm/BAAI/bge-small-en-v1.5",
        input=["Hello from litellm!"],
        api_base="http://my-fake-vllm.com/v1",
        metadata={"trace_id": trace_id},
    )
    await _flush_langfuse()
    assert upstream.call_count == 1
    expected_path: Final = Path(__file__).with_name("langfuse_expected_request_body") / "embedding_with_vllm.json"
    assert TypeAdapter(dict[str, object]).validate_json(upstream.calls.last.request.content) == json.loads(
        expected_path.read_text()
    )
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_langfuse_logging_with_router(langfuse_export: tuple[respx.Route, str]) -> None:
    route, trace_id = langfuse_export
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "Hello! How can I assist you today?",
                    "api_key": "test_api_key",
                },
            }
        ]
    )
    await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response=litellm.ModelResponse(
            choices=[],
            usage=litellm.Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            model="gpt-3.5-turbo",
            object="chat.completion",
            created=1723081200,
        ).model_dump(),
        metadata={"trace_id": trace_id},
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_router.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_e2e_sync(langfuse_export: tuple[respx.Route, str]) -> None:
    route, trace_id = langfuse_export
    litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "sync callback route"}],
        mock_response="Hello from litellm",
        metadata={"trace_id": trace_id},
    )
    await _flush_langfuse()
    assert route.call_count == 1
    assert str(route.calls[0].request.url).endswith("/api/public/otel/v1/traces")
    assert _exported_spans(route)
