import asyncio
import json
import logging
import os
import threading
from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue

logging.basicConfig(level=logging.DEBUG)

import litellm
from litellm.integrations.langfuse.langfuse_sdk import resolve_observation_id, resolve_trace_id
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

litellm.num_retries = 3
litellm.success_callback = ["langfuse"]
os.environ["LANGFUSE_DEBUG"] = "True"

import pytest
import pytest_asyncio

LANGFUSE_EXPORT_POST: Final = "requests.Session.post"
LANGFUSE_EXPORT_PATH: Final = "/api/public/otel/v1/traces"

_LITELLM_OWNED_ATTRIBUTES: Final = frozenset(
    {
        "langfuse.internal.is_app_root",
        "langfuse.observation.completion_start_time",
        "langfuse.observation.metadata.applied_guardrails",
        "langfuse.observation.metadata.cache_hit",
        "langfuse.observation.metadata.hidden_params",
        "langfuse.observation.metadata.litellm_response_cost",
        "langfuse.observation.metadata.requester_metadata",
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


def _exported_spans(mock_post: MagicMock) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    for call in mock_post.call_args_list:
        assert call.kwargs["url"].endswith(LANGFUSE_EXPORT_PATH), call.kwargs["url"]
        request = ExportTraceServiceRequest.FromString(call.kwargs["data"])
        for resource_spans in request.resource_spans:
            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    spans.append(
                        {
                            "name": span.name,
                            "trace_id": span.trace_id.hex(),
                            "span_id": span.span_id.hex(),
                            "parent_span_id": span.parent_span_id.hex() or None,
                            "attributes": {
                                attribute.key: _decode_attribute(attribute.value) for attribute in span.attributes
                            },
                        }
                    )
    return spans


def _comparable(span: Mapping[str, object]) -> dict[str, object]:
    attributes = span["attributes"]
    assert isinstance(attributes, dict)
    return {
        "name": span["name"],
        "parent_span_id": None if attributes.get("langfuse.internal.as_root") else span["parent_span_id"],
        "attributes": {key: value for key, value in sorted(attributes.items()) if key not in _LITELLM_OWNED_ATTRIBUTES},
    }


def assert_langfuse_request_matches_expected(
    spans: list[dict[str, object]],
    expected_file_name: str,
    trace_id: str,
):
    """Compare the generation langfuse exported for ``trace_id`` with the expected JSON file."""
    pwd = os.path.dirname(os.path.realpath(__file__))
    expected_body_path = os.path.join(pwd, "langfuse_expected_request_body", expected_file_name)
    with open(expected_body_path, "r") as f:
        expected_generation = json.load(f)

    otel_trace_id: Final = resolve_trace_id(trace_id)
    generations: Final = [
        span
        for span in spans
        if span["trace_id"] == otel_trace_id and span["attributes"]["langfuse.observation.type"] == "generation"  # pyright: ignore[reportIndexIssue]  # built as dict in _exported_spans
    ]
    assert len(generations) == 1, (
        f"Expected exactly one generation for trace_id={trace_id} ({otel_trace_id}), "
        f"got {len(generations)}. Spans: {json.dumps(spans, indent=2)}"
    )

    actual_generation: Final = _comparable(generations[0])
    assert actual_generation == expected_generation, (
        f"Difference in exported generation: {json.dumps(actual_generation, indent=2)} "
        f"!= {json.dumps(expected_generation, indent=2)}"
    )


class TestLangfuseLogging:
    @pytest_asyncio.fixture
    async def mock_setup(self):
        """Common setup for Langfuse logging tests"""
        from litellm._uuid import uuid

        mock_post = MagicMock(return_value=MagicMock(ok=True, status_code=200))

        litellm.set_verbose = True
        litellm.success_callback = ["langfuse"]

        return {"trace_id": f"litellm-test-{uuid.uuid4()!s}", "mock_post": mock_post}

    async def _verify_langfuse_call(
        self,
        mock_post,
        expected_file_name: str,
        trace_id: str,
    ):
        """Wait for the batch processor to export, then compare the generation it shipped."""
        otel_trace_id: Final = resolve_trace_id(trace_id)
        for _ in range(100):
            if any(span["trace_id"] == otel_trace_id for span in _exported_spans(mock_post)):
                break
            await asyncio.sleep(0.1)

        assert mock_post.call_count >= 1, "langfuse exported nothing"
        assert_langfuse_request_matches_expected(
            _exported_spans(mock_post),
            expected_file_name,
            trace_id,
        )

    @pytest.mark.asyncio
    @pytest.mark.flaky(retries=3, delay=1)
    async def test_langfuse_logging_completion(self, mock_setup):
        """Test Langfuse logging for chat completion"""
        setup = mock_setup
        with patch(LANGFUSE_EXPORT_POST, setup["mock_post"]):
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response="Hello! How can I assist you today?",
                metadata={"trace_id": setup["trace_id"]},
            )
            await self._verify_langfuse_call(setup["mock_post"], "completion.json", setup["trace_id"])

    @pytest.mark.asyncio
    @pytest.mark.flaky(retries=3, delay=1)
    async def test_langfuse_logging_completion_with_tags(self, mock_setup):
        """Test Langfuse logging for chat completion with tags"""
        setup = mock_setup
        with patch(LANGFUSE_EXPORT_POST, setup["mock_post"]):
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response="Hello! How can I assist you today?",
                metadata={
                    "trace_id": setup["trace_id"],
                    "tags": ["test_tag", "test_tag_2"],
                },
            )
            await self._verify_langfuse_call(setup["mock_post"], "completion_with_tags.json", setup["trace_id"])

    @pytest.mark.asyncio
    @pytest.mark.flaky(retries=3, delay=1)
    async def test_langfuse_logging_completion_with_tags_stream(self, mock_setup):
        """Test Langfuse logging for chat completion with tags"""
        setup = mock_setup
        with patch(LANGFUSE_EXPORT_POST, setup["mock_post"]):
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response="Hello! How can I assist you today?",
                metadata={
                    "trace_id": setup["trace_id"],
                    "tags": ["test_tag_stream", "test_tag_2_stream"],
                },
            )
            await self._verify_langfuse_call(
                setup["mock_post"],
                "completion_with_tags_stream.json",
                setup["trace_id"],
            )

    @pytest.mark.asyncio
    @pytest.mark.flaky(retries=3, delay=1)
    async def test_langfuse_generation_id_metadata_names_the_exported_observation(self, mock_setup):
        """v2 let callers pick the generation id; v4 only has span ids, so the requested id must become one."""
        setup = mock_setup
        with patch(LANGFUSE_EXPORT_POST, setup["mock_post"]):
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response="Hello! How can I assist you today?",
                metadata={"trace_id": setup["trace_id"], "generation_id": "my-generation"},
            )
            await self._verify_langfuse_call(setup["mock_post"], "completion.json", setup["trace_id"])

        generation: Final = next(
            span
            for span in _exported_spans(setup["mock_post"])
            if span["trace_id"] == resolve_trace_id(setup["trace_id"])
        )
        assert generation["span_id"] == resolve_observation_id("my-generation")

    @pytest.mark.asyncio
    @pytest.mark.flaky(retries=3, delay=1)
    async def test_langfuse_logging_completion_with_langfuse_metadata(self, mock_setup):
        """Test Langfuse logging for chat completion with metadata for langfuse"""
        setup = mock_setup
        with patch(LANGFUSE_EXPORT_POST, setup["mock_post"]):
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response="Hello! How can I assist you today?",
                metadata={
                    "trace_id": setup["trace_id"],
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
            await self._verify_langfuse_call(
                setup["mock_post"],
                "completion_with_langfuse_metadata.json",
                setup["trace_id"],
            )

    @pytest.mark.asyncio
    @pytest.mark.flaky(retries=3, delay=1)
    async def test_langfuse_logging_with_non_serializable_metadata(self, mock_setup):
        """Test Langfuse logging with metadata that requires preparation (Pydantic models, sets, etc)"""
        import datetime

        from pydantic import BaseModel

        class UserPreferences(BaseModel):
            favorite_colors: set[str]
            last_login: datetime.datetime
            settings: dict

        setup = mock_setup

        test_metadata = {
            "user_prefs": UserPreferences(
                favorite_colors={"red", "blue"},
                last_login=datetime.datetime.now(),
                settings={"theme": "dark", "notifications": True},
            ),
            "nested_set": {
                "inner_set": {1, 2, 3},
                "inner_pydantic": UserPreferences(
                    favorite_colors={"green", "yellow"},
                    last_login=datetime.datetime.now(),
                    settings={"theme": "light"},
                ),
            },
            "trace_id": setup["trace_id"],
        }

        with patch(LANGFUSE_EXPORT_POST, setup["mock_post"]):
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response="Hello! How can I assist you today?",
                metadata=test_metadata,
            )

            await self._verify_langfuse_call(
                setup["mock_post"],
                "completion_with_complex_metadata.json",
                setup["trace_id"],
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_metadata, response_json_file",
        [
            ({"a": 1, "b": 2, "c": 3}, "simple_metadata.json"),
            (
                {"a": {"nested_a": 1}, "b": {"nested_b": 2}},
                "nested_metadata.json",
            ),
            ({"a": [1, 2, 3], "b": {4, 5, 6}}, "simple_metadata2.json"),
            (
                {"a": (1, 2), "b": frozenset([3, 4]), "c": {"d": [5, 6]}},
                "simple_metadata3.json",
            ),
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
            (
                {"list": ["list", "not", "a", "dict"]},
                "complex_metadata_2.json",
            ),
            ({}, "empty_metadata.json"),
        ],
    )
    @pytest.mark.flaky(retries=6, delay=1)
    async def test_langfuse_logging_with_various_metadata_types(self, mock_setup, test_metadata, response_json_file):
        """Test Langfuse logging with various metadata types including non-serializable objects"""
        setup = mock_setup

        if test_metadata is not None:
            test_metadata["trace_id"] = setup["trace_id"]

        with patch(LANGFUSE_EXPORT_POST, setup["mock_post"]):
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response="Hello! How can I assist you today?",
                metadata=test_metadata,
            )

            await self._verify_langfuse_call(
                setup["mock_post"],
                response_json_file,
                setup["trace_id"],
            )

    @pytest.mark.asyncio
    @pytest.mark.flaky(retries=3, delay=1)
    async def test_langfuse_logging_completion_with_malformed_llm_response(self, mock_setup):
        """Test Langfuse logging for chat completion with malformed LLM response"""
        setup = mock_setup
        litellm._turn_on_debug()
        with patch(LANGFUSE_EXPORT_POST, setup["mock_post"]):
            mock_response = litellm.ModelResponse(
                choices=[],
                usage=litellm.Usage(
                    prompt_tokens=10,
                    completion_tokens=10,
                    total_tokens=20,
                ),
                model="gpt-3.5-turbo",
                object="chat.completion",
                created=1723081200,
            ).model_dump()
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response=mock_response,
                metadata={"trace_id": setup["trace_id"]},
            )
            await self._verify_langfuse_call(setup["mock_post"], "completion_with_no_choices.json", setup["trace_id"])

    @pytest.mark.asyncio
    @pytest.mark.flaky(retries=3, delay=1)
    async def test_langfuse_logging_completion_with_bedrock_llm_response(self, mock_setup):
        """Test Langfuse logging for chat completion with malformed LLM response"""
        setup = mock_setup
        litellm._turn_on_debug()
        with patch(LANGFUSE_EXPORT_POST, setup["mock_post"]):
            mock_response = litellm.ModelResponse(
                choices=[],
                usage=litellm.Usage(
                    prompt_tokens=10,
                    completion_tokens=10,
                    total_tokens=20,
                ),
                model="anthropic.claude-haiku-4-5-20251001-v1:0",
                object="chat.completion",
                created=1723081200,
            ).model_dump()
            await litellm.acompletion(
                model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response=mock_response,
                metadata={"trace_id": setup["trace_id"]},
                aws_access_key_id="fake-key",
                aws_secret_access_key="fake-key",
                aws_region="us-east-1",
            )
            await self._verify_langfuse_call(
                setup["mock_post"],
                "completion_with_bedrock_call.json",
                setup["trace_id"],
            )

    @pytest.mark.asyncio
    @pytest.mark.flaky(retries=3, delay=1)
    async def test_langfuse_logging_completion_with_vertex_llm_response(self, mock_setup):
        """Test Langfuse logging for chat completion with malformed LLM response"""
        setup = mock_setup
        litellm._turn_on_debug()
        with patch(LANGFUSE_EXPORT_POST, setup["mock_post"]):
            mock_response = litellm.ModelResponse(
                choices=[],
                usage=litellm.Usage(
                    prompt_tokens=10,
                    completion_tokens=10,
                    total_tokens=20,
                ),
                model="vertex/gemini-2.0-flash-001",
                object="chat.completion",
                created=1723081200,
            ).model_dump()
            await litellm.acompletion(
                model="vertex_ai/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response=mock_response,
                metadata={"trace_id": setup["trace_id"]},
                vertex_credentials="my-mock-credentials",
                api_key="my-mock-credentials-2",
            )
            await self._verify_langfuse_call(
                setup["mock_post"],
                "completion_with_vertex_call.json",
                setup["trace_id"],
            )

    @pytest.mark.asyncio
    @pytest.mark.flaky(retries=3, delay=1)
    async def test_langfuse_logging_vllm_embedding(self, mock_setup):
        """
        Test that the request sent to the vllm embedding endpoint is correct.

        Verifies the request body matches the expected JSON fixture,
        including that the hosted_vllm/ prefix is stripped from the model name
        and that no unexpected fields (e.g. encoding_format) are included.
        """
        setup = mock_setup

        vllm_response_data = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "model": "BAAI/bge-small-en-v1.5",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
        mock_vllm_response = httpx.Response(
            status_code=200,
            json=vllm_response_data,
        )

        mock_async_client = AsyncHTTPHandler()
        mock_async_client.post = AsyncMock(return_value=mock_vllm_response)

        with patch(LANGFUSE_EXPORT_POST, setup["mock_post"]):
            await litellm.aembedding(
                model="hosted_vllm/BAAI/bge-small-en-v1.5",
                input=["Hello from litellm!"],
                api_base="http://my-fake-vllm.com/v1",
                metadata={"trace_id": setup["trace_id"]},
                client=mock_async_client,
            )

        # Verify the request sent to vllm matches the expected JSON fixture
        assert mock_async_client.post.call_count == 1
        actual_vllm_request = mock_async_client.post.call_args.kwargs["json"]

        pwd = os.path.dirname(os.path.realpath(__file__))
        expected_body_path = os.path.join(pwd, "langfuse_expected_request_body", "embedding_with_vllm.json")
        with open(expected_body_path, "r") as f:
            expected_vllm_request = json.load(f)

        assert actual_vllm_request == expected_vllm_request, (
            f"vllm request body mismatch:\n"
            f"actual:   {json.dumps(actual_vllm_request, indent=2)}\n"
            f"expected: {json.dumps(expected_vllm_request, indent=2)}"
        )

    @pytest.mark.asyncio
    @pytest.mark.flaky(retries=3, delay=1)
    async def test_langfuse_logging_with_router(self, mock_setup):
        """Test Langfuse logging with router"""
        litellm._turn_on_debug()
        router = litellm.Router(
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
        with patch(LANGFUSE_EXPORT_POST, mock_setup["mock_post"]):
            mock_response = litellm.ModelResponse(
                choices=[],
                usage=litellm.Usage(
                    prompt_tokens=10,
                    completion_tokens=10,
                    total_tokens=20,
                ),
                model="gpt-3.5-turbo",
                object="chat.completion",
                created=1723081200,
            ).model_dump()
            await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response=mock_response,
                metadata={"trace_id": mock_setup["trace_id"]},
            )
            await self._verify_langfuse_call(
                mock_setup["mock_post"],
                "completion_with_router.json",
                mock_setup["trace_id"],
            )
