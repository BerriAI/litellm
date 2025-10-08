import asyncio
import copy
import json
import logging
import os
import sys
from typing import Any, Optional
from unittest.mock import MagicMock, patch
import threading

logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import completion
from litellm.caching import InMemoryCache

litellm.num_retries = 3
litellm.success_callback = ["langfuse"]
os.environ["LANGFUSE_DEBUG"] = "True"
import time

import pytest
import pytest_asyncio


def assert_langfuse_request_matches_expected(
    actual_request_body: dict,
    expected_file_name: str,
    trace_id: Optional[str] = None,
):
    """
    Helper function to compare actual Langfuse request body with expected JSON file.

    Args:
        actual_request_body (dict): The actual request body received from the API call
        expected_file_name (str): Name of the JSON file containing expected request body (e.g., "transcription.json")
    """
    # Get the current directory and read the expected request body
    pwd = os.path.dirname(os.path.realpath(__file__))
    expected_body_path = os.path.join(
        pwd, "langfuse_expected_request_body", expected_file_name
    )

    with open(expected_body_path, "r") as f:
        expected_request_body = json.load(f)

    # Filter out events that don't match the trace_id
    if trace_id:
        actual_request_body["batch"] = [
            item
            for item in actual_request_body["batch"]
            if (item["type"] == "trace-create" and item["body"].get("id") == trace_id)
            or (
                item["type"] == "generation-create"
                and item["body"].get("traceId") == trace_id
            )
        ]

    print(
        "actual_request_body after filtering", json.dumps(actual_request_body, indent=4)
    )

    # Replace dynamic values in actual request body
    for item in actual_request_body["batch"]:

        # Replace IDs with expected IDs
        if item["type"] == "trace-create":
            item["id"] = expected_request_body["batch"][0]["id"]
            item["body"]["id"] = expected_request_body["batch"][0]["body"]["id"]
            item["timestamp"] = expected_request_body["batch"][0]["timestamp"]
            item["body"]["timestamp"] = expected_request_body["batch"][0]["body"][
                "timestamp"
            ]
        elif item["type"] == "generation-create":
            item["id"] = expected_request_body["batch"][1]["id"]
            item["body"]["id"] = expected_request_body["batch"][1]["body"]["id"]
            item["timestamp"] = expected_request_body["batch"][1]["timestamp"]
            item["body"]["startTime"] = expected_request_body["batch"][1]["body"][
                "startTime"
            ]
            item["body"]["endTime"] = expected_request_body["batch"][1]["body"][
                "endTime"
            ]
            item["body"]["completionStartTime"] = expected_request_body["batch"][1][
                "body"
            ]["completionStartTime"]
            if trace_id is None:
                print("popping traceId")
                item["body"].pop("traceId")
            else:
                item["body"]["traceId"] = trace_id
                expected_request_body["batch"][1]["body"]["traceId"] = trace_id

    # Replace SDK version with expected version
    actual_request_body["batch"][0]["body"].pop("release", None)
    actual_request_body["metadata"]["sdk_version"] = expected_request_body["metadata"][
        "sdk_version"
    ]
    # replace "public_key" with expected public key
    actual_request_body["metadata"]["public_key"] = expected_request_body["metadata"][
        "public_key"
    ]
    actual_request_body["batch"][1]["body"]["metadata"] = expected_request_body[
        "batch"
    ][1]["body"]["metadata"]
    actual_request_body["metadata"]["sdk_integration"] = expected_request_body[
        "metadata"
    ]["sdk_integration"]
    actual_request_body["metadata"]["batch_size"] = expected_request_body["metadata"][
        "batch_size"
    ]
    # Assert the entire request body matches
    assert (
        actual_request_body == expected_request_body
    ), f"Difference in request bodies: {json.dumps(actual_request_body, indent=2)} != {json.dumps(expected_request_body, indent=2)}"


class TestLangfuseLogging:
    @pytest_asyncio.fixture
    async def mock_setup(self):
        """Common setup for Langfuse logging tests"""
        from litellm._uuid import uuid
        from unittest.mock import AsyncMock, patch
        import httpx

        # Create a mock Response object
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}

        # Create mock for httpx.Client.post
        mock_post = AsyncMock()
        mock_post.return_value = mock_response

        litellm.set_verbose = True
        litellm.success_callback = ["langfuse"]

        return {"trace_id": f"litellm-test-{str(uuid.uuid4())}", "mock_post": mock_post}

    async def _verify_langfuse_call(
        self,
        mock_post,
        expected_file_name: str,
        trace_id: str,
    ):
        """Helper method to verify Langfuse API calls"""
        await asyncio.sleep(3)

        # Verify the call
        assert mock_post.call_count >= 1
        url = mock_post.call_args[0][0]
        request_body = mock_post.call_args[1].get("content")

        # Parse the JSON string into a dict for assertions
        actual_request_body = json.loads(request_body)

        print("\nMocked Request Details:")
        print(f"URL: {url}")
        print(f"Request Body: {json.dumps(actual_request_body, indent=4)}")

        assert url == "https://us.cloud.langfuse.com/api/public/ingestion"
        assert_langfuse_request_matches_expected(
            actual_request_body,
            expected_file_name,
            trace_id,
        )

    @pytest.mark.asyncio
    async def test_langfuse_logging_completion(self, mock_setup):
        """Test Langfuse logging for chat completion"""
        setup = mock_setup
        with patch("httpx.Client.post", setup["mock_post"]):
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response="Hello! How can I assist you today?",
                metadata={"trace_id": setup["trace_id"]},
            )
            await self._verify_langfuse_call(
                setup["mock_post"], "completion.json", setup["trace_id"]
            )

    @pytest.mark.asyncio
    async def test_langfuse_logging_completion_with_tags(self, mock_setup):
        """Test Langfuse logging for chat completion with tags"""
        setup = mock_setup
        with patch("httpx.Client.post", setup["mock_post"]):
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response="Hello! How can I assist you today?",
                metadata={
                    "trace_id": setup["trace_id"],
                    "tags": ["test_tag", "test_tag_2"],
                },
            )
            await self._verify_langfuse_call(
                setup["mock_post"], "completion_with_tags.json", setup["trace_id"]
            )

    @pytest.mark.asyncio
    async def test_langfuse_logging_completion_with_tags_stream(self, mock_setup):
        """Test Langfuse logging for chat completion with tags"""
        setup = mock_setup
        with patch("httpx.Client.post", setup["mock_post"]):
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
    async def test_langfuse_logging_completion_with_langfuse_metadata(self, mock_setup):
        """Test Langfuse logging for chat completion with metadata for langfuse"""
        setup = mock_setup
        with patch("httpx.Client.post", setup["mock_post"]):
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
    async def test_langfuse_logging_with_non_serializable_metadata(self, mock_setup):
        """Test Langfuse logging with metadata that requires preparation (Pydantic models, sets, etc)"""
        from pydantic import BaseModel
        from typing import Set
        import datetime

        class UserPreferences(BaseModel):
            favorite_colors: Set[str]
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

        with patch("httpx.Client.post", setup["mock_post"]):
            response = await litellm.acompletion(
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
    async def test_langfuse_logging_with_various_metadata_types(
        self, mock_setup, test_metadata, response_json_file
    ):
        """Test Langfuse logging with various metadata types including non-serializable objects"""
        import threading

        setup = mock_setup

        if test_metadata is not None:
            test_metadata["trace_id"] = setup["trace_id"]

        with patch("httpx.Client.post", setup["mock_post"]):
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
    async def test_langfuse_logging_completion_with_malformed_llm_response(
        self, mock_setup
    ):
        """Test Langfuse logging for chat completion with malformed LLM response"""
        setup = mock_setup
        litellm._turn_on_debug()
        with patch("httpx.Client.post", setup["mock_post"]):
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
            await self._verify_langfuse_call(
                setup["mock_post"], "completion_with_no_choices.json", setup["trace_id"]
            )
    
    @pytest.mark.asyncio
    async def test_langfuse_logging_completion_with_bedrock_llm_response(
        self, mock_setup
    ):
        """Test Langfuse logging for chat completion with malformed LLM response"""
        setup = mock_setup
        litellm._turn_on_debug()
        with patch("httpx.Client.post", setup["mock_post"]):
            mock_response = litellm.ModelResponse(
                choices=[],
                usage=litellm.Usage(
                    prompt_tokens=10,
                    completion_tokens=10,
                    total_tokens=20,
                ),
                model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                object="chat.completion",
                created=1723081200,
            ).model_dump()
            await litellm.acompletion(
                model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response=mock_response,
                metadata={"trace_id": setup["trace_id"]},
                aws_access_key_id="fake-key",
                aws_secret_access_key="fake-key",
                aws_region="us-east-1",
            )
            await self._verify_langfuse_call(
                setup["mock_post"], "completion_with_bedrock_call.json", setup["trace_id"]
            )
    @pytest.mark.asyncio
    async def test_langfuse_logging_completion_with_vertex_llm_response(
        self, mock_setup
    ):
        """Test Langfuse logging for chat completion with malformed LLM response"""
        setup = mock_setup
        litellm._turn_on_debug()
        with patch("httpx.Client.post", setup["mock_post"]):
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
                setup["mock_post"], "completion_with_vertex_call.json", setup["trace_id"]
            )

    @pytest.mark.asyncio
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
                    }
                }
            ]
        )
        with patch("httpx.Client.post", mock_setup["mock_post"]):
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
                mock_setup["mock_post"], "completion_with_router.json", mock_setup["trace_id"]
            )
