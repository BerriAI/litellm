import asyncio
import copy
import json
import logging
import os
import sys
from typing import Any, Optional
from unittest.mock import MagicMock, patch

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
    @pytest.fixture
    async def mock_setup(self):
        """Common setup for Langfuse logging tests"""
        import uuid
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
        await asyncio.sleep(1)

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
        setup = await mock_setup  # Await the fixture
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
        setup = await mock_setup  # Await the fixture
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
        setup = await mock_setup  # Await the fixture
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
