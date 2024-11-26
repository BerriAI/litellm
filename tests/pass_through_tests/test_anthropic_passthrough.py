"""
This test ensures that the proxy can passthrough anthropic requests

"""

import pytest
import anthropic
import aiohttp
import asyncio

client = anthropic.Anthropic(
    base_url="http://0.0.0.0:4000/anthropic", api_key="sk-1234"
)


def test_anthropic_basic_completion():
    print("making basic completion request to anthropic passthrough")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Say 'hello test' and nothing else"}],
        extra_body={
            "litellm_metadata": {
                "tags": ["test-tag-1", "test-tag-2"],
            }
        },
    )
    print(response)


def test_anthropic_streaming():
    print("making streaming request to anthropic passthrough")
    collected_output = []

    with client.messages.stream(
        max_tokens=10,
        messages=[
            {"role": "user", "content": "Say 'hello stream test' and nothing else"}
        ],
        model="claude-3-5-sonnet-20241022",
        extra_body={
            "litellm_metadata": {
                "tags": ["test-tag-stream-1", "test-tag-stream-2"],
            }
        },
    ) as stream:
        for text in stream.text_stream:
            collected_output.append(text)

    full_response = "".join(collected_output)
    print(full_response)


@pytest.mark.asyncio
async def test_anthropic_basic_completion_with_headers():
    print("making basic completion request to anthropic passthrough with aiohttp")

    headers = {
        "Authorization": f"Bearer sk-1234",
        "Content-Type": "application/json",
        "Anthropic-Version": "2023-06-01",
    }

    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Say 'hello test' and nothing else"}],
        "litellm_metadata": {
            "tags": ["test-tag-1", "test-tag-2"],
        },
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://0.0.0.0:4000/anthropic/v1/messages", json=payload, headers=headers
        ) as response:
            response_text = await response.text()
            print(f"Response text: {response_text}")

            response_json = await response.json()
            response_headers = response.headers
            litellm_call_id = response_headers.get("x-litellm-call-id")

            print(f"LiteLLM Call ID: {litellm_call_id}")

            # Wait for spend to be logged
            await asyncio.sleep(15)

            # Check spend logs for this specific request
            async with session.get(
                f"http://0.0.0.0:4000/spend/logs?request_id={litellm_call_id}",
                headers={"Authorization": "Bearer sk-1234"},
            ) as spend_response:
                print("text spend response")
                print(f"Spend response: {spend_response}")
                spend_data = await spend_response.json()
                print(f"Spend data: {spend_data}")
                assert spend_data is not None, "Should have spend data for the request"

                log_entry = spend_data[
                    0
                ]  # Get the first (and should be only) log entry

                # Basic existence checks
                assert spend_data is not None, "Should have spend data for the request"
                assert isinstance(log_entry, dict), "Log entry should be a dictionary"

                # Request metadata assertions
                assert (
                    log_entry["request_id"] == litellm_call_id
                ), "Request ID should match"
                assert (
                    log_entry["call_type"] == "pass_through_endpoint"
                ), "Call type should be pass_through_endpoint"
                assert (
                    log_entry["api_base"] == "https://api.anthropic.com/v1/messages"
                ), "API base should be Anthropic's endpoint"

                # Token and spend assertions
                assert log_entry["spend"] > 0, "Spend value should not be None"
                assert isinstance(
                    log_entry["spend"], (int, float)
                ), "Spend should be a number"
                assert log_entry["total_tokens"] > 0, "Should have some tokens"
                assert log_entry["prompt_tokens"] > 0, "Should have prompt tokens"
                assert (
                    log_entry["completion_tokens"] > 0
                ), "Should have completion tokens"
                assert (
                    log_entry["total_tokens"]
                    == log_entry["prompt_tokens"] + log_entry["completion_tokens"]
                ), "Total tokens should equal prompt + completion"

                # Time assertions
                assert all(
                    key in log_entry
                    for key in ["startTime", "endTime", "completionStartTime"]
                ), "Should have all time fields"
                assert (
                    log_entry["startTime"] < log_entry["endTime"]
                ), "Start time should be before end time"

                # Metadata assertions
                assert (
                    str(log_entry["cache_hit"]).lower() != "true"
                ), "Cache should be off"
                assert log_entry["request_tags"] == [
                    "test-tag-1",
                    "test-tag-2",
                ], "Tags should match input"
                assert (
                    "user_api_key" in log_entry["metadata"]
                ), "Should have user API key in metadata"

                assert "claude" in log_entry["model"]


@pytest.mark.asyncio
async def test_anthropic_streaming_with_headers():
    print("making streaming request to anthropic passthrough with aiohttp")

    headers = {
        "Authorization": f"Bearer sk-1234",
        "Content-Type": "application/json",
        "Anthropic-Version": "2023-06-01",
    }

    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": "Say 'hello stream test' and nothing else"}
        ],
        "stream": True,
        "litellm_metadata": {
            "tags": ["test-tag-stream-1", "test-tag-stream-2"],
        },
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://0.0.0.0:4000/anthropic/v1/messages", json=payload, headers=headers
        ) as response:
            print("response status")
            print(response.status)
            assert response.status == 200, "Response should be successful"
            response_headers = response.headers
            print(f"Response headers: {response_headers}")
            litellm_call_id = response_headers.get("x-litellm-call-id")
            print(f"LiteLLM Call ID: {litellm_call_id}")

            collected_output = []
            async for line in response.content:
                if line:
                    text = line.decode("utf-8").strip()
                    if text.startswith("data: "):
                        collected_output.append(text[6:])  # Remove 'data: ' prefix

            print("Collected output:", "".join(collected_output))

            # Wait for spend to be logged
            await asyncio.sleep(20)

            # Check spend logs for this specific request
            async with session.get(
                f"http://0.0.0.0:4000/spend/logs?request_id={litellm_call_id}",
                headers={"Authorization": "Bearer sk-1234"},
            ) as spend_response:
                spend_data = await spend_response.json()
                print(f"Spend data: {spend_data}")
                assert spend_data is not None, "Should have spend data for the request"

                log_entry = spend_data[
                    0
                ]  # Get the first (and should be only) log entry

                # Basic existence checks
                assert spend_data is not None, "Should have spend data for the request"
                assert isinstance(log_entry, dict), "Log entry should be a dictionary"

                # Request metadata assertions
                assert (
                    log_entry["request_id"] == litellm_call_id
                ), "Request ID should match"
                assert (
                    log_entry["call_type"] == "pass_through_endpoint"
                ), "Call type should be pass_through_endpoint"
                assert (
                    log_entry["api_base"] == "https://api.anthropic.com/v1/messages"
                ), "API base should be Anthropic's endpoint"

                # Token and spend assertions
                assert log_entry["spend"] > 0, "Spend value should not be None"
                assert isinstance(
                    log_entry["spend"], (int, float)
                ), "Spend should be a number"
                assert log_entry["total_tokens"] > 0, "Should have some tokens"
                assert (
                    log_entry["completion_tokens"] > 0
                ), "Should have completion tokens"
                assert (
                    log_entry["total_tokens"]
                    == log_entry["prompt_tokens"] + log_entry["completion_tokens"]
                ), "Total tokens should equal prompt + completion"

                # Time assertions
                assert all(
                    key in log_entry
                    for key in ["startTime", "endTime", "completionStartTime"]
                ), "Should have all time fields"
                assert (
                    log_entry["startTime"] < log_entry["endTime"]
                ), "Start time should be before end time"

                # Metadata assertions
                assert (
                    str(log_entry["cache_hit"]).lower() != "true"
                ), "Cache should be off"
                assert log_entry["request_tags"] == [
                    "test-tag-stream-1",
                    "test-tag-stream-2",
                ], "Tags should match input"
                assert (
                    "user_api_key" in log_entry["metadata"]
                ), "Should have user API key in metadata"

                assert "claude" in log_entry["model"]
