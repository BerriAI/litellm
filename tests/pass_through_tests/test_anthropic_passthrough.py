"""
This test ensures that the proxy can passthrough anthropic requests
"""

import pytest
import anthropic
import aiohttp
import asyncio
import json


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=2)
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
            print(
                "non-streaming response",
                json.dumps(response_json, indent=4, default=str),
            )
            reported_usage = response_json.get("usage", None)
            # fix null checks for reported_usage
            anthropic_api_input_tokens = reported_usage.get("input_tokens", None) if reported_usage else None
            anthropic_api_output_tokens = reported_usage.get("output_tokens", None) if reported_usage else None
            litellm_call_id = response_headers.get("x-litellm-call-id")

            print(f"LiteLLM Call ID: {litellm_call_id}")

            # Wait for spend to be logged
            await asyncio.sleep(15)

            # Check spend logs for this specific request with retry logic
            spend_data = None
            max_retries = 2
            for attempt in range(max_retries):
                print(f"Attempt {attempt + 1}/{max_retries} to check spend logs")

                async with session.get(
                    f"http://0.0.0.0:4000/spend/logs?request_id={litellm_call_id}",
                    headers={"Authorization": "Bearer sk-1234"},
                ) as spend_response:
                    print("text spend response")
                    print(f"Spend response: {spend_response}")
                    spend_data = await spend_response.json()
                    print(f"Spend data: {spend_data}")

                    # Check if spend data exists and has entries
                    if spend_data and len(spend_data) > 0:
                        print("Spend logs found!")
                        break
                    else:
                        print("Spend logs not found yet...")
                        if (
                            attempt < max_retries - 1
                        ):  # Don't wait after the last attempt
                            print("Waiting 10 seconds before retry...")
                            await asyncio.sleep(10)

            assert spend_data is not None, "Should have spend data for the request"
            assert len(spend_data) > 0, "Should have at least one spend log entry"

            log_entry = spend_data[0]  # Get the first (and should be only) log entry

            # Basic existence checks
            assert spend_data is not None, "Should have spend data for the request"
            assert isinstance(log_entry, dict), "Log entry should be a dictionary"

            # Request metadata assertions
            assert log_entry["request_id"] == litellm_call_id, "Request ID should match"
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
                log_entry["prompt_tokens"] == anthropic_api_input_tokens
            ), f"Should have prompt tokens matching anthropic api. Expected {anthropic_api_input_tokens} but got {log_entry['prompt_tokens']}"
            assert (
                log_entry["completion_tokens"] == anthropic_api_output_tokens
            ), f"Should have completion tokens matching anthropic api. Expected {anthropic_api_output_tokens} but got {log_entry['completion_tokens']}"
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
            assert str(log_entry["cache_hit"]).lower() != "true", "Cache should be off"
            assert log_entry["request_tags"] == [
                "test-tag-1",
                "test-tag-2",
            ], "Tags should match input"
            assert (
                "user_api_key" in log_entry["metadata"]
            ), "Should have user API key in metadata"

            assert "claude" in log_entry["model"]
            assert log_entry["custom_llm_provider"] == "anthropic"


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
            "user": "test-user-1",
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
            anthropic_api_usage_chunks = []
            for chunk in collected_output:
                chunk_json = json.loads(chunk)
                if "usage" in chunk_json:
                    anthropic_api_usage_chunks.append(chunk_json["usage"])
                elif "message" in chunk_json and "usage" in chunk_json["message"]:
                    anthropic_api_usage_chunks.append(chunk_json["message"]["usage"])

            print(
                "anthropic_api_usage_chunks",
                json.dumps(anthropic_api_usage_chunks, indent=4, default=str),
            )

            print("anthropic_api_usage_chunks: ", anthropic_api_usage_chunks)
            # Get the most recent value of input tokens (iterate backwards to find last non-zero value)
            anthropic_api_input_tokens = 0
            for usage in reversed(anthropic_api_usage_chunks):
                if usage.get("input_tokens", 0) > 0:
                    anthropic_api_input_tokens = usage.get("input_tokens", 0)
                    break
            anthropic_api_output_tokens = 0
            for usage in reversed(anthropic_api_usage_chunks):
                if usage.get("output_tokens", 0) > 0:
                    anthropic_api_output_tokens = usage.get("output_tokens", 0)
                    break

            print("anthropic_api_input_tokens", anthropic_api_input_tokens)
            print("anthropic_api_output_tokens", anthropic_api_output_tokens)

            # Wait for spend to be logged
            await asyncio.sleep(20)

            # Check spend logs for this specific request with retry logic
            spend_data = None
            max_retries = 2
            for attempt in range(max_retries):
                print(f"Attempt {attempt + 1}/{max_retries} to check spend logs")

                async with session.get(
                    f"http://0.0.0.0:4000/spend/logs?request_id={litellm_call_id}",
                    headers={"Authorization": "Bearer sk-1234"},
                ) as spend_response:
                    spend_data = await spend_response.json()
                    print(f"Spend data: {spend_data}")

                    # Check if spend data exists and has entries
                    if spend_data and len(spend_data) > 0:
                        print("Spend logs found!")
                        break
                    else:
                        print("Spend logs not found yet...")
                        if (
                            attempt < max_retries - 1
                        ):  # Don't wait after the last attempt
                            print("Waiting 10 seconds before retry...")
                            await asyncio.sleep(10)

            assert spend_data is not None, "Should have spend data for the request"
            assert len(spend_data) > 0, "Should have at least one spend log entry"

            log_entry = spend_data[0]  # Get the first (and should be only) log entry

            # Basic existence checks
            assert spend_data is not None, "Should have spend data for the request"
            assert isinstance(log_entry, dict), "Log entry should be a dictionary"

            # Request metadata assertions
            assert log_entry["request_id"] == litellm_call_id, "Request ID should match"
            assert (
                log_entry["call_type"] == "pass_through_endpoint"
            ), "Call type should be pass_through_endpoint"
            # assert (
            #     log_entry["api_base"] == "https://api.anthropic.com/v1/messages"
            # ), "API base should be Anthropic's endpoint"

            # Token and spend assertions
            assert log_entry["spend"] > 0, "Spend value should not be None"
            assert isinstance(
                log_entry["spend"], (int, float)
            ), "Spend should be a number"
            assert log_entry["total_tokens"] > 0, "Should have some tokens"
            assert (
                log_entry["prompt_tokens"] == anthropic_api_input_tokens
            ), f"Should have prompt tokens matching anthropic api. Expected {anthropic_api_input_tokens} but got {log_entry['prompt_tokens']}"
            assert (
                log_entry["completion_tokens"] == anthropic_api_output_tokens
            ), f"Should have completion tokens matching anthropic api. Expected {anthropic_api_output_tokens} but got {log_entry['completion_tokens']}"
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
            assert str(log_entry["cache_hit"]).lower() != "true", "Cache should be off"
            assert log_entry["request_tags"] == [
                "test-tag-stream-1",
                "test-tag-stream-2",
            ], "Tags should match input"
            assert (
                "user_api_key" in log_entry["metadata"]
            ), "Should have user API key in metadata"

            assert "claude" in log_entry["model"]

            assert log_entry["end_user"] == "test-user-1"
            assert log_entry["custom_llm_provider"] == "anthropic"
