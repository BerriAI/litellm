import pytest
from openai import OpenAI, BadRequestError, AsyncOpenAI
import asyncio
import httpx


def generate_key_sync():
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}

    with httpx.Client() as client:
        response = client.post(
            url,
            headers=headers,
            json={
                "models": [
                    "gpt-4",
                    "text-embedding-ada-002",
                    "dall-e-2",
                    "fake-openai-endpoint-2",
                    "mistral-embed",
                    "non-existent-model",
                ],
            },
        )
        response_text = response.text

        print(response_text)
        print()

        if response.status_code != 200:
            raise Exception(
                f"Request did not return a 200 status code: {response.status_code}"
            )

        response_data = response.json()
        return response_data["key"]


def test_chat_completion_bad_model():
    key = generate_key_sync()
    client = OpenAI(api_key=key, base_url="http://0.0.0.0:4000")

    with pytest.raises(BadRequestError) as excinfo:
        client.chat.completions.create(
            model="non-existent-model", messages=[{"role": "user", "content": "Hello!"}]
        )
    print(f"Chat completion error: {excinfo.value}")


def test_completion_bad_model():
    key = generate_key_sync()
    client = OpenAI(api_key=key, base_url="http://0.0.0.0:4000")

    with pytest.raises(BadRequestError) as excinfo:
        client.completions.create(model="non-existent-model", prompt="Hello!")
    print(f"Completion error: {excinfo.value}")


def test_embeddings_bad_model():
    key = generate_key_sync()
    client = OpenAI(api_key=key, base_url="http://0.0.0.0:4000")

    with pytest.raises(BadRequestError) as excinfo:
        client.embeddings.create(model="non-existent-model", input="Hello world")
    print(f"Embeddings error: {excinfo.value}")


def test_images_bad_model():
    key = generate_key_sync()
    client = OpenAI(api_key=key, base_url="http://0.0.0.0:4000")

    with pytest.raises(BadRequestError) as excinfo:
        client.images.generate(
            model="non-existent-model", prompt="A cute baby sea otter"
        )
    print(f"Images error: {excinfo.value}")


@pytest.mark.asyncio
async def test_async_chat_completion_bad_model():
    key = generate_key_sync()
    async_client = AsyncOpenAI(api_key=key, base_url="http://0.0.0.0:4000")

    with pytest.raises(BadRequestError) as excinfo:
        await async_client.chat.completions.create(
            model="non-existent-model", messages=[{"role": "user", "content": "Hello!"}]
        )
    print(f"Async chat completion error: {excinfo.value}")


@pytest.mark.parametrize(
    "curl_command",
    [
        'curl http://0.0.0.0:4000/v1/chat/completions -H \'Content-Type: application/json\' -H \'Authorization: Bearer sk-1234\' -d \'{"messages":[{"role":"user","content":"Hello!"}]}\'',
        "curl http://0.0.0.0:4000/v1/completions -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-1234' -d '{\"prompt\":\"Hello!\"}'",
        "curl http://0.0.0.0:4000/v1/embeddings -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-1234' -d '{\"input\":\"Hello world\"}'",
        "curl http://0.0.0.0:4000/v1/images/generations -H 'Content-Type: application/json' -H 'Authorization: Bearer sk-1234' -d '{\"prompt\":\"A cute baby sea otter\"}'",
    ],
    ids=["chat", "completions", "embeddings", "images"],
)
def test_missing_model_parameter_curl(curl_command):
    import subprocess
    import json

    # Run the curl command and capture the output
    key = generate_key_sync()
    curl_command = curl_command.replace("sk-1234", key)
    result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
    # Parse the JSON response
    response = json.loads(result.stdout)

    # Check that we got an error response
    assert "error" in response
    print("error in response", json.dumps(response, indent=4))

    assert "litellm.BadRequestError" in response["error"]["message"]


@pytest.mark.asyncio
async def test_chat_completion_bad_model_with_spend_logs():
    """
    Tests that Error Logs are created for failed requests
    """
    import json

    key = generate_key_sync()

    # Use httpx to make the request and capture headers
    url = "http://0.0.0.0:4000/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": "non-existent-model",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=payload)

        # Extract the litellm call ID from headers
        litellm_call_id = response.headers.get("x-litellm-call-id")
        print(f"Status code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"LiteLLM Call ID: {litellm_call_id}")

        # Parse the JSON response body
        try:
            response_body = response.json()
            print(f"Error response: {json.dumps(response_body, indent=4)}")
        except json.JSONDecodeError:
            print(f"Could not parse response body as JSON: {response.text}")

    assert (
        litellm_call_id is not None
    ), "Failed to get LiteLLM Call ID from response headers"
    print("waiting for flushing error log to db....")
    await asyncio.sleep(15)

    # Now query the spend logs
    url = "http://0.0.0.0:4000/spend/logs?request_id=" + litellm_call_id
    headers = {"Authorization": f"Bearer sk-1234", "Content-Type": "application/json"}

    with httpx.Client() as client:
        response = client.get(
            url,
            headers=headers,
        )

        assert (
            response.status_code == 200
        ), f"Failed to get spend logs: {response.status_code}"

        spend_logs = response.json()

        # Print the spend logs payload
        print(f"Spend logs response: {json.dumps(spend_logs, indent=4)}")

        # Verify we have logs for the failed request
        assert len(spend_logs) > 0, "No spend logs found"

        # Check if the error is recorded in the logs
        log_entry = spend_logs[0]  # Should be the specific log for our litellm_call_id

        # Verify the structure of the log entry
        assert log_entry["request_id"] == litellm_call_id
        assert log_entry["model"] == "non-existent-model"
        assert log_entry["model_group"] == "non-existent-model"
        assert log_entry["spend"] == 0.0
        assert log_entry["total_tokens"] == 0
        assert log_entry["prompt_tokens"] == 0
        assert log_entry["completion_tokens"] == 0

        # Verify metadata fields
        assert log_entry["metadata"]["status"] == "failure"
        assert "user_api_key" in log_entry["metadata"]
        assert "error_information" in log_entry["metadata"]

        # Verify error information
        error_info = log_entry["metadata"]["error_information"]
        assert "traceback" in error_info
        assert error_info["error_code"] == "400"
        assert error_info["error_class"] == "BadRequestError"
        assert "litellm.BadRequestError" in error_info["error_message"]
        assert "non-existent-model" in error_info["error_message"]

        # Verify request details
        assert log_entry["cache_hit"] == "False"
        assert log_entry["response"] == {}
