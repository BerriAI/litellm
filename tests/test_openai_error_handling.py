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


def test_moderations_bad_model():
    key = generate_key_sync()
    client = OpenAI(api_key=key, base_url="http://0.0.0.0:4000")

    with pytest.raises(BadRequestError) as excinfo:
        client.moderations.create(
            model="non-existent-model", input="I want to harm someone."
        )
    print(f"Moderations error: {excinfo.value}")


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
