import pytest
from openai import OpenAI, BadRequestError
import asyncio

client = OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")


def test_chat_completion_bad_model():
    with pytest.raises(BadRequestError) as excinfo:
        client.chat.completions.create(
            model="non-existent-model", messages=[{"role": "user", "content": "Hello!"}]
        )
    print(f"Chat completion error: {excinfo.value}")


def test_completion_bad_model():
    with pytest.raises(BadRequestError) as excinfo:
        client.completions.create(model="non-existent-model", prompt="Hello!")
    print(f"Completion error: {excinfo.value}")


def test_embeddings_bad_model():
    with pytest.raises(BadRequestError) as excinfo:
        client.embeddings.create(model="non-existent-model", input="Hello world")
    print(f"Embeddings error: {excinfo.value}")


def test_images_bad_model():
    with pytest.raises(BadRequestError) as excinfo:
        client.images.generate(
            model="non-existent-model", prompt="A cute baby sea otter"
        )
    print(f"Images error: {excinfo.value}")


def test_moderations_bad_model():
    with pytest.raises(BadRequestError) as excinfo:
        client.moderations.create(
            model="non-existent-model", input="I want to harm someone."
        )
    print(f"Moderations error: {excinfo.value}")


@pytest.mark.asyncio
async def test_async_chat_completion_bad_model():
    from openai import AsyncOpenAI

    async_client = AsyncOpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

    with pytest.raises(BadRequestError) as excinfo:
        await async_client.chat.completions.create(
            model="non-existent-model", messages=[{"role": "user", "content": "Hello!"}]
        )
    print(f"Async chat completion error: {excinfo.value}")
