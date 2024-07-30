# What is this?
## This tests if the proxy fallbacks work as expected
import pytest
import asyncio
import aiohttp
from large_text import text


async def generate_key(
    session,
    i,
    models: list,
    calling_key="sk-1234",
):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {
        "Authorization": f"Bearer {calling_key}",
        "Content-Type": "application/json",
    }
    data = {
        "models": models,
    }

    print(f"data: {data}")

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"Response {i} (Status code: {status}):")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request {i} did not return a 200 status code: {status}")

        return await response.json()


async def chat_completion(session, key: str, model: str, messages: list, **kwargs):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {"model": model, "messages": messages, **kwargs}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


@pytest.mark.asyncio
async def test_chat_completion():
    """
    make chat completion call with prompt > context window. expect it to work with fallback
    """
    async with aiohttp.ClientSession() as session:
        model = "gpt-3.5-turbo"
        messages = [
            {"role": "system", "content": text},
            {"role": "user", "content": "Who was Alexander?"},
        ]
        await chat_completion(
            session=session, key="sk-1234", model=model, messages=messages
        )


@pytest.mark.parametrize("has_access", [True, False])
@pytest.mark.asyncio
async def test_chat_completion_client_fallbacks(has_access):
    """
    make chat completion call with prompt > context window. expect it to work with fallback
    """

    async with aiohttp.ClientSession() as session:
        models = ["gpt-3.5-turbo"]

        if has_access:
            models.append("gpt-instruct")

        ## CREATE KEY WITH MODELS
        generated_key = await generate_key(session=session, i=0, models=models)
        calling_key = generated_key["key"]
        model = "gpt-3.5-turbo"
        messages = [
            {"role": "user", "content": "Who was Alexander?"},
        ]

        ## CALL PROXY
        try:
            await chat_completion(
                session=session,
                key=calling_key,
                model=model,
                messages=messages,
                mock_testing_fallbacks=True,
                fallbacks=["gpt-instruct"],
            )
            if not has_access:
                pytest.fail(
                    "Expected this to fail, submitted fallback model that key did not have access to"
                )
        except Exception as e:
            if has_access:
                pytest.fail("Expected this to work: {}".format(str(e)))
