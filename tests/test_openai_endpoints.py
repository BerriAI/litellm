# What this tests ?
## Tests /chat/completions by generating a key and then making a chat completions request
import pytest
import asyncio
import aiohttp


async def generate_key(session):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": ["gpt-4", "text-embedding-ada-002", "dall-e-2"],
        "duration": None,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


async def new_user(session):
    url = "http://0.0.0.0:4000/user/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": ["gpt-4", "text-embedding-ada-002", "dall-e-2"],
        "duration": None,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


async def chat_completion(session, key):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    }

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
    - Create key
    Make chat completion call
    - Create user
    make chat completion call
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        await chat_completion(session=session, key=key)
        key_gen = await new_user(session=session)
        key_2 = key_gen["key"]
        await chat_completion(session=session, key=key_2)


async def completion(session, key):
    url = "http://0.0.0.0:4000/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {"model": "gpt-4", "prompt": "Hello!"}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")


@pytest.mark.asyncio
async def test_completion():
    """
    - Create key
    Make chat completion call
    - Create user
    make chat completion call
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        await completion(session=session, key=key)
        key_gen = await new_user(session=session)
        key_2 = key_gen["key"]
        await completion(session=session, key=key_2)


async def embeddings(session, key):
    url = "http://0.0.0.0:4000/embeddings"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "text-embedding-ada-002",
        "input": ["hello world"],
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")


@pytest.mark.asyncio
async def test_embeddings():
    """
    - Create key
    Make embeddings call
    - Create user
    make embeddings call
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        await embeddings(session=session, key=key)
        key_gen = await new_user(session=session)
        key_2 = key_gen["key"]
        await embeddings(session=session, key=key_2)


async def image_generation(session, key):
    url = "http://0.0.0.0:4000/images/generations"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "dall-e-2",
        "prompt": "A cute baby sea otter",
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")


@pytest.mark.asyncio
async def test_image_generation():
    """
    - Create key
    Make embeddings call
    - Create user
    make embeddings call
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        await image_generation(session=session, key=key)
        key_gen = await new_user(session=session)
        key_2 = key_gen["key"]
        await image_generation(session=session, key=key_2)
