# What this tests ?
## Tests /models and /model/* endpoints

import pytest
import asyncio
import aiohttp


async def generate_key(session, models=[]):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": models,
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


async def get_models(session, key):
    url = "http://0.0.0.0:4000/models"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")


@pytest.mark.asyncio
async def test_get_models():
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        await get_models(session=session, key=key)


async def add_models(session, model_id="123"):
    url = "http://0.0.0.0:4000/model/new"
    headers = {
        "Authorization": f"Bearer sk-1234",
        "Content-Type": "application/json",
    }

    data = {
        "model_name": "azure-gpt-3.5",
        "litellm_params": {
            "model": "azure/chatgpt-v-2",
            "api_key": "os.environ/AZURE_API_KEY",
            "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com/",
            "api_version": "2023-05-15",
        },
        "model_info": {"id": model_id},
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"Add models {response_text}")
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")


async def get_model_info(session, key):
    """
    Make sure only models user has access to are returned
    """
    url = "http://0.0.0.0:4000/model/info"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    async with session.get(url, headers=headers) as response:
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
        "model": "azure-gpt-3.5",
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


@pytest.mark.asyncio
async def test_add_models():
    """
    Add model
    Call new model
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        await add_models(session=session)
        await chat_completion(session=session, key=key)


@pytest.mark.asyncio
async def test_get_models():
    """
    Get models user has access to
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, models=["gpt-4"])
        key = key_gen["key"]
        response = await get_model_info(session=session, key=key)
        models = [m["model_name"] for m in response["data"]]
        for m in models:
            assert m == "gpt-4"


async def delete_model(session, model_id="123"):
    """
    Make sure only models user has access to are returned
    """
    url = "http://0.0.0.0:4000/model/delete"
    headers = {
        "Authorization": f"Bearer sk-1234",
        "Content-Type": "application/json",
    }
    data = {"id": model_id}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


@pytest.mark.asyncio
async def test_delete_models():
    """
    Get models user has access to
    """
    model_id = "12345"
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        await add_models(session=session, model_id=model_id)
        await chat_completion(session=session, key=key)
        await delete_model(session=session, model_id=model_id)
