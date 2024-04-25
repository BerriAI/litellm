# What this tests ?
## Tests /models and /model/* endpoints

import pytest
import asyncio
import aiohttp
import os
import dotenv
from dotenv import load_dotenv

load_dotenv()


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
        print("response from /models")
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


async def add_models(session, model_id="123", model_name="azure-gpt-3.5"):
    url = "http://0.0.0.0:4000/model/new"
    headers = {
        "Authorization": f"Bearer sk-1234",
        "Content-Type": "application/json",
    }

    data = {
        "model_name": model_name,
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

        response_json = await response.json()
        return response_json


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


async def chat_completion(session, key, model="azure-gpt-3.5"):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
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
async def test_add_and_delete_models():
    """
    - Add model
    - Call new model -> expect to pass
    - Delete model
    - Call model -> expect to fail
    """
    import uuid

    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        model_id = f"12345_{uuid.uuid4()}"
        model_name = f"{uuid.uuid4()}"
        response = await add_models(
            session=session, model_id=model_id, model_name=model_name
        )
        assert response["model_id"] == model_id
        await asyncio.sleep(10)
        await chat_completion(session=session, key=key, model=model_name)
        await delete_model(session=session, model_id=model_id)
        try:
            await chat_completion(session=session, key=key, model=model_name)
            pytest.fail(f"Expected call to fail.")
        except:
            pass


async def add_model_for_health_checking(session, model_id="123"):
    url = "http://0.0.0.0:4000/model/new"
    headers = {
        "Authorization": f"Bearer sk-1234",
        "Content-Type": "application/json",
    }

    data = {
        "model_name": f"azure-model-health-check-{model_id}",
        "litellm_params": {
            "model": "azure/chatgpt-v-2",
            "api_key": os.getenv("AZURE_API_KEY"),
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


async def get_model_info_v2(session, key):
    url = "http://0.0.0.0:4000/v2/model/info"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()
        print("response from v2/model/info")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")


async def get_model_health(session, key, model_name):
    url = "http://0.0.0.0:4000/health?model=" + model_name
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.json()
        print("response from /health?model=", model_name)
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
    return response_text


@pytest.mark.asyncio
async def test_add_model_run_health():
    """
    Add model
    Call /model/info and v2/model/info
    -> Admin UI calls v2/model/info
    Call /chat/completions
    Call /health
    -> Ensure the health check for the endpoint is working as expected
    """
    import uuid

    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        master_key = "sk-1234"
        model_id = str(uuid.uuid4())
        model_name = f"azure-model-health-check-{model_id}"
        print("adding model", model_name)
        await add_model_for_health_checking(session=session, model_id=model_id)
        await asyncio.sleep(30)
        print("calling /model/info")
        await get_model_info(session=session, key=key)
        print("calling v2/model/info")
        await get_model_info_v2(session=session, key=key)

        print("calling /chat/completions -> expect to work")
        await chat_completion(session=session, key=key, model=model_name)

        print("calling /health?model=", model_name)
        _health_info = await get_model_health(
            session=session, key=master_key, model_name=model_name
        )
        _healthy_endpooint = _health_info["healthy_endpoints"][0]

        assert _health_info["healthy_count"] == 1
        assert (
            _healthy_endpooint["model"] == "azure/chatgpt-v-2"
        )  # this is the model that got added

        # cleanup
        await delete_model(session=session, model_id=model_id)
