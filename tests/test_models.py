# What this tests ?
## Tests /models and /model/* endpoints

import pytest
import asyncio
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()


async def generate_key(session, models=[]):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}", "Content-Type": "application/json"}
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


async def add_models(
    session, model_id="123", model_name="azure-gpt-3.5", key=os.environ["LITELLM_MASTER_KEY"], team_id=None
):
    url = "http://0.0.0.0:4000/model/new"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    data = {
        "model_name": model_name,
        "litellm_params": {
            "model": "openai/gpt-4.1-nano",
            "api_key": "os.environ/OPENAI_API_KEY",
        },
        "model_info": {"id": model_id},
    }

    if team_id:
        data["model_info"]["team_id"] = team_id

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        print(f"Add models {response_text}")
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        response_json = await response.json()
        return response_json


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


async def delete_model(session, model_id="123", key=os.environ["LITELLM_MASTER_KEY"]):
    """
    Make sure only models user has access to are returned
    """
    url = "http://0.0.0.0:4000/model/delete"
    headers = {
        "Authorization": f"Bearer {key}",
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


@pytest.mark.skip(
    reason="Requires live proxy + OPENAI_API_KEY. Deterministic mock version in tests/unit/proxy/management_endpoints/test_model_management_endpoints.py::TestAddAndDeleteModelLifecycle"
)
@pytest.mark.asyncio
async def test_add_and_delete_models():
    """
    - Add model
    - Call new model -> expect to pass
    - Delete model
    - Call model -> expect to fail
    """
    from litellm._uuid import uuid

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
        except Exception:
            pass


