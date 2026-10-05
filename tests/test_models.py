# What this tests ?
## Tests /models and /model/* endpoints

import pytest
import asyncio
import aiohttp
import os
import dotenv
from typing import Final
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


async def get_models(session, key, only_model_access_groups=False):
    url = "http://0.0.0.0:4000/models"
    if only_model_access_groups:
        url += "?only_model_access_groups=True"
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
        return await response.json()


@pytest.mark.asyncio
async def test_get_models_multiple_tests():
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        models = await get_models(session=session, key=key)
        print(f"\n\nmodels: {models}")
        assert len(models["data"]) > 0

        ## Test only_model_access_groups
        new_response = await get_models(
            session=session, key=key, only_model_access_groups=True
        )
        print(f"\n\nnew_response: {new_response}")
        assert (
            len(new_response["data"]) == 0
        )  # no model access groups set on config.yaml


async def add_models(
    session, model_id="123", model_name="azure-gpt-3.5", key="sk-1234", team_id=None
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


async def get_model_info(session, key, litellm_model_id=None):
    """
    Make sure only models user has access to are returned
    """
    if litellm_model_id:
        url = f"http://0.0.0.0:4000/model/info?litellm_model_id={litellm_model_id}"
    else:
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


async def get_model_group_info(session, key):
    url = "http://0.0.0.0:4000/model_group/info"
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


@pytest.mark.asyncio
async def test_get_specific_model():
    """
    Return specific model info

    Ensure value of model_info is same as on `/model/info` (no id set)
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, models=["gpt-4"])
        key = key_gen["key"]
        response = await get_model_info(session=session, key=key)
        models = [m["model_name"] for m in response["data"]]
        model_specific_info = None
        for idx, m in enumerate(models):
            assert m == "gpt-4"
            litellm_model_id = response["data"][idx]["model_info"]["id"]
            model_specific_info = response["data"][idx]
        assert litellm_model_id is not None
        response = await get_model_info(
            session=session, key=key, litellm_model_id=litellm_model_id
        )
        assert response["data"][0]["model_info"]["id"] == litellm_model_id
        assert (
            response["data"][0] == model_specific_info
        ), "Model info is not the same. Got={}, Expected={}".format(
            response["data"][0], model_specific_info
        )


async def delete_model(session, model_id="123", key="sk-1234"):
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


@pytest.mark.asyncio
async def test_get_personal_models_for_user():
    """
    Test /models endpoint with team
    """
    from tests.test_users import new_user

    async with aiohttp.ClientSession() as session:
        # Creat a user
        user_data = await new_user(session=session, i=0, models=["gpt-3.5-turbo"])
        user_id = user_data["user_id"]
        user_api_key = user_data["key"]

        model_group_info = await get_model_group_info(session=session, key=user_api_key)
        print(model_group_info)

        assert len(model_group_info["data"]) == 1
        assert model_group_info["data"][0]["model_group"] == "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_model_group_info_e2e():
    """
    Test /model/group/info endpoint
    """
    async with aiohttp.ClientSession() as session:
        models = await get_models(session=session, key="sk-1234")
        print(models)

        model_group_info = await get_model_group_info(session=session, key="sk-1234")
        print(model_group_info)

        model_groups: Final = [m["model_group"] for m in model_group_info["data"]]

        assert "anthropic/*" not in model_groups, (
            f"Expected 'anthropic/*' to be expanded, but it was returned verbatim: {model_groups}"
        )
        assert any(m.startswith("anthropic/") for m in model_groups), (
            f"Expected concrete anthropic models from the 'anthropic/*' config entry, got: {model_groups}"
        )


