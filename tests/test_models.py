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


async def update_model(
    session, model_id="123", model_name="azure-gpt-3.5", key="sk-1234"
):
    url = "http://0.0.0.0:4000/model/update"
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


async def add_model_for_health_checking(session, model_id="123"):
    url = "http://0.0.0.0:4000/model/new"
    headers = {
        "Authorization": f"Bearer sk-1234",
        "Content-Type": "application/json",
    }

    data = {
        "model_name": f"azure-model-health-check-{model_id}",
        "litellm_params": {
            "model": "gpt-4.1-nano",
            "api_key": os.getenv("OPENAI_API_KEY"),
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


async def get_specific_model_info_v2(session, key, model_name):
    url = "http://0.0.0.0:4000/v2/model/info?debug=True&model=" + model_name
    print("running /model/info check for model=", model_name)

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

        _json_response = await response.json()
        print("JSON response from /v2/model/info?model=", model_name, _json_response)

        _model_info = _json_response["data"]
        assert len(_model_info) == 1, f"Expected 1 model, got {len(_model_info)}"

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return _model_info[0]


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
    from litellm._uuid import uuid

    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        master_key = "sk-1234"
        model_id = str(uuid.uuid4())
        model_name = f"azure-model-health-check-{model_id}"
        print("adding model", model_name)
        await add_model_for_health_checking(session=session, model_id=model_id)
        _old_model_info = await get_specific_model_info_v2(
            session=session, key=key, model_name=model_name
        )
        print("model info before test", _old_model_info)

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
            _healthy_endpooint["model"] == "gpt-4.1-nano"
        )  # this is the model that got added

        # assert httpx client is is unchanges

        await asyncio.sleep(10)

        _model_info_after_test = await get_specific_model_info_v2(
            session=session, key=key, model_name=model_name
        )

        print("model info after test", _model_info_after_test)
        old_openai_client = _old_model_info["openai_client"]
        new_openai_client = _model_info_after_test["openai_client"]
        print("old openai client", old_openai_client)
        print("new openai client", new_openai_client)

        """
        PROD TEST - This is extremly important 
        The OpenAI client used should be the same after 30 seconds
        It is a serious bug if the openai client does not match here
        """
        assert (
            old_openai_client == new_openai_client
        ), "OpenAI client does not match for the same model after 30 seconds"

        # cleanup
        await delete_model(session=session, model_id=model_id)


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

        expected_models = [
            "anthropic/claude-3-5-haiku-20241022",
            "anthropic/claude-3-opus-20240229",
        ]

        model_group_info = await get_model_group_info(session=session, key="sk-1234")
        print(model_group_info)

        has_anthropic_claude_3_5_haiku = False
        has_anthropic_claude_3_opus = False
        for model in model_group_info["data"]:
            if model["model_group"] == "anthropic/claude-3-5-haiku-20241022":
                has_anthropic_claude_3_5_haiku = True
            if model["model_group"] == "anthropic/claude-3-opus-20240229":
                has_anthropic_claude_3_opus = True

        assert has_anthropic_claude_3_5_haiku and has_anthropic_claude_3_opus


@pytest.mark.asyncio
async def test_team_model_e2e():
    """
    Test team model e2e

    - create team
    - create user
    - add user to team as admin
    - add model to team
    - update model
    - delete model
    """
    from tests.test_users import new_user
    from tests.test_team import new_team
    from litellm._uuid import uuid

    async with aiohttp.ClientSession() as session:
        # Creat a user
        user_data = await new_user(session=session, i=0)
        user_id = user_data["user_id"]
        user_api_key = user_data["key"]

        # Create a team
        member_list = [
            {"role": "admin", "user_id": user_id},
        ]
        team_data = await new_team(session=session, member_list=member_list, i=0)
        team_id = team_data["team_id"]

        model_id = str(uuid.uuid4())
        model_name = "my-test-model"
        # Add model to team
        model_data = await add_models(
            session=session,
            model_id=model_id,
            model_name=model_name,
            key=user_api_key,
            team_id=team_id,
        )
        model_id = model_data["model_id"]

        # Update model
        model_data = await update_model(
            session=session, model_id=model_id, model_name=model_name, key=user_api_key
        )
        model_id = model_data["model_id"]

        # Delete model
        await delete_model(session=session, model_id=model_id, key=user_api_key)
