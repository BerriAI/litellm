# What this tests ?
## Tests /key endpoints.

import pytest
import asyncio, time, uuid
import aiohttp
from openai import AsyncOpenAI
import sys, os
from typing import Optional

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path
import litellm
from litellm.proxy._types import LitellmUserRoles


async def generate_team(
    session, models: Optional[list] = None, team_id: Optional[str] = None
):
    url = "http://0.0.0.0:4000/team/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    if team_id is None:
        team_id = "litellm-dashboard"
    data = {"team_id": team_id, "models": models}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"Response (Status code: {status}):")
        print(response_text)
        print()
        _json_response = await response.json()
        return _json_response


async def generate_user(
    session,
    user_role="app_owner",
):
    url = "http://0.0.0.0:4000/user/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "user_role": user_role,
        "team_id": "litellm-dashboard",
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"Response (Status code: {status}):")
        print(response_text)
        print()
        _json_response = await response.json()
        return _json_response


async def generate_key(
    session,
    i,
    budget=None,
    budget_duration=None,
    models=["azure-models", "gpt-4", "dall-e-3"],
    max_parallel_requests: Optional[int] = None,
    user_id: Optional[str] = None,
    team_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    calling_key="sk-1234",
):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {
        "Authorization": f"Bearer {calling_key}",
        "Content-Type": "application/json",
    }
    data = {
        "models": models,
        "aliases": {"mistral-7b": "gpt-3.5-turbo"},
        "duration": None,
        "max_budget": budget,
        "budget_duration": budget_duration,
        "max_parallel_requests": max_parallel_requests,
        "user_id": user_id,
        "team_id": team_id,
        "metadata": metadata,
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


@pytest.mark.asyncio
async def test_key_gen():
    async with aiohttp.ClientSession() as session:
        tasks = [generate_key(session, i) for i in range(1, 11)]
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_simple_key_gen():
    async with aiohttp.ClientSession() as session:
        key_data = await generate_key(session, i=0)
        key = key_data["key"]
        assert key_data["token"] is not None
        assert key_data["token"] != key
        assert key_data["token_id"] is not None
        assert key_data["created_at"] is not None
        assert key_data["updated_at"] is not None


@pytest.mark.asyncio
async def test_key_gen_bad_key():
    """
    Test if you can create a key with a non-admin key, even with UI setup
    """
    async with aiohttp.ClientSession() as session:
        ## LOGIN TO UI
        form_data = {"username": "admin", "password": "sk-1234"}
        async with session.post(
            "http://0.0.0.0:4000/login", data=form_data
        ) as response:
            assert (
                response.status == 200
            )  # Assuming the endpoint returns a 500 status code for error handling
            text = await response.text()
            print(text)
        ## create user key with admin key -> expect to work
        key_data = await generate_key(session=session, i=0, user_id="user-1234")
        key = key_data["key"]
        ## create new key with user key -> expect to fail
        try:
            await generate_key(
                session=session, i=0, user_id="user-1234", calling_key=key
            )
            pytest.fail("Expected to fail")
        except Exception as e:
            pass


async def update_key(session, get_key, metadata: Optional[dict] = None):
    """
    Make sure only models user has access to are returned
    """
    url = "http://0.0.0.0:4000/key/update"
    headers = {
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json",
    }
    data = {"key": get_key}

    if metadata is not None:
        data["metadata"] = metadata
    else:
        data.update({"models": ["gpt-4"], "duration": "120s"})

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


async def update_proxy_budget(session):
    """
    Make sure only models user has access to are returned
    """
    url = "http://0.0.0.0:4000/user/update"
    headers = {
        "Authorization": f"Bearer sk-1234",
        "Content-Type": "application/json",
    }
    data = {"user_id": "litellm-proxy-budget", "spend": 0}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


async def chat_completion(session, key, model="gpt-4"):
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

    for i in range(3):
        try:
            async with session.post(url, headers=headers, json=data) as response:
                status = response.status
                response_text = await response.text()

                print(response_text)
                print()

                if status != 200:
                    raise Exception(
                        f"Request did not return a 200 status code: {status}. Response: {response_text}"
                    )

                return await response.json()
        except Exception as e:
            if "Request did not return a 200 status code" in str(e):
                raise e
            else:
                pass


async def image_generation(session, key, model="dall-e-3"):
    url = "http://0.0.0.0:4000/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "prompt": "A cute baby sea otter",
    }

    for i in range(3):
        try:
            async with session.post(url, headers=headers, json=data) as response:
                status = response.status
                response_text = await response.text()
                print("/images/generations response", response_text)

                print()

                if status != 200:
                    raise Exception(
                        f"Request did not return a 200 status code: {status}. Response: {response_text}"
                    )

                return await response.json()
        except Exception as e:
            if "Request did not return a 200 status code" in str(e):
                raise e
            else:
                pass


async def chat_completion_streaming(session, key, model="gpt-4"):
    client = AsyncOpenAI(api_key=key, base_url="http://0.0.0.0:4000")
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": f"Hello! {time.time()}"},
    ]
    prompt_tokens = litellm.token_counter(model="gpt-35-turbo", messages=messages)
    data = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    response = await client.chat.completions.create(**data)

    content = ""
    async for chunk in response:
        content += chunk.choices[0].delta.content or ""

    print(f"content: {content}")

    completion_tokens = litellm.token_counter(
        model="gpt-35-turbo", text=content, count_response_tokens=True
    )

    return prompt_tokens, completion_tokens


@pytest.mark.parametrize("metadata", [{"test": "new"}, {}])
@pytest.mark.asyncio
async def test_key_update(metadata):
    """
    Create key
    Update key with new model
    Test key w/ model
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, i=0, metadata={"test": "test"})
        key = key_gen["key"]
        assert key_gen["metadata"]["test"] == "test"
        updated_key = await update_key(
            session=session,
            get_key=key,
            metadata=metadata,
        )
        print(f"updated_key['metadata']: {updated_key['metadata']}")
        assert updated_key["metadata"] == metadata
        await update_proxy_budget(session=session)  # resets proxy spend
        await chat_completion(session=session, key=key)


async def delete_key(session, get_key, auth_key="sk-1234"):
    """
    Delete key
    """
    url = "http://0.0.0.0:4000/key/delete"
    headers = {
        "Authorization": f"Bearer {auth_key}",
        "Content-Type": "application/json",
    }
    data = {"keys": [get_key]}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


@pytest.mark.asyncio
async def test_key_delete():
    """
    Delete key
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, i=0)
        key = key_gen["key"]
        await delete_key(
            session=session,
            get_key=key,
        )


async def get_key_info(session, call_key, get_key=None):
    """
    Make sure only models user has access to are returned
    """
    if get_key is None:
        url = "http://0.0.0.0:4000/key/info"
    else:
        url = f"http://0.0.0.0:4000/key/info?key={get_key}"
    headers = {
        "Authorization": f"Bearer {call_key}",
        "Content-Type": "application/json",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()
        print(response_text)
        print()

        if status != 200:
            if call_key != get_key:
                return status
            else:
                print(f"call_key: {call_key}; get_key: {get_key}")
                raise Exception(
                    f"Request did not return a 200 status code: {status}. Responses {response_text}"
                )
        return await response.json()


async def get_model_list(session, call_key, endpoint: str = "/v1/models"):
    """
    Make sure only models user has access to are returned
    """
    url = "http://0.0.0.0:4000" + endpoint
    headers = {
        "Authorization": f"Bearer {call_key}",
        "Content-Type": "application/json",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()
        print(response_text)
        print()

        if status != 200:
            raise Exception(
                f"Request did not return a 200 status code: {status}. Responses {response_text}"
            )
        return await response.json()


async def get_model_info(session, call_key):
    """
    Make sure only models user has access to are returned
    """
    url = "http://0.0.0.0:4000/model/info"
    headers = {
        "Authorization": f"Bearer {call_key}",
        "Content-Type": "application/json",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()
        print(response_text)
        print()

        if status != 200:
            raise Exception(
                f"Request did not return a 200 status code: {status}. Responses {response_text}"
            )
        return await response.json()


@pytest.mark.asyncio
async def test_key_info():
    """
    Get key info
    - as admin -> 200
    - as key itself -> 200
    - as non existent key -> 404
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, i=0)
        key = key_gen["key"]
        # as admin #
        await get_key_info(session=session, get_key=key, call_key="sk-1234")
        # as key itself #
        await get_key_info(session=session, get_key=key, call_key=key)

        # as key itself, use the auth param, and no query key needed
        await get_key_info(session=session, call_key=key)
        # as random key #
        random_key = f"sk-{uuid.uuid4()}"
        status = await get_key_info(session=session, get_key=random_key, call_key=key)
        assert status == 404


@pytest.mark.asyncio
async def test_model_info():
    """
    Get model info for models key has access to
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, i=0)
        key = key_gen["key"]
        # as admin #
        admin_models = await get_model_info(session=session, call_key="sk-1234")
        admin_models = admin_models["data"]
        # as key itself #
        user_models = await get_model_info(session=session, call_key=key)
        user_models = user_models["data"]

        assert len(admin_models) > len(user_models)
        assert len(user_models) > 0


async def get_spend_logs(session, request_id):
    url = f"http://0.0.0.0:4000/spend/logs?request_id={request_id}"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


@pytest.mark.skip(reason="Hanging on ci/cd")
@pytest.mark.asyncio
async def test_key_info_spend_values():
    """
    Test to ensure spend is correctly calculated
    - create key
    - make completion call
    - assert cost is expected value
    """

    async def retry_request(func, *args, _max_attempts=5, **kwargs):
        for attempt in range(_max_attempts):
            try:
                return await func(*args, **kwargs)
            except aiohttp.client_exceptions.ClientOSError as e:
                if attempt + 1 == _max_attempts:
                    raise  # re-raise the last ClientOSError if all attempts failed
                print(f"Attempt {attempt+1} failed, retrying...")

    async with aiohttp.ClientSession() as session:
        ## Test Spend Update ##
        # completion
        key_gen = await generate_key(session=session, i=0)
        key = key_gen["key"]
        response = await chat_completion(session=session, key=key)
        await asyncio.sleep(5)
        spend_logs = await retry_request(
            get_spend_logs, session=session, request_id=response["id"]
        )
        print(f"spend_logs: {spend_logs}")
        completion_tokens = spend_logs[0]["completion_tokens"]
        prompt_tokens = spend_logs[0]["prompt_tokens"]
        print(f"prompt_tokens: {prompt_tokens}; completion_tokens: {completion_tokens}")

        litellm.set_verbose = True
        prompt_cost, completion_cost = litellm.cost_per_token(
            model="gpt-35-turbo",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            custom_llm_provider="azure",
        )
        print("prompt_cost: ", prompt_cost, "completion_cost: ", completion_cost)
        response_cost = prompt_cost + completion_cost
        print(f"response_cost: {response_cost}")
        await asyncio.sleep(5)  # allow db log to be updated
        key_info = await get_key_info(session=session, get_key=key, call_key=key)
        print(
            f"response_cost: {response_cost}; key_info spend: {key_info['info']['spend']}"
        )
        rounded_response_cost = round(response_cost, 8)
        rounded_key_info_spend = round(key_info["info"]["spend"], 8)
        assert (
            rounded_response_cost == rounded_key_info_spend
        ), f"Expected cost= {rounded_response_cost} != Tracked Cost={rounded_key_info_spend}"


@pytest.mark.asyncio
@pytest.mark.flaky(retries=6, delay=2)
@pytest.mark.skip(reason="Temporarily skipping due to model change. Will be updated soon.")
async def test_aaaaakey_info_spend_values_streaming():
    """
    Test to ensure spend is correctly calculated.
    - create key
    - make completion call
    - assert cost is expected value
    """
    async with aiohttp.ClientSession() as session:
        ## streaming - azure
        key_gen = await generate_key(session=session, i=0)
        new_key = key_gen["key"]
        prompt_tokens, completion_tokens = await chat_completion_streaming(
            session=session, key=new_key
        )
        print(f"prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}")
        prompt_cost, completion_cost = litellm.cost_per_token(
            model="azure/gpt-4o",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        response_cost = prompt_cost + completion_cost
        await asyncio.sleep(8)  # allow db log to be updated
        print(f"new_key: {new_key}")
        key_info = await get_key_info(
            session=session, get_key=new_key, call_key=new_key
        )
        print(
            f"response_cost: {response_cost}; key_info spend: {key_info['info']['spend']}"
        )
        rounded_response_cost = round(response_cost, 8)
        rounded_key_info_spend = round(key_info["info"]["spend"], 8)
        assert (
            rounded_response_cost == rounded_key_info_spend
        ), f"Expected={rounded_response_cost}, Got={rounded_key_info_spend}"

@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.asyncio
async def test_key_info_spend_values_image_generation():
    """
    Test to ensure spend is correctly calculated
    - create key
    - make image gen call
    - assert cost is expected value
    """

    async def retry_request(func, *args, _max_attempts=5, **kwargs):
        for attempt in range(_max_attempts):
            try:
                return await func(*args, **kwargs)
            except aiohttp.client_exceptions.ClientOSError as e:
                if attempt + 1 == _max_attempts:
                    raise  # re-raise the last ClientOSError if all attempts failed
                print(f"Attempt {attempt+1} failed, retrying...")

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        ## Test Spend Update ##
        # completion
        key_gen = await generate_key(session=session, i=0)
        key = key_gen["key"]
        response = await image_generation(session=session, key=key)
        await asyncio.sleep(5)
        key_info = await retry_request(
            get_key_info, session=session, get_key=key, call_key=key
        )
        spend = key_info["info"]["spend"]
        assert spend > 0


@pytest.mark.skip(reason="Frequent check on ci/cd leads to read timeout issue.")
@pytest.mark.asyncio
async def test_key_with_budgets():
    """
    - Create key with budget and 5min duration
    - Get 'reset_at' value
    - wait 10min (budget reset runs every 10mins.)
    - Check if value updated
    """
    from litellm.proxy.utils import hash_token

    async def retry_request(func, *args, _max_attempts=5, **kwargs):
        for attempt in range(_max_attempts):
            try:
                return await func(*args, **kwargs)
            except aiohttp.client_exceptions.ClientOSError as e:
                if attempt + 1 == _max_attempts:
                    raise  # re-raise the last ClientOSError if all attempts failed
                print(f"Attempt {attempt+1} failed, retrying...")

    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(
            session=session, i=0, budget=10, budget_duration="5s"
        )
        key = key_gen["key"]
        hashed_token = hash_token(token=key)
        print(f"hashed_token: {hashed_token}")
        key_info = await get_key_info(session=session, get_key=key, call_key=key)
        reset_at_init_value = key_info["info"]["budget_reset_at"]
        reset_at_new_value = None
        i = 0
        for i in range(3):
            await asyncio.sleep(70)
            key_info = await retry_request(
                get_key_info, session=session, get_key=key, call_key=key
            )
            reset_at_new_value = key_info["info"]["budget_reset_at"]
            try:
                assert reset_at_init_value != reset_at_new_value
                break
            except Exception:
                i + 1
                await asyncio.sleep(10)
        assert reset_at_init_value != reset_at_new_value


@pytest.mark.asyncio
async def test_key_crossing_budget():
    """
    - Create key with budget with budget=0.00000001
    - make a /chat/completions call
    - wait 5s
    - make a /chat/completions call - should fail with key crossed it's budget

    - Check if value updated
    """
    from litellm.proxy.utils import hash_token

    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, i=0, budget=0.0000001)
        key = key_gen["key"]
        hashed_token = hash_token(token=key)
        print(f"hashed_token: {hashed_token}")

        response = await chat_completion(session=session, key=key)
        print("response 1: ", response)
        await asyncio.sleep(10)
        try:
            response = await chat_completion(session=session, key=key)
            pytest.fail("Should have failed - Key crossed it's budget")
        except Exception as e:
            assert "Budget has been exceeded!" in str(e)


@pytest.mark.skip(reason="AWS Suspended Account")
@pytest.mark.asyncio
async def test_key_info_spend_values_sagemaker():
    """
    Tests the sync streaming loop to ensure spend is correctly calculated.
    - create key
    - make completion call
    - assert cost is expected value
    """
    async with aiohttp.ClientSession() as session:
        ## streaming - sagemaker
        key_gen = await generate_key(session=session, i=0, models=[])
        new_key = key_gen["key"]
        prompt_tokens, completion_tokens = await chat_completion_streaming(
            session=session, key=new_key, model="sagemaker-completion-model"
        )
        await asyncio.sleep(5)  # allow db log to be updated
        key_info = await get_key_info(
            session=session, get_key=new_key, call_key=new_key
        )
        rounded_key_info_spend = round(key_info["info"]["spend"], 8)
        assert rounded_key_info_spend > 0
        # assert rounded_response_cost == rounded_key_info_spend


@pytest.mark.asyncio
async def test_key_rate_limit():
    """
    Tests backoff/retry logic on parallel request error.
    - Create key with max parallel requests 0
    - run 2 requests -> both fail
    - Create key with max parallel request 1
    - run 2 requests
    - both should succeed
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, i=0, max_parallel_requests=0)
        new_key = key_gen["key"]
        try:
            await chat_completion(session=session, key=new_key)
            pytest.fail(f"Expected this call to fail")
        except Exception as e:
            pass
        key_gen = await generate_key(session=session, i=0, max_parallel_requests=1)
        new_key = key_gen["key"]
        try:
            await chat_completion(session=session, key=new_key)
        except Exception as e:
            pytest.fail(f"Expected this call to work - {str(e)}")


@pytest.mark.asyncio
async def test_key_delete_ui():
    """
    Admin UI flow - DO NOT DELETE
    -> Create a key with user_id = "ishaan"
    -> Log on Admin UI, delete the key for user "ishaan"
    -> This should work, since we're on the admin UI and role == "proxy_admin
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, i=0, user_id="ishaan-smart")
        key = key_gen["key"]

        # generate a admin UI key
        team = await generate_team(session=session)
        admin_ui_key = await generate_user(
            session=session, user_role=LitellmUserRoles.PROXY_ADMIN.value
        )
        print(
            "trying to delete key=",
            key,
            "using key=",
            admin_ui_key["key"],
            " to auth in",
        )

        await delete_key(
            session=session,
            get_key=key,
            auth_key=admin_ui_key["key"],
        )


@pytest.mark.parametrize("model_access", ["all-team-models", "gpt-3.5-turbo"])
@pytest.mark.parametrize("model_access_level", ["key", "team"])
@pytest.mark.parametrize("model_endpoint", ["/v1/models", "/model/info"])
@pytest.mark.asyncio
async def test_key_model_list(model_access, model_access_level, model_endpoint):
    """
    Test if `/v1/models` works as expected.
    """
    async with aiohttp.ClientSession() as session:
        _models = [] if model_access == "all-team-models" else [model_access]
        team_id = "litellm_dashboard_{}".format(uuid.uuid4())
        new_team = await generate_team(
            session=session,
            models=_models if model_access_level == "team" else None,
            team_id=team_id,
        )
        key_gen = await generate_key(
            session=session,
            i=0,
            team_id=team_id,
            models=_models if model_access_level == "key" else [],
        )
        key = key_gen["key"]
        print(f"key: {key}")

        model_list = await get_model_list(
            session=session, call_key=key, endpoint=model_endpoint
        )
        print(f"model_list: {model_list}")

        if model_access == "all-team-models":
            if model_endpoint == "/v1/models":
                assert not isinstance(model_list["data"][0]["id"], list)
                assert isinstance(model_list["data"][0]["id"], str)
            elif model_endpoint == "/model/info":
                assert isinstance(model_list["data"], list)
                assert len(model_list["data"]) > 0
        if model_access == "gpt-3.5-turbo":
            if model_endpoint == "/v1/models":
                assert (
                    len(model_list["data"]) == 1
                ), "model_access={}, model_access_level={}".format(
                    model_access, model_access_level
                )
                assert model_list["data"][0]["id"] == model_access
            elif model_endpoint == "/model/info":
                assert isinstance(model_list["data"], list)
                assert len(model_list["data"]) == 1


@pytest.mark.asyncio
async def test_key_user_not_in_db():
    """
    - Create a key with unique user-id (not in db)
    - Check if key can make `/chat/completion` call
    """
    my_unique_user = str(uuid.uuid4())
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(
            session=session,
            i=0,
            user_id=my_unique_user,
        )
        key = key_gen["key"]
        try:
            await chat_completion(session=session, key=key)
        except Exception as e:
            pytest.fail(f"Expected this call to work - {str(e)}")


@pytest.mark.asyncio
async def test_key_over_budget():
    """
    Test if key over budget is handled as expected.
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, i=0, budget=0.0000001)
        key = key_gen["key"]
        try:
            await chat_completion(session=session, key=key)
        except Exception as e:
            pytest.fail(f"Expected this call to work - {str(e)}")

        ## CALL `/models` - expect to work
        model_list = await get_key_info(session=session, get_key=key, call_key=key)
        ## CALL `/chat/completions` - expect to fail    
        try:
            await chat_completion(session=session, key=key)
            pytest.fail("Expected this call to fail")
        except Exception as e:
            assert "Budget has been exceeded!" in str(e)
