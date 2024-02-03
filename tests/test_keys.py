# What this tests ?
## Tests /key endpoints.

import pytest
import asyncio, time
import aiohttp
from openai import AsyncOpenAI
import sys, os

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path
import litellm


async def generate_key(
    session, i, budget=None, budget_duration=None, models=["azure-models", "gpt-4"]
):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": models,
        "aliases": {"mistral-7b": "gpt-3.5-turbo"},
        "duration": None,
        "max_budget": budget,
        "budget_duration": budget_duration,
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


async def update_key(session, get_key):
    """
    Make sure only models user has access to are returned
    """
    url = "http://0.0.0.0:4000/key/update"
    headers = {
        "Authorization": f"Bearer sk-1234",
        "Content-Type": "application/json",
    }
    data = {"key": get_key, "models": ["gpt-4"]}

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


@pytest.mark.asyncio
async def test_key_update():
    """
    Create key
    Update key with new model
    Test key w/ model
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session, i=0)
        key = key_gen["key"]
        await update_key(
            session=session,
            get_key=key,
        )
        await update_proxy_budget(session=session)  # resets proxy spend
        await chat_completion(session=session, key=key)


async def delete_key(session, get_key):
    """
    Delete key
    """
    url = "http://0.0.0.0:4000/key/delete"
    headers = {
        "Authorization": f"Bearer sk-1234",
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


@pytest.mark.asyncio
async def test_key_info():
    """
    Get key info
    - as admin -> 200
    - as key itself -> 200
    - as random key -> 403
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
        key_gen = await generate_key(session=session, i=0)
        random_key = key_gen["key"]
        status = await get_key_info(session=session, get_key=key, call_key=random_key)
        assert status == 403


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


@pytest.mark.asyncio
async def test_key_info_spend_values():
    """
    Test to ensure spend is correctly calculated
    - create key
    - make completion call
    - assert cost is expected value
    """
    async with aiohttp.ClientSession() as session:
        ## Test Spend Update ##
        # completion
        key_gen = await generate_key(session=session, i=0)
        key = key_gen["key"]
        response = await chat_completion(session=session, key=key)
        await asyncio.sleep(5)
        spend_logs = await get_spend_logs(session=session, request_id=response["id"])
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
        assert rounded_response_cost == rounded_key_info_spend


@pytest.mark.asyncio
async def test_key_info_spend_values_streaming():
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
            model="azure/gpt-35-turbo",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        response_cost = prompt_cost + completion_cost
        await asyncio.sleep(5)  # allow db log to be updated
        print(f"new_key: {new_key}")
        key_info = await get_key_info(
            session=session, get_key=new_key, call_key=new_key
        )
        print(
            f"response_cost: {response_cost}; key_info spend: {key_info['info']['spend']}"
        )
        rounded_response_cost = round(response_cost, 8)
        rounded_key_info_spend = round(key_info["info"]["spend"], 8)
        assert rounded_response_cost == rounded_key_info_spend


@pytest.mark.asyncio
async def test_key_with_budgets():
    """
    - Create key with budget and 5s duration
    - Get 'reset_at' value
    - wait 5s
    - Check if value updated
    """
    from litellm.proxy.utils import hash_token

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
        while i < 3:
            await asyncio.sleep(30)
            key_info = await get_key_info(session=session, get_key=key, call_key=key)
            reset_at_new_value = key_info["info"]["budget_reset_at"]
            try:
                assert reset_at_init_value != reset_at_new_value
                break
            except:
                i + 1
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
        await asyncio.sleep(2)
        try:
            response = await chat_completion(session=session, key=key)
            pytest.fail("Should have failed - Key crossed it's budget")
        except Exception as e:
            assert "ExceededTokenBudget: Current spend for token:" in str(e)


@pytest.mark.asyncio
async def test_key_zinfo_spend_values_sagemaker():
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
