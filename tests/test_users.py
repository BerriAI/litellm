# What this tests ?
## Tests /user endpoints.
import pytest
import asyncio
import aiohttp
import time
from openai import AsyncOpenAI
from test_team import list_teams
from typing import Optional


async def new_user(session, i, user_id=None, budget=None, budget_duration=None):
    url = "http://0.0.0.0:4000/user/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": ["azure-models"],
        "aliases": {"mistral-7b": "gpt-3.5-turbo"},
        "duration": None,
        "max_budget": budget,
        "budget_duration": budget_duration,
    }

    if user_id is not None:
        data["user_id"] = user_id

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
async def test_user_new():
    """
    Make 20 parallel calls to /user/new. Assert all worked.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [new_user(session, i) for i in range(1, 11)]
        await asyncio.gather(*tasks)


async def get_user_info(session, get_user, call_user, view_all: Optional[bool] = None):
    """
    Make sure only models user has access to are returned
    """
    if view_all is True:
        url = "http://0.0.0.0:4000/user/info"
    else:
        url = f"http://0.0.0.0:4000/user/info?user_id={get_user}"
    headers = {
        "Authorization": f"Bearer {call_user}",
        "Content-Type": "application/json",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()
        print(response_text)
        print()

        if status != 200:
            if call_user != get_user:
                return status
            else:
                print(f"call_user: {call_user}; get_user: {get_user}")
                raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


@pytest.mark.asyncio
async def test_user_info():
    """
    Get user info
    - as admin
    - as user themself
    - as random
    """
    get_user = f"krrish_{time.time()}@berri.ai"
    async with aiohttp.ClientSession() as session:
        key_gen = await new_user(session, 0, user_id=get_user)
        key = key_gen["key"]
        ## as admin ##
        resp = await get_user_info(
            session=session, get_user=get_user, call_user="sk-1234"
        )
        assert isinstance(resp["user_info"], dict)
        assert len(resp["user_info"]) > 0
        ## as user themself ##
        resp = await get_user_info(session=session, get_user=get_user, call_user=key)
        assert isinstance(resp["user_info"], dict)
        assert len(resp["user_info"]) > 0
        # as random user #
        key_gen = await new_user(session=session, i=0)
        random_key = key_gen["key"]
        status = await get_user_info(
            session=session, get_user=get_user, call_user=random_key
        )
        assert status == 403


@pytest.mark.asyncio
async def test_user_update():
    """
    Create user
    Update user access to new model
    Make chat completion call
    """
    pass


@pytest.mark.skip(reason="Frequent check on ci/cd leads to read timeout issue.")
@pytest.mark.asyncio
async def test_users_budgets_reset():
    """
    - Create key with budget and 5s duration
    - Get 'reset_at' value
    - wait 5s
    - Check if value updated
    """
    get_user = f"krrish_{time.time()}@berri.ai"
    async with aiohttp.ClientSession() as session:
        key_gen = await new_user(
            session, 0, user_id=get_user, budget=10, budget_duration="5s"
        )
        key = key_gen["key"]
        user_info = await get_user_info(
            session=session, get_user=get_user, call_user=key
        )
        reset_at_init_value = user_info["user_info"]["budget_reset_at"]
        i = 0
        reset_at_new_value = None
        while i < 3:
            await asyncio.sleep(70)
            user_info = await get_user_info(
                session=session, get_user=get_user, call_user=key
            )
            reset_at_new_value = user_info["user_info"]["budget_reset_at"]
            try:
                assert reset_at_init_value != reset_at_new_value
                break
            except Exception:
                i + 1
        assert reset_at_init_value != reset_at_new_value


async def chat_completion(session, key, model="gpt-4"):
    client = AsyncOpenAI(api_key=key, base_url="http://0.0.0.0:4000")
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": f"Hello! {time.time()}"},
    ]

    data = {
        "model": model,
        "messages": messages,
    }
    response = await client.chat.completions.create(**data)


async def chat_completion_streaming(session, key, model="gpt-4"):
    client = AsyncOpenAI(api_key=key, base_url="http://0.0.0.0:4000")
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": f"Hello! {time.time()}"},
    ]

    data = {"model": model, "messages": messages, "stream": True}
    response = await client.chat.completions.create(**data)
    async for chunk in response:
        continue


@pytest.mark.skip(reason="Global proxy now tracked via `/global/spend/logs`")
@pytest.mark.asyncio
async def test_global_proxy_budget_update():
    """
    - Get proxy current spend
    - Make chat completion call (normal)
    - Assert spend increased
    - Make chat completion call (streaming)
    - Assert spend increased
    """
    get_user = f"litellm-proxy-budget"
    async with aiohttp.ClientSession() as session:
        user_info = await get_user_info(
            session=session, get_user=get_user, call_user="sk-1234"
        )
        original_spend = user_info["user_info"]["spend"]
        await chat_completion(session=session, key="sk-1234")
        await asyncio.sleep(5)  # let db update
        user_info = await get_user_info(
            session=session, get_user=get_user, call_user="sk-1234"
        )
        new_spend = user_info["user_info"]["spend"]
        print(f"new_spend: {new_spend}; original_spend: {original_spend}")
        assert new_spend > original_spend
        await chat_completion_streaming(session=session, key="sk-1234")
        await asyncio.sleep(5)  # let db update
        user_info = await get_user_info(
            session=session, get_user=get_user, call_user="sk-1234"
        )
        new_new_spend = user_info["user_info"]["spend"]
        print(f"new_spend: {new_spend}; original_spend: {original_spend}")
        assert new_new_spend > new_spend
