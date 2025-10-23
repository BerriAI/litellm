# What this tests ?
## Tests /user endpoints.
import pytest
import asyncio
import aiohttp
import time
from openai import AsyncOpenAI
from tests.test_team import list_teams
from typing import Optional
from tests.test_keys import generate_key
from fastapi import HTTPException


async def new_user(
    session, i, user_id=None, budget=None, budget_duration=None, models=None
):
    url = "http://0.0.0.0:4000/user/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": models or ["azure-models"],
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


@pytest.mark.asyncio
async def test_user_model_access():
    """
    - Create user with model access
    - Create key with user
    - Call model that user has access to -> should work
    - Call wildcard model that user has access to -> should work
    - Call model that user does not have access to -> should fail
    - Call wildcard model that user does not have access to -> should fail
    """
    import openai

    async with aiohttp.ClientSession() as session:
        get_user = f"krrish_{time.time()}@berri.ai"
        await new_user(
            session=session,
            i=0,
            user_id=get_user,
            models=["good-model", "anthropic/*"],
        )

        result = await generate_key(
            session=session,
            i=0,
            user_id=get_user,
            models=[],  # assign no models. Allow inheritance from user
        )
        key = result["key"]

        await chat_completion(
            session=session,
            key=key,
            model="anthropic/claude-3-5-haiku-20241022",
        )

        await chat_completion(
            session=session,
            key=key,
            model="good-model",
        )

        with pytest.raises(openai.AuthenticationError):
            await chat_completion(
                session=session,
                key=key,
                model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            )

        with pytest.raises(openai.AuthenticationError):
            await chat_completion(
                session=session,
                key=key,
                model="groq/claude-3-5-haiku-20241022",
            )


import json
from litellm._uuid import uuid
import pytest
import aiohttp
from typing import Dict, Tuple


async def setup_test_users(session: aiohttp.ClientSession) -> Tuple[Dict, Dict]:
    """
    Create two test users and an additional key for the first user.
    Returns tuple of (user1_data, user2_data) where each contains user info and keys.
    """
    # Create two test users
    user1 = await new_user(
        session=session,
        i=0,
        budget=100,
        budget_duration="30d",
        models=["anthropic.claude-3-5-sonnet-20240620-v1:0"],
    )

    user2 = await new_user(
        session=session,
        i=1,
        budget=100,
        budget_duration="30d",
        models=["anthropic.claude-3-5-sonnet-20240620-v1:0"],
    )

    print("\nCreated two test users:")
    print(f"User 1 ID: {user1['user_id']}")
    print(f"User 2 ID: {user2['user_id']}")

    # Create an additional key for user1
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {user1['key']}",
    }

    key_payload = {
        "user_id": user1["user_id"],
        "duration": "7d",
        "key_alias": f"test_key_{uuid.uuid4()}",
        "models": ["anthropic.claude-3-5-sonnet-20240620-v1:0"],
    }

    print("\nGenerating additional key for user1...")
    key_response = await session.post(
        f"http://0.0.0.0:4000/key/generate", headers=headers, json=key_payload
    )

    assert key_response.status == 200, "Failed to generate additional key for user1"
    user1_additional_key = await key_response.json()

    print(f"\nGenerated key details:")
    print(json.dumps(user1_additional_key, indent=2))

    # Return both users' data including the additional key
    return {
        "user_data": user1,
        "additional_key": user1_additional_key,
        "headers": headers,
    }, {
        "user_data": user2,
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {user2['key']}",
        },
    }


async def print_response_details(response: aiohttp.ClientResponse) -> None:
    """Helper function to print response details"""
    print("\nResponse Details:")
    print(f"Status Code: {response.status}")
    print("\nResponse Content:")
    try:
        formatted_json = json.dumps(await response.json(), indent=2)
        print(formatted_json)
    except json.JSONDecodeError:
        print(await response.text())


@pytest.mark.asyncio
async def test_key_update_user_isolation():
    """Test that a user cannot update a key that belongs to another user"""
    async with aiohttp.ClientSession() as session:
        user1_data, user2_data = await setup_test_users(session)

        # Try to update the key to belong to user2
        update_payload = {
            "key": user1_data["additional_key"]["key"],
            "user_id": user2_data["user_data"][
                "user_id"
            ],  # Attempting to change ownership
            "metadata": {"purpose": "testing_user_isolation", "environment": "test"},
        }

        print("\nAttempting to update key ownership to user2...")
        update_response = await session.post(
            f"http://0.0.0.0:4000/key/update",
            headers=user1_data["headers"],  # Using user1's headers
            json=update_payload,
        )

        await print_response_details(update_response)

        # Verify update attempt was rejected
        assert (
            update_response.status == 403
        ), "Request should have been rejected with 403 status code"


@pytest.mark.asyncio
async def test_key_delete_user_isolation():
    """Test that a user cannot delete a key that belongs to another user"""
    async with aiohttp.ClientSession() as session:
        user1_data, user2_data = await setup_test_users(session)

        # Try to delete user1's additional key using user2's credentials
        delete_payload = {
            "keys": [user1_data["additional_key"]["key"]],
        }

        print("\nAttempting to delete user1's key using user2's credentials...")
        delete_response = await session.post(
            f"http://0.0.0.0:4000/key/delete",
            headers=user2_data["headers"],
            json=delete_payload,
        )

        await print_response_details(delete_response)

        # Verify delete attempt was rejected
        assert (
            delete_response.status == 403
        ), "Request should have been rejected with 403 status code"
