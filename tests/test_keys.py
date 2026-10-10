# What this tests ?
## Tests /key endpoints.

import pytest
import asyncio
import aiohttp
import os
from typing import Optional


async def generate_key(
    session,
    i,
    budget=None,
    budget_duration=None,
    models=["azure-models", "gpt-4", "gpt-image-1"],
    max_parallel_requests: Optional[int] = None,
    user_id: Optional[str] = None,
    team_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    calling_key=os.environ["LITELLM_MASTER_KEY"],
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
async def test_key_gen_bad_key():
    """
    Test if you can create a key with a non-admin key, even with UI setup
    """
    async with aiohttp.ClientSession() as session:
        ## LOGIN TO UI
        form_data = {"username": "admin", "password": os.environ["LITELLM_MASTER_KEY"]}
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


        # assert rounded_response_cost == rounded_key_info_spend
