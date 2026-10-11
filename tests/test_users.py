import os
# What this tests ?
## Tests /user endpoints.
import pytest
import asyncio
import aiohttp
import time
from typing import Optional


async def new_user(
    session, i, user_id=None, budget=None, budget_duration=None, models=None
):
    url = "http://0.0.0.0:4000/user/new"
    headers = {"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}", "Content-Type": "application/json"}
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


import pytest


