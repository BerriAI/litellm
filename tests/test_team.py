# What this tests ?
## Tests /team endpoints.
import pytest
import asyncio
import aiohttp
import time
from openai import AsyncOpenAI


async def new_team(
    session,
    i,
):
    url = "http://0.0.0.0:4000/team/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "team_alias": "my-new-team",
        "admins": ["user-1234"],
        "members": ["user-1234"],
    }
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
async def test_team_new():
    """
    Make 20 parallel calls to /user/new. Assert all worked.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [new_team(session, i) for i in range(1, 11)]
        await asyncio.gather(*tasks)


async def get_team_info(session, get_team, call_key):
    url = f"http://0.0.0.0:4000/team/info?team_id={get_team}"
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
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


@pytest.mark.asyncio
async def test_team_info():
    async with aiohttp.ClientSession() as session:
        new_team_data = await new_team(
            session,
            0,
        )
        team_id = new_team_data["team_id"]
        ## as admin ##
        await get_team_info(session=session, get_team=team_id, call_key="sk-1234")
