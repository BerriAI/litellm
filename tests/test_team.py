# What this tests ?
## Tests /team endpoints.
import pytest
import asyncio
import aiohttp
import time, uuid
from openai import AsyncOpenAI


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


async def generate_key(
    session,
    i,
    budget=None,
    budget_duration=None,
    models=["azure-models", "gpt-4", "dall-e-3"],
    team_id=None,
):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": models,
        "duration": None,
        "max_budget": budget,
        "budget_duration": budget_duration,
    }
    if team_id is not None:
        data["team_id"] = team_id

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


async def new_team(session, i, user_id=None, member_list=None):
    url = "http://0.0.0.0:4000/team/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "team_alias": "my-new-team",
    }
    if user_id is not None:
        data["members_with_roles"] = [{"role": "user", "user_id": user_id}]
    elif member_list is not None:
        data["members_with_roles"] = member_list

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"Response {i} (Status code: {status}):")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request {i} did not return a 200 status code: {status}")

        return await response.json()


async def update_team(
    session,
    i,
    team_id,
    user_id=None,
    member_list=None,
):
    url = "http://0.0.0.0:4000/team/update"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "team_id": team_id,
    }
    if user_id is not None:
        data["members_with_roles"] = [{"role": "user", "user_id": user_id}]
    elif member_list is not None:
        data["members_with_roles"] = member_list

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"Response {i} (Status code: {status}):")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request {i} did not return a 200 status code: {status}")

        return await response.json()


async def delete_team(
    session,
    i,
    team_id,
):
    url = "http://0.0.0.0:4000/team/delete"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "team_ids": [team_id],
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
    user_id = f"{uuid.uuid4()}"
    async with aiohttp.ClientSession() as session:
        new_user(session=session, i=0, user_id=user_id)
        tasks = [new_team(session, i, user_id=user_id) for i in range(1, 11)]
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


@pytest.mark.asyncio
async def test_team_update():
    """
    - Create team with 1 admin, 1 user
    - Create new user
    - Replace existing user with new user in team
    """
    async with aiohttp.ClientSession() as session:
        ## Create admin
        admin_user = f"{uuid.uuid4()}"
        await new_user(session=session, i=0, user_id=admin_user)
        ## Create normal user
        normal_user = f"{uuid.uuid4()}"
        await new_user(session=session, i=0, user_id=normal_user)
        ## Create team with 1 admin and 1 user
        member_list = [
            {"role": "admin", "user_id": admin_user},
            {"role": "user", "user_id": normal_user},
        ]
        team_data = await new_team(session=session, i=0, member_list=member_list)
        ## Create new normal user
        new_normal_user = f"{uuid.uuid4()}"
        await new_user(session=session, i=0, user_id=new_normal_user)
        ## Update member list
        member_list = [
            {"role": "admin", "user_id": admin_user},
            {"role": "user", "user_id": new_normal_user},
        ]
        team_data = await update_team(
            session=session, i=0, member_list=member_list, team_id=team_data["team_id"]
        )


@pytest.mark.asyncio
async def test_team_delete():
    """
    - Create team
    - Create key for team
    - Check if key works
    - Delete team
    """
    async with aiohttp.ClientSession() as session:
        ## Create admin
        admin_user = f"{uuid.uuid4()}"
        await new_user(session=session, i=0, user_id=admin_user)
        ## Create normal user
        normal_user = f"{uuid.uuid4()}"
        await new_user(session=session, i=0, user_id=normal_user)
        ## Create team with 1 admin and 1 user
        member_list = [
            {"role": "admin", "user_id": admin_user},
            {"role": "user", "user_id": normal_user},
        ]
        team_data = await new_team(session=session, i=0, member_list=member_list)
        ## Create key
        key_gen = await generate_key(session=session, i=0, team_id=team_data["team_id"])
        key = key_gen["key"]
        ## Test key
        response = await chat_completion(session=session, key=key)
        ## Delete team
        await delete_team(session=session, i=0, team_id=team_data["team_id"])
