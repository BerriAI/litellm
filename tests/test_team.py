# What this tests ?
## Tests /team endpoints.
import pytest
import asyncio
import aiohttp
import time, uuid
from openai import AsyncOpenAI


async def new_user(
    session, i, user_id=None, budget=None, budget_duration=None, models=["azure-models"]
):
    url = "http://0.0.0.0:4000/user/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": models,
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


async def add_member(session, i, team_id, user_id=None, user_email=None):
    url = "http://0.0.0.0:4000/team/member_add"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"team_id": team_id, "member": {"role": "user"}}
    if user_email is not None:
        data["member"]["user_email"] = user_email
    elif user_id is not None:
        data["member"]["user_id"] = user_id

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"ADD MEMBER Response {i} (Status code: {status}):")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request {i} did not return a 200 status code: {status}")

        return await response.json()


async def delete_member(session, i, team_id, user_id):
    url = "http://0.0.0.0:4000/team/member_delete"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"team_id": team_id, "user_id": user_id}

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


async def new_team(session, i, user_id=None, member_list=None, model_aliases=None):
    import json

    url = "http://0.0.0.0:4000/team/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"team_alias": "my-new-team"}
    if user_id is not None:
        data["members_with_roles"] = [{"role": "user", "user_id": user_id}]
    elif member_list is not None:
        data["members_with_roles"] = member_list

    if model_aliases is not None:
        data["model_aliases"] = model_aliases

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


async def update_team(session, i, team_id, user_id=None, member_list=None, **kwargs):
    url = "http://0.0.0.0:4000/team/update"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"team_id": team_id, **kwargs}
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


async def get_global_teams_spend(
    session,
    i
):
    url = "http://0.0.0.0:4000/global/spend/teams"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()

        print(f"Response {i} (Status code: {status}):")
        print(response_text)

        if status != 200:
            raise Exception(f'Request {i} did not return a 200 status code: {status}')

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
    """
    Scenario 1:
    - test with admin key -> expect to work
    Scenario 2:
    - test with team key -> expect to work
    Scenario 3:
    - test with non-team key -> expect to fail
    """
    async with aiohttp.ClientSession() as session:
        """
        Scenario 1 - as admin
        """
        new_team_data = await new_team(
            session,
            0,
        )
        team_id = new_team_data["team_id"]
        ## as admin ##
        await get_team_info(session=session, get_team=team_id, call_key="sk-1234")
        """
        Scenario 2 - as team key
        """
        key_gen = await generate_key(session=session, i=0, team_id=team_id)
        key = key_gen["key"]

        await get_team_info(session=session, get_team=team_id, call_key=key)

        """
        Scenario 3 - as non-team key
        """
        key_gen = await generate_key(session=session, i=0)
        key = key_gen["key"]

        try:
            await get_team_info(session=session, get_team=team_id, call_key=key)
            pytest.fail(f"Expected call to fail")
        except Exception as e:
            pass


"""
- Create team 
- Add user (user exists in db)
- Update team 
- Check if it works
"""

"""
- Create team
- Add user (user doesn't exist in db)
- Update team 
- Check if it works
"""


@pytest.mark.asyncio
async def test_team_update_sc_2():
    """
    - Create team
    - Add 1 user (doesn't exist in db)
    - Change team alias
    - Check if it works
    - Assert team object unchanged besides team alias
    """
    async with aiohttp.ClientSession() as session:
        ## Create admin
        admin_user = f"{uuid.uuid4()}"
        await new_user(session=session, i=0, user_id=admin_user)
        ## Create team with 1 admin and 1 user
        member_list = [
            {"role": "admin", "user_id": admin_user},
        ]
        team_data = await new_team(session=session, i=0, member_list=member_list)
        ## Create new normal user
        new_normal_user = f"krrish_{uuid.uuid4()}@berri.ai"
        await add_member(
            session=session,
            i=0,
            team_id=team_data["team_id"],
            user_id=None,
            user_email=new_normal_user,
        )

        ## CHANGE TEAM ALIAS

        new_team_data = await update_team(
            session=session, i=0, team_id=team_data["team_id"], team_alias="my-new-team"
        )

        assert new_team_data["data"]["team_alias"] == "my-new-team"
        print(f"team_data: {team_data}")
        ## assert rest of object is the same
        for k, v in new_team_data["data"].items():
            if (
                k == "members_with_roles"
            ):  # assert 1 more member (role: "user", user_email: $user_email)
                len(new_team_data["data"][k]) == len(team_data[k]) + 1
            elif (
                k == "created_at"
                or k == "updated_at"
                or k == "model_spend"
                or k == "model_max_budget"
                or k == "model_id"
                or k == "litellm_organization_table"
                or k == "litellm_model_table"
            ):
                pass
            else:
                assert new_team_data["data"][k] == team_data[k]


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


@pytest.mark.asyncio
async def test_member_delete():
    """
    - Create team
    - Add member
    - Get team info (check if member in team)
    - Delete member
    - Get team info (check if member in team)
    """
    async with aiohttp.ClientSession() as session:
        # Create Team
        ## Create admin
        admin_user = f"{uuid.uuid4()}"
        await new_user(session=session, i=0, user_id=admin_user)
        ## Create normal user
        normal_user = f"{uuid.uuid4()}"
        print(f"normal_user: {normal_user}")
        await new_user(session=session, i=0, user_id=normal_user)
        ## Create team with 1 admin and 1 user
        member_list = [
            {"role": "admin", "user_id": admin_user},
            {"role": "user", "user_id": normal_user},
        ]
        team_data = await new_team(session=session, i=0, member_list=member_list)
        print(f"team_data: {team_data}")
        member_id_list = []
        for member in team_data["members_with_roles"]:
            member_id_list.append(member["user_id"])

        assert normal_user in member_id_list
        # Delete member
        updated_team_data = await delete_member(
            session=session, i=0, team_id=team_data["team_id"], user_id=normal_user
        )
        print(f"updated_team_data: {updated_team_data}")
        member_id_list = []
        for member in updated_team_data["members_with_roles"]:
            member_id_list.append(member["user_id"])

        assert normal_user not in member_id_list


@pytest.mark.asyncio
async def test_team_alias():
    """
    - Create team w/ model alias
    - Create key for team
    - Check if key works
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
        team_data = await new_team(
            session=session,
            i=0,
            member_list=member_list,
            model_aliases={"cheap-model": "gpt-3.5-turbo"},
        )
        ## Create key
        key_gen = await generate_key(
            session=session, i=0, team_id=team_data["team_id"], models=["gpt-3.5-turbo"]
        )
        key = key_gen["key"]
        ## Test key
        response = await chat_completion(session=session, key=key, model="cheap-model")

@pytest.mark.asyncio
async def test_global_teams_spend():
    """
    - Create team w/ model alias
    - Create key for team
    - Check if key works
    - Get global teams spend
    - Check if team id and alias both exist
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
        team_data = await new_team(
            session=session,
            i=0,
            member_list=member_list,
            model_aliases={"cheap-model": "gpt-3.5-turbo"},
        )

        ## Create key
        key_gen = await generate_key(
            session=session, i=0, team_id=team_data["team_id"], models=["gpt-3.5-turbo"]
        )
        key = key_gen["key"]

        ## Test key
        response = await chat_completion(session=session, key=key, model="cheap-model")

    # NOTE: /global/spend/teams takes a few seconds to update with newly created teams
    time.sleep(3)
    async with aiohttp.ClientSession() as session:
        ## Get team spend
        spend_data = await get_global_teams_spend(session=session, i=0)

        if "team_id" not in spend_data['total_spend_per_team'][0] or "team_alias" not in spend_data['total_spend_per_team'][0]:
            raise Exception(f'Total spend per team entries missing team_id or team_alias: {spend_data["total_spend_per_team"][0]}')

        team_id = team_data['team_id']
        team_alias = team_data['team_alias']

        if team_alias not in spend_data['teams']:
            raise Exception(f'Team alias "{team_alias}" not in global spend teams list: {spend_data["teams"]}')

        team_id_set = {team["team_id"] for team in spend_data['total_spend_per_team']}
        if team_id not in team_id_set:
            raise Exception(f'Team ID "{team_id}" not in global spend total_spend_per_team list: {team_id_set}')
