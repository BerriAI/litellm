# What this tests ?
## Tests /team endpoints.
import pytest
import asyncio
import aiohttp
import time, uuid
from openai import AsyncOpenAI
from typing import Optional


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


async def new_user(
    session,
    i,
    user_id=None,
    budget=None,
    budget_duration=None,
    models=["azure-models"],
    team_id=None,
    user_email=None,
):
    url = "http://0.0.0.0:4000/user/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": models,
        "aliases": {"mistral-7b": "gpt-3.5-turbo"},
        "duration": None,
        "max_budget": budget,
        "budget_duration": budget_duration,
        "user_email": user_email,
    }

    if user_id is not None:
        data["user_id"] = user_id

    if team_id is not None:
        data["team_id"] = team_id

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"Response {i} (Status code: {status}):")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request {i} did not return a 200 status code: {status}")

        return await response.json()


async def add_member(
    session, i, team_id, user_id=None, user_email=None, max_budget=None, members=None
):
    url = "http://0.0.0.0:4000/team/member_add"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"team_id": team_id, "member": {"role": "user"}}
    if user_email is not None:
        data["member"]["user_email"] = user_email
    elif user_id is not None:
        data["member"]["user_id"] = user_id
    elif members is not None:
        data["member"] = members

    if max_budget is not None:
        data["max_budget_in_team"] = max_budget

    print("sent data: {}".format(data))
    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"ADD MEMBER Response {i} (Status code: {status}):")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request {i} did not return a 200 status code: {status}")

        return await response.json()


async def delete_member(session, i, team_id, user_id=None, user_email=None):
    url = "http://0.0.0.0:4000/team/member_delete"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"team_id": team_id}
    if user_id is not None:
        data["user_id"] = user_id
    elif user_email is not None:
        data["user_email"] = user_email

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


async def list_teams(
    session,
    i,
):
    url = "http://0.0.0.0:4000/team/list"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}

    async with session.get(url, headers=headers) as response:
        status = response.status
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
            pytest.fail("Expected call to fail")
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
    - Add 3 users (doesn't exist in db)
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
        ## Create 10 normal users
        members = [
            {"role": "user", "user_id": f"krrish_{uuid.uuid4()}@berri.ai"}
            for _ in range(10)
        ]
        await add_member(
            session=session, i=0, team_id=team_data["team_id"], members=members
        )
        ## ASSERT TEAM SIZE
        team_info = await get_team_info(
            session=session, get_team=team_data["team_id"], call_key="sk-1234"
        )

        assert len(team_info["team_info"]["members_with_roles"]) == 12

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
async def test_team_member_add_email():
    from test_users import get_user_info

    async with aiohttp.ClientSession() as session:
        ## Create admin
        admin_user = f"{uuid.uuid4()}"
        await new_user(session=session, i=0, user_id=admin_user)
        ## Create team with 1 admin and 1 user
        member_list = [
            {"role": "admin", "user_id": admin_user},
        ]
        team_data = await new_team(session=session, i=0, member_list=member_list)
        ## Add 1 user via email
        user_email = "krrish{}@berri.ai".format(uuid.uuid4())
        new_user_info = await new_user(session=session, i=0, user_email=user_email)
        new_member = {"role": "user", "user_email": user_email}
        await add_member(
            session=session, i=0, team_id=team_data["team_id"], members=[new_member]
        )

        ## check user info to confirm user is in team
        updated_user_info = await get_user_info(
            session=session, get_user=new_user_info["user_id"], call_user="sk-1234"
        )

        print(updated_user_info)

        ## check if team in user table
        is_team_in_list: bool = False
        for team in updated_user_info["teams"]:
            if team_data["team_id"] == team["team_id"]:
                is_team_in_list = True
        assert is_team_in_list


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


@pytest.mark.parametrize("dimension", ["user_id", "user_email"])
@pytest.mark.asyncio
async def test_member_delete(dimension):
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
        normal_user_email = "{}@berri.ai".format(normal_user)
        print(f"normal_user: {normal_user}")
        await new_user(
            session=session, i=0, user_id=normal_user, user_email=normal_user_email
        )
        ## Create team with 1 admin and 1 user
        member_list = [
            {"role": "admin", "user_id": admin_user},
        ]
        if dimension == "user_id":
            member_list.append({"role": "user", "user_id": normal_user})
        elif dimension == "user_email":
            member_list.append({"role": "user", "user_email": normal_user_email})
        team_data = await new_team(session=session, i=0, member_list=member_list)

        user_in_team = False
        for member in team_data["members_with_roles"]:
            if dimension == "user_id" and member["user_id"] == normal_user:
                user_in_team = True
            elif (
                dimension == "user_email" and member["user_email"] == normal_user_email
            ):
                user_in_team = True

        assert (
            user_in_team is True
        ), "User not in team. Team list={}, User details - id={}, email={}. Dimension={}".format(
            team_data["members_with_roles"], normal_user, normal_user_email, dimension
        )
        # Delete member
        if dimension == "user_id":
            updated_team_data = await delete_member(
                session=session, i=0, team_id=team_data["team_id"], user_id=normal_user
            )
        elif dimension == "user_email":
            updated_team_data = await delete_member(
                session=session,
                i=0,
                team_id=team_data["team_id"],
                user_email=normal_user_email,
            )
        print(f"updated_team_data: {updated_team_data}")
        user_in_team = False
        for member in team_data["members_with_roles"]:
            if dimension == "user_id" and member["user_id"] == normal_user:
                user_in_team = True
            elif (
                dimension == "user_email" and member["user_email"] == normal_user_email
            ):
                user_in_team = True

        assert user_in_team is True


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
async def test_users_in_team_budget():
    """
    - Create Team
    - Create User
    - Add User to team with budget = 0.0000001
    - Make Call 1 -> pass
    - Make Call 2 -> fail
    """
    get_user = f"krrish_{time.time()}@berri.ai"
    async with aiohttp.ClientSession() as session:
        team = await new_team(session, 0, user_id=get_user)
        print("New team=", team)
        key_gen = await new_user(
            session,
            0,
            user_id=get_user,
            budget=10,
            budget_duration="5s",
            team_id=team["team_id"],
            models=["fake-openai-endpoint"],
        )
        key = key_gen["key"]

        # Add user to team
        await add_member(
            session, 0, team_id=team["team_id"], user_id=get_user, max_budget=0.0000001
        )

        # Call 1
        result = await chat_completion(session, key, model="fake-openai-endpoint")
        print("Call 1 passed", result)

        await asyncio.sleep(2)

        # Call 2
        try:
            await chat_completion(session, key, model="fake-openai-endpoint")
            pytest.fail(
                "Call 2 should have failed. The user crossed their budget within their team"
            )
        except Exception as e:
            print("got exception, this is expected")
            print(e)
            assert "Budget has been exceeded" in str(e)

        ## Check user info
        user_info = await get_user_info(session, get_user, call_user="sk-1234")

        assert (
            user_info["teams"][0]["team_memberships"][0]["litellm_budget_table"][
                "max_budget"
            ]
            == 0.0000001
        )
