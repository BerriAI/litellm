# What this tests ?
## Tests /organization endpoints.
import pytest
import asyncio
import aiohttp
import time, uuid
from openai import AsyncOpenAI


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
            raise Exception(
                f"Request {i} did not return a 200 status code: {status}, response: {response_text}"
            )

        return await response.json()


async def new_organization(session, i, organization_alias, max_budget=None):
    url = "http://0.0.0.0:4000/organization/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "organization_alias": organization_alias,
        "models": ["azure-models"],
        "max_budget": max_budget,
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


async def add_member_to_org(
    session, i, organization_id, user_id, user_role="internal_user"
):
    url = "http://0.0.0.0:4000/organization/member_add"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "organization_id": organization_id,
        "member": {
            "user_id": user_id,
            "role": user_role,
        },
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


async def update_member_role(
    session, i, organization_id, user_id, user_role="internal_user"
):
    url = "http://0.0.0.0:4000/organization/member_update"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "organization_id": organization_id,
        "user_id": user_id,
        "role": user_role,
    }

    async with session.patch(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"Response {i} (Status code: {status}):")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request {i} did not return a 200 status code: {status}")

        return await response.json()


async def delete_member_from_org(session, i, organization_id, user_id):
    url = "http://0.0.0.0:4000/organization/member_delete"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "organization_id": organization_id,
        "user_id": user_id,
    }

    async with session.delete(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"Response {i} (Status code: {status}):")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request {i} did not return a 200 status code: {status}")

        return await response.json()


async def delete_organization(session, i, organization_id):
    url = "http://0.0.0.0:4000/organization/delete"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"organization_ids": [organization_id]}

    async with session.delete(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(f"Response {i} (Status code: {status}):")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request {i} did not return a 200 status code: {status}")

        return await response.json()


async def list_organization(session, i):
    url = "http://0.0.0.0:4000/organization/list"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_json = await response.json()

        print(f"Response {i} (Status code: {status}):")
        print()

        if status != 200:
            raise Exception(f"Request {i} did not return a 200 status code: {status}")

        # Assert that budget info is returned for each organization
        for org in response_json:
            assert "litellm_budget_table" in org, "Missing budget info in organization response"
            # Optionally also check that it's not null
            assert org["litellm_budget_table"] is not None, "Budget info is None"

        return response_json

@pytest.mark.flaky(retries=5, delay=1)
@pytest.mark.asyncio
async def test_organization_new():
    """
    Make 20 parallel calls to /organization/new. Assert all worked.
    """
    organization_alias = f"Organization: {uuid.uuid4()}"
    async with aiohttp.ClientSession() as session:
        tasks = [
            new_organization(
                session=session, i=0, organization_alias=organization_alias
            )
            for i in range(1, 20)
        ]
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_organization_list():
    """
    create 2 new Organizations
    check if the Organization list is not empty
    """
    organization_alias = f"Organization: {uuid.uuid4()}"
    async with aiohttp.ClientSession() as session:
        tasks = [
            new_organization(
                session=session, i=0, organization_alias=organization_alias
            )
            for i in range(1, 2)
        ]
        await asyncio.gather(*tasks)

        response_json = await list_organization(session, i=0)
        print(len(response_json))

        if len(response_json) == 0:
            raise Exception("Return empty list of organization")


@pytest.mark.asyncio
async def test_organization_delete():
    """
    create a new organization
    delete the organization
    check if the Organization list is set
    """
    organization_alias = f"Organization: {uuid.uuid4()}"
    async with aiohttp.ClientSession() as session:
        tasks = [
            new_organization(
                session=session, i=0, organization_alias=organization_alias
            )
        ]
        await asyncio.gather(*tasks)

        response_json = await list_organization(session, i=0)
        print(len(response_json))

        organization_id = response_json[0]["organization_id"]
        await delete_organization(session, i=0, organization_id=organization_id)

        response_json = await list_organization(session, i=0)
        print(len(response_json))


@pytest.mark.asyncio
async def test_organization_member_flow():
    """
    create a new organization
    add a new member to the organization
    check if the member is added to the organization
    update the member's role in the organization
    delete the member from the organization
    check if the member is deleted from the organization
    """
    organization_alias = f"Organization: {uuid.uuid4()}"
    async with aiohttp.ClientSession() as session:
        response_json = await new_organization(
            session=session, i=0, organization_alias=organization_alias
        )
        organization_id = response_json["organization_id"]

        response_json = await list_organization(session, i=0)
        print(len(response_json))

        new_user_response_json = await new_user(
            session=session, i=0, user_email=f"test_user_{uuid.uuid4()}@example.com"
        )
        user_id = new_user_response_json["user_id"]

        await add_member_to_org(
            session, i=0, organization_id=organization_id, user_id=user_id
        )

        response_json = await list_organization(session, i=0)
        print(len(response_json))

        for orgs in response_json:
            tmp_organization_id = orgs["organization_id"]
            if (
                tmp_organization_id is not None
                and tmp_organization_id == organization_id
            ):
                user_id = orgs["members"][0]["user_id"]

        response_json = await list_organization(session, i=0)
        print(len(response_json))

        await update_member_role(
            session,
            i=0,
            organization_id=organization_id,
            user_id=user_id,
            user_role="org_admin",
        )

        response_json = await list_organization(session, i=0)
        print(len(response_json))

        await delete_member_from_org(
            session, i=0, organization_id=organization_id, user_id=user_id
        )

        response_json = await list_organization(session, i=0)
        print(len(response_json))
