# What this tests ?
## Tests /organization endpoints.
import pytest
import asyncio
import aiohttp
import time, uuid
from openai import AsyncOpenAI


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
