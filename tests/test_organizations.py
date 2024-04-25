# What this tests ?
## Tests /organization endpoints.
from typing import Optional, Any

import pytest
import asyncio
import aiohttp
import time, uuid

from aiohttp import ClientSession
from openai import AsyncOpenAI


async def new_organization(
    session: ClientSession,
    i: int,
    organization_alias: str,
    max_budget: Optional[float] = None,
) -> Any:
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
async def test_organization_new() -> None:
    """
    Make 20 parallel calls to /user/new. Assert all worked.
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
