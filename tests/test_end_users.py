import os
# What is this?
## Unit tests for the /end_users/* endpoints
import pytest
import asyncio
import aiohttp
from litellm._uuid import uuid

"""
- `/end_user/new` 
- `/end_user/info` 
"""


async def new_end_user(
    session,
    i,
    user_id=str(uuid.uuid4()),
    model_region=None,
    default_model=None,
    budget_id=None,
):
    url = "http://0.0.0.0:4000/end_user/new"
    headers = {"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}", "Content-Type": "application/json"}
    data = {
        "user_id": user_id,
        "allowed_model_region": model_region,
        "default_model": default_model,
    }

    if budget_id is not None:
        data["budget_id"] = budget_id
    print("end user data: {}".format(data))

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
async def test_end_user_new():
    """
    Make 20 parallel calls to /user/new. Assert all worked.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [new_end_user(session, i, str(uuid.uuid4())) for i in range(1, 11)]
        await asyncio.gather(*tasks)


