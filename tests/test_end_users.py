# What is this?
## Unit tests for the /end_users/* endpoints
import pytest
import asyncio
import aiohttp
import time
import uuid
from openai import AsyncOpenAI
from typing import Optional

"""
- `/end_user/new` 
- `/end_user/info` 
"""


async def chat_completion_with_headers(session, key, model="gpt-4"):
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

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        response_header_check(
            response
        )  # calling the function to check response headers

        raw_headers = response.raw_headers
        raw_headers_json = {}

        for (
            item
        ) in (
            response.raw_headers
        ):  # ((b'date', b'Fri, 19 Apr 2024 21:17:29 GMT'), (), )
            raw_headers_json[item[0].decode("utf-8")] = item[1].decode("utf-8")

        return raw_headers_json


async def generate_key(
    session,
    i,
    budget=None,
    budget_duration=None,
    models=["azure-models", "gpt-4", "dall-e-3"],
    max_parallel_requests: Optional[int] = None,
    user_id: Optional[str] = None,
    team_id: Optional[str] = None,
    calling_key="sk-1234",
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


async def new_end_user(
    session, i, user_id=str(uuid.uuid4()), model_region=None, default_model=None
):
    url = "http://0.0.0.0:4000/end_user/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "user_id": user_id,
        "allowed_model_region": model_region,
        "default_model": default_model,
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
async def test_end_user_new():
    """
    Make 20 parallel calls to /user/new. Assert all worked.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [new_end_user(session, i, str(uuid.uuid4())) for i in range(1, 11)]
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_end_user_specific_region():
    """
    - Specify region user can make calls in
    - Make a generic call
    - assert returned api base is for model in region

    Repeat 3 times
    """
    key: str = ""
    ## CREATE USER ##
    async with aiohttp.ClientSession() as session:
        end_user_obj = await new_end_user(
            session=session,
            i=0,
            user_id=str(uuid.uuid4()),
            model_region="eu",
        )

        ## MAKE CALL ##
        key_gen = await generate_key(session=session, i=0, models=["gpt-3.5-turbo"])

        key = key_gen["key"]

    for _ in range(3):
        client = AsyncOpenAI(api_key=key, base_url="http://0.0.0.0:4000")

        print("SENDING USER PARAM - {}".format(end_user_obj["user_id"]))
        result = await client.chat.completions.with_raw_response.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey!"}],
            user=end_user_obj["user_id"],
        )

        assert (
            result.headers.get("x-litellm-model-api-base")
            == "https://my-endpoint-europe-berri-992.openai.azure.com/"
        )
