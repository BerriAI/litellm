# What this tests ?
## Set tags on a team and then make a request to /chat/completions
import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Union
from litellm._uuid import uuid

LITELLM_MASTER_KEY = "sk-1234"


async def chat_completion(
    session, key, model: Union[str, List] = "fake-openai-endpoint"
):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    print("headers=", headers)
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"Hello! {str(uuid.uuid4())}"},
        ],
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        if status != 200:
            raise Exception(response_text)

        return await response.json(), response.headers


async def create_team_with_tags(session, key, tags: List[str]):
    url = "http://0.0.0.0:4000/team/new"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "tags": tags,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        if status != 200:
            raise Exception(response_text)

        return await response.json()


async def create_key_with_team(session, key, team_id: str):
    url = f"http://0.0.0.0:4000/key/generate"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "team_id": team_id,
    }
    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        if status != 200:
            raise Exception(response_text)

        return await response.json()


async def model_info_get_call(session, key, model_id: str):
    # make get call pass "litellm_model_id" in query params
    url = f"http://0.0.0.0:4000/model/info?litellm_model_id={model_id}"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()

        if status != 200:
            raise Exception(response_text)

        return await response.json()


@pytest.mark.asyncio()
async def test_team_tag_routing():
    async with aiohttp.ClientSession() as session:
        key = LITELLM_MASTER_KEY
        team_a_data = await create_team_with_tags(session, key, ["teamA"])
        print("team_a_data=", team_a_data)
        team_a_id = team_a_data["team_id"]

        team_b_data = await create_team_with_tags(session, key, ["teamB"])
        print("team_b_data=", team_b_data)
        team_b_id = team_b_data["team_id"]

        key_with_team_a = await create_key_with_team(session, key, team_a_id)
        print("key_with_team_a=", key_with_team_a)
        _key_with_team_a = key_with_team_a["key"]
        for _ in range(5):
            response_a, headers = await chat_completion(
                session=session, key=_key_with_team_a
            )

            headers = dict(headers)
            print(response_a)
            print(headers)
            assert (
                headers["x-litellm-model-id"] == "team-a-model"
            ), "Model ID should be teamA"

        key_with_team_b = await create_key_with_team(session, key, team_b_id)
        _key_with_team_b = key_with_team_b["key"]
        for _ in range(5):
            response_b, headers = await chat_completion(session, _key_with_team_b)
            headers = dict(headers)
            print(response_b)
            print(headers)
            assert (
                headers["x-litellm-model-id"] == "team-b-model"
            ), "Model ID should be teamB"


@pytest.mark.asyncio()
async def test_chat_completion_with_no_tags():
    async with aiohttp.ClientSession() as session:
        key = LITELLM_MASTER_KEY
        response, headers = await chat_completion(session, key)
        headers = dict(headers)
        print(response)
        print(headers)
        assert response is not None
