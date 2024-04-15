# What this tests ?
## Tests /models and /model/* endpoints

import pytest
import asyncio
import aiohttp
import os
import dotenv
from dotenv import load_dotenv
import pytest

load_dotenv()


async def generate_key(session, models=[], team_id=None):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": models,
        "duration": None,
        "team_id": team_id,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


async def chat_completion(session, key, model="azure-gpt-3.5", request_metadata=None):
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
        "metadata": request_metadata,
    }

    print("data sent in test=", data)

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")


@pytest.mark.asyncio
async def test_team_logging():
    """
    -> Team 1 logs to project 1
    -> Create Key
    -> Make chat/completions call
    -> Fetch logs from langfuse
    """
    try:
        async with aiohttp.ClientSession() as session:

            key = await generate_key(
                session, models=["fake-openai-endpoint"], team_id="team-1"
            )  # team-1 logs to project 1

            import uuid

            _trace_id = f"trace-{uuid.uuid4()}"
            _request_metadata = {
                "trace_id": _trace_id,
            }

            await chat_completion(
                session,
                key["key"],
                model="fake-openai-endpoint",
                request_metadata=_request_metadata,
            )

            # Test - if the logs were sent to the correct team on langfuse
            import langfuse

            langfuse_client = langfuse.Langfuse(
                public_key=os.getenv("LANGFUSE_PROJECT1_PUBLIC"),
                secret_key=os.getenv("LANGFUSE_PROJECT1_SECRET"),
            )

            await asyncio.sleep(10)

            print(f"searching for trace_id={_trace_id} on langfuse")

            generations = langfuse_client.get_generations(trace_id=_trace_id).data
            print(generations)
            assert len(generations) == 1
    except Exception as e:
        pytest.fail(f"Unexpected error: {str(e)}")


@pytest.mark.asyncio
async def test_team_2logging():
    """
    -> Team 1 logs to project 2
    -> Create Key
    -> Make chat/completions call
    -> Fetch logs from langfuse
    """
    try:
        async with aiohttp.ClientSession() as session:

            key = await generate_key(
                session, models=["fake-openai-endpoint"], team_id="team-2"
            )  # team-1 logs to project 1

            import uuid

            _trace_id = f"trace-{uuid.uuid4()}"
            _request_metadata = {
                "trace_id": _trace_id,
            }

            await chat_completion(
                session,
                key["key"],
                model="fake-openai-endpoint",
                request_metadata=_request_metadata,
            )

            # Test - if the logs were sent to the correct team on langfuse
            import langfuse

            langfuse_client = langfuse.Langfuse(
                public_key=os.getenv("LANGFUSE_PROJECT2_PUBLIC"),
                secret_key=os.getenv("LANGFUSE_PROJECT2_SECRET"),
            )

            await asyncio.sleep(10)

            print(f"searching for trace_id={_trace_id} on langfuse")

            generations = langfuse_client.get_generations(trace_id=_trace_id).data
            print("Team 2 generations", generations)

            # team-2 should have 1 generation with this trace id
            assert len(generations) == 1

            # team-1 should have 0 generations with this trace id
            langfuse_client_1 = langfuse.Langfuse(
                public_key=os.getenv("LANGFUSE_PROJECT1_PUBLIC"),
                secret_key=os.getenv("LANGFUSE_PROJECT1_SECRET"),
            )

            generations_team_1 = langfuse_client_1.get_generations(
                trace_id=_trace_id
            ).data
            print("Team 1 generations", generations_team_1)

            assert len(generations_team_1) == 0

    except Exception as e:
        pytest.fail("Team 2 logging failed: " + str(e))
