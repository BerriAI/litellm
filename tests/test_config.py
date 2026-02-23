# What this tests ?
## Tests /config/update + Test /chat/completions -> assert logs are sent to Langfuse

import pytest
import asyncio
import aiohttp
import os
import dotenv
from dotenv import load_dotenv
import pytest

load_dotenv()


async def config_update(session):
    url = "http://0.0.0.0:4000/config/update"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "litellm_settings": {
            "success_callback": ["langfuse"],
        },
        "environment_variables": {
            "LANGFUSE_HOST": os.environ["LANGFUSE_HOST"],
            "LANGFUSE_PUBLIC_KEY": os.environ["LANGFUSE_PUBLIC_KEY"],
            "LANGFUSE_SECRET_KEY": os.environ["LANGFUSE_SECRET_KEY"],
        },
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
@pytest.mark.skip(
    reason="langfuse apis are flaky, we unit test team / key based logging in test_langfuse_unit_tests.py"
)
async def test_team_logging():
    """
    1. Add Langfuse as a callback with /config/update
    2. Call /chat/completions
    3. Assert the logs are sent to Langfuse
    """
    try:
        async with aiohttp.ClientSession() as session:

            # Add Langfuse as a callback with /config/update
            await config_update(session)

            # 2. Call /chat/completions with a specific trace id
            from litellm._uuid import uuid

            _trace_id = f"trace-{uuid.uuid4()}"
            _request_metadata = {
                "trace_id": _trace_id,
            }

            await chat_completion(
                session,
                key="sk-1234",
                model="fake-openai-endpoint",
                request_metadata=_request_metadata,
            )

            # Test - if the logs were sent to the correct team on langfuse
            import langfuse

            langfuse_client = langfuse.Langfuse(
                host=os.getenv("LANGFUSE_HOST"),
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            )

            await asyncio.sleep(10)

            print(f"searching for trace_id={_trace_id} on langfuse")

            generations = langfuse_client.get_generations(trace_id=_trace_id).data

            # 1 generation with this trace id
            assert len(generations) == 1

    except Exception as e:
        pytest.fail("Team 2 logging failed: " + str(e))
