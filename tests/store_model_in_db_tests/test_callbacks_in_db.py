"""
PROD TEST - DO NOT Delete this Test

e2e test for langfuse callback in DB 
- Add langfuse callback to DB - with /config/update
- wait 20 seconds for the callback to be loaded into the instance 
- Make a /chat/completions request to the proxy
- Check if the request is logged in Langfuse
"""

import pytest
import asyncio
import aiohttp
import os
import dotenv
from dotenv import load_dotenv
import pytest
from openai import AsyncOpenAI, APIConnectionError
from openai.types.chat import ChatCompletion

load_dotenv()

# used for testing
LANGFUSE_BASE_URL = "https://exampleopenaiendpoint-production-c715.up.railway.app"
PROXY_BASE_URL = "http://127.0.0.1:4000"


async def wait_for_proxy_ready(session, timeout: int = 60):
    for _ in range(timeout):
        try:
            async with session.get(f"{PROXY_BASE_URL}/health/liveliness") as response:
                if response.status == 200:
                    return
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(1)
    raise RuntimeError(f"Proxy at {PROXY_BASE_URL} not ready after {timeout}s")


async def config_update(session, routing_strategy=None):
    url = f"{PROXY_BASE_URL}/config/update"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    print("routing_strategy: ", routing_strategy)
    data = {
        "litellm_settings": {"success_callback": ["langfuse"]},
        "environment_variables": {
            "LANGFUSE_PUBLIC_KEY": "any-public-key",
            "LANGFUSE_SECRET_KEY": "any-secret-key",
            "LANGFUSE_HOST": LANGFUSE_BASE_URL,
        },
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print("status: ", status)

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


async def check_langfuse_request(response_id: str):
    async with aiohttp.ClientSession() as session:
        url = f"{LANGFUSE_BASE_URL}/langfuse/trace/{response_id}"
        async with session.get(url) as response:
            response_json = await response.json()
            assert response.status == 200, f"Expected status 200, got {response.status}"
            assert (
                response_json["exists"] == True
            ), f"Request {response_id} not found in Langfuse traces"
            assert response_json["request_id"] == response_id, f"Request ID mismatch"


async def make_chat_completions_request() -> ChatCompletion:
    client = AsyncOpenAI(api_key="sk-1234", base_url=PROXY_BASE_URL)
    last_error = None
    for _ in range(10):
        try:
            response = await client.chat.completions.create(
                model="fake-openai-endpoint",
                messages=[{"role": "user", "content": "Hello, world!"}],
            )
            print(response)
            return response
        except APIConnectionError as e:
            last_error = e
            await asyncio.sleep(2)
    raise AssertionError(
        f"Proxy at {PROXY_BASE_URL} unreachable after retries: {last_error!r}"
    )


@pytest.mark.flaky(reruns=2, reruns_delay=5)
@pytest.mark.asyncio
async def test_e2e_langfuse_callbacks_in_db():

    async with aiohttp.ClientSession() as session:
        # add langfuse callback to DB
        await config_update(session)

        # wait 20 seconds for the callback to be loaded into the instance
        await asyncio.sleep(20)
        await wait_for_proxy_ready(session)

        # make a /chat/completions request to the proxy
        response = await make_chat_completions_request()
        print(response)
        response_id = response.id
        print("response_id: ", response_id)

    await asyncio.sleep(20)
    # check if the request is logged in Langfuse
    await check_langfuse_request(response_id)
