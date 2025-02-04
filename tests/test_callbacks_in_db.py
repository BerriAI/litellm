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
from openai import AsyncOpenAI

load_dotenv()


async def config_update(session, routing_strategy=None):
    url = "http://0.0.0.0:4000/config/update"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    print("routing_strategy: ", routing_strategy)
    data = {
        "litellm_settings": {"success_callback": ["langfuse"]},
        "environment_variables": {
            "LANGFUSE_PUBLIC_KEY": "pk-lf-e02aaea3-8668-4c9f-8c69-771a4ea1f5c9",
            "LANGFUSE_SECRET_KEY": "sk-lf-2480d7c9-f135-4b48-bf7e-2e6af14bedda",
            "LANGFUSE_HOST": "https://us.cloud.langfuse.com",
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


async def make_chat_completions_request(session):
    client = AsyncOpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")
    response = await client.chat.completions.create(
        model="fake-openai-endpoint",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )
    print(response)


@pytest.mark.asyncio
async def test_e2e_langfuse_callbacks_in_db():

    session = aiohttp.ClientSession()

    # add langfuse callback to DB
    # await config_update(session)

    # wait 20 seconds for the callback to be loaded into the instance
    # await asyncio.sleep(20)

    # make a /chat/completions request to the proxy
    await make_chat_completions_request(session)
