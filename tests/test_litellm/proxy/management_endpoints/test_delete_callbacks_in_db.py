"""
PROD TEST - DO NOT Delete this Test

e2e test for delete callback functionality in DB 
- Add langfuse callback to DB - with /config/update
- Wait for the callback to be loaded into the instance 
- Delete the callback using /config/callback/delete
- Verify the callback is removed
"""

import pytest
import asyncio
import aiohttp
from dotenv import load_dotenv
import pytest

load_dotenv()

# used for testing
LANGFUSE_BASE_URL = "https://exampleopenaiendpoint-production-c715.up.railway.app"


async def config_update(session):
    url = "http://0.0.0.0:4000/config/update"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
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

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


async def delete_callback(session, callback_name: str):
    url = "http://0.0.0.0:4000/config/callback/delete"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "callback_name": callback_name
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        if status != 200:
            raise Exception(f"Delete request did not return a 200 status code: {status}")
        return await response.json()


async def get_config(session):
    url = "http://0.0.0.0:4000/get/config/callbacks"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


@pytest.mark.asyncio
async def test_e2e_delete_callbacks_in_db():

    session = aiohttp.ClientSession()

    # add langfuse callback to DB
    await config_update(session)

    # wait 20 seconds for the callback to be loaded into the instance
    await asyncio.sleep(20)

    # delete the langfuse callback
    delete_response = await delete_callback(session, "langfuse")
    
    # verify delete response
    assert "message" in delete_response
    assert "langfuse" in delete_response.get("removed_callback", "")
    assert "langfuse" not in delete_response.get("remaining_callbacks", [])

    # get config and verify callback is deleted
    config_data = await get_config(session)
    callback_names = [callback["name"] for callback in config_data.get("callbacks", [])]
    assert "langfuse" not in callback_names

    await session.close() 