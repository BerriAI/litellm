# What this tests ?
## Makes sure the number of callbacks on the proxy don't increase over time
## Num callbacks should be a fixed number at t=0 and t=10, t=20
"""
PROD TEST - DO NOT Delete this Test
"""

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
        "router_settings": {
            "routing_strategy": ["latency-based-routing"],
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


async def get_active_callbacks(session):
    url = "http://0.0.0.0:4000/health/readiness"
    headers = {
        "Content-Type": "application/json",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()
        print("response from /health/readiness")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        _json_response = await response.json()

        _num_callbacks = _json_response["num_callbacks"]
        print("current number of callbacks: ", _num_callbacks)
        return _num_callbacks


@pytest.mark.asyncio
async def test_add_model_run_health():
    """ """
    import uuid

    async with aiohttp.ClientSession() as session:
        num_callbacks_1 = await get_active_callbacks(session=session)

        await asyncio.sleep(30)

        num_callbacks_2 = await get_active_callbacks(session=session)

        await asyncio.sleep(30)

        num_callbacks_3 = await get_active_callbacks(session=session)

        assert num_callbacks_1 == num_callbacks_2 == num_callbacks_3
