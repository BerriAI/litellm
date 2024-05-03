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


async def config_update(session, routing_strategy=None):
    url = "http://0.0.0.0:4000/config/update"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    print("routing_strategy: ", routing_strategy)
    data = {
        "router_settings": {
            "routing_strategy": routing_strategy,
        },
        "general_settings": {
            "alert_to_webhook_url": {
                "llm_exceptions": "https://hooks.slack.com/services/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"
            },
            "alert_types": ["llm_exceptions", "db_exceptions"],
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
    url = "http://0.0.0.0:4000/active/callbacks"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-1234",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()
        print("response from /active/callbacks")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        _json_response = await response.json()

        _num_callbacks = _json_response["num_callbacks"]
        _num_alerts = _json_response["num_alerting"]
        print("current number of callbacks: ", _num_callbacks)
        print("current number of alerts: ", _num_alerts)
        return _num_callbacks, _num_alerts


async def get_current_routing_strategy(session):
    url = "http://0.0.0.0:4000/get/config/callbacks"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-1234",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        _json_response = await response.json()
        print("JSON response: ", _json_response)

        router_settings = _json_response["router_settings"]
        print("Router settings: ", router_settings)
        routing_strategy = router_settings["routing_strategy"]
        return routing_strategy


@pytest.mark.asyncio
async def test_check_num_callbacks():
    """
    Test 1:  num callbacks should NOT increase over time
    -> check current callbacks
    -> sleep for 30s
    -> check current callbacks
    -> sleep for 30s
    -> check current callbacks
    """
    import uuid

    async with aiohttp.ClientSession() as session:
        num_callbacks_1, _ = await get_active_callbacks(session=session)
        assert (
            num_callbacks_1 > 0
        )  # /health/readiness returns 0 when some calculation goes wrong

        await asyncio.sleep(30)

        num_callbacks_2, _ = await get_active_callbacks(session=session)

        assert num_callbacks_1 == num_callbacks_2

        await asyncio.sleep(30)

        num_callbacks_3, _ = await get_active_callbacks(session=session)

        assert num_callbacks_1 == num_callbacks_2 == num_callbacks_3


@pytest.mark.asyncio
async def test_check_num_callbacks_on_lowest_latency():
    """
    Test 1:  num callbacks should NOT increase over time
    -> Update to lowest latency
    -> check current callbacks
    -> sleep for 30s
    -> check current callbacks
    -> sleep for 30s
    -> check current callbacks
    -> update back to original routing-strategy
    """
    import uuid

    async with aiohttp.ClientSession() as session:

        original_routing_strategy = await get_current_routing_strategy(session=session)
        await config_update(session=session, routing_strategy="latency-based-routing")

        num_callbacks_1, num_alerts_1 = await get_active_callbacks(session=session)
        assert (
            num_callbacks_1 > 0
        )  # /health/readiness returns 0 when some calculation goes wrong

        await asyncio.sleep(30)

        num_callbacks_2, num_alerts_2 = await get_active_callbacks(session=session)

        assert num_callbacks_1 == num_callbacks_2

        await asyncio.sleep(30)

        num_callbacks_3, num_alerts_3 = await get_active_callbacks(session=session)

        assert num_callbacks_1 == num_callbacks_2 == num_callbacks_3

        assert num_alerts_1 == num_alerts_2 == num_alerts_3

        await config_update(session=session, routing_strategy=original_routing_strategy)
