# What this tests?
## Tests /health + /routes endpoints.

import pytest
import asyncio
import aiohttp


async def health(session, call_key):
    url = "http://0.0.0.0:4000/health"
    headers = {
        "Authorization": f"Bearer {call_key}",
        "Content-Type": "application/json",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()

        print(f"Response (Status code: {status}):")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        return await response.json()


async def generate_key(session):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": ["gpt-4", "text-embedding-ada-002", "dall-e-2"],
        "duration": None,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


@pytest.mark.asyncio
async def test_health():
    """
    - Call /health
    """
    async with aiohttp.ClientSession() as session:
        # as admin #
        all_healthy_models = await health(session=session, call_key="sk-1234")
        total_model_count = (
            all_healthy_models["healthy_count"] + all_healthy_models["unhealthy_count"]
        )
        assert total_model_count > 0


@pytest.mark.asyncio
async def test_health_readiness():
    """
    Check if 200
    """
    async with aiohttp.ClientSession() as session:
        url = "http://0.0.0.0:4000/health/readiness"
        async with session.get(url) as response:
            status = response.status
            response_json = await response.json()

            print(response_json)
            assert "litellm_version" in response_json
            assert "status" in response_json

            if status != 200:
                raise Exception(f"Request did not return a 200 status code: {status}")


@pytest.mark.asyncio
async def test_health_liveliness():
    """
    Check if 200
    """
    async with aiohttp.ClientSession() as session:
        url = "http://0.0.0.0:4000/health/liveliness"
        async with session.get(url) as response:
            status = response.status
            response_text = await response.text()

            print(response_text)
            print()

            if status != 200:
                raise Exception(f"Request did not return a 200 status code: {status}")


@pytest.mark.asyncio
async def test_routes():
    """
    Check if 200
    """
    async with aiohttp.ClientSession() as session:
        url = "http://0.0.0.0:4000/routes"
        async with session.get(url) as response:
            status = response.status
            response_text = await response.text()

            print(response_text)
            print()

            if status != 200:
                raise Exception(f"Request did not return a 200 status code: {status}")
