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
        "models": ["gpt-4", "text-embedding-ada-002", "gpt-image-1"],
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
        # GPT-image-1 cost audit: enumerate every healthy endpoint so we can
        # confirm which proxy model entries (e.g. dall-e-2 / openai-dall-e-3
        # aliased to gpt-image-1) trigger live image_generation calls per
        # /health invocation.
        try:
            for ep in all_healthy_models.get("healthy_endpoints") or []:
                if not isinstance(ep, dict):
                    continue
                print(
                    f"[GPT_IMAGE_AUDIT] source=health_endpoint "
                    f"model_in_config={ep.get('model_name')!r} "
                    f"underlying_model={ep.get('model')!r}",
                    flush=True,
                )
        except Exception as _audit_exc:
            print(
                f"[GPT_IMAGE_AUDIT] health endpoint audit failed: {_audit_exc}",
                flush=True,
            )


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
            assert "status" in response_json

            if status != 200:
                raise Exception(f"Request did not return a 200 status code: {status}")


@pytest.mark.asyncio
async def test_health_readiness_details():
    """
    Check if authenticated readiness diagnostics expose version metadata.
    """
    async with aiohttp.ClientSession() as session:
        url = "http://0.0.0.0:4000/health/readiness/details"
        headers = {"Authorization": "Bearer sk-1234"}
        async with session.get(url, headers=headers) as response:
            status = response.status
            response_json = await response.json()

            print(response_json)
            assert "status" in response_json
            assert "litellm_version" in response_json

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
