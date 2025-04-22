"""
This test ensures that the proxy starts and serves requests even with a bad license.


in ci/cd config.yml, we set the license to "bad-license"
"""

import pytest
import aiohttp
from typing import Optional


@pytest.mark.asyncio
async def test_health_and_chat_completion():
    """
    Test health endpoints and chat completion:
    1. Check /health/readiness
    2. Check /health/liveness
    3. Make a chat completion call
    """
    async with aiohttp.ClientSession() as session:
        # Test readiness endpoint
        async with session.get("http://0.0.0.0:4000/health/readiness") as response:
            assert response.status == 200
            readiness_response = await response.json()
            assert readiness_response["status"] == "connected"

        # Test liveness endpoint
        async with session.get("http://0.0.0.0:4000/health/liveness") as response:
            assert response.status == 200
            liveness_response = await response.json()
            print("liveness_response", liveness_response)

        # Make a chat completion call
        url = "http://0.0.0.0:4000/chat/completions"
        headers = {
            "Authorization": "Bearer sk-1234",
            "Content-Type": "application/json",
        }
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        async with session.post(url, headers=headers, json=data) as response:
            assert response.status == 200
            completion_response = await response.json()
            assert "choices" in completion_response
