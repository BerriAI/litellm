"""
This test ensures that the proxy starts and serves requests even with a bad license.


in ci/cd config.yml, we set the license to "bad-license"
"""

import os
from typing import Final

import aiohttp
import pytest
from tests.integration._support.provider import SharedProvider
from tests.integration._support.wire import Reply


PROXY_URL: Final = os.environ.get("INTEGRATION_PROXY_URL", "http://0.0.0.0:4000")
INTEGRATION_RIG: Final = "INTEGRATION_PROXY_URL" in os.environ
MODEL: Final = "openai/gpt-5.4" if INTEGRATION_RIG else "gpt-4"


@pytest.mark.asyncio
async def test_health_and_chat_completion(provider: SharedProvider) -> None:
    base_url: Final = PROXY_URL.rstrip("/")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/health/readiness") as response:
            assert response.status == 200, await response.text()
            readiness_response: Final = await response.json()
            assert readiness_response["status"] in ("healthy", "connected")
        async with session.get(f"{base_url}/health/liveness") as response:
            assert response.status == 200, await response.text()
        headers: Final = {
            "Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}",
            "Content-Type": "application/json",
        }
        if INTEGRATION_RIG:
            provider.expect(
                Reply(
                    status=200,
                    body=b'{"id":"chatcmpl-scripted","object":"chat.completion","created":1700000000,"model":"gpt-5.4","choices":[{"index":0,"message":{"role":"assistant","content":"scripted integration reply"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":5,"total_tokens":9}}',
                )
            )
        data: Final = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }
        async with session.post(f"{base_url}/chat/completions", headers=headers, json=data) as response:
            assert response.status == 200, await response.text()
            completion_response: Final = await response.json()
            assert "choices" in completion_response
    if INTEGRATION_RIG:
        requests: Final = provider.received()
        assert len(requests) == 1, requests
        assert requests[0].method == "POST", requests[0]
        assert requests[0].target.endswith("/chat/completions"), requests[0]
