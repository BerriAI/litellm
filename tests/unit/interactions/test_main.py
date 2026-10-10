import base64
import json
from typing import Final

import httpx
import pytest
from respx import MockRouter

import litellm
import litellm.interactions as interactions


@pytest.fixture
def api_key():
    return "test-api-key"


class TestGoogleInteractionsCreate:
    @pytest.mark.usefixtures("fake_provider_credentials")
    def test_missing_model_and_agent(self, api_key):
        """Test error when neither model nor agent is provided."""
        with pytest.raises((ValueError, litellm.APIConnectionError)):
            interactions.create(
                input="Hello",
                api_key=api_key,
            )


class TestInteractionsAcreateOffline:
    @pytest.fixture(autouse=True)
    def _httpx_only_transport(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")

    @pytest.mark.usefixtures("fake_provider_credentials")
    @pytest.mark.asyncio
    async def test_acreate_simple_gemini(self, respx_mock: MockRouter) -> None:
        route: Final = respx_mock.post("https://generativelanguage.googleapis.com/v1beta/interactions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "interaction-offline",
                    "object": "interaction",
                    "model": "gemini-2.5-flash",
                    "status": "completed",
                    "steps": [{"type": "model_output", "content": [{"type": "text", "text": "299792458"}]}],
                    "usage": {"input_tokens": 6, "output_tokens": 3},
                },
            )
        )
        response: Final = await interactions.acreate(
            model="gemini/gemini-2.5-flash",
            input="What is the speed of light?",
            api_key="gemini-offline",
        )
        body: Final = json.loads(route.calls.last.request.content)
        assert body["model"] == "gemini-2.5-flash"
        assert body["input"] == "What is the speed of light?"
        assert response.id == "interaction-offline"
        assert response.status == "completed"

    @pytest.mark.usefixtures("fake_provider_credentials")
    @pytest.mark.asyncio
    async def test_acreate_simple_litellm_responses_bridge(self, respx_mock: MockRouter) -> None:
        route: Final = respx_mock.post("https://api.openai.com/v1/responses").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "resp-offline",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-4o",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "299792458 m/s"}],
                        }
                    ],
                    "usage": {"input_tokens": 6, "output_tokens": 3, "total_tokens": 9},
                },
            )
        )
        response: Final = await interactions.acreate(
            model="gpt-4o",
            input="What is the speed of light?",
            api_key="sk-offline",
        )
        body: Final = json.loads(route.calls.last.request.content)
        assert body["model"] == "gpt-4o"
        serialized: Final = json.dumps(body)
        assert "What is the speed of light?" in serialized
        assert "response_id:resp-offline" in base64.b64decode(response.id.removeprefix("resp_")).decode()
        assert response.status == "completed"
