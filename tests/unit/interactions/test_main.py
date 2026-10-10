import base64
import json
from typing import Final

import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import interactions


def _openai_response_body(response_id: str, text: str) -> dict[str, object]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": "gpt-5.6",
        "output": [
            {
                "id": f"msg-{response_id}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 6,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 7,
        },
        "user": None,
        "metadata": {},
    }


@pytest.fixture
def api_key():
    return "test-api-key"


class TestGoogleInteractionsCreate:
    @classmethod
    def _responses_stream_body(cls) -> str:
        delta_event: Final = {
            "type": "response.output_text.delta",
            "delta": "Hello",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
        }
        completed_event: Final = {
            "type": "response.completed",
            "response": _openai_response_body("resp-stream", "Hello"),
        }
        return (
            f"event: response.output_text.delta\ndata: {json.dumps(delta_event)}\n\n"
            f"event: response.completed\ndata: {json.dumps(completed_event)}\n\n"
        )

    @pytest.mark.usefixtures("fake_provider_credentials")
    def test_missing_model_and_agent(self, api_key):
        """Test error when neither model nor agent is provided."""
        with pytest.raises((ValueError, litellm.APIConnectionError)):
            interactions.create(
                input="Hello",
                api_key=api_key,
            )

    @pytest.mark.parametrize(
        ("model", "api_key", "url", "expected_body", "response_body"),
        [
            pytest.param(
                "gemini/gemini-2.5-flash",
                "gemini-offline",
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                {"model": "gemini-2.5-flash", "input": "Hello, how are you?"},
                {
                    "id": "interaction-gemini",
                    "object": "interaction",
                    "model": "gemini-2.5-flash",
                    "status": "completed",
                    "created": "2026-05-01T00:00:00Z",
                    "updated": "2026-05-01T00:00:00Z",
                    "outputs": [{"type": "text", "text": "4"}],
                    "usage": {"total_input_tokens": 6, "total_output_tokens": 1},
                },
                id="gemini",
            ),
            pytest.param(
                "gpt-5.6",
                "sk-offline",
                "https://api.openai.com/v1/responses",
                {"model": "gpt-5.6", "input": "Hello, how are you?"},
                _openai_response_body("resp-bridge", "4"),
                id="responses-bridge",
            ),
        ],
    )
    @pytest.mark.usefixtures("fake_provider_credentials")
    def test_create_simple_string_input(
        self,
        respx_mock: MockRouter,
        model: str,
        api_key: str,
        url: str,
        expected_body: dict[str, object],
        response_body: dict[str, object],
    ) -> None:
        route: Final = respx_mock.post(url).mock(return_value=httpx.Response(200, json=response_body))

        response: Final = interactions.create(
            model=model,
            input="Hello, how are you?",
            api_key=api_key,
        )

        assert json.loads(route.calls.last.request.content) == expected_body
        assert response.id
        assert response.object == "interaction"
        assert response.model == ("gpt-5.6" if model == "gpt-5.6" else "gemini-2.5-flash")
        assert response.status == "completed"
        assert response.created
        assert response.updated
        assert len(response.outputs) == 1
        assert response.outputs[0]["text"] == "4"
        assert response.usage["total_input_tokens"] == 6
        assert response.usage["total_output_tokens"] == 1

    @pytest.mark.parametrize(
        ("model", "api_key", "url", "expected_body", "response_body"),
        [
            pytest.param(
                "gemini/gemini-2.5-flash",
                "gemini-offline",
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                {
                    "model": "gemini-2.5-flash",
                    "input": "Return the required token.",
                    "system_instruction": "Reply with exactly this token: keep-system-instruction",
                },
                {
                    "id": "interaction-system-instruction",
                    "object": "interaction",
                    "model": "gemini-2.5-flash",
                    "status": "completed",
                    "outputs": [{"type": "text", "text": "keep-system-instruction"}],
                },
                id="gemini",
            ),
            pytest.param(
                "gpt-5.6",
                "sk-offline",
                "https://api.openai.com/v1/responses",
                {
                    "model": "gpt-5.6",
                    "input": "Return the required token.",
                    "instructions": "Reply with exactly this token: keep-system-instruction",
                },
                _openai_response_body("resp-system-instruction", "keep-system-instruction"),
                id="responses-bridge",
            ),
        ],
    )
    @pytest.mark.usefixtures("fake_provider_credentials")
    def test_create_with_system_instruction(
        self,
        respx_mock: MockRouter,
        model: str,
        api_key: str,
        url: str,
        expected_body: dict[str, object],
        response_body: dict[str, object],
    ) -> None:
        route: Final = respx_mock.post(url).mock(return_value=httpx.Response(200, json=response_body))

        response: Final = interactions.create(
            model=model,
            input="Return the required token.",
            system_instruction="Reply with exactly this token: keep-system-instruction",
            api_key=api_key,
        )

        assert json.loads(route.calls.last.request.content) == expected_body
        assert response.status == "completed"

    @pytest.mark.parametrize(
        ("model", "api_key", "url", "expected_body", "sse_body"),
        [
            pytest.param(
                "gemini/gemini-2.5-flash",
                "gemini-offline",
                "https://generativelanguage.googleapis.com/v1beta/interactions?alt=sse",
                {"model": "gemini-2.5-flash", "input": "Stream an answer.", "stream": True},
                'data: {"event_type":"step.delta","delta":{"type":"text","text":"Hello"}}\n\n',
                id="gemini",
            ),
            pytest.param(
                "gpt-5.6",
                "sk-offline",
                "https://api.openai.com/v1/responses",
                {"model": "gpt-5.6", "input": "Stream an answer.", "stream": True},
                None,
                id="responses-bridge",
            ),
        ],
    )
    @pytest.mark.usefixtures("fake_provider_credentials")
    def test_create_streaming(
        self,
        respx_mock: MockRouter,
        model: str,
        api_key: str,
        url: str,
        expected_body: dict[str, object],
        sse_body: str | None,
    ) -> None:
        stream_body: Final = sse_body or self._responses_stream_body()
        route: Final = respx_mock.post(url).mock(
            return_value=httpx.Response(200, text=stream_body, headers={"content-type": "text/event-stream"})
        )

        response_stream: Final = interactions.create(
            model=model,
            input="Stream an answer.",
            stream=True,
            api_key=api_key,
        )
        chunks: Final = tuple(response_stream)

        assert json.loads(route.calls.last.request.content) == expected_body
        assert any(
            chunk.event_type == "step.delta" and chunk.delta == {"type": "text", "text": "Hello"} for chunk in chunks
        )

    @pytest.mark.usefixtures("fake_provider_credentials")
    def test_invalid_model_raises_not_found(self, respx_mock: MockRouter) -> None:
        route: Final = respx_mock.post("https://generativelanguage.googleapis.com/v1beta/interactions").mock(
            return_value=httpx.Response(
                404,
                json={"error": {"code": 404, "message": "model not found", "status": "NOT_FOUND"}},
            )
        )

        with pytest.raises(litellm.NotFoundError):
            interactions.create(
                model="gemini/invalid-model-name-xyz",
                input="Hello",
                api_key="gemini-offline",
            )

        assert json.loads(route.calls.last.request.content) == {
            "model": "invalid-model-name-xyz",
            "input": "Hello",
        }


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
        assert body == {
            "model": "gemini-2.5-flash",
            "input": "What is the speed of light?",
        }
        assert response.id == "interaction-offline"
        assert response.status == "completed"

    @pytest.mark.usefixtures("fake_provider_credentials")
    @pytest.mark.asyncio
    async def test_acreate_streaming_gemini(self, respx_mock: MockRouter) -> None:
        route: Final = respx_mock.post("https://generativelanguage.googleapis.com/v1beta/interactions?alt=sse").mock(
            return_value=httpx.Response(
                200,
                text='data: {"event_type":"step.delta","delta":{"type":"text","text":"Hello"}}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        )

        response_stream: Final = await interactions.acreate(
            model="gemini/gemini-2.5-flash",
            input="Stream an answer.",
            stream=True,
            api_key="gemini-offline",
        )
        chunks: Final = [chunk async for chunk in response_stream]

        assert json.loads(route.calls.last.request.content) == {
            "model": "gemini-2.5-flash",
            "input": "Stream an answer.",
            "stream": True,
        }
        assert any(
            chunk.event_type == "step.delta" and chunk.delta == {"type": "text", "text": "Hello"} for chunk in chunks
        )

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
                    "model": "gpt-5.6",
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
            model="gpt-5.6",
            input="What is the speed of light?",
            api_key="sk-offline",
        )
        body: Final = json.loads(route.calls.last.request.content)
        assert body["model"] == "gpt-5.6"
        serialized: Final = json.dumps(body)
        assert "What is the speed of light?" in serialized
        assert "response_id:resp-offline" in base64.b64decode(response.id.removeprefix("resp_")).decode()
        assert response.status == "completed"
