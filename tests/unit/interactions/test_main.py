import base64
import json
from datetime import datetime
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


def _bridge_interaction_id(response_id: str) -> str:
    encoded: Final = f"litellm:custom_llm_provider:openai;model_id:None;response_id:{response_id}".encode()
    return f"resp_{base64.b64encode(encoded).decode()}"


def _sse_event(event_type: str, sequence_number: int, payload: dict[str, object]) -> str:
    data: Final = json.dumps({"type": event_type, "sequence_number": sequence_number, **payload})
    return f"event: {event_type}\ndata: {data}\n\n"


def _responses_stream_body() -> str:
    completed: Final = _openai_response_body("resp-stream", "Hello")
    in_progress: Final = {**completed, "status": "in_progress", "output": [], "usage": None}
    text_location: Final = {"item_id": "msg-resp-stream", "output_index": 0, "content_index": 0}
    empty_part: Final = {"type": "output_text", "text": "", "annotations": []}
    item: Final = {"id": "msg-resp-stream", "type": "message", "role": "assistant"}
    events: Final = (
        ("response.created", {"response": in_progress}),
        ("response.in_progress", {"response": in_progress}),
        ("response.output_item.added", {"output_index": 0, "item": {**item, "status": "in_progress", "content": []}}),
        ("response.content_part.added", {**text_location, "part": empty_part}),
        ("response.output_text.delta", {**text_location, "delta": "Hello"}),
        ("response.output_text.done", {**text_location, "text": "Hello"}),
        ("response.content_part.done", {**text_location, "part": {**empty_part, "text": "Hello"}}),
        (
            "response.output_item.done",
            {"output_index": 0, "item": {**item, "status": "completed", "content": [{**empty_part, "text": "Hello"}]}},
        ),
        ("response.completed", {"response": completed}),
    )
    return "".join(_sse_event(event_type, index, payload) for index, (event_type, payload) in enumerate(events))


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

    @pytest.mark.parametrize(
        ("model", "api_key", "url", "expected_body", "response_body", "expected_response"),
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
                {
                    "id": _bridge_interaction_id("resp-bridge"),
                    "object": "interaction",
                    "model": "gpt-5.6",
                    "status": "completed",
                    "created": datetime.fromtimestamp(1700000000).isoformat(),
                    "updated": datetime.fromtimestamp(1700000000).isoformat(),
                    "outputs": [{"type": "text", "text": "4"}],
                    "steps": [{"type": "model_output", "content": [{"type": "text", "text": "4"}]}],
                    "usage": {"total_input_tokens": 6, "total_output_tokens": 1},
                },
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
        expected_response: dict[str, object],
    ) -> None:
        route: Final = respx_mock.post(url).mock(return_value=httpx.Response(200, json=response_body))

        response: Final = interactions.create(
            model=model,
            input="Hello, how are you?",
            api_key=api_key,
        )

        assert json.loads(route.calls.last.request.content) == expected_body
        assert response.model_dump(exclude_none=True) == expected_response

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
        ("model", "api_key", "url", "expected_body", "sse_body", "expected_chunks"),
        [
            pytest.param(
                "gemini/gemini-2.5-flash",
                "gemini-offline",
                "https://generativelanguage.googleapis.com/v1beta/interactions?alt=sse",
                {"model": "gemini-2.5-flash", "input": "Stream an answer.", "stream": True},
                'data: {"event_type":"step.delta","delta":{"type":"text","text":"Hello"}}\n\n',
                ({"event_type": "step.delta", "object": "interaction", "delta": {"type": "text", "text": "Hello"}},),
                id="gemini",
            ),
            pytest.param(
                "gpt-5.6",
                "sk-offline",
                "https://api.openai.com/v1/responses",
                {"model": "gpt-5.6", "input": "Stream an answer.", "stream": True},
                _responses_stream_body(),
                (
                    {
                        "event_type": "interaction.created",
                        "id": _bridge_interaction_id("resp-stream"),
                        "object": "interaction",
                        "model": "gpt-5.6",
                        "status": "in_progress",
                    },
                    {
                        "event_type": "step.start",
                        "object": "interaction",
                        "index": 0,
                        "step": {"type": "model_output", "content": []},
                    },
                    {
                        "event_type": "step.delta",
                        "object": "interaction",
                        "delta": {"type": "text", "text": "Hello"},
                        "index": 0,
                    },
                    {"event_type": "step.stop", "object": "interaction", "index": 0},
                    {
                        "event_type": "interaction.completed",
                        "id": _bridge_interaction_id("resp-stream"),
                        "object": "interaction",
                        "model": "gpt-5.6",
                        "status": "completed",
                        "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Hello"}]}],
                    },
                ),
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
        sse_body: str,
        expected_chunks: tuple[dict[str, object], ...],
    ) -> None:
        route: Final = respx_mock.post(url).mock(
            return_value=httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})
        )

        response_stream: Final = interactions.create(
            model=model,
            input="Stream an answer.",
            stream=True,
            api_key=api_key,
        )
        chunks: Final = tuple(chunk.model_dump(exclude_none=True) for chunk in response_stream)

        assert json.loads(route.calls.last.request.content) == expected_body
        assert chunks == expected_chunks

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
        assert response.model_dump(exclude_none=True) == {
            "id": "interaction-offline",
            "object": "interaction",
            "model": "gemini-2.5-flash",
            "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": "299792458"}]}],
            "usage": {"input_tokens": 6, "output_tokens": 3},
        }

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
        chunks: Final = [chunk.model_dump(exclude_none=True) async for chunk in response_stream]

        assert json.loads(route.calls.last.request.content) == {
            "model": "gemini-2.5-flash",
            "input": "Stream an answer.",
            "stream": True,
        }
        assert chunks == [
            {"event_type": "step.delta", "object": "interaction", "delta": {"type": "text", "text": "Hello"}},
        ]

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
