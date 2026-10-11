import base64
import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter
from respx import MockRouter

import litellm
from litellm import interactions

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


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


@pytest.mark.parametrize(
    ("model", "api_key", "url", "response_body", "expected_body"),
    [
        (
            "gemini/gemini-2.5-flash",
            "gemini-offline",
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            {
                "id": "interaction-sync",
                "object": "interaction",
                "model": "gemini-2.5-flash",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "four"}]}],
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
            {"model": "gemini-2.5-flash", "input": "What is 2 + 2?"},
        ),
        (
            "gpt-4o",
            "sk-offline",
            "https://api.openai.com/v1/responses",
            {
                "id": "resp-sync",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "four"}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
            },
            {"model": "gpt-4o", "input": "What is 2 + 2?"},
        ),
    ],
)
def test_create_simple_string_input_for_google_and_responses(
    respx_mock: MockRouter,
    model: str,
    api_key: str,
    url: str,
    response_body: Mapping[str, JsonValue],
    expected_body: Mapping[str, JsonValue],
) -> None:
    route: Final = respx_mock.post(url).mock(return_value=httpx.Response(200, json=response_body))

    response: Final = interactions.create(
        model=model,
        input="What is 2 + 2?",
        api_key=api_key,
    )

    assert _JSON_OBJECT.validate_json(route.calls.last.request.content) == expected_body
    assert response.status == "completed"


@pytest.mark.parametrize(
    ("model", "api_key", "url", "response_body", "expected_body"),
    [
        (
            "gemini/gemini-2.5-flash",
            "gemini-offline",
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            {
                "id": "interaction-instruction",
                "object": "interaction",
                "model": "gemini-2.5-flash",
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": "ahoy"}]}],
            },
            {
                "model": "gemini-2.5-flash",
                "input": "What are you?",
                "system_instruction": "Answer like a pirate.",
            },
        ),
        (
            "gpt-4o",
            "sk-offline",
            "https://api.openai.com/v1/responses",
            {
                "id": "resp-instruction",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ahoy"}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
            },
            {"model": "gpt-4o", "input": "What are you?", "instructions": "Answer like a pirate."},
        ),
    ],
)
def test_create_with_system_instruction_for_google_and_responses(
    respx_mock: MockRouter,
    model: str,
    api_key: str,
    url: str,
    response_body: Mapping[str, JsonValue],
    expected_body: Mapping[str, JsonValue],
) -> None:
    route: Final = respx_mock.post(url).mock(return_value=httpx.Response(200, json=response_body))

    response: Final = interactions.create(
        model=model,
        input="What are you?",
        system_instruction="Answer like a pirate.",
        api_key=api_key,
    )

    assert _JSON_OBJECT.validate_json(route.calls.last.request.content) == expected_body
    assert response.status == "completed"


def test_create_streaming_returns_interaction_events(
    respx_mock: MockRouter,
) -> None:
    route: Final = respx_mock.post(
        "https://generativelanguage.googleapis.com/v1beta/interactions?alt=sse"
    ).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"event_type":"step.delta","delta":{"type":"text_delta","text":"1, 2, 3"}}\n\n'
                b'data: {"event_type":"interaction.completed","id":"interaction-stream","status":"completed"}\n\n'
            ),
        )
    )

    chunks: Final = tuple(
        interactions.create(
            model="gemini/gemini-2.5-flash",
            input="Count from 1 to 3.",
            stream=True,
            api_key="gemini-offline",
        )
    )

    assert tuple((chunk.event_type, chunk.delta) for chunk in chunks) == (
        ("step.delta", {"type": "text_delta", "text": "1, 2, 3"}),
        ("interaction.completed", None),
    )
    assert _JSON_OBJECT.validate_json(route.calls.last.request.content) == {
        "model": "gemini-2.5-flash",
        "input": "Count from 1 to 3.",
        "stream": True,
    }
