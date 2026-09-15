import json
from pathlib import Path

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


def _prism_client(requests: list[httpx.Request], response_body: dict[str, object]) -> HTTPHandler:
    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_body)

    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))


def test_prism_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("PRISM_API_KEY", "prism-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="prism/deepseek-v4-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-v4-flash"
    assert provider == "prism"
    assert api_key == "prism-test-key"
    assert api_base == "https://api.prisminference.com/v1"


def test_prism_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("PRISM_API_KEY", "prism-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="prism/deepseek-v4-flash",
        custom_llm_provider=None,
        api_base="https://prism.internal.example/v1",
        api_key="prism-explicit-key",
    )

    assert provider == "prism"
    assert api_key == "prism-explicit-key"
    assert api_base == "https://prism.internal.example/v1"


@pytest.mark.parametrize(
    ("model", "input_cost", "output_cost", "max_output_tokens"),
    [
        ("prism/deepseek-v4.1-flash", 0.30, 1.20, 384_000),
        ("prism/deepseek-v4-flash", 0.14, 0.28, 393_216),
    ],
)
def test_prism_model_cost_and_capabilities(
    model: str,
    input_cost: float,
    output_cost: float,
    max_output_tokens: int,
):
    from litellm.cost_calculator import cost_per_token

    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="prism",
    )
    model_info = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(input_cost)
    assert completion_cost == pytest.approx(output_cost)
    assert model_info["cache_read_input_token_cost"] == pytest.approx(7e-08)
    assert model_info["max_input_tokens"] == 1_000_000
    assert model_info["max_output_tokens"] == max_output_tokens
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_native_streaming"] is True
    assert model_info["supports_reasoning"] is True
    assert model_info["supports_response_schema"] is True


def test_prism_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    prism = next(provider for provider in providers if provider["litellm_provider"] == "prism")

    assert prism["provider"] == "PRISM"
    assert prism["provider_display_name"] == "Prism"
    assert prism["default_model_placeholder"] == "prism/deepseek-v4.1-flash"
    assert {field["key"]: field["required"] for field in prism["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_prism_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["prism"]["endpoints"] == {
        "chat_completions": True,
        "messages": True,
        "responses": True,
        "embeddings": False,
        "image_generations": False,
        "audio_transcriptions": False,
        "audio_speech": False,
        "moderations": False,
        "batches": False,
        "rerank": False,
        "a2a": False,
    }


def test_prism_responses_request():
    requests: list[httpx.Request] = []
    client = _prism_client(
        requests,
        {
            "id": "resp_prism",
            "object": "response",
            "created_at": 1_789_550_000,
            "model": "deepseek-v4-flash",
            "status": "completed",
            "output": [
                {
                    "id": "msg_prism",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello from Prism",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 4,
                "output_tokens": 3,
                "total_tokens": 7,
            },
        },
    )

    response = litellm.responses(
        model="prism/deepseek-v4-flash",
        input="Say hello",
        api_key="prism-test-key",
        client=client,
    )

    request = requests[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://api.prisminference.com/v1/responses"
    assert request.headers["authorization"] == "Bearer prism-test-key"
    assert body["model"] == "deepseek-v4-flash"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Prism"


@pytest.mark.asyncio
async def test_prism_anthropic_messages_request():
    requests: list[httpx.Request] = []
    response_body: dict[str, object] = {
        "id": "msg_prism",
        "type": "message",
        "role": "assistant",
        "model": "deepseek-v4-flash",
        "content": [{"type": "text", "text": "Hello from Prism"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 4, "output_tokens": 3},
    }

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_body)

    client = AsyncHTTPHandler()
    await client.close()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

    response = await litellm.anthropic.messages.acreate(
        model="prism/deepseek-v4-flash",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=32,
        api_key="prism-test-key",
        client=client,
    )

    request = requests[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://api.prisminference.com/v1/messages"
    assert request.headers["authorization"] == "Bearer prism-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "deepseek-v4-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Prism"
