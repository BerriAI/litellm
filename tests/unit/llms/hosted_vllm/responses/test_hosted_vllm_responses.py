"""
Tests for hosted_vllm responses API support.

Regression test for: https://github.com/BerriAI/litellm/issues
Bug: client.responses.create() raised TypeError: 'NoneType' object is not a mapping
when extra_body=None was passed through the responses→completion pipeline for
hosted_vllm (and any OpenAI-compatible provider using add_provider_specific_params_to_optional_params).
"""

import json
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
from openai.types.responses import EasyInputMessage

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.hosted_vllm.responses.transformation import (
    HostedVLLMResponsesAPIConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def _make_mock_responses_api_response(content: str = "Hello! I'm doing well.") -> dict:
    return {
        "id": "resp-test123",
        "object": "response",
        "created_at": 1234567890,
        "model": "Qwen/Qwen3-8B",
        "output": [
            {
                "type": "message",
                "id": "msg-test123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        ],
        "status": "completed",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
    }


def _make_mock_http_client(response_body: dict) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = response_body
    mock_response.text = json.dumps(response_body)
    mock_client.post.return_value = mock_response
    return mock_client


def test_hosted_vllm_responses_create_with_string_input():
    """
    Test that hosted_vllm routes directly to the native /v1/responses endpoint
    when the Responses API config is registered, and correctly parses the response.
    """
    mock_client = _make_mock_http_client(_make_mock_responses_api_response("I'm doing well, thanks!"))

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.get_httpx_client",
        return_value=mock_client,
    ):
        response = litellm.responses(
            model="hosted_vllm/Qwen/Qwen3-8B",
            input="Hello, how are you?",
            api_base="https://test-vllm.example.com/v1",
            api_key="test-key",
        )

    from litellm.types.llms.openai import ResponsesAPIResponse

    assert response is not None
    assert isinstance(response, ResponsesAPIResponse)
    assert len(response.output) > 0
    output_message = response.output[0]
    assert output_message.role == "assistant"  # type: ignore[union-attr]
    assert len(output_message.content) > 0  # type: ignore[union-attr]
    assert "well" in output_message.content[0].text  # type: ignore[union-attr]


def test_hosted_vllm_responses_create_with_explicit_none_extra_body():
    """
    Directly verify the fix in add_provider_specific_params_to_optional_params:
    extra_body=None must not crash when building optional_params.
    """
    from litellm.utils import get_optional_params

    # This should not raise TypeError: 'NoneType' object is not a mapping
    optional_params = get_optional_params(
        model="Qwen/Qwen3-8B",
        custom_llm_provider="hosted_vllm",
        extra_body=None,
    )

    # extra_body=None should be normalized to an empty dict (or absent)
    assert optional_params.get("extra_body") is not None or "extra_body" not in optional_params


def test_hosted_vllm_provider_config_registration():
    """Test that ProviderConfigManager returns HostedVLLMResponsesAPIConfig for hosted_vllm."""
    config = ProviderConfigManager.get_provider_responses_api_config(
        model="hosted_vllm/Qwen/Qwen3-8B",
        provider=LlmProviders.HOSTED_VLLM,
    )

    assert config is not None
    assert isinstance(config, HostedVLLMResponsesAPIConfig)
    assert config.custom_llm_provider == LlmProviders.HOSTED_VLLM


def test_hosted_vllm_responses_api_url():
    """Test get_complete_url() constructs the correct URL."""
    config = HostedVLLMResponsesAPIConfig()

    # api_base without /v1
    url = config.get_complete_url(
        api_base="http://localhost:8000",
        litellm_params={},
    )
    assert url == "http://localhost:8000/v1/responses"

    # api_base with /v1
    url_with_v1 = config.get_complete_url(
        api_base="http://localhost:8000/v1",
        litellm_params={},
    )
    assert url_with_v1 == "http://localhost:8000/v1/responses"

    # api_base with trailing slash
    url_with_slash = config.get_complete_url(
        api_base="http://localhost:8000/v1/",
        litellm_params={},
    )
    assert url_with_slash == "http://localhost:8000/v1/responses"


def test_hosted_vllm_responses_api_url_requires_api_base():
    """Test get_complete_url() raises ValueError when api_base is not set."""
    config = HostedVLLMResponsesAPIConfig()

    with pytest.raises(ValueError, match="api_base not set"):
        config.get_complete_url(
            api_base=None,
            litellm_params={},
        )


def test_hosted_vllm_validate_environment_default_api_key():
    """Test validate_environment() defaults to 'fake-api-key' when no key is provided."""
    config = HostedVLLMResponsesAPIConfig()

    headers = config.validate_environment(
        headers={},
        model="Qwen/Qwen3-8B",
        litellm_params=GenericLiteLLMParams(),
    )

    assert headers.get("Authorization") == "Bearer fake-api-key"


def test_hosted_vllm_validate_environment_custom_api_key():
    """Test validate_environment() uses the provided api_key."""
    config = HostedVLLMResponsesAPIConfig()

    headers = config.validate_environment(
        headers={},
        model="Qwen/Qwen3-8B",
        litellm_params=GenericLiteLLMParams(api_key="my-custom-key"),
    )

    assert headers.get("Authorization") == "Bearer my-custom-key"


VLLM_API_BASE: Final = "https://vllm.invalid/v1"
CODEX_INSTRUCTIONS: Final = "You are Codex, a coding agent. Follow the repository's AGENTS.md."


def _sent_input(input: object) -> object:
    request: Final = HostedVLLMResponsesAPIConfig().transform_responses_api_request(
        model="kimi-k3",
        input=input,
        response_api_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    return request["input"]


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        pytest.param(
            [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": CODEX_INSTRUCTIONS}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "fix the failing test"}],
                },
            ],
            [
                {"type": "message", "role": "system", "content": [{"type": "input_text", "text": CODEX_INSTRUCTIONS}]},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "fix the failing test"}],
                },
            ],
            id="codex-leading-developer-item",
        ),
        pytest.param(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "developer", "content": "Answer in French from now on."},
                {"role": "user", "content": "how are you?"},
            ],
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "Answer in French from now on."},
                {"role": "user", "content": "how are you?"},
            ],
            id="mid-conversation-developer-item",
        ),
        pytest.param(
            [
                EasyInputMessage(role="developer", content=CODEX_INSTRUCTIONS, type="message"),
                EasyInputMessage(role="user", content="hi", type="message"),
            ],
            [
                {"role": "system", "content": CODEX_INSTRUCTIONS, "type": "message"},
                {"role": "user", "content": "hi", "type": "message"},
            ],
            id="pydantic-developer-item",
        ),
    ],
)
def test_hosted_vllm_responses_sends_developer_items_as_system(input: object, expected: object) -> None:
    assert _sent_input(input) == expected


def test_hosted_vllm_responses_keeps_everything_but_the_developer_role() -> None:
    input: Final = [
        {"type": "message", "role": "developer", "id": "msg_dev_1", "status": "completed", "content": "Be terse."},
        {"type": "message", "role": "system", "content": "You are helpful."},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city": "Paris"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "21C"},
    ]
    assert _sent_input(input) == [
        {"type": "message", "role": "system", "id": "msg_dev_1", "status": "completed", "content": "Be terse."},
        *input[1:],
    ]


def test_hosted_vllm_responses_passes_string_input_and_instructions_through() -> None:
    request: Final = HostedVLLMResponsesAPIConfig().transform_responses_api_request(
        model="kimi-k3",
        input="hello",
        response_api_optional_request_params={"instructions": CODEX_INSTRUCTIONS},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert request["input"] == "hello"
    assert request["instructions"] == CODEX_INSTRUCTIONS


@pytest.fixture
def vllm_responses_route(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> respx.Route:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    return respx_mock.post(f"{VLLM_API_BASE}/responses").mock(
        return_value=httpx.Response(200, json=_make_mock_responses_api_response())
    )


@pytest.mark.asyncio
async def test_hosted_vllm_aresponses_sends_codex_developer_item_as_system_on_the_wire(
    vllm_responses_route: respx.Route,
) -> None:
    response: Final = await litellm.aresponses(
        model="hosted_vllm/kimi-k3",
        input=[
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": CODEX_INSTRUCTIONS}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix the failing test"}]},
        ],
        api_base=VLLM_API_BASE,
    )
    body: Final = json.loads(vllm_responses_route.calls.last.request.content)
    assert [item["role"] for item in body["input"]] == ["system", "user"]
    assert body["input"][0]["content"] == [{"type": "input_text", "text": CODEX_INSTRUCTIONS}]
    assert response.output[0].role == "assistant"  # type: ignore[union-attr]
