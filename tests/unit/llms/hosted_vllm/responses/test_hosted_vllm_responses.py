"""
Tests for hosted_vllm responses API support.

Regression test for: https://github.com/BerriAI/litellm/issues
Bug: client.responses.create() raised TypeError: 'NoneType' object is not a mapping
when extra_body=None was passed through the responses→completion pipeline for
hosted_vllm (and any OpenAI-compatible provider using add_provider_specific_params_to_optional_params).
"""

import json
from collections.abc import Callable, Mapping
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx


import litellm
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import CONVERTED_SYSTEM_NOTE
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


_LATER_DEVELOPER_INPUT: Final = (
    {"role": "user", "content": "Hi there"},
    {"role": "developer", "content": "Answer with exactly one word."},
    {"role": "user", "content": "What is the capital of France?"},
)


_VLLM_RESPONSES_URL: Final = "https://test-vllm.example.com/v1/responses"


def _vllm_responses_route(output_text: str) -> respx.Route:
    return respx.post(_VLLM_RESPONSES_URL).mock(
        return_value=httpx.Response(200, json=_make_mock_responses_api_response(output_text))
    )


def _sent_body(route: respx.Route) -> Mapping[str, object]:
    return json.loads(route.calls.last.request.content)


@respx.mock
def test_hosted_vllm_responses_folds_a_later_developer_item_into_instructions_for_a_system_first_model(
    declare_mid_conversation_system: Callable[[str, bool], None],
) -> None:
    declare_mid_conversation_system("hosted_vllm/Qwen/Qwen3.6-27B", False)
    route: Final = _vllm_responses_route("Paris")
    litellm.responses(
        model="hosted_vllm/Qwen/Qwen3.6-27B",
        instructions="You are a terse assistant.",
        input=list(_LATER_DEVELOPER_INPUT),
        api_base="https://test-vllm.example.com/v1",
    )
    body: Final = _sent_body(route)
    assert body["instructions"] == "You are a terse assistant.\n\nAnswer with exactly one word."
    assert tuple(body["input"]) == (_LATER_DEVELOPER_INPUT[0], _LATER_DEVELOPER_INPUT[2])


@respx.mock
def test_hosted_vllm_responses_folds_for_a_deployment_declared_system_first_and_sends_a_closing_developer_as_user() -> (
    None
):
    assistant_turn: Final = {"type": "message", "role": "assistant", "content": "Paris"}
    route: Final = _vllm_responses_route("PONG")
    litellm.responses(
        model="hosted_vllm/Qwen/Qwen3-8B",
        instructions="You are a terse assistant.",
        input=[*_LATER_DEVELOPER_INPUT, assistant_turn, {"role": "developer", "content": "Reply with PONG."}],
        model_info={"id": "deployment-1", "supports_mid_conversation_system": False},
        api_base="https://test-vllm.example.com/v1",
    )
    body: Final = _sent_body(route)
    assert body["instructions"] == "You are a terse assistant.\n\nAnswer with exactly one word."
    assert tuple(body["input"]) == (
        _LATER_DEVELOPER_INPUT[0],
        _LATER_DEVELOPER_INPUT[2],
        assistant_turn,
        {"role": "user", "content": "Reply with PONG.", "type": "message"},
    )


@respx.mock
def test_hosted_vllm_responses_forwards_a_later_developer_item_in_place_for_an_undeclared_model() -> None:
    route: Final = _vllm_responses_route("Paris")
    litellm.responses(
        model="hosted_vllm/Qwen/Qwen3-8B",
        instructions="You are a terse assistant.",
        input=list(_LATER_DEVELOPER_INPUT),
        api_base="https://test-vllm.example.com/v1",
    )
    body: Final = _sent_body(route)
    assert body["instructions"] == "You are a terse assistant."
    assert tuple(body["input"]) == _LATER_DEVELOPER_INPUT


@respx.mock
def test_hosted_vllm_responses_forwards_a_developer_item_without_content_on_a_deployment_declared_system_first() -> (
    None
):
    route: Final = _vllm_responses_route("Paris")
    input_items: Final = (
        {"role": "user", "content": "What is the capital of France?"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Paris."}]},
        {"type": "message", "role": "developer"},
    )
    litellm.responses(
        model="hosted_vllm/Qwen/Qwen3-8B",
        input=list(input_items),
        model_info={"id": "deployment-1", "supports_mid_conversation_system": False},
        api_base="https://test-vllm.example.com/v1",
    )
    assert tuple(_sent_body(route)["input"]) == input_items


_SYSTEM_FIRST_DEPLOYMENT: Final = {"id": "deployment-1", "supports_mid_conversation_system": False}
_FIRST_USER_ITEM: Final = {"role": "user", "content": "Hi there"}
_ASSISTANT_ITEM: Final = {"type": "message", "role": "assistant", "content": "Hello!"}
_NEXT_USER_ITEM: Final = {"role": "user", "content": "What is the capital of France?"}
_FUNCTION_CALL_ITEM: Final = {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"}
_FUNCTION_CALL_OUTPUT_ITEM: Final = {"type": "function_call_output", "call_id": "call_1", "output": "21C"}
_CUSTOM_TOOL_CALL_ITEM: Final = {"type": "custom_tool_call", "call_id": "call_2", "name": "apply_patch", "input": "*"}
_CUSTOM_TOOL_CALL_OUTPUT_ITEM: Final = {"type": "custom_tool_call_output", "call_id": "call_2", "output": "Done"}
_REFERENCE_IMAGE_PART: Final = {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo=", "detail": "auto"}


def _reminder_as_user_item(*parts: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": CONVERTED_SYSTEM_NOTE}, *parts],
    }


_REMINDER_AS_USER_ITEM: Final = _reminder_as_user_item({"type": "input_text", "text": "Reply in French."})


@respx.mock
@pytest.mark.parametrize(
    ("instructions", "input_items", "expected_instructions", "expected_input"),
    [
        pytest.param(
            None,
            (_FIRST_USER_ITEM, _ASSISTANT_ITEM, {"role": "system", "content": "Reply in French."}, _NEXT_USER_ITEM),
            None,
            (_FIRST_USER_ITEM, _ASSISTANT_ITEM, _REMINDER_AS_USER_ITEM, _NEXT_USER_ITEM),
            id="string-content-without-a-leading-system",
        ),
        pytest.param(
            None,
            (
                _FIRST_USER_ITEM,
                _ASSISTANT_ITEM,
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Match this image."}, _REFERENCE_IMAGE_PART],
                },
                _NEXT_USER_ITEM,
            ),
            None,
            (
                _FIRST_USER_ITEM,
                _ASSISTANT_ITEM,
                _reminder_as_user_item({"type": "input_text", "text": "Match this image."}, _REFERENCE_IMAGE_PART),
                _NEXT_USER_ITEM,
            ),
            id="list-content-keeps-every-part",
        ),
        pytest.param(
            None,
            (_FIRST_USER_ITEM, _ASSISTANT_ITEM, {"role": "system", "content": "Reply in French."}),
            None,
            (_FIRST_USER_ITEM, _ASSISTANT_ITEM, _REMINDER_AS_USER_ITEM),
            id="closing-after-the-assistant",
        ),
        pytest.param(
            None,
            (
                _FIRST_USER_ITEM,
                _FUNCTION_CALL_ITEM,
                {"role": "system", "content": "Reply in French."},
                _FUNCTION_CALL_OUTPUT_ITEM,
            ),
            None,
            (_FIRST_USER_ITEM, _FUNCTION_CALL_ITEM, _FUNCTION_CALL_OUTPUT_ITEM, _REMINDER_AS_USER_ITEM),
            id="between-a-function-call-and-its-output",
        ),
        pytest.param(
            None,
            (
                _FIRST_USER_ITEM,
                _CUSTOM_TOOL_CALL_ITEM,
                {"role": "system", "content": "Reply in French."},
                _CUSTOM_TOOL_CALL_OUTPUT_ITEM,
            ),
            None,
            (_FIRST_USER_ITEM, _CUSTOM_TOOL_CALL_ITEM, _CUSTOM_TOOL_CALL_OUTPUT_ITEM, _REMINDER_AS_USER_ITEM),
            id="between-a-custom-tool-call-and-its-output",
        ),
        pytest.param(
            None,
            (
                {"role": "system", "content": "You are terse."},
                _FIRST_USER_ITEM,
                _ASSISTANT_ITEM,
                {"role": "developer", "content": "Answer with exactly one word."},
                {"role": "system", "content": "Reply in French."},
                _NEXT_USER_ITEM,
            ),
            "You are terse.\n\nAnswer with exactly one word.",
            (_FIRST_USER_ITEM, _ASSISTANT_ITEM, _REMINDER_AS_USER_ITEM, _NEXT_USER_ITEM),
            id="with-a-leading-system-and-a-later-developer",
        ),
        pytest.param(
            "You are terse.",
            (_FIRST_USER_ITEM, {"role": "system", "content": "Reply in French."}, _NEXT_USER_ITEM),
            "You are terse.",
            (_FIRST_USER_ITEM, _REMINDER_AS_USER_ITEM, _NEXT_USER_ITEM),
            id="with-instructions",
        ),
    ],
)
def test_hosted_vllm_responses_sends_a_later_system_item_as_a_user_turn_in_place_on_a_system_first_deployment(
    instructions: str | None,
    input_items: tuple[Mapping[str, object], ...],
    expected_instructions: str | None,
    expected_input: tuple[Mapping[str, object], ...],
) -> None:
    route: Final = _vllm_responses_route("Paris")
    litellm.responses(
        model="hosted_vllm/Qwen/Qwen3.8-27B",
        instructions=instructions,
        input=list(input_items),
        model_info=_SYSTEM_FIRST_DEPLOYMENT,
        api_base="https://test-vllm.example.com/v1",
    )
    body: Final = _sent_body(route)
    assert body.get("instructions") == expected_instructions
    assert tuple(body["input"]) == expected_input


_LATER_SYSTEM_INPUT: Final = (
    _FIRST_USER_ITEM,
    _FUNCTION_CALL_ITEM,
    {"role": "system", "content": "Reply in French."},
    _FUNCTION_CALL_OUTPUT_ITEM,
    _ASSISTANT_ITEM,
    {"role": "system", "content": [{"type": "input_text", "text": "Reply in French."}]},
)


@respx.mock
@pytest.mark.parametrize(
    ("input_value", "model_info"),
    [
        pytest.param(list(_LATER_SYSTEM_INPUT), None, id="undeclared"),
        pytest.param(
            list(_LATER_SYSTEM_INPUT),
            {"id": "deployment-1", "supports_mid_conversation_system": True},
            id="declared-true",
        ),
        pytest.param(
            [_FIRST_USER_ITEM, _FUNCTION_CALL_ITEM, _FUNCTION_CALL_OUTPUT_ITEM, _ASSISTANT_ITEM, _NEXT_USER_ITEM],
            _SYSTEM_FIRST_DEPLOYMENT,
            id="system-first-without-a-system-item",
        ),
        pytest.param(
            [_FIRST_USER_ITEM, _ASSISTANT_ITEM, {"type": "message", "role": "system"}],
            _SYSTEM_FIRST_DEPLOYMENT,
            id="system-first-with-a-system-item-without-content",
        ),
        pytest.param("What is the capital of France?", _SYSTEM_FIRST_DEPLOYMENT, id="system-first-string-input"),
    ],
)
def test_hosted_vllm_responses_forwards_the_input_unchanged_unless_a_system_first_deployment_gets_a_later_system_item(
    input_value: str | list[Mapping[str, object]], model_info: Mapping[str, object] | None
) -> None:
    route: Final = _vllm_responses_route("Paris")
    litellm.responses(
        model="hosted_vllm/Qwen/Qwen3.8-27B",
        instructions="You are terse.",
        input=input_value,
        model_info=model_info,
        api_base="https://test-vllm.example.com/v1",
    )
    body: Final = _sent_body(route)
    assert body["instructions"] == "You are terse."
    assert body["input"] == input_value


@respx.mock
def test_hosted_vllm_responses_preserves_cache_salt_isolation_inputs() -> None:
    route: Final = _vllm_responses_route("ok")
    requests: Final[tuple[Mapping[str, str], ...]] = (
        {"cache_salt": "salt-a"},
        {"cache_salt": "salt-b"},
        {},
        {"cache_salt": "salt-a"},
    )

    def sent_body(kwargs: Mapping[str, str]) -> Mapping[str, object]:
        litellm.responses(
            model="hosted_vllm/Qwen/Qwen3-8B",
            input="long prompt",
            api_base="https://test-vllm.example.com/v1",
            **kwargs,
        )
        return _sent_body(route)

    assert [sent_body(kwargs) for kwargs in requests] == [
        {"input": "long prompt", "cache_salt": "salt-a", "model": "Qwen/Qwen3-8B"},
        {"input": "long prompt", "cache_salt": "salt-b", "model": "Qwen/Qwen3-8B"},
        {"input": "long prompt", "model": "Qwen/Qwen3-8B"},
        {"input": "long prompt", "cache_salt": "salt-a", "model": "Qwen/Qwen3-8B"},
    ]


@respx.mock
def test_openai_responses_still_drops_cache_salt() -> None:
    route: Final = respx.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json=_make_mock_responses_api_response("ok"))
    )

    litellm.responses(model="openai/gpt-4o", input="long prompt", cache_salt="salt-a", api_key="test-key")

    assert "cache_salt" not in _sent_body(route)
