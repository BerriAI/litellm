"""
Tests for ChatGPT subscription Responses API transformation

Source: litellm/llms/chatgpt/responses/transformation.py
"""

import json
from collections.abc import Generator
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai.types.responses import ResponseInputMessageItem, ResponseInputText, ResponseOutputMessage, ResponseOutputText
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.llms.chatgpt.common_utils import get_chatgpt_default_instructions
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OPENAI_RESPONSES_API_MIN_MAX_OUTPUT_TOKENS
from litellm.main import responses_api_bridge_check
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


class TestChatGPTResponsesAPITransformation:
    @pytest.mark.parametrize(
        ("requested_tier", "expected_tier"),
        [("default", "default"), ("priority", "priority"), ("fast", "priority")],
    )
    @pytest.mark.parametrize("effort", ["low", "high"])
    def test_chatgpt_preserves_service_tier(self, requested_tier: str, expected_tier: str, effort: str) -> None:
        config: Final = ChatGPTResponsesAPIConfig()
        request: Final = config.transform_responses_api_request(
            model="chatgpt/gpt-6.1-sol",
            input=[{"role": "user", "content": "Reply with OK"}],
            response_api_optional_request_params={
                "service_tier": requested_tier,
                "reasoning": {"effort": effort},
                "max_output_tokens": 16,
                "prompt_cache_options": {"ttl": "30m"},
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["service_tier"] == expected_tier
        assert request["reasoning"] == {"effort": effort}
        assert request["stream"] is True
        assert request["store"] is False
        assert request["max_output_tokens"] == 16
        assert "prompt_cache_options" not in request

    @pytest.mark.parametrize("requested_tier", [None, "auto", "flex", "unknown"])
    def test_chatgpt_does_not_introduce_unsupported_service_tier(self, requested_tier: str | None) -> None:
        config: Final = ChatGPTResponsesAPIConfig()
        request: Final = config.transform_responses_api_request(
            model="chatgpt/gpt-6.1-sol",
            input=[{"role": "user", "content": "Reply with OK"}],
            response_api_optional_request_params={} if requested_tier is None else {"service_tier": requested_tier},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "service_tier" not in request

    def test_chatgpt_sends_system_input_items_as_developer(self) -> None:
        system_string_item: Final = {"role": "system", "content": "Be brief."}
        system_blocks_item: Final = {
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": "Answer in French."}],
        }
        user_item: Final = {"role": "user", "content": "Hello"}
        developer_item: Final = {"role": "developer", "content": "Use metric units."}
        request: Final = ChatGPTResponsesAPIConfig().transform_responses_api_request(
            model="chatgpt/gpt-6.1-sol",
            input=[system_string_item, user_item, system_blocks_item, developer_item],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["input"] == [
            {**system_string_item, "role": "developer"},
            user_item,
            {**system_blocks_item, "role": "developer"},
            developer_item,
        ]

    def test_chatgpt_accepts_sdk_response_objects_replayed_as_input(self) -> None:
        prior_reply: Final = ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(type="output_text", text="Hi!", annotations=[])],
        )
        system_item: Final = ResponseInputMessageItem(
            id="msg_2",
            type="message",
            role="system",
            content=[ResponseInputText(type="input_text", text="Answer in French.")],
        )
        user_item: Final = {"role": "user", "content": "Again"}
        request: Final = ChatGPTResponsesAPIConfig().transform_responses_api_request(
            model="chatgpt/gpt-6.1-sol",
            input=[prior_reply, system_item, user_item],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["input"] == [
            prior_reply.model_dump(exclude_none=True),
            {**system_item.model_dump(exclude_none=True), "role": "developer"},
            user_item,
        ]

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-6-sol",
            "chatgpt/gpt-6-luna",
            "chatgpt/gpt-6-astra",
            "chatgpt/gpt-6.1-sol",
            "chatgpt/gpt-5.5",
            "chatgpt/gpt-5.6-luna",
            "chatgpt/gpt-5.6-sol",
            "chatgpt/gpt-5.6-terra",
            "chatgpt/gpt-5.4",
            "chatgpt/gpt-5.4-pro",
            "chatgpt/gpt-5.3-chat-latest",
            "chatgpt/gpt-5.3-instant",
            "chatgpt/gpt-5.3-codex",
            "chatgpt/gpt-5.3-codex-spark",
        ],
    )
    def test_chatgpt_provider_config_registration(self, model_name):
        config = ProviderConfigManager.get_provider_responses_api_config(
            model=model_name,
            provider=LlmProviders.CHATGPT,
        )

        assert config is not None
        assert isinstance(config, ChatGPTResponsesAPIConfig)
        assert config.custom_llm_provider == LlmProviders.CHATGPT

    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt-6-sol",
            "gpt-6-luna",
            "gpt-6-astra",
            "gpt-6.1-sol",
            "gpt-5.5",
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
        ],
    )
    def test_chatgpt_models_bridge_chat_completions_to_responses(
        self, model_name: str, local_model_cost_map: None
    ) -> None:
        """A chat completions request for these models must take the Responses bridge.

        These models also exist as openai chat models, so an unregistered
        chatgpt model resolves to mode "chat" here and never reaches the bridge.
        """
        model_info, resolved_model = responses_api_bridge_check(
            model=model_name,
            custom_llm_provider="chatgpt",
        )

        assert model_info["mode"] == "responses"
        assert resolved_model == model_name

    @patch("litellm.llms.chatgpt.responses.transformation.Authenticator")
    def test_chatgpt_responses_endpoint_url(self, mock_authenticator_class):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_api_base.return_value = "https://chatgpt.example.com"
        mock_authenticator_class.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()

        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://chatgpt.example.com/responses"

        custom_url = config.get_complete_url(api_base="https://custom.chatgpt.com", litellm_params={})
        assert custom_url == "https://custom.chatgpt.com/responses"

        url_with_slash = config.get_complete_url(api_base="https://chatgpt.example.com/", litellm_params={})
        assert url_with_slash == "https://chatgpt.example.com/responses"

    @patch("litellm.llms.chatgpt.responses.transformation.Authenticator")
    def test_validate_environment_headers(self, mock_authenticator_class):
        mock_auth_instance = MagicMock()
        mock_auth_instance.get_access_token.return_value = "access-123"
        mock_auth_instance.get_account_id.return_value = "acct-123"
        mock_authenticator_class.return_value = mock_auth_instance

        config = ChatGPTResponsesAPIConfig()
        litellm_params = GenericLiteLLMParams(litellm_session_id="session-123")
        headers = config.validate_environment(
            headers={"originator": "custom-origin"},
            model="gpt-5.2",
            litellm_params=litellm_params,
        )

        assert headers["Authorization"] == "Bearer access-123"
        assert headers["ChatGPT-Account-Id"] == "acct-123"
        assert headers["originator"] == "custom-origin"
        assert headers["content-type"] == "application/json"
        assert headers["accept"] == "text/event-stream"
        assert headers["session_id"] == "session-123"

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.2-codex",
            "chatgpt/gpt-5.3-codex",
        ],
    )
    def test_chatgpt_forces_streaming_and_reasoning_include(self, model_name):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model=model_name,
            input="hi",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["stream"] is True
        assert "reasoning.encrypted_content" in request["include"]
        assert request["instructions"].startswith("You are Codex, based on GPT-5.")

    @pytest.mark.parametrize(
        "model_name",
        [
            "chatgpt/gpt-5.2-codex",
            "chatgpt/gpt-5.3-codex-spark",
        ],
    )
    def test_chatgpt_drops_unsupported_responses_params(self, model_name):
        config = ChatGPTResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model=model_name,
            input="hi",
            response_api_optional_request_params={
                # unsupported by ChatGPT Codex
                "user": "user_123",
                "temperature": 0.2,
                "top_p": 0.9,
                "context_management": [{"type": "compaction", "compact_threshold": 200000}],
                "metadata": {"foo": "bar"},
                "stream_options": {"include_usage": True},
                # supported and should be preserved
                "max_output_tokens": 123,
                "truncation": "auto",
                "previous_response_id": "resp_123",
                "reasoning": {"effort": "medium"},
                "tools": [{"type": "function", "function": {"name": "hello"}}],
                "tool_choice": {"type": "function", "function": {"name": "hello"}},
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "user" not in request
        assert "temperature" not in request
        assert "top_p" not in request
        assert "context_management" not in request
        assert "metadata" not in request
        assert "stream_options" not in request

        assert request["max_output_tokens"] == 123
        assert request["truncation"] == "auto"
        assert request["previous_response_id"] == "resp_123"
        assert request["reasoning"] == {"effort": "medium"}
        assert request["tools"] == [{"type": "function", "function": {"name": "hello"}}]
        assert request["tool_choice"] == {
            "type": "function",
            "function": {"name": "hello"},
        }

    @pytest.mark.parametrize(
        ("model_name", "response_model"),
        [
            ("chatgpt/gpt-5.2-codex", "gpt-5.2-codex"),
            ("chatgpt/gpt-5.3-codex", "gpt-5.3-codex"),
        ],
    )
    def test_chatgpt_non_stream_sse_response_parsing(self, model_name: str, response_model: str):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": response_model,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ],
        }
        sse_body = "\n".join(
            [
                f"data: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse_body)
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model=model_name,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Hello!"

    @pytest.mark.parametrize(
        ("model_name", "response_model"),
        [
            ("chatgpt/gpt-5.2-codex", "gpt-5.2-codex"),
            ("chatgpt/gpt-5.3-codex", "gpt-5.3-codex"),
        ],
    )
    def test_chatgpt_non_stream_sse_response_recovers_output_items(self, model_name: str, response_model: str):
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": response_model,
            "output": [],
        }
        streamed_output_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello from stream!"}],
        }
        sse_body = "\n".join(
            [
                f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': streamed_output_item})}",
                f"data: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse_body)
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model=model_name,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Hello from stream!"

    def test_chatgpt_non_stream_sse_recovers_whitespace_padded_chunks(self):
        """Chunks with leading whitespace before `data:` must still parse.

        `_strip_sse_data_from_chunk` only matches the prefix at position 0,
        so without an outer `.strip()` such chunks would fail JSON parsing
        and silently drop the contained event.
        """
        config = ChatGPTResponsesAPIConfig()
        response_payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-5.4",
            "output": [],
        }
        streamed_output_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Recovered from padded"}],
        }
        sse_body = "\n".join(
            [
                f"   data:  {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': streamed_output_item})}   ",
                f"\tdata: {json.dumps({'type': 'response.completed', 'response': response_payload})}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(200, headers={"content-type": "text/event-stream"}, text=sse_body)
        logging_obj = MagicMock()

        parsed = config.transform_response_api_response(
            model="chatgpt/gpt-5.4",
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert parsed.output_text == "Recovered from padded"

    @pytest.mark.parametrize(
        "error_chunk",
        [
            {
                "type": "response.failed",
                "response": {"error": {"message": "ChatGPT upstream failed"}},
            },
            {
                "type": "error",
                "error": {"message": "ChatGPT upstream failed"},
            },
            {"type": "error", "code": "server_error", "message": "ChatGPT upstream failed"},
        ],
    )
    def test_chatgpt_non_stream_sse_response_raises_openai_error(self, error_chunk):
        config = ChatGPTResponsesAPIConfig()
        sse_body = "\n".join(
            [
                f"data: {json.dumps(error_chunk)}",
                "data: [DONE]",
                "",
            ]
        )
        raw_response = httpx.Response(502, headers={"content-type": "text/event-stream"}, text=sse_body)
        logging_obj = MagicMock()

        with pytest.raises(OpenAIError) as exc_info:
            config.transform_response_api_response(
                model="chatgpt/gpt-5.4",
                raw_response=raw_response,
                logging_obj=logging_obj,
            )

        assert exc_info.value.message == "ChatGPT upstream failed"
        assert exc_info.value.status_code == 502


@pytest.fixture
def chatgpt_token_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "auth.json").write_text(
        json.dumps({"access_token": "token", "account_id": "acct", "expires_at": 4102444800})
    )
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)


def _completed_codex_upstream(output_text: str, incomplete_reason: str | None = None) -> MagicMock:
    status: Final = "completed" if incomplete_reason is None else "incomplete"
    completed: Final = {
        "type": f"response.{status}",
        "response": {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "status": status,
            "incomplete_details": None if incomplete_reason is None else {"reason": incomplete_reason},
            "model": "gpt-5.5",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": output_text, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    }
    return MagicMock(
        side_effect=lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"event: response.{status}\ndata: {json.dumps(completed)}\n\n".encode(),
        )
    )


_CITY_SCHEMA: Final = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"],
    "additionalProperties": False,
}


@pytest.mark.usefixtures("chatgpt_token_dir")
def test_responses_text_reaches_the_codex_backend_whole() -> None:
    text: Final = {
        "format": {"type": "json_schema", "name": "city", "schema": _CITY_SCHEMA, "strict": True},
        "verbosity": "low",
    }
    upstream: Final = _completed_codex_upstream('{"city": "Paris", "country": "France"}')
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(upstream)))

    litellm.responses(model="chatgpt/gpt-5.5", input="Where is the Eiffel Tower?", text=text, client=client)

    assert json.loads(upstream.call_args.args[0].content)["text"] == text


_CONVERSATION: Final = (
    {"role": "user", "content": "Where is the Eiffel Tower?"},
    {"role": "assistant", "content": "Paris"},
    {"role": "user", "content": "And the Louvre?"},
)


@pytest.mark.usefixtures("chatgpt_token_dir")
@pytest.mark.parametrize(
    ("client_input", "forwarded_input"),
    [
        ("Where is the Eiffel Tower?", [{"role": "user", "content": "Where is the Eiffel Tower?"}]),
        (list(_CONVERSATION), list(_CONVERSATION)),
    ],
    ids=["string", "list"],
)
def test_responses_input_reaches_the_codex_backend_as_a_message_list(
    client_input: str | list[dict[str, str]],
    forwarded_input: list[dict[str, str]],
) -> None:
    upstream: Final = _completed_codex_upstream("Paris")
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(upstream)))

    litellm.responses(model="chatgpt/gpt-5.5", input=client_input, client=client)

    assert json.loads(upstream.call_args.args[0].content)["input"] == forwarded_input


@pytest.mark.usefixtures("chatgpt_token_dir")
@pytest.mark.parametrize(
    ("response_format", "expected_format"),
    [
        (
            {"type": "json_schema", "json_schema": {"name": "city", "schema": _CITY_SCHEMA, "strict": True}},
            {"type": "json_schema", "name": "city", "schema": _CITY_SCHEMA, "strict": True},
        ),
        ({"type": "json_object"}, {"type": "json_object"}),
    ],
)
def test_chat_response_format_reaches_the_codex_backend_as_text_format(
    response_format: dict[str, object], expected_format: dict[str, object]
) -> None:
    upstream: Final = _completed_codex_upstream('{"city": "Paris", "country": "France"}')
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(upstream)))

    litellm.completion(
        model="chatgpt/gpt-5.5",
        messages=[{"role": "user", "content": "Where is the Eiffel Tower? Answer in JSON."}],
        response_format=response_format,
        client=client,
    )

    assert json.loads(upstream.call_args.args[0].content)["text"]["format"] == expected_format


_JSON_INSTRUCTION: Final = "Reply in JSON with the keys city and country."


class _ForwardedRequest(TypedDict):
    instructions: ReadOnly[str]
    input: ReadOnly[list[dict[str, object]]]


_FORWARDED_REQUEST: Final = TypeAdapter(_ForwardedRequest)


def _send_responses_with_instructions(
    instructions: str,
    text_format: dict[str, object],
    user_content: str | list[dict[str, str]] = "Paris is the capital of France. Return the city and country.",
    earlier_turns: tuple[dict[str, object], ...] = (),
) -> _ForwardedRequest:
    upstream: Final = _completed_codex_upstream('{"city": "Paris", "country": "France"}')
    litellm.responses(
        model="chatgpt/gpt-6.1-sol",
        instructions=instructions,
        input=[*earlier_turns, {"role": "user", "content": user_content}],
        text={"format": text_format},
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(upstream))),
    )
    return _FORWARDED_REQUEST.validate_json(upstream.call_args.args[0].content)


@pytest.mark.usefixtures("chatgpt_token_dir")
@pytest.mark.parametrize(
    ("instructions", "earlier_turns"),
    [
        (_JSON_INSTRUCTION, ()),
        (f"{get_chatgpt_default_instructions()}\n\n{_JSON_INSTRUCTION}", ()),
        (
            _JSON_INSTRUCTION,
            (
                {"role": "user", "content": "I will ask you about a city."},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Sure, I will answer as a json object."}],
                },
            ),
        ),
    ],
    ids=["caller_only", "with_codex_base", "only_assistant_turn_says_json"],
)
def test_json_object_instructions_reach_the_codex_backend_as_an_input_message(
    instructions: str, earlier_turns: tuple[dict[str, object], ...]
) -> None:
    forwarded: Final = _send_responses_with_instructions(
        instructions, {"type": "json_object"}, earlier_turns=earlier_turns
    )

    assert forwarded["input"][0] == {"role": "developer", "content": _JSON_INSTRUCTION}
    assert forwarded["instructions"] == get_chatgpt_default_instructions()


@pytest.mark.usefixtures("chatgpt_token_dir")
@pytest.mark.parametrize(
    ("instructions", "user_content"),
    [
        (_JSON_INSTRUCTION, "Paris is the capital of France. Return the city and country as JSON."),
        (_JSON_INSTRUCTION, [{"type": "input_text", "text": "Paris is the capital of France. Answer as JSON."}]),
        ("Answer in French.", "Paris is the capital of France. Return the city and country."),
    ],
    ids=["input_says_json", "input_part_says_json", "instructions_lack_json"],
)
def test_json_object_instructions_stay_in_the_codex_instructions_when_moving_them_cannot_help(
    instructions: str, user_content: str | list[dict[str, str]]
) -> None:
    forwarded: Final = _send_responses_with_instructions(instructions, {"type": "json_object"}, user_content)

    assert forwarded["instructions"] == f"{get_chatgpt_default_instructions()}\n\n{instructions}"
    assert forwarded["input"] == [{"role": "user", "content": user_content}]


@pytest.mark.usefixtures("chatgpt_token_dir")
def test_json_schema_instructions_stay_in_the_codex_instructions() -> None:
    forwarded: Final = _send_responses_with_instructions(
        _JSON_INSTRUCTION, {"type": "json_schema", "name": "city", "schema": _CITY_SCHEMA, "strict": True}
    )

    assert forwarded["instructions"] == f"{get_chatgpt_default_instructions()}\n\n{_JSON_INSTRUCTION}"
    assert all(item.get("role") != "developer" for item in forwarded["input"])


@pytest.mark.usefixtures("chatgpt_token_dir")
@pytest.mark.parametrize(
    ("max_tokens", "expected_max_output_tokens"),
    [(300, 300), (OPENAI_RESPONSES_API_MIN_MAX_OUTPUT_TOKENS - 1, OPENAI_RESPONSES_API_MIN_MAX_OUTPUT_TOKENS)],
)
def test_chat_max_tokens_reaches_the_codex_backend_no_lower_than_its_minimum(
    max_tokens: int, expected_max_output_tokens: int
) -> None:
    upstream: Final = _completed_codex_upstream("OK")
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(upstream)))

    litellm.completion(
        model="chatgpt/gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=max_tokens,
        client=client,
    )

    assert json.loads(upstream.call_args.args[0].content)["max_output_tokens"] == expected_max_output_tokens


@pytest.mark.usefixtures("chatgpt_token_dir")
def test_non_stream_chat_cut_off_by_max_tokens_returns_the_partial_answer() -> None:
    upstream: Final = _completed_codex_upstream('{"capitals": ["Paris", "Ber', incomplete_reason="max_output_tokens")
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(upstream)))

    response: Final = litellm.completion(
        model="chatgpt/gpt-5.5",
        messages=[{"role": "user", "content": "List capitals as JSON."}],
        max_tokens=16,
        client=client,
    )

    assert response.choices[0].finish_reason == "length"
    assert response.choices[0].message.content == '{"capitals": ["Paris", "Ber'
