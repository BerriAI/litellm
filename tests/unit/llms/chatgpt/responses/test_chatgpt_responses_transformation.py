"""
Tests for ChatGPT subscription Responses API transformation

Source: litellm/llms/chatgpt/responses/transformation.py
"""

import base64
import json
from collections.abc import Generator
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.llms.openai.common_utils import OpenAIError
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
        assert "max_output_tokens" not in request
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
                "max_output_tokens": 123,
                "stream_options": {"include_usage": True},
                # supported and should be preserved
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
        assert "max_output_tokens" not in request
        assert "stream_options" not in request

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


def _fake_chatgpt_oauth_token(account_id: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
    payload = (
        base64.urlsafe_b64encode(
            json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    return f"{header}.{payload}.fake-signature"


def _gateway_authenticator() -> MagicMock:
    authenticator = MagicMock()
    authenticator.get_access_token.return_value = "gateway-access-token"
    authenticator.get_account_id.return_value = "acct-gateway"
    return authenticator


def _config_with(authenticator: MagicMock) -> ChatGPTResponsesAPIConfig:
    return ChatGPTResponsesAPIConfig(authenticator=authenticator)


class TestChatGPTClientCredentialForwarding:
    def test_client_chatgpt_oauth_wins_over_configured_key_and_gateway_login(self):
        authenticator = _gateway_authenticator()
        token = _fake_chatgpt_oauth_token("acct-alice")

        headers = _config_with(authenticator).validate_environment(
            headers={"Authorization": f"Bearer {token}", "originator": "codex_cli_rs"},
            model="gpt-5.4",
            litellm_params=GenericLiteLLMParams(api_key="configured-gateway-key", litellm_session_id="session-1"),
        )

        assert headers["Authorization"] == f"Bearer {token}"
        assert headers["ChatGPT-Account-Id"] == "acct-alice"
        assert headers["originator"] == "codex_cli_rs"
        assert headers["accept"] == "text/event-stream"
        assert [name for name in headers if name.lower() == "authorization"] == ["Authorization"]
        authenticator.get_access_token.assert_not_called()

    def test_explicit_account_header_wins_over_jwt_claim(self):
        token = _fake_chatgpt_oauth_token("acct-from-jwt")

        headers = _config_with(_gateway_authenticator()).validate_environment(
            headers={"authorization": f"Bearer {token}", "chatgpt-account-id": "acct-from-header"},
            model="gpt-5.4",
            litellm_params=GenericLiteLLMParams(),
        )

        assert headers["ChatGPT-Account-Id"] == "acct-from-header"
        assert [name for name in headers if name.lower() == "chatgpt-account-id"] == ["ChatGPT-Account-Id"]

    @pytest.mark.parametrize(
        "bearer", ["Bearer sk-proj-fake-platform-key", "Bearer not-a-jwt", "Bearer a.bm90LWpzb24.c"]
    )
    def test_non_chatgpt_bearer_falls_back_to_configured_key(self, bearer, tmp_path):
        authenticator = Authenticator(auth_file=str(tmp_path / "missing-default.json"))

        headers = ChatGPTResponsesAPIConfig(authenticator=authenticator).validate_environment(
            headers={"Authorization": bearer},
            model="gpt-5.4",
            litellm_params=GenericLiteLLMParams(api_key="configured-gateway-key"),
        )

        assert "ChatGPT-Account-Id" not in headers
        assert headers["Authorization"] == "Bearer configured-gateway-key"

    def test_configured_key_is_used_without_client_credential(self):
        authenticator = _gateway_authenticator()

        headers = _config_with(authenticator).validate_environment(
            headers={},
            model="gpt-5.4",
            litellm_params=GenericLiteLLMParams(api_key="configured-gateway-key"),
        )

        assert headers["Authorization"] == "Bearer configured-gateway-key"
        authenticator.get_access_token.assert_not_called()

    def test_gateway_login_is_used_without_client_credential_or_configured_key(self):
        headers = _config_with(_gateway_authenticator()).validate_environment(
            headers={},
            model="gpt-5.4",
            litellm_params=GenericLiteLLMParams(),
        )

        assert headers["Authorization"] == "Bearer gateway-access-token"
        assert headers["ChatGPT-Account-Id"] == "acct-gateway"

    def test_two_clients_never_share_a_credential(self):
        config = _config_with(_gateway_authenticator())
        alice = _fake_chatgpt_oauth_token("acct-alice")
        bob = _fake_chatgpt_oauth_token("acct-bob")

        forwarded = [
            config.validate_environment(
                headers={"Authorization": f"Bearer {token}"}, model="gpt-5.4", litellm_params=GenericLiteLLMParams()
            )
            for token in (alice, bob, alice)
        ]

        assert [h["Authorization"] for h in forwarded] == [f"Bearer {alice}", f"Bearer {bob}", f"Bearer {alice}"]
        assert [h["ChatGPT-Account-Id"] for h in forwarded] == ["acct-alice", "acct-bob", "acct-alice"]


class TestChatGPTResponsesMultiAccount:
    @pytest.fixture(autouse=True)
    def fixed_clock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("time.time", lambda: 1700000000)

    @staticmethod
    def _write_auth_record(path: Path, token: str, account_id: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "access_token": token,
                    "account_id": account_id,
                    "expires_at": 4102444800,
                }
            )
        )

    def test_validate_environment_uses_per_model_auth_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        default_dir: Final = tmp_path / "default"
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(default_dir))
        monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
        self._write_auth_record(default_dir / "auth.json", "token-default", "acct-default")
        file_a: Final = tmp_path / "account-a" / "auth.json"
        file_b: Final = tmp_path / "account-b" / "auth.json"
        self._write_auth_record(file_a, "token-a", "acct-a")
        self._write_auth_record(file_b, "token-b", "acct-b")

        config: Final = ChatGPTResponsesAPIConfig()

        headers_a: Final = config.validate_environment(
            headers={},
            model="gpt-6.1-sol",
            litellm_params=GenericLiteLLMParams(chatgpt_auth_file=str(file_a)),
        )
        headers_b: Final = config.validate_environment(
            headers={},
            model="gpt-6.1-sol",
            litellm_params=GenericLiteLLMParams(chatgpt_auth_file=str(file_b)),
        )
        headers_default: Final = config.validate_environment(
            headers={},
            model="gpt-6.1-sol",
            litellm_params=GenericLiteLLMParams(),
        )

        assert headers_a["Authorization"] == "Bearer token-a"
        assert headers_a["ChatGPT-Account-Id"] == "acct-a"
        assert headers_b["Authorization"] == "Bearer token-b"
        assert headers_b["ChatGPT-Account-Id"] == "acct-b"
        assert headers_default["Authorization"] == "Bearer token-default"
        assert headers_default["ChatGPT-Account-Id"] == "acct-default"
