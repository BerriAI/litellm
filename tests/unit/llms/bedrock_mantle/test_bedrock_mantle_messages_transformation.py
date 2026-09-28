"""
Unit tests for the bedrock_mantle native Anthropic Messages route.

Mantle serves its Claude models only on `/anthropic/v1/messages` (the OpenAI
paths reject them), so `bedrock_mantle/anthropic.claude-*` requests on
/v1/messages must hit that endpoint directly instead of the chat-completions
bridge. These tests lock the dispatcher gate, the URL derivation from the
OpenAI-surface base that get_llm_provider pre-fills, the version header, the
Bearer/SigV4 auth chain, and the wire request through the public entrypoint.
"""

import json
from unittest.mock import MagicMock

import httpx
import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock_mantle.messages.transformation import (
    BedrockMantleAnthropicMessagesConfig,
    build_mantle_native_messages_url,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager

MESSAGES_PATH = "/anthropic/v1/messages"


@pytest.fixture(autouse=True)
def _httpx_transport_with_fresh_clients(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())


@pytest.fixture(autouse=True)
def _no_ambient_mantle_env(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
    monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)
    monkeypatch.delenv("BEDROCK_MANTLE_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION_NAME", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)


def _anthropic_response() -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "anthropic.claude-sonnet-5",
            "content": [{"type": "text", "text": "pong"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 1},
        },
    )


_SSE_EVENTS = (
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "type": "message",
                "role": "assistant",
                "model": "anthropic.claude-sonnet-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        },
    ),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "pong"}},
    ),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
    ("message_stop", {"type": "message_stop"}),
)


def _sse_response() -> httpx.Response:
    body = "".join(f"event: {event}\ndata: {json.dumps(payload)}\n\n" for event, payload in _SSE_EVENTS).encode()
    return httpx.Response(status_code=200, content=body, headers={"content-type": "text/event-stream"})


def _mantle_messages_route(region: str) -> respx.Route:
    return respx.post(f"https://bedrock-mantle.{region}.api.aws{MESSAGES_PATH}")


def _sent_body(route: respx.Route) -> dict:
    return json.loads(route.calls.last.request.content)


class TestDispatch:
    def test_claude_models_get_the_native_messages_config(self):
        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="anthropic.claude-sonnet-5", provider=litellm.LlmProviders.BEDROCK_MANTLE
        )
        assert isinstance(config, BedrockMantleAnthropicMessagesConfig)
        assert config.custom_llm_provider == "bedrock_mantle"

    @pytest.mark.parametrize("model", ["openai.gpt-5.6-sol", "openai.gpt-oss-120b-1:0", "google.gemma-4-31b"])
    def test_non_claude_models_keep_the_bridge(self, model):
        assert (
            ProviderConfigManager.get_provider_anthropic_messages_config(
                model=model, provider=litellm.LlmProviders.BEDROCK_MANTLE
            )
            is None
        )


class TestURL:
    @pytest.mark.parametrize(
        "api_base",
        [
            "https://bedrock-mantle.us-east-1.api.aws/v1",
            "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
            "https://bedrock-mantle.us-east-1.api.aws/openai/v1/",
            "https://bedrock-mantle.us-east-1.api.aws",
            "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
        ],
    )
    def test_prefilled_openai_base_becomes_the_messages_endpoint(self, api_base):
        url = build_mantle_native_messages_url(api_base, {"aws_region_name": "us-east-1"})
        assert url == f"https://bedrock-mantle.us-east-1.api.aws{MESSAGES_PATH}"

    def test_aws_region_name_wins_over_the_prefilled_host_region(self):
        url = build_mantle_native_messages_url(
            "https://bedrock-mantle.us-east-1.api.aws/v1", {"aws_region_name": "us-east-2"}
        )
        assert url == f"https://bedrock-mantle.us-east-2.api.aws{MESSAGES_PATH}"

    def test_host_region_is_used_when_no_region_param(self):
        url = build_mantle_native_messages_url("https://bedrock-mantle.eu-west-1.api.aws/v1", {})
        assert url == f"https://bedrock-mantle.eu-west-1.api.aws{MESSAGES_PATH}"

    def test_custom_host_is_preserved(self):
        url = build_mantle_native_messages_url("https://vpce-abc.bedrock-mantle.example.com/v1", {})
        assert url == f"https://vpce-abc.bedrock-mantle.example.com{MESSAGES_PATH}"

    def test_env_base_is_used_without_api_base(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_API_BASE", "https://mantle-proxy.internal/openai/v1")
        assert build_mantle_native_messages_url(None, {}) == f"https://mantle-proxy.internal{MESSAGES_PATH}"

    def test_default_host_comes_from_mantle_region_env(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_REGION", "ap-northeast-1")
        assert (
            build_mantle_native_messages_url(None, {})
            == f"https://bedrock-mantle.ap-northeast-1.api.aws{MESSAGES_PATH}"
        )

    def test_config_get_complete_url_reads_litellm_params(self):
        config = BedrockMantleAnthropicMessagesConfig()
        url = config.get_complete_url(
            api_base="https://bedrock-mantle.us-east-1.api.aws/v1",
            api_key=None,
            model="anthropic.claude-sonnet-5",
            optional_params={},
            litellm_params={"aws_region_name": "us-west-2"},
        )
        assert url == f"https://bedrock-mantle.us-west-2.api.aws{MESSAGES_PATH}"


class TestEnvironment:
    def _validate(self, headers: dict, litellm_params: dict) -> dict:
        config = BedrockMantleAnthropicMessagesConfig()
        merged, _ = config.validate_anthropic_messages_environment(
            headers=headers,
            model="anthropic.claude-sonnet-5",
            messages=[],
            optional_params={},
            litellm_params=litellm_params,
        )
        return merged

    def test_adds_the_anthropic_version_header(self):
        assert self._validate({}, {})["anthropic-version"] == "2023-06-01"

    def test_keeps_a_caller_supplied_version_header(self):
        merged = self._validate({"Anthropic-Version": "2024-01-01"}, {})
        assert merged["Anthropic-Version"] == "2024-01-01"
        assert "anthropic-version" not in merged

    def test_project_id_becomes_the_workspace_header(self):
        assert self._validate({}, {"aws_bedrock_project_id": "proj_123"})["anthropic-workspace"] == "proj_123"


class TestRequestBody:
    def test_body_carries_model_and_stream_but_not_the_invoke_version(self):
        config = BedrockMantleAnthropicMessagesConfig()
        body = config.transform_anthropic_messages_request(
            model="anthropic.claude-sonnet-5",
            messages=[{"role": "user", "content": "ping"}],
            anthropic_messages_optional_request_params={"max_tokens": 8, "stream": True},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["model"] == "anthropic.claude-sonnet-5"
        assert body["stream"] is True
        assert body["max_tokens"] == 8
        assert "anthropic_version" not in body

    def test_body_omits_stream_when_not_streaming(self):
        config = BedrockMantleAnthropicMessagesConfig()
        body = config.transform_anthropic_messages_request(
            model="anthropic.claude-sonnet-5",
            messages=[{"role": "user", "content": "ping"}],
            anthropic_messages_optional_request_params={"max_tokens": 8},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert "stream" not in body


class TestAuth:
    def test_bearer_from_api_key_skips_aws_credentials(self):
        signer = BaseAWSLLM()
        signer.get_credentials = MagicMock(side_effect=AssertionError("must not resolve AWS credentials"))
        config = BedrockMantleAnthropicMessagesConfig(aws_signer=signer)
        headers, signed = config.sign_request(
            headers={"anthropic-version": "2023-06-01"},
            optional_params={},
            request_data={"model": "anthropic.claude-sonnet-5"},
            api_base=f"https://bedrock-mantle.us-east-1.api.aws{MESSAGES_PATH}",
            api_key="arg-bearer",
        )
        assert headers["Authorization"] == "Bearer arg-bearer"
        assert headers["anthropic-version"] == "2023-06-01"
        assert signed == b'{"model": "anthropic.claude-sonnet-5"}'

    def test_bearer_from_mantle_env_key(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_MANTLE_API_KEY", "env-bearer")
        config = BedrockMantleAnthropicMessagesConfig()
        headers, _ = config.sign_request(
            headers={},
            optional_params={},
            request_data={},
            api_base=f"https://bedrock-mantle.us-east-1.api.aws{MESSAGES_PATH}",
            api_key=None,
        )
        assert headers["Authorization"] == "Bearer env-bearer"

    def test_sigv4_scope_is_pinned_to_the_url_host_region(self):
        config = BedrockMantleAnthropicMessagesConfig()
        headers, signed = config.sign_request(
            headers={"anthropic-version": "2023-06-01"},
            optional_params={
                "aws_access_key_id": "AKIAEXAMPLE",
                "aws_secret_access_key": "c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
                "aws_region_name": "us-east-1",
            },
            request_data={"model": "anthropic.claude-sonnet-5"},
            api_base=f"https://bedrock-mantle.us-west-2.api.aws{MESSAGES_PATH}",
            api_key=None,
        )
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "/us-west-2/bedrock/aws4_request" in headers["Authorization"]
        assert signed == b'{"model": "anthropic.claude-sonnet-5"}'


class TestWireRequest:
    @pytest.mark.asyncio
    @respx.mock
    async def test_claude_request_hits_the_native_messages_endpoint(self):
        route = _mantle_messages_route("us-east-1").mock(return_value=_anthropic_response())

        response = await litellm.anthropic_messages(
            model="bedrock_mantle/anthropic.claude-sonnet-5",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
            api_key="test-bearer",
            aws_region_name="us-east-1",
        )

        assert response["content"][0]["text"] == "pong"
        assert route.call_count == 1
        sent = route.calls.last.request
        assert sent.headers["authorization"] == "Bearer test-bearer"
        assert sent.headers["anthropic-version"] == "2023-06-01"
        assert "x-api-key" not in sent.headers
        body = _sent_body(route)
        assert body["model"] == "anthropic.claude-sonnet-5"
        assert body["messages"] == [{"role": "user", "content": "ping"}]
        assert "anthropic_version" not in body
        assert "stream" not in body

    @pytest.mark.asyncio
    @respx.mock
    async def test_region_prefix_selects_the_host_and_is_not_sent_as_model(self):
        route = _mantle_messages_route("us-east-2").mock(return_value=_anthropic_response())

        await litellm.anthropic_messages(
            model="bedrock_mantle/us-east-2/anthropic.claude-haiku-4-5",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
            api_key="test-bearer",
        )

        assert route.call_count == 1
        assert _sent_body(route)["model"] == "anthropic.claude-haiku-4-5"

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_sends_stream_and_passes_the_sse_through(self):
        route = _mantle_messages_route("us-east-1").mock(return_value=_sse_response())

        response = await litellm.anthropic_messages(
            model="bedrock_mantle/anthropic.claude-sonnet-5",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
            stream=True,
            api_key="test-bearer",
            aws_region_name="us-east-1",
        )
        raw = b"".join([chunk async for chunk in response])

        assert route.call_count == 1
        assert _sent_body(route)["stream"] is True
        text = raw.decode()
        assert "event: message_start" in text
        assert '"text": "pong"' in text
        assert "event: message_stop" in text

    @pytest.mark.asyncio
    @respx.mock
    async def test_sigv4_request_signs_against_the_messages_url(self):
        route = _mantle_messages_route("us-east-1").mock(return_value=_anthropic_response())

        await litellm.anthropic_messages(
            model="bedrock_mantle/anthropic.claude-sonnet-5",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
            aws_access_key_id="AKIAEXAMPLE",
            aws_secret_access_key="c2VjcmV0LXRlc3Qtc2VjcmV0LXRlc3Qtc2VjcmV0",
            aws_region_name="us-east-1",
        )

        assert route.call_count == 1
        authorization = route.calls.last.request.headers["authorization"]
        assert authorization.startswith("AWS4-HMAC-SHA256")
        assert "/us-east-1/bedrock/aws4_request" in authorization


def _sent_betas(route: respx.Route) -> list[str]:
    return route.calls.last.request.headers["anthropic-beta"].split(",")


@pytest.mark.usefixtures("local_beta_headers_config")
class TestBetaHeadersOnTheWire:
    async def _send(self, **request_params) -> respx.Route:
        route = _mantle_messages_route("us-east-1").mock(return_value=_anthropic_response())
        await litellm.anthropic_messages(
            model="bedrock_mantle/anthropic.claude-sonnet-5",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
            api_key="test-bearer",
            aws_region_name="us-east-1",
            **request_params,
        )
        return route

    @pytest.mark.asyncio
    @respx.mock
    async def test_betas_mantle_accepts_reach_it_in_the_header(self):
        route = await self._send(
            extra_headers={
                "anthropic-beta": "claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27"
            }
        )

        assert _sent_betas(route) == [
            "claude-code-20250219",
            "context-management-2025-06-27",
            "interleaved-thinking-2025-05-14",
        ]

    @pytest.mark.asyncio
    @respx.mock
    async def test_betas_a_proxy_client_sends_reach_mantle_filtered(self):
        from litellm.proxy.litellm_pre_call_utils import add_provider_specific_headers_to_request

        proxy_request_data: dict = {}
        add_provider_specific_headers_to_request(
            data=proxy_request_data,
            headers={
                "anthropic-beta": "claude-code-20250219,fast-mode-2026-02-01,interleaved-thinking-2025-05-14",
                "anthropic-version": "2023-06-01",
                "user-agent": "claude-cli/2.1.239",
            },
        )

        route = await self._send(**proxy_request_data)

        assert _sent_betas(route) == ["claude-code-20250219", "interleaved-thinking-2025-05-14"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_betas_mantle_rejects_are_dropped_before_the_request(self):
        route = await self._send(
            extra_headers={"anthropic-beta": "code-execution-2025-08-25,context-1m-2025-08-07,files-api-2025-04-14"}
        )

        assert _sent_betas(route) == ["context-1m-2025-08-07"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_no_beta_header_is_sent_when_every_value_is_rejected(self):
        route = await self._send(extra_headers={"anthropic-beta": "code-execution-2025-08-25"})

        assert "anthropic-beta" not in route.calls.last.request.headers

    @pytest.mark.asyncio
    @respx.mock
    async def test_advanced_tool_use_is_renamed_to_the_beta_mantle_knows(self):
        route = await self._send(extra_headers={"anthropic-beta": "advanced-tool-use-2025-11-20"})

        assert "tool-search-tool-2025-10-19" in _sent_betas(route)
        assert "advanced-tool-use-2025-11-20" not in _sent_betas(route)

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_feature_beta_joins_the_callers_betas_in_the_header(self):
        route = await self._send(
            extra_headers={"anthropic-beta": "context-1m-2025-08-07"},
            context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
        )

        assert _sent_betas(route) == ["context-1m-2025-08-07", "context-management-2025-06-27"]
        assert _sent_body(route)["context_management"] == {"edits": [{"type": "clear_tool_uses_20250919"}]}

    @pytest.mark.asyncio
    @respx.mock
    async def test_safeguards_reach_mantle_with_the_dangerous_tool_use_beta(self):
        """Mantle answers 400 "safeguards: Extra inputs are not permitted" when the field
        arrives without dangerous-tool-use-2026-09-03 (probed 2026-09-21), so the beta
        has to ride along even when the client never sent the header."""
        safeguards = [{"type": "dangerous_tool_use", "classifier_context": {"v": 1, "permission_mode": "auto"}}]

        route = await self._send(safeguards=safeguards)

        assert _sent_betas(route) == ["dangerous-tool-use-2026-09-03"]
        assert _sent_body(route)["safeguards"] == safeguards

    @pytest.mark.asyncio
    @respx.mock
    async def test_betas_and_version_never_travel_in_the_body(self):
        route = await self._send(
            extra_headers={"anthropic-beta": "context-1m-2025-08-07"},
            context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
            anthropic_version="bedrock-2023-05-31",
        )

        body = _sent_body(route)
        assert "anthropic_beta" not in body
        assert "anthropic_version" not in body
        assert route.calls.last.request.headers["anthropic-version"] == "2023-06-01"

    @pytest.mark.asyncio
    @respx.mock
    async def test_clear_thinking_edit_is_forwarded_with_thinking_on(self):
        edits = [{"type": "clear_thinking_20251015", "keep": "all"}, {"type": "clear_tool_uses_20250919"}]
        route = await self._send(
            context_management={"edits": edits},
            thinking={"type": "adaptive"},
        )

        body = _sent_body(route)
        assert body["context_management"] == {"edits": edits}
        assert body["thinking"] == {"type": "adaptive"}
        assert "context-management-2025-06-27" in _sent_betas(route)

    @pytest.mark.asyncio
    @respx.mock
    async def test_tools_reach_mantle_unchanged(self):
        tools = [
            {
                "name": "get_weather",
                "description": "Look up the weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            }
        ]
        route = await self._send(tools=tools, tool_choice={"type": "auto"})

        body = _sent_body(route)
        assert body["tools"] == tools
        assert body["tool_choice"] == {"type": "auto"}
