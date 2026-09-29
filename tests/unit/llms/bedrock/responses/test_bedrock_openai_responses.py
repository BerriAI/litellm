"""Native OpenAI Responses API on the bedrock-runtime endpoint.

Without this config the bedrock provider has no Responses config, so /v1/responses
falls back to the Chat Completions bridge and rides Converse.
"""

import json
import logging
from importlib.resources import files
from unittest.mock import patch

import pytest

import litellm
from litellm.llms.bedrock.common_utils import bedrock_supports_openai_responses
from litellm.llms.bedrock.responses.transformation import BedrockOpenAIResponsesConfig
from litellm.responses.file_search.emulated_handler import should_use_emulated_file_search
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

MODEL = "global.openai.gpt-5.6-sol"


def _cfg():
    return BedrockOpenAIResponsesConfig()


class TestCompleteURL:
    def test_default_host_and_path(self):
        url = _cfg().get_complete_url(None, {"aws_region_name": "us-east-1"})
        assert url == "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses"

    def test_region_is_honoured(self):
        url = _cfg().get_complete_url(None, {"aws_region_name": "eu-west-1"})
        assert url == "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1/responses"

    @pytest.mark.parametrize(
        "api_base",
        [
            "https://proxy.example.com",
            "https://proxy.example.com/",
            "https://proxy.example.com/openai/v1",
            "https://proxy.example.com/openai/v1/responses",
            "https://proxy.example.com/v1",
            "https://proxy.example.com/v1/responses",
            "https://proxy.example.com/responses",
        ],
    )
    def test_custom_host_is_preserved_and_path_never_doubles(self, api_base):
        url = _cfg().get_complete_url(api_base, {"aws_region_name": "us-east-1"})
        assert url == "https://proxy.example.com/openai/v1/responses"

    def test_runtime_endpoint_param_is_honoured(self):
        url = _cfg().get_complete_url(
            None, {"aws_region_name": "us-east-1", "aws_bedrock_runtime_endpoint": "https://vpce.example.com"}
        )
        assert url == "https://vpce.example.com/openai/v1/responses"


class TestAuth:
    def test_bearer_token_is_used_when_present(self):
        headers = _cfg().validate_environment({}, MODEL, GenericLiteLLMParams(api_key="sk-bedrock"))
        assert headers["Authorization"] == "Bearer sk-bedrock"

    def test_no_authorization_header_without_a_token(self, monkeypatch):
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        headers = _cfg().validate_environment({}, MODEL, GenericLiteLLMParams())
        assert "Authorization" not in headers

    def test_sigv4_is_skipped_when_a_bearer_token_is_present(self):
        """Bedrock API keys are Bearer; signing on top would be wrong."""
        headers, body = _cfg().sign_request(
            headers={"Authorization": "Bearer sk-bedrock"},
            optional_params={},
            request_data={},
            api_base="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses",
            api_key="sk-bedrock",
        )
        assert headers["Authorization"] == "Bearer sk-bedrock"
        assert body is None


class TestProviderIdentity:
    def test_reports_the_bedrock_provider(self):
        """Cost tracking and callbacks key off this, so it must stay `bedrock` rather
        than becoming a separate provider."""
        assert _cfg().custom_llm_provider == LlmProviders.BEDROCK


class TestSigV4Fallback:
    def test_signs_with_sigv4_when_no_bearer_token_is_present(self, monkeypatch):
        """No Bedrock API key means SigV4 over the standard credential chain. Static
        credentials are set in the environment so signing stays a local computation."""
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
        headers, body = _cfg().sign_request(
            headers={"content-type": "application/json"},
            optional_params={"aws_region_name": "us-east-1"},
            request_data={"model": MODEL, "input": "hi"},
            api_base="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses",
            api_key=None,
        )
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "Credential=AKIAIOSFODNN7EXAMPLE" in headers["Authorization"]


class TestErrorClass:
    """Bedrock's request id must survive; the OpenAI base builds a blank response."""

    def test_amzn_request_id_is_preserved(self):
        error = _cfg().get_error_class(
            error_message="boom",
            status_code=500,
            headers={"x-amzn-RequestId": "req-500"},
        )
        assert error.status_code == 500
        assert error.response.headers["x-amzn-requestid"] == "req-500"


class TestPriceMapGate:
    def test_absent_model_has_no_signal(self):
        assert bedrock_supports_openai_responses(MODEL, {}) is False

    def test_none_model_is_false(self):
        assert bedrock_supports_openai_responses(None, {}) is False

    def test_signal_on_the_bare_key(self):
        cost = {MODEL: {"supported_endpoints": ["/v1/responses"]}}
        assert bedrock_supports_openai_responses(MODEL, cost) is True

    def test_signal_on_the_bedrock_prefixed_key(self):
        cost = {f"bedrock/{MODEL}": {"supported_endpoints": ["/v1/responses"]}}
        assert bedrock_supports_openai_responses(MODEL, cost) is True

    def test_other_endpoints_do_not_count(self):
        cost = {MODEL: {"supported_endpoints": ["/v1/messages"]}}
        assert bedrock_supports_openai_responses(MODEL, cost) is False


class TestForModelGate:
    """The capability decision lives on the adapter, not in the shared dispatch."""

    def test_returns_a_config_for_a_signalled_model(self):
        with patch.object(  # test-quality-ok: the gate reads the global cost map by design; no injection point exists
            litellm, "model_cost", {MODEL: {"supported_endpoints": ["/v1/responses"]}}
        ):
            assert isinstance(BedrockOpenAIResponsesConfig.for_model(MODEL), BedrockOpenAIResponsesConfig)

    def test_returns_none_for_an_unsignalled_model(self):
        with patch.object(  # test-quality-ok: the gate reads the global cost map by design; no injection point exists
            litellm, "model_cost", {}
        ):
            assert BedrockOpenAIResponsesConfig.for_model(MODEL) is None

    def test_returns_none_for_no_model(self):
        with patch.object(  # test-quality-ok: the gate reads the global cost map by design; no injection point exists
            litellm, "model_cost", {}
        ):
            assert BedrockOpenAIResponsesConfig.for_model(None) is None


class TestProviderResolution:
    """model_cost is patched explicitly: it is populated at import time from a GitHub
    fetch unless LITELLM_LOCAL_MODEL_COST_MAP is set, and conftest's monkeypatch of
    that variable lands after import — so these must not read the global."""

    def test_signalled_model_resolves_to_the_bedrock_responses_config(self):
        with patch.object(  # test-quality-ok: resolution reads the global cost map by design; no HTTP boundary or injection point exists
            litellm, "model_cost", {MODEL: {"supported_endpoints": ["/v1/responses"]}}
        ):
            cfg = ProviderConfigManager.get_provider_responses_api_config(model=MODEL, provider=LlmProviders.BEDROCK)
        assert isinstance(cfg, BedrockOpenAIResponsesConfig)

    def test_unsignalled_model_keeps_the_existing_bridge(self):
        """Claude on Bedrock has no OpenAI surface; it must keep falling through to
        the chat-completions bridge exactly as before."""
        with patch.object(litellm, "model_cost", {}):  # test-quality-ok: resolution reads the global cost map by design
            cfg = ProviderConfigManager.get_provider_responses_api_config(
                model="anthropic.claude-3-haiku-20240307-v1:0", provider=LlmProviders.BEDROCK
            )
        assert cfg is None

    @pytest.mark.parametrize(
        ("family", "variants"),
        [("gpt-5.6", ("sol", "terra", "luna")), ("gpt-6", ("astra", "sol", "luna"))],
    )
    def test_the_shipped_price_map_signals_the_openai_families(self, family: str, variants: tuple[str, ...]):
        """Reads the bundled backup directly rather than the network-fetched global."""
        shipped = json.loads(
            files("litellm").joinpath("model_prices_and_context_window_backup.json").read_text(encoding="utf-8")
        )
        for prefix in ("us", "global"):
            for variant in variants:
                model = f"{prefix}.openai.{family}-{variant}"
                assert bedrock_supports_openai_responses(model, shipped) is True, model


class TestUnsupportedToolDrop:
    """Codex sends a web_search tool on every turn; bedrock-runtime 400s the whole request over it."""

    _WEB_SEARCH_TOOL = {"type": "web_search", "external_web_access": False}
    _SHELL_TOOL = {"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {}}}
    _NAMESPACE_TOOL = {
        "type": "namespace",
        "name": "multi_agent_v1",
        "tools": [{"type": "function", "name": "spawn_agent"}],
    }

    def _outbound_tools(self, tools: list[dict]) -> object:
        params = _cfg().map_openai_params(response_api_optional_params={"tools": tools}, model=MODEL, drop_params=False)
        body = _cfg().transform_responses_api_request(
            model=MODEL,
            input="count the lines",
            response_api_optional_request_params=params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        return body.get("tools")

    def test_codex_default_tools_reach_the_endpoint_without_web_search(self, caplog):
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            outbound = self._outbound_tools([self._SHELL_TOOL, self._WEB_SEARCH_TOOL, self._NAMESPACE_TOOL])
        assert outbound == [self._SHELL_TOOL, self._NAMESPACE_TOOL]
        dropped = [r.getMessage() for r in caplog.records if "dropping unsupported tool type" in r.getMessage()]
        assert len(dropped) == 1 and "web_search" in dropped[0]

    def test_only_unsupported_tools_means_no_tools_key(self):
        assert self._outbound_tools([self._WEB_SEARCH_TOOL, {"type": "web_search_preview"}]) is None

    def test_supported_tools_are_not_logged_as_dropped(self, caplog):
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            outbound = self._outbound_tools([self._SHELL_TOOL, {"type": "custom", "name": "exec"}])
        assert outbound == [self._SHELL_TOOL, {"type": "custom", "name": "exec"}]
        assert not [r for r in caplog.records if "dropping unsupported tool type" in r.getMessage()]


class TestFileSearchEmulation:
    """bedrock-runtime runs no server-side tools, so a file_search tool must take the emulated path."""

    def test_file_search_tool_is_routed_to_emulation(self):
        tools = [{"type": "file_search", "vector_store_ids": ["vs_1"]}]
        assert should_use_emulated_file_search(tools, _cfg()) is True

    def test_plain_function_tools_skip_emulation(self):
        tools = [{"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {}}}]
        assert should_use_emulated_file_search(tools, _cfg()) is False


class TestCodexHistoryNormalization:
    def test_history_items_the_endpoint_rejects_are_rewritten(self):
        body = _cfg().transform_responses_api_request(
            model=MODEL,
            input=[
                {"type": "agent_message", "content": [{"type": "output_text", "text": "prior"}]},
                {"type": "context_compaction", "encrypted_content": "abc"},
                {"type": "local_shell_call", "call_id": "c1", "action": {"command": ["ls"]}},
                {"role": "user", "content": "carry on"},
            ],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert [i.get("type") or i.get("role") for i in body["input"]] == [
            "message",
            "compaction",
            "function_call",
            "user",
        ]

    def test_a_first_turn_request_is_untouched(self):
        """The rejected types are history items, so turn one exercises none of this."""
        original = [{"role": "user", "content": "first turn"}]
        body = _cfg().transform_responses_api_request(
            model=MODEL,
            input=list(original),
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["input"] == original


class TestBackgroundDrop:
    """The Converse bridge answered `background` requests synchronously; bedrock-runtime 400s the parameter."""

    def test_background_is_dropped_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            params = _cfg().map_openai_params(
                response_api_optional_params={"background": True, "max_output_tokens": 64},
                model=MODEL,
                drop_params=False,
            )
        assert params == {"max_output_tokens": 64}
        dropped = [r.getMessage() for r in caplog.records if "dropping unsupported parameter" in r.getMessage()]
        assert len(dropped) == 1 and "background" in dropped[0]

    def test_without_background_nothing_is_dropped_or_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            params = _cfg().map_openai_params(
                response_api_optional_params={"max_output_tokens": 64}, model=MODEL, drop_params=False
            )
        assert params == {"max_output_tokens": 64}
        assert not [r for r in caplog.records if "dropping unsupported parameter" in r.getMessage()]


def _never_fetch(url: str) -> str:
    raise AssertionError(f"unexpected sync fetch of {url}")


async def _never_fetch_async(url: str) -> str:
    raise AssertionError(f"unexpected async fetch of {url}")


class TestRemoteImageInlining:
    """The Converse bridge downloaded http(s) image URLs; bedrock-runtime accepts only data: and s3://."""

    _REMOTE = "https://example.com/grapes.png"
    _DATA_URI = "data:image/png;base64,QUJD"
    _INLINED = "data:image/png;base64,ZmV0Y2hlZA=="

    def _input(self, remote: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is this?"},
                    {"type": "input_image", "image_url": remote, "detail": "auto"},
                    {"type": "input_image", "image_url": remote},
                    {"type": "input_image", "image_url": self._DATA_URI},
                    {"type": "input_image", "image_url": "s3://bucket/grapes.png"},
                    {"type": "input_image", "file_id": "file-1"},
                ],
            },
            {"role": "assistant", "content": "plain string content"},
        ]

    def test_sync_transform_fetches_each_remote_url_once_and_inlines_it(self):
        fetched: list[str] = []

        def fetch(url: str) -> str:
            fetched.append(url)
            return self._INLINED

        body = BedrockOpenAIResponsesConfig(
            fetch_image=fetch, async_fetch_image=_never_fetch_async
        ).transform_responses_api_request(
            model=MODEL,
            input=self._input(self._REMOTE),
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["input"] == self._input(self._INLINED)
        assert fetched == [self._REMOTE]

    def test_tool_output_lists_are_inlined_and_string_outputs_are_untouched(self):
        fetched: list[str] = []

        def fetch(url: str) -> str:
            fetched.append(url)
            return self._INLINED

        def tool_turn(remote: str) -> list[dict]:
            return [
                {"type": "function_call", "call_id": "call_1", "name": "fetch_chart", "arguments": "{}"},
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_text", "text": "the chart"},
                        {"type": "input_image", "image_url": remote},
                    ],
                },
                {"type": "function_call_output", "call_id": "call_2", "output": "https://example.com/plain-text.png"},
                {"role": "user", "content": [{"type": "input_image", "image_url": remote}]},
            ]

        body = BedrockOpenAIResponsesConfig(
            fetch_image=fetch, async_fetch_image=_never_fetch_async
        ).transform_responses_api_request(
            model=MODEL,
            input=tool_turn(self._REMOTE),
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["input"] == tool_turn(self._INLINED)
        assert fetched == [self._REMOTE]

    def test_computer_screenshot_outputs_are_inlined(self):
        fetched: list[str] = []

        def fetch(url: str) -> str:
            fetched.append(url)
            return self._INLINED

        def computer_turn(remote: str) -> list[dict]:
            return [
                {"type": "computer_call", "call_id": "call_1", "id": "cu_1", "actions": [{"type": "screenshot"}]},
                {
                    "type": "computer_call_output",
                    "call_id": "call_1",
                    "output": {"type": "computer_screenshot", "image_url": remote},
                },
                {
                    "type": "computer_call_output",
                    "call_id": "call_2",
                    "output": {"type": "computer_screenshot", "file_id": "file-1"},
                },
                {
                    "type": "computer_call_output",
                    "call_id": "call_3",
                    "output": {"type": "computer_screenshot", "image_url": self._DATA_URI},
                },
            ]

        body = BedrockOpenAIResponsesConfig(
            fetch_image=fetch, async_fetch_image=_never_fetch_async
        ).transform_responses_api_request(
            model=MODEL,
            input=computer_turn(self._REMOTE),
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["input"] == computer_turn(self._INLINED)
        assert fetched == [self._REMOTE]

    @pytest.mark.asyncio
    async def test_async_transform_fetches_with_the_async_fetcher(self):
        fetched: list[str] = []

        async def fetch(url: str) -> str:
            fetched.append(url)
            return self._INLINED

        body = await BedrockOpenAIResponsesConfig(
            fetch_image=_never_fetch, async_fetch_image=fetch
        ).async_transform_responses_api_request(
            model=MODEL,
            input=self._input(self._REMOTE),
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert body["input"] == self._input(self._INLINED)
        assert fetched == [self._REMOTE]

    @pytest.mark.asyncio
    async def test_inputs_without_remote_images_never_fetch(self):
        cfg = BedrockOpenAIResponsesConfig(fetch_image=_never_fetch, async_fetch_image=_never_fetch_async)
        local_only = self._input(self._DATA_URI)
        sync_body = cfg.transform_responses_api_request(
            model=MODEL,
            input=local_only,
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        async_body = await cfg.async_transform_responses_api_request(
            model=MODEL,
            input="a plain string prompt",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert sync_body["input"] == local_only
        assert async_body["input"] == "a plain string prompt"

    @pytest.mark.asyncio
    async def test_inlining_runs_before_codex_history_normalization(self):
        async def fetch(url: str) -> str:
            return self._INLINED

        body = await BedrockOpenAIResponsesConfig(
            fetch_image=_never_fetch, async_fetch_image=fetch
        ).async_transform_responses_api_request(
            model=MODEL,
            input=[
                {"type": "agent_message", "content": [{"type": "output_text", "text": "prior"}]},
                {"role": "user", "content": [{"type": "input_image", "image_url": self._REMOTE}]},
            ],
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert [i.get("type") or i.get("role") for i in body["input"]] == ["message", "user"]
        assert body["input"][1]["content"] == [{"type": "input_image", "image_url": self._INLINED}]
