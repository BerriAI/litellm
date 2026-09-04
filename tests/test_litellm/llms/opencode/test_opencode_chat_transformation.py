"""
Unit tests for the OpenCode Zen and OpenCode Go providers.

OpenCode rejects inference requests that arrive without an `x-opencode-session` header, so the
tests below pin both the value LiteLLM picks for it and the fact that it reaches the wire.
API docs: https://opencode.ai/docs/zen and https://opencode.ai/docs/go
"""

import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.opencode.chat.transformation import (
    OpenCodeGoChatConfig,
    OpenCodeGoMessagesChatConfig,
    OpenCodeZenChatConfig,
    OpenCodeZenGeminiChatConfig,
    OpenCodeZenMessagesChatConfig,
)
from litellm.llms.opencode.responses.transformation import (
    OpenCodeGoResponsesAPIConfig,
    OpenCodeZenResponsesAPIConfig,
)
from litellm.llms.opencode.common_utils import (
    OPENCODE_SESSION_HEADER,
    opencode_endpoint_for_model,
    resolve_opencode_api_key,
    resolve_opencode_session_id,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

ZEN_API_BASE: Final = "https://opencode.ai/zen/v1"
GO_API_BASE: Final = "https://opencode.ai/zen/go/v1"


class RecordingHTTPHandler(HTTPHandler):
    """Injected in place of the real client so the outbound request can be asserted on."""

    def __init__(self, response_model: str):
        super().__init__()
        self.response_model = response_model
        self.requests: tuple[Mapping[str, object], ...] = ()

    def post(self, url, data=None, headers=None, **kwargs):
        raw_body: Final = data.decode("utf-8") if isinstance(data, bytes) else data
        self.requests = (
            *self.requests,
            {"url": url, "headers": dict(headers or {}), "body": json.loads(raw_body or "{}")},
        )
        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1733529600,
                "model": self.response_model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            request=httpx.Request("POST", url),
        )


def _headers_for(config, litellm_params, headers=None):
    return config.validate_environment(
        headers=dict(headers or {}),
        model="kimi-k3",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params=litellm_params,
        api_key="fake-key",
    )


class TestSessionIdResolution:
    @pytest.mark.parametrize(
        "litellm_params, expected",
        [
            pytest.param(
                {
                    "litellm_session_id": "session-1",
                    "metadata": {"session_id": "meta-1"},
                    "litellm_trace_id": "trace-1",
                    "litellm_call_id": "call-1",
                },
                "session-1",
                id="litellm_session_id-wins",
            ),
            pytest.param(
                {"metadata": {"session_id": "meta-1"}, "litellm_trace_id": "trace-1", "litellm_call_id": "call-1"},
                "meta-1",
                id="metadata-session-id-next",
            ),
            pytest.param(
                {"litellm_trace_id": "trace-1", "litellm_call_id": "call-1"},
                "trace-1",
                id="trace-id-next",
            ),
            pytest.param({"litellm_call_id": "call-1"}, "call-1", id="call-id-last"),
        ],
    )
    def test_precedence_order(self, litellm_params, expected):
        assert resolve_opencode_session_id(litellm_params) == expected

    @pytest.mark.parametrize(
        "litellm_params",
        [
            pytest.param({}, id="nothing-available"),
            pytest.param({"litellm_session_id": None, "litellm_call_id": None}, id="explicit-nones"),
            pytest.param({"litellm_session_id": ""}, id="empty-string-is-not-an-id"),
            pytest.param({"litellm_session_id": 42}, id="non-string-is-not-an-id"),
            pytest.param({"metadata": "not-a-mapping"}, id="metadata-not-a-mapping"),
        ],
    )
    def test_returns_none_when_no_usable_id(self, litellm_params):
        assert resolve_opencode_session_id(litellm_params) is None

    def test_empty_session_id_falls_through_to_next_candidate(self):
        assert resolve_opencode_session_id({"litellm_session_id": "", "litellm_call_id": "call-1"}) == "call-1"


class TestSessionHeader:
    @pytest.mark.parametrize("config", [OpenCodeZenChatConfig(), OpenCodeGoChatConfig()])
    def test_header_is_set_from_session_id(self, config):
        headers = _headers_for(config, {"litellm_session_id": "session-1"})
        assert headers[OPENCODE_SESSION_HEADER] == "session-1"

    @pytest.mark.parametrize("config", [OpenCodeZenChatConfig(), OpenCodeGoChatConfig()])
    def test_header_absent_when_no_id_is_available(self, config):
        assert OPENCODE_SESSION_HEADER not in _headers_for(config, {})

    @pytest.mark.parametrize("supplied_name", ["x-opencode-session", "X-OpenCode-Session"])
    def test_caller_supplied_header_is_never_overwritten(self, supplied_name):
        headers = _headers_for(
            OpenCodeGoChatConfig(),
            {"litellm_session_id": "session-1"},
            headers={supplied_name: "caller-owned"},
        )
        assert headers[supplied_name] == "caller-owned"
        assert "session-1" not in headers.values()

    def test_auth_and_content_type_are_preserved(self):
        headers = _headers_for(OpenCodeGoChatConfig(), {"litellm_session_id": "session-1"})
        assert headers["Authorization"] == "Bearer fake-key"
        assert headers["Content-Type"] == "application/json"


class TestProviderRegistration:
    def test_provider_enums_exist(self):
        assert LlmProviders.OPENCODE == "opencode"
        assert LlmProviders.OPENCODE_GO == "opencode_go"

    @pytest.mark.parametrize("provider", ["opencode", "opencode_go"])
    def test_provider_in_provider_list(self, provider):
        assert provider in litellm.provider_list

    @pytest.mark.parametrize(
        "model, expected_provider, expected_api_base",
        [
            ("opencode/claude-opus-5", "opencode", ZEN_API_BASE),
            ("opencode_go/kimi-k3", "opencode_go", GO_API_BASE),
        ],
    )
    def test_model_prefix_resolves_provider_and_default_api_base(self, model, expected_provider, expected_api_base):
        _, provider, _, api_base = litellm.get_llm_provider(model=model)
        assert provider == expected_provider
        assert api_base == expected_api_base

    @pytest.mark.parametrize(
        "api_base, expected_provider",
        [
            (ZEN_API_BASE, "opencode"),
            (GO_API_BASE, "opencode_go"),
        ],
    )
    def test_api_base_infers_provider(self, api_base, expected_provider):
        _, provider, _, _ = litellm.get_llm_provider(model="kimi-k3", api_base=api_base)
        assert provider == expected_provider

    @pytest.mark.parametrize(
        "config",
        [
            OpenCodeZenChatConfig(),
            OpenCodeGoChatConfig(),
            OpenCodeZenMessagesChatConfig(),
            OpenCodeGoMessagesChatConfig(),
            OpenCodeZenGeminiChatConfig(),
        ],
    )
    def test_one_account_key_authenticates_every_surface(self, monkeypatch, config):
        """One OpenCode account key covers Zen and Go, so every config reads the same variable."""
        monkeypatch.setenv("OPENCODE_API_KEY", "account-key")
        assert resolve_opencode_api_key(None) == "account-key"
        _, resolved = OpenCodeZenChatConfig()._get_openai_compatible_provider_info(None, None)
        assert resolved == "account-key"

    def test_api_key_is_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        assert resolve_opencode_api_key(None) is None

    def test_explicit_api_key_wins_over_the_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_API_KEY", "account-key")
        assert resolve_opencode_api_key("explicit-key") == "explicit-key"

    @pytest.mark.parametrize(
        "config, expected_api_base",
        [
            (OpenCodeZenChatConfig(), ZEN_API_BASE),
            (OpenCodeGoChatConfig(), GO_API_BASE),
        ],
    )
    def test_default_api_base_needs_no_configuration(self, monkeypatch, config, expected_api_base):
        monkeypatch.setenv("OPENCODE_API_KEY", "account-key")
        api_base, _ = config._get_openai_compatible_provider_info(None, None)
        assert api_base == expected_api_base

    @pytest.mark.parametrize("config", [OpenCodeZenChatConfig(), OpenCodeGoChatConfig()])
    def test_explicit_api_base_wins_over_default(self, config):
        api_base, _ = config._get_openai_compatible_provider_info("https://proxy.internal/v1", "k")
        assert api_base == "https://proxy.internal/v1"


class TestCompletionWiring:
    @pytest.mark.parametrize(
        "model, expected_url",
        [
            ("opencode/kimi-k3", f"{ZEN_API_BASE}/chat/completions"),
            ("opencode_go/kimi-k3", f"{GO_API_BASE}/chat/completions"),
            ("opencode_go/glm-5.3-flash", f"{GO_API_BASE}/chat/completions"),
        ],
    )
    def test_session_header_reaches_the_wire(self, local_model_cost_map, model, expected_url):
        client = RecordingHTTPHandler(response_model=model.split("/", 1)[1])
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            api_key="fake-key",
            litellm_session_id="session-42",
            client=client,
        )

        assert response.choices[0].message.content == "ok"
        assert len(client.requests) == 1
        assert client.requests[0]["url"] == expected_url
        assert client.requests[0]["headers"][OPENCODE_SESSION_HEADER] == "session-42"
        assert OPENCODE_SESSION_HEADER not in client.requests[0]["body"]

    def test_session_header_is_stable_across_turns_of_one_session(self, local_model_cost_map):
        client = RecordingHTTPHandler(response_model="kimi-k3")
        for turn in ("first", "second"):
            litellm.completion(
                model="opencode_go/kimi-k3",
                messages=[{"role": "user", "content": turn}],
                api_key="fake-key",
                litellm_session_id="session-42",
                client=client,
            )

        sent = [r["headers"][OPENCODE_SESSION_HEADER] for r in client.requests]
        assert sent == ["session-42", "session-42"]

    def test_header_present_even_without_a_caller_supplied_session_id(self, local_model_cost_map):
        client = RecordingHTTPHandler(response_model="kimi-k3")
        litellm.completion(
            model="opencode_go/kimi-k3",
            messages=[{"role": "user", "content": "hello"}],
            api_key="fake-key",
            client=client,
        )

        assert client.requests[0]["headers"].get(OPENCODE_SESSION_HEADER)


class TestPricing:
    @pytest.mark.parametrize(
        "model, provider",
        [
            ("opencode/claude-opus-5", "opencode"),
            ("opencode_go/kimi-k3", "opencode_go"),
        ],
    )
    def test_models_are_registered_under_their_provider(self, local_model_cost_map, model, provider):
        assert litellm.get_model_info(model)["litellm_provider"] == provider

    def test_go_pricing_matches_published_rates(self, local_model_cost_map):
        info = litellm.get_model_info("opencode_go/kimi-k3")
        assert info["input_cost_per_token"] == pytest.approx(3.0 / 1_000_000)
        assert info["output_cost_per_token"] == pytest.approx(15.0 / 1_000_000)
        assert info["cache_read_input_token_cost"] == pytest.approx(0.3 / 1_000_000)
        assert info["max_input_tokens"] == 1_048_576

    def test_zen_pricing_matches_published_rates(self, local_model_cost_map):
        info = litellm.get_model_info("opencode/gpt-5.6-luna")
        assert info["input_cost_per_token"] == pytest.approx(0.2 / 1_000_000)
        assert info["output_cost_per_token"] == pytest.approx(1.2 / 1_000_000)
        assert info["input_cost_per_token_above_272k_tokens"] == pytest.approx(0.4 / 1_000_000)

    def test_tiered_pricing_applies_above_the_context_threshold(self, local_model_cost_map):
        from litellm.types.utils import Choices, Message, ModelResponse, Usage

        def cost_for(prompt_tokens):
            response = ModelResponse(
                id="x",
                model="grok-4.6",
                object="chat.completion",
                created=0,
                choices=[Choices(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
                usage=Usage(
                    prompt_tokens=prompt_tokens, completion_tokens=1_000, total_tokens=prompt_tokens + 1_000
                ),
            )
            return litellm.completion_cost(completion_response=response, model="opencode_go/grok-4.6")

        assert cost_for(100_000) == pytest.approx(100_000 * 2e-06 + 1_000 * 6e-06)
        assert cost_for(300_000) == pytest.approx(300_000 * 4e-06 + 1_000 * 1.2e-05)


class TestEndpointRouting:
    """
    OpenCode splits its catalogue across four wire formats, per model rather than per family,
    and serving a model on the wrong one returns 500 or `not supported for format oa-compat`.
    Every expectation below was verified against the live API with a real key, except the
    Zen-only gemini and claude rows, which come from OpenCode's published endpoint table.
    """

    @pytest.mark.parametrize(
        "model, expected_endpoint",
        [
            pytest.param("opencode_go/glm-5.3-flash", "/v1/chat/completions", id="go-glm-chat"),
            pytest.param("opencode_go/kimi-k3", "/v1/chat/completions", id="go-kimi-chat"),
            pytest.param("opencode_go/deepseek-v4-pro", "/v1/chat/completions", id="go-deepseek-chat"),
            pytest.param("opencode_go/gpt-5.6-luna", "/v1/responses", id="go-gpt-responses"),
            pytest.param("opencode_go/grok-4.6", "/v1/responses", id="go-grok-responses"),
            pytest.param("opencode_go/grok-4.5", "/v1/responses", id="go-grok45-responses-undocumented"),
            pytest.param("opencode_go/muse-spark-1.3-contributor", "/v1/responses", id="go-muse-responses"),
            pytest.param("opencode_go/minimax-m3", "/v1/messages", id="go-minimax-messages"),
            pytest.param("opencode_go/minimax-m2.7", "/v1/messages", id="go-minimax27-messages"),
            pytest.param("opencode_go/qwen3.7-max", "/v1/messages", id="go-qwen-messages"),
            pytest.param("opencode_go/qwen3.5-plus", "/v1/messages", id="go-qwen35-messages-undocumented"),
            pytest.param("opencode/claude-opus-5", "/v1/messages", id="zen-claude-messages"),
            pytest.param("opencode/gemini-3.1-pro", "/v1/models:generateContent", id="zen-gemini-models"),
            pytest.param("opencode/gpt-5.5", "/v1/responses", id="zen-gpt-responses"),
            pytest.param("opencode/kimi-k3", "/v1/chat/completions", id="zen-kimi-chat"),
        ],
    )
    def test_cost_map_records_the_endpoint(self, local_model_cost_map, model, expected_endpoint):
        assert litellm.model_cost[model]["supported_endpoints"] == [expected_endpoint]

    @pytest.mark.parametrize(
        "model, expected_endpoint",
        [
            ("opencode_go/minimax-m3", "/v1/messages"),
            ("opencode_go/glm-5.3-flash", "/v1/chat/completions"),
        ],
    )
    def test_endpoint_lookup_reads_the_cost_map(self, local_model_cost_map, model, expected_endpoint):
        provider, bare_model = model.split("/", 1)
        assert opencode_endpoint_for_model(provider, bare_model) == expected_endpoint

    def test_unknown_model_falls_back_to_chat_completions(self, local_model_cost_map):
        assert opencode_endpoint_for_model("opencode_go", "not-a-real-model") == "/v1/chat/completions"

    @pytest.mark.parametrize(
        "provider, model, expected_config",
        [
            (LlmProviders.OPENCODE_GO, "glm-5.3-flash", OpenCodeGoChatConfig),
            (LlmProviders.OPENCODE_GO, "minimax-m3", OpenCodeGoMessagesChatConfig),
            (LlmProviders.OPENCODE_GO, "qwen3.7-max", OpenCodeGoMessagesChatConfig),
            (LlmProviders.OPENCODE, "claude-opus-5", OpenCodeZenMessagesChatConfig),
            (LlmProviders.OPENCODE, "gemini-3.1-pro", OpenCodeZenGeminiChatConfig),
            (LlmProviders.OPENCODE, "kimi-k3", OpenCodeZenChatConfig),
        ],
    )
    def test_provider_picks_the_matching_wire_format(self, local_model_cost_map, provider, model, expected_config):
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(model=model, provider=provider)
        assert type(config) is expected_config

    @pytest.mark.parametrize(
        "model", ["opencode_go/gpt-5.6-luna", "opencode_go/grok-4.6", "opencode/gpt-5.5", "opencode/grok-4.6"]
    )
    def test_responses_models_are_marked_for_the_bridge(self, local_model_cost_map, model):
        assert litellm.get_model_info(model)["mode"] == "responses"

    @pytest.mark.parametrize("model", ["opencode_go/minimax-m3", "opencode/claude-opus-5", "opencode/gemini-3.1-pro"])
    def test_non_responses_models_stay_chat_mode(self, local_model_cost_map, model):
        assert litellm.get_model_info(model)["mode"] == "chat"


class TestMessagesWireFormat:
    @pytest.mark.parametrize(
        "config, expected_url",
        [
            (OpenCodeGoMessagesChatConfig(), f"{GO_API_BASE}/messages"),
            (OpenCodeZenMessagesChatConfig(), f"{ZEN_API_BASE}/messages"),
        ],
    )
    def test_url_targets_the_messages_endpoint(self, config, expected_url):
        url = config.get_complete_url(
            api_base=None, api_key="k", model="minimax-m3", optional_params={}, litellm_params={}
        )
        assert url == expected_url

    def test_auth_uses_x_api_key_not_bearer(self, monkeypatch):
        """OpenCode's /messages rejects `Authorization: Bearer` with `Missing API key`."""
        monkeypatch.setenv("OPENCODE_API_KEY", "go-key")
        headers = OpenCodeGoMessagesChatConfig().validate_environment(
            headers={},
            model="minimax-m3",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={"litellm_session_id": "session-1"},
        )
        assert headers["x-api-key"] == "go-key"
        assert "Authorization" not in headers
        assert headers[OPENCODE_SESSION_HEADER] == "session-1"


class TestGeminiWireFormat:
    def test_url_targets_generate_content(self):
        url = OpenCodeZenGeminiChatConfig().get_complete_url(
            api_base=None, api_key="k", model="gemini-3.1-pro", optional_params={}, litellm_params={}
        )
        assert url == f"{ZEN_API_BASE}/models/gemini-3.1-pro:generateContent"

    def test_streaming_url_targets_stream_generate_content(self):
        url = OpenCodeZenGeminiChatConfig().get_complete_url(
            api_base=None, api_key="k", model="gemini-3.1-pro", optional_params={}, litellm_params={}, stream=True
        )
        assert url == f"{ZEN_API_BASE}/models/gemini-3.1-pro:streamGenerateContent?alt=sse"

    def test_auth_and_session_header(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_API_KEY", "zen-key")
        headers = OpenCodeZenGeminiChatConfig().validate_environment(
            headers={},
            model="gemini-3.1-pro",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={"litellm_session_id": "session-1"},
        )
        assert headers["x-goog-api-key"] == "zen-key"
        assert "Authorization" not in headers
        assert headers[OPENCODE_SESSION_HEADER] == "session-1"


class RecordingGeminiHTTPHandler(HTTPHandler):
    """Gemini answers in Google's own shape, so the canned reply differs from the OpenAI one."""

    def __init__(self):
        super().__init__()
        self.requests: tuple[Mapping[str, object], ...] = ()

    def post(self, url, data=None, headers=None, **kwargs):
        raw_body: Final = data.decode("utf-8") if isinstance(data, bytes) else data
        self.requests = (
            *self.requests,
            {"url": url, "headers": dict(headers or {}), "body": json.loads(raw_body or "{}")},
        )
        return httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "OK"}], "role": "model"}, "finishReason": "STOP", "index": 0}
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6},
                "modelVersion": "gemini-3.5-flash-lite",
            },
            request=httpx.Request("POST", url),
        )


class TestGeminiRequestBody:
    def test_transform_request_builds_a_gemini_body(self):
        """VertexGeminiConfig.transform_request raises NotImplementedError; the override must not."""
        body = OpenCodeZenGeminiChatConfig().transform_request(
            model="gemini-3.5-flash-lite",
            messages=[{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
        assert body["system_instruction"] == {"parts": [{"text": "Be terse."}]}

    def test_completion_reaches_generate_content_with_the_session_header(self, local_model_cost_map):
        client = RecordingGeminiHTTPHandler()
        response = litellm.completion(
            model="opencode/gemini-3.5-flash-lite",
            messages=[{"role": "user", "content": "hi"}],
            api_key="fake-key",
            litellm_session_id="session-77",
            client=client,
        )

        assert response.choices[0].message.content == "OK"
        assert len(client.requests) == 1
        sent = client.requests[0]
        assert sent["url"] == f"{ZEN_API_BASE}/models/gemini-3.5-flash-lite:generateContent"
        assert sent["headers"][OPENCODE_SESSION_HEADER] == "session-77"
        assert sent["headers"]["x-goog-api-key"] == "fake-key"
        assert sent["body"]["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]


class TestTieredPricingPropagation:
    """The 256k tier fields are not named in ModelInfoBase's constructor; a generic pass copies them."""

    @pytest.mark.parametrize(
        "model, field",
        [
            ("opencode_go/qwen3.7-plus", "input_cost_per_token_above_256k_tokens"),
            ("opencode_go/qwen3.7-plus", "output_cost_per_token_above_256k_tokens"),
            ("opencode_go/qwen3.7-plus", "cache_read_input_token_cost_above_256k_tokens"),
            ("opencode_go/qwen3.6-plus", "cache_creation_input_token_cost_above_256k_tokens"),
        ],
    )
    def test_256k_tier_fields_survive_model_info(self, local_model_cost_map, model, field):
        raw = litellm.model_cost[model][field]
        assert litellm.get_model_info(model)[field] == raw

    def test_256k_tier_is_actually_billed(self, local_model_cost_map):
        from litellm.types.utils import Choices, Message, ModelResponse, Usage

        model = "opencode_go/qwen3.7-plus"
        raw = litellm.model_cost[model]

        def cost_for(prompt_tokens):
            response = ModelResponse(
                id="x", model="qwen3.7-plus", object="chat.completion", created=0,
                choices=[Choices(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
                usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=1_000,
                            total_tokens=prompt_tokens + 1_000),
            )
            return litellm.completion_cost(completion_response=response, model=model)

        below = 100_000 * raw["input_cost_per_token"] + 1_000 * raw["output_cost_per_token"]
        above = 300_000 * raw["input_cost_per_token_above_256k_tokens"] + (
            1_000 * raw["output_cost_per_token_above_256k_tokens"]
        )
        assert cost_for(100_000) == pytest.approx(below)
        assert cost_for(300_000) == pytest.approx(above)
        assert cost_for(300_000) > cost_for(100_000) * 2


class FailingHTTPHandler(HTTPHandler):
    """Drives the dispatch helper's error path, which logs the failure before re-raising."""

    def post(self, url, data=None, headers=None, **kwargs):
        raise httpx.ConnectError("connection refused")


class TestDispatchErrorPath:
    @pytest.mark.parametrize("model", ["opencode_go/kimi-k3", "opencode/claude-opus-5"])
    def test_upstream_failure_surfaces_to_the_caller(self, local_model_cost_map, model):
        with pytest.raises(litellm.InternalServerError, match="connection refused"):
            litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                api_key="fake-key",
                litellm_session_id="session-err",
                client=FailingHTTPHandler(),
            )


class TestResponsesWireFormat:
    """OpenCode serves its OpenAI-native models on /responses; /chat/completions answers them 500."""

    @pytest.mark.parametrize(
        "config, expected_provider, expected_url",
        [
            (OpenCodeZenResponsesAPIConfig(), LlmProviders.OPENCODE, f"{ZEN_API_BASE}/responses"),
            (OpenCodeGoResponsesAPIConfig(), LlmProviders.OPENCODE_GO, f"{GO_API_BASE}/responses"),
        ],
    )
    def test_provider_and_default_url(self, config, expected_provider, expected_url):
        assert config.custom_llm_provider == expected_provider
        assert config.get_complete_url(api_base=None, litellm_params={}) == expected_url

    @pytest.mark.parametrize(
        "config", [OpenCodeZenResponsesAPIConfig(), OpenCodeGoResponsesAPIConfig()]
    )
    def test_explicit_api_base_wins_and_trailing_slash_is_trimmed(self, config):
        assert config.get_complete_url(api_base="https://proxy.internal/v1/", litellm_params={}) == (
            "https://proxy.internal/v1/responses"
        )

    def test_auth_uses_bearer_and_stamps_the_session_header(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_API_KEY", "account-key")
        headers = OpenCodeGoResponsesAPIConfig().validate_environment(
            headers={},
            model="gpt-5.6-luna",
            litellm_params=GenericLiteLLMParams(litellm_session_id="session-9"),
        )
        assert headers["Authorization"] == "Bearer account-key"
        assert headers["Content-Type"] == "application/json"
        assert headers[OPENCODE_SESSION_HEADER] == "session-9"

    def test_request_api_key_wins_over_the_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_API_KEY", "account-key")
        headers = OpenCodeZenResponsesAPIConfig().validate_environment(
            headers={},
            model="gpt-5.5",
            litellm_params=GenericLiteLLMParams(api_key="request-key"),
        )
        assert headers["Authorization"] == "Bearer request-key"

    def test_caller_supplied_session_header_is_not_overwritten(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_API_KEY", "account-key")
        headers = OpenCodeZenResponsesAPIConfig().validate_environment(
            headers={"X-OpenCode-Session": "caller-owned"},
            model="gpt-5.5",
            litellm_params=GenericLiteLLMParams(litellm_session_id="session-9"),
        )
        assert headers["X-OpenCode-Session"] == "caller-owned"
        assert "session-9" not in headers.values()

    def test_missing_litellm_params_still_produces_headers(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_API_KEY", "account-key")
        headers = OpenCodeGoResponsesAPIConfig().validate_environment(
            headers={}, model="gpt-5.6-luna", litellm_params=None
        )
        assert headers["Authorization"] == "Bearer account-key"

    @pytest.mark.parametrize(
        "provider, expected_config",
        [
            (LlmProviders.OPENCODE, OpenCodeZenResponsesAPIConfig),
            (LlmProviders.OPENCODE_GO, OpenCodeGoResponsesAPIConfig),
        ],
    )
    def test_provider_manager_returns_the_responses_config(self, provider, expected_config):
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_responses_api_config(provider=provider, model="gpt-5.6-luna")
        assert type(config) is expected_config


class TestChatConfigProviderNames:
    @pytest.mark.parametrize(
        "config, expected",
        [
            (OpenCodeZenChatConfig(), "opencode"),
            (OpenCodeGoChatConfig(), "opencode_go"),
            (OpenCodeZenMessagesChatConfig(), "opencode"),
            (OpenCodeGoMessagesChatConfig(), "opencode_go"),
            (OpenCodeZenGeminiChatConfig(), "opencode"),
        ],
    )
    def test_custom_llm_provider(self, config, expected):
        assert config.custom_llm_provider == expected

    def test_messages_config_honours_an_explicit_api_base(self):
        url = OpenCodeGoMessagesChatConfig().get_complete_url(
            api_base="https://proxy.internal/v1/messages",
            api_key="k",
            model="minimax-m3",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://proxy.internal/v1/messages"


class RecordingAnthropicHTTPHandler(HTTPHandler):
    """OpenCode's /messages answers in Anthropic's shape, so the canned reply differs again."""

    def __init__(self, response_model: str):
        super().__init__()
        self.response_model = response_model
        self.requests: tuple[Mapping[str, object], ...] = ()

    def post(self, url, data=None, headers=None, **kwargs):
        raw_body: Final = data.decode("utf-8") if isinstance(data, bytes) else data
        self.requests = (
            *self.requests,
            {"url": url, "headers": dict(headers or {}), "body": json.loads(raw_body or "{}")},
        )
        return httpx.Response(
            status_code=200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": self.response_model,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
            request=httpx.Request("POST", url),
        )


class TestMessagesCompletionWiring:
    """Claude and the qwen/minimax families are served on /messages, not /chat/completions."""

    @pytest.mark.parametrize(
        "model, expected_url",
        [
            ("opencode/claude-opus-5", f"{ZEN_API_BASE}/messages"),
            ("opencode/claude-haiku-4-5", f"{ZEN_API_BASE}/messages"),
            ("opencode_go/minimax-m3", f"{GO_API_BASE}/messages"),
            ("opencode_go/qwen3.7-max", f"{GO_API_BASE}/messages"),
        ],
    )
    def test_session_header_reaches_the_wire(self, local_model_cost_map, model, expected_url):
        client = RecordingAnthropicHTTPHandler(response_model=model.split("/", 1)[1])
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            api_key="fake-key",
            litellm_session_id="session-42",
            client=client,
        )

        assert response.choices[0].message.content == "ok"
        assert len(client.requests) == 1
        sent = client.requests[0]
        assert sent["url"] == expected_url
        assert sent["headers"][OPENCODE_SESSION_HEADER] == "session-42"
        assert sent["headers"]["x-api-key"] == "fake-key"
        assert "Authorization" not in sent["headers"]

    def test_body_is_anthropic_shaped_not_openai_shaped(self, local_model_cost_map):
        """A regression here means the model is being sent to /chat/completions, which answers 500."""
        client = RecordingAnthropicHTTPHandler(response_model="claude-opus-5")
        litellm.completion(
            model="opencode/claude-opus-5",
            messages=[{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hello"}],
            api_key="fake-key",
            max_tokens=16,
            client=client,
        )

        body = client.requests[0]["body"]
        assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        assert body["system"] == [{"type": "text", "text": "Be terse."}]
        assert body["max_tokens"] == 16
