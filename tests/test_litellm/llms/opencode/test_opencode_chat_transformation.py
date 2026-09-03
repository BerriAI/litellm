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
    OpenCodeZenChatConfig,
)
from litellm.llms.opencode.common_utils import (
    OPENCODE_SESSION_HEADER,
    resolve_opencode_session_id,
)
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
        "config, expected_api_base",
        [
            (OpenCodeZenChatConfig(), ZEN_API_BASE),
            (OpenCodeGoChatConfig(), GO_API_BASE),
        ],
    )
    def test_api_key_env_vars(self, monkeypatch, config, expected_api_base):
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-alias-key")
        assert config._get_openai_compatible_provider_info(None, None) == (expected_api_base, "zen-alias-key")
        monkeypatch.setenv("OPENCODE_API_KEY", "primary-key")
        assert config._get_openai_compatible_provider_info(None, None) == (expected_api_base, "primary-key")

    @pytest.mark.parametrize("config", [OpenCodeZenChatConfig(), OpenCodeGoChatConfig()])
    def test_explicit_api_base_wins_over_default(self, config):
        api_base, _ = config._get_openai_compatible_provider_info("https://proxy.internal/v1", "k")
        assert api_base == "https://proxy.internal/v1"


class TestCompletionWiring:
    @pytest.mark.parametrize(
        "model, expected_url",
        [
            ("opencode/claude-opus-5", f"{ZEN_API_BASE}/chat/completions"),
            ("opencode_go/kimi-k3", f"{GO_API_BASE}/chat/completions"),
        ],
    )
    def test_session_header_reaches_the_wire(self, model, expected_url):
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

    def test_session_header_is_stable_across_turns_of_one_session(self):
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

    def test_header_present_even_without_a_caller_supplied_session_id(self):
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
