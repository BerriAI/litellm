import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.databricks.ai_gateway import (
    AI_GATEWAY_PATHS,
    ProviderFamily,
    bare_endpoint_name,
    build_anthropic_messages_url,
    build_chat_url,
    build_gemini_generate_content_url,
    clear_gateway_cache,
    detect_family,
    gateway_available,
    has_explicit_custom_path,
    is_gateway_base,
    is_serving_endpoints_base,
    normalize_gateway_base,
    parse_use_ai_gateway_flag,
    resolve_surface,
    workspace_host_from_base,
)


class TestDetectFamily:
    @pytest.mark.parametrize(
        "model",
        [
            "databricks/databricks-claude-3-7-sonnet",
            "databricks-claude-sonnet-4",
            "databricks/claude-opus-4",
            "MY-CLAUDE-ENDPOINT",
        ],
    )
    def test_claude_routes_anthropic(self, model):
        assert detect_family(model) == ProviderFamily.ANTHROPIC

    @pytest.mark.parametrize(
        "model",
        [
            "databricks/databricks-gemini-2-5-pro",
            "databricks-gemini-flash",
            "gemini-1.5-pro",
        ],
    )
    def test_gemini_routes_gemini(self, model):
        assert detect_family(model) == ProviderFamily.GEMINI

    @pytest.mark.parametrize(
        "model",
        [
            "databricks/databricks-gpt-5",
            "databricks-gpt-5-mini",
            "gpt-4o",
            "gpt-5.1",
        ],
    )
    def test_gpt_n_routes_openai_responses(self, model):
        assert detect_family(model) == ProviderFamily.OPENAI_RESPONSES

    @pytest.mark.parametrize(
        "model",
        [
            "databricks/databricks-gpt-oss-120b",
            "databricks-gpt-oss-20b",
            "gpt-oss-120b",
        ],
    )
    def test_gpt_oss_is_not_responses(self, model):
        # gpt-oss has no native Responses API -> must fall through to unified chat
        assert detect_family(model) == ProviderFamily.OPENAI

    @pytest.mark.parametrize(
        "model",
        [
            "databricks/databricks-llama-4-maverick",
            "databricks-meta-llama-3-1-70b-instruct",
            "databricks/qwen2-5-coder",
            "databricks-gemma-3-12b",
            "some-random-endpoint",
        ],
    )
    def test_others_route_unified_openai(self, model):
        assert detect_family(model) == ProviderFamily.OPENAI


class TestBareEndpointName:
    def test_strips_databricks_prefix(self):
        assert bare_endpoint_name("databricks/databricks-claude-3-7-sonnet") == (
            "databricks-claude-3-7-sonnet"
        )

    def test_no_prefix_unchanged(self):
        assert bare_endpoint_name("databricks-gpt-5") == "databricks-gpt-5"


class TestNormalizeGatewayBase:
    def test_appends_ai_gateway_to_host(self):
        assert (
            normalize_gateway_base("https://my-ws.databricks.com")
            == "https://my-ws.databricks.com/ai-gateway"
        )

    def test_trailing_slash_handled(self):
        assert (
            normalize_gateway_base("https://my-ws.databricks.com/")
            == "https://my-ws.databricks.com/ai-gateway"
        )

    def test_idempotent_for_gateway_base(self):
        assert (
            normalize_gateway_base("https://my-ws.databricks.com/ai-gateway")
            == "https://my-ws.databricks.com/ai-gateway"
        )

    def test_replaces_serving_endpoints_suffix(self):
        assert (
            normalize_gateway_base("https://my-ws.databricks.com/serving-endpoints")
            == "https://my-ws.databricks.com/ai-gateway"
        )


class TestWorkspaceHostFromBase:
    @pytest.mark.parametrize(
        "api_base",
        [
            "https://my-ws.databricks.com/ai-gateway",
            "https://my-ws.databricks.com/serving-endpoints",
            "https://my-ws.databricks.com/ai-gateway/",
            "https://my-ws.databricks.com",
        ],
    )
    def test_recovers_host(self, api_base):
        assert workspace_host_from_base(api_base) == "https://my-ws.databricks.com"


class TestSurfaceDetection:
    def test_is_gateway_base(self):
        assert is_gateway_base("https://h/ai-gateway")
        assert is_gateway_base("https://h/ai-gateway/")
        assert not is_gateway_base("https://h/serving-endpoints")
        assert not is_gateway_base(None)

    def test_is_serving_endpoints_base(self):
        assert is_serving_endpoints_base("https://h/serving-endpoints")
        assert not is_serving_endpoints_base("https://h/ai-gateway")
        assert not is_serving_endpoints_base(None)


class TestGatewayPaths:
    def test_expected_paths_present(self):
        assert AI_GATEWAY_PATHS["chat"] == "/mlflow/v1/chat/completions"
        assert AI_GATEWAY_PATHS["supervisor_responses"] == "/mlflow/v1/responses"
        assert AI_GATEWAY_PATHS["openai_responses"] == "/openai/v1/responses"
        assert AI_GATEWAY_PATHS["anthropic_messages"] == "/anthropic/v1/messages"
        assert "{endpoint}" in AI_GATEWAY_PATHS["gemini_generate_content"]


class TestHasExplicitCustomPath:
    @pytest.mark.parametrize(
        "api_base,expected",
        [
            (None, False),
            ("https://h.databricks.com", False),
            ("https://h.databricks.com/", False),
            ("https://h.databricks.com/serving-endpoints", False),
            ("https://h.databricks.com/ai-gateway", False),
            ("https://h.databricks.com/my/custom/endpoint", True),
            ("https://h.databricks.com/serving-endpoints/ep/invocations", True),
        ],
    )
    def test_classification(self, api_base, expected):
        assert has_explicit_custom_path(api_base) is expected


class TestParseUseAiGatewayFlag:
    def test_default_is_none_auto(self):
        assert parse_use_ai_gateway_flag({}, {}) is None

    def test_optional_params_wins_over_litellm_params(self):
        assert (
            parse_use_ai_gateway_flag(
                {"databricks_use_ai_gateway": False},
                {"databricks_use_ai_gateway": True},
            )
            is True
        )

    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, True),
            (False, False),
            ("true", True),
            ("False", False),
            ("auto", None),
            ("on", True),
            ("0", False),
        ],
    )
    def test_coercion(self, value, expected):
        assert (
            parse_use_ai_gateway_flag({"databricks_use_ai_gateway": value}, {})
            is expected
        )

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_USE_AI_GATEWAY", "false")
        assert parse_use_ai_gateway_flag({}, {}) is False
        monkeypatch.setenv("DATABRICKS_USE_AI_GATEWAY", "true")
        assert parse_use_ai_gateway_flag({}, {}) is True


class TestResolveSurface:
    def _gw(self, value):
        return lambda: value

    def test_explicit_serving_endpoints_wins(self):
        # Even with the gateway available and flag forcing gateway, an explicit
        # /serving-endpoints base stays on serving-endpoints (strict Q4).
        assert (
            resolve_surface(
                "https://h/serving-endpoints", True, self._gw(True)
            )
            == "serving_endpoints"
        )

    def test_explicit_gateway_base(self):
        assert (
            resolve_surface("https://h/ai-gateway", False, self._gw(False))
            == "ai_gateway"
        )

    def test_flag_true_forces_gateway_no_probe(self):
        probe_called = {"n": 0}

        def probe():
            probe_called["n"] += 1
            return False

        assert resolve_surface(None, True, probe) == "ai_gateway"
        assert probe_called["n"] == 0

    def test_flag_false_forces_serving_no_probe(self):
        assert resolve_surface(None, False, self._gw(True)) == "serving_endpoints"

    def test_auto_defaults_gateway_when_available(self):
        assert resolve_surface(None, None, self._gw(True)) == "ai_gateway"

    def test_auto_falls_back_when_unavailable(self):
        assert resolve_surface(None, None, self._gw(False)) == "serving_endpoints"


class TestBuildChatUrl:
    def test_gateway_chat_url(self):
        assert (
            build_chat_url("https://h.databricks.com", "ai_gateway")
            == "https://h.databricks.com/ai-gateway/mlflow/v1/chat/completions"
        )

    def test_serving_endpoints_chat_url(self):
        assert (
            build_chat_url("https://h.databricks.com", "serving_endpoints")
            == "https://h.databricks.com/serving-endpoints/chat/completions"
        )

    def test_host_already_suffixed_is_normalized(self):
        assert (
            build_chat_url("https://h.databricks.com/serving-endpoints", "ai_gateway")
            == "https://h.databricks.com/ai-gateway/mlflow/v1/chat/completions"
        )


class TestBuildNativeUrls:
    def test_anthropic_messages_url(self):
        assert build_anthropic_messages_url("https://h.databricks.com") == (
            "https://h.databricks.com/ai-gateway/anthropic/v1/messages"
        )

    def test_anthropic_messages_url_from_serving_endpoints(self):
        assert build_anthropic_messages_url(
            "https://h.databricks.com/serving-endpoints"
        ) == "https://h.databricks.com/ai-gateway/anthropic/v1/messages"

    def test_anthropic_messages_url_idempotent(self):
        full = "https://h.databricks.com/ai-gateway/anthropic/v1/messages"
        assert build_anthropic_messages_url(full) == full

    def test_gemini_url_non_stream(self):
        assert build_gemini_generate_content_url(
            "https://h.databricks.com", "databricks/databricks-gemini-2-5-pro"
        ) == (
            "https://h.databricks.com/ai-gateway/gemini/v1beta/models/"
            "databricks-gemini-2-5-pro:generateContent"
        )

    def test_gemini_url_stream(self):
        assert build_gemini_generate_content_url(
            "https://h.databricks.com",
            "databricks/databricks-gemini-2-5-pro",
            stream=True,
        ) == (
            "https://h.databricks.com/ai-gateway/gemini/v1beta/models/"
            "databricks-gemini-2-5-pro:streamGenerateContent?alt=sse"
        )


class TestGatewayAvailableCache:
    def setup_method(self):
        clear_gateway_cache()

    def teardown_method(self):
        clear_gateway_cache()

    def test_probe_called_once_then_cached(self):
        calls = {"n": 0}

        def probe(host, headers):
            calls["n"] += 1
            return True

        host = "https://h.databricks.com"
        assert gateway_available(host, probe_fn=probe) is True
        assert gateway_available(host, probe_fn=probe) is True
        assert calls["n"] == 1  # second call served from cache

    def test_negative_result_cached(self):
        def probe(host, headers):
            return False

        host = "https://h.databricks.com"
        assert gateway_available(host, probe_fn=probe) is False

    def test_probe_exception_assumes_available(self):
        def probe(host, headers):
            raise RuntimeError("connection refused")

        host = "https://h.databricks.com"
        # Transient failure should not permanently disable the gateway.
        assert gateway_available(host, probe_fn=probe) is True

    def test_cache_is_per_host(self):
        results = {"a": True, "b": False}

        def probe(host, headers):
            return results["a"] if "host-a" in host else results["b"]

        assert gateway_available("https://host-a.databricks.com", probe_fn=probe) is True
        assert gateway_available("https://host-b.databricks.com", probe_fn=probe) is False
