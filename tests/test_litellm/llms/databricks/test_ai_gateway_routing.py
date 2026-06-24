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
    gateway_known_absent,
    has_explicit_custom_path,
    is_gateway_absent_error,
    is_gateway_base,
    is_serving_endpoints_base,
    mark_gateway_absent,
    mark_gateway_present,
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
    """resolve_surface is pure + no-network: optimistic gateway-first, consulting
    only the per-host absence cache for the auto case."""

    HOST = "https://h.databricks.com"

    def setup_method(self):
        clear_gateway_cache()

    def teardown_method(self):
        clear_gateway_cache()

    def test_explicit_serving_endpoints_wins(self):
        # Even with the flag forcing gateway, an explicit /serving-endpoints base
        # stays on serving-endpoints (strict Q4).
        assert (
            resolve_surface("https://h/serving-endpoints", True, self.HOST)
            == "serving_endpoints"
        )

    def test_explicit_gateway_base(self):
        assert resolve_surface("https://h/ai-gateway", False, self.HOST) == "ai_gateway"

    def test_flag_true_forces_gateway(self):
        assert resolve_surface(None, True, self.HOST) == "ai_gateway"

    def test_flag_false_forces_serving(self):
        # Even if cached present, an explicit False forces serving-endpoints.
        mark_gateway_present(self.HOST)
        assert resolve_surface(None, False, self.HOST) == "serving_endpoints"

    def test_auto_defaults_gateway_optimistically(self):
        # Unknown host -> optimistic gateway, no probe.
        assert resolve_surface(None, None, self.HOST) == "ai_gateway"

    def test_auto_uses_serving_when_host_known_absent(self):
        mark_gateway_absent(self.HOST)
        assert resolve_surface(None, None, self.HOST) == "serving_endpoints"

    def test_auto_uses_gateway_when_host_known_present(self):
        mark_gateway_present(self.HOST)
        assert resolve_surface(None, None, self.HOST) == "ai_gateway"


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


class TestReactiveGatewayCache:
    """The per-host cache is populated reactively (no preflight probe)."""

    def setup_method(self):
        clear_gateway_cache()

    def teardown_method(self):
        clear_gateway_cache()

    def test_unknown_host_not_absent(self):
        # Unknown host is NOT "known absent" -> resolve_surface stays optimistic.
        assert gateway_known_absent("https://h.databricks.com") is False

    def test_mark_absent_then_known_absent(self):
        host = "https://h.databricks.com"
        mark_gateway_absent(host)
        assert gateway_known_absent(host) is True

    def test_mark_present_then_not_absent(self):
        host = "https://h.databricks.com"
        mark_gateway_present(host)
        assert gateway_known_absent(host) is False

    def test_cache_is_per_host(self):
        mark_gateway_absent("https://host-a.databricks.com")
        assert gateway_known_absent("https://host-a.databricks.com") is True
        assert gateway_known_absent("https://host-b.databricks.com") is False

    def test_mark_absent_normalizes_suffix(self):
        # Marking via a suffixed base and reading via the bare host agree.
        mark_gateway_absent("https://h.databricks.com/serving-endpoints")
        assert gateway_known_absent("https://h.databricks.com") is True


class TestIsGatewayAbsentError:
    def _exc(self, status=None, message=""):
        e = Exception(message)
        if status is not None:
            e.status_code = status  # type: ignore[attr-defined]
        e.message = message  # type: ignore[attr-defined]
        return e

    def test_bare_404_is_not_absent(self):
        # A bare 404 conflates "no gateway" with "no such model" -> must NOT demote
        # the whole host. Only 501 / ENDPOINT_NOT_FOUND are conclusive.
        assert is_gateway_absent_error(self._exc(status=404, message="not found")) is False

    def test_404_with_endpoint_not_found_marker_is_absent(self):
        # Real absent gateway path returns 404 ENDPOINT_NOT_FOUND (per live matrix).
        assert (
            is_gateway_absent_error(
                self._exc(status=404, message="ENDPOINT_NOT_FOUND: no path")
            )
            is True
        )

    def test_model_level_404_is_not_absent(self):
        # A genuinely-missing model returns 404 RESOURCE_DOES_NOT_EXIST -> not host absence.
        assert (
            is_gateway_absent_error(
                self._exc(status=404, message="RESOURCE_DOES_NOT_EXIST: no model")
            )
            is False
        )

    def test_501_is_absent(self):
        assert is_gateway_absent_error(self._exc(status=501)) is True

    def test_endpoint_not_found_marker_is_absent(self):
        assert (
            is_gateway_absent_error(self._exc(message="ENDPOINT_NOT_FOUND: no path"))
            is True
        )

    def test_400_without_marker_is_not_absent(self):
        # A plain 400 (e.g. bad params) must NOT be treated as gateway-absent.
        assert is_gateway_absent_error(self._exc(status=400, message="bad input")) is False

    def test_500_without_absent_marker_is_not_absent(self):
        assert (
            is_gateway_absent_error(self._exc(status=500, message="internal_error"))
            is False
        )
