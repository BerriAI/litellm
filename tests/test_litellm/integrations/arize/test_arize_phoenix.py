import unittest
from unittest.mock import MagicMock, patch

import pytest

from litellm.integrations.arize.arize_phoenix import (
    ArizePhoenixConfig,
    ArizePhoenixLogger,
)
from litellm.integrations.arize._utils import ArizeOTELAttributes


class TestArizePhoenixConfig(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "PHOENIX_API_KEY": "test_api_key",
            "PHOENIX_COLLECTOR_HTTP_ENDPOINT": "http://test.endpoint",
        },
    )
    def test_get_arize_phoenix_config_http(self):
        # Call the function to get the configuration
        config = ArizePhoenixLogger.get_arize_phoenix_config()

        # Verify the configuration - now uses standard Authorization Bearer format
        self.assertEqual(config.otlp_auth_headers, "Authorization=Bearer test_api_key")
        self.assertEqual(config.endpoint, "http://test.endpoint/v1/traces")
        self.assertEqual(config.protocol, "otlp_http")

    @patch.dict(
        "os.environ",
        {
            "PHOENIX_API_KEY": "test_api_key",
            "PHOENIX_COLLECTOR_ENDPOINT": "grpc://test.endpoint",
        },
    )
    def test_get_arize_phoenix_config_grpc(self):
        # Call the function to get the configuration
        config = ArizePhoenixLogger.get_arize_phoenix_config()

        # Verify the configuration - now uses standard Authorization Bearer format
        self.assertEqual(config.otlp_auth_headers, "Authorization=Bearer test_api_key")
        self.assertEqual(config.endpoint, "grpc://test.endpoint")
        self.assertEqual(config.protocol, "otlp_grpc")

    @patch.dict(
        "os.environ",
        {
            "PHOENIX_API_KEY": "test_api_key",
            "PHOENIX_COLLECTOR_ENDPOINT": "http://localhost:6006",
        },
    )
    def test_get_arize_phoenix_config_http_local(self):
        # Test with local Phoenix instance
        config = ArizePhoenixLogger.get_arize_phoenix_config()

        # Should automatically append /v1/traces to local endpoint
        self.assertEqual(config.otlp_auth_headers, "Authorization=Bearer test_api_key")
        self.assertEqual(config.endpoint, "http://localhost:6006/v1/traces")
        self.assertEqual(config.protocol, "otlp_http")

    @patch.dict(
        "os.environ",
        {
            "PHOENIX_COLLECTOR_ENDPOINT": "http://localhost:4317",
        },
        clear=True,
    )
    def test_get_arize_phoenix_config_grpc_no_api_key(self):
        # Test gRPC endpoint detection and no API key (for local development)
        config = ArizePhoenixLogger.get_arize_phoenix_config()

        # No API key should be fine for local development
        self.assertIsNone(config.otlp_auth_headers)
        self.assertEqual(config.endpoint, "http://localhost:4317")
        self.assertEqual(config.protocol, "otlp_grpc")

    @patch.dict("os.environ", {}, clear=True)
    def test_get_arize_phoenix_config_defaults_to_local(self):
        # Test that it defaults to local Phoenix when no config is provided
        config = ArizePhoenixLogger.get_arize_phoenix_config()

        # Should default to localhost
        self.assertEqual(config.endpoint, "http://localhost:6006/v1/traces")
        self.assertEqual(config.protocol, "otlp_http")
        # No auth headers when no API key is provided for local instance
        self.assertIsNone(config.otlp_auth_headers)


@pytest.mark.parametrize(
    "env_vars, expected_headers, expected_endpoint, expected_protocol",
    [
        pytest.param(
            {"PHOENIX_API_KEY": "test_api_key"},
            "Authorization=Bearer test_api_key",
            "http://localhost:6006/v1/traces",
            "otlp_http",
            id="default to http protocol and self-hosted Phoenix endpoint",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_HTTP_ENDPOINT": "", "PHOENIX_API_KEY": "test_api_key"},
            "Authorization=Bearer test_api_key",
            "http://localhost:6006/v1/traces",
            "otlp_http",
            id="empty string/unset endpoint will default to http protocol and self-hosted Phoenix endpoint",
        ),
        pytest.param(
            {
                "PHOENIX_COLLECTOR_HTTP_ENDPOINT": "http://localhost:4318",
                "PHOENIX_COLLECTOR_ENDPOINT": "http://localhost:4317",
                "PHOENIX_API_KEY": "test_api_key",
            },
            "Authorization=Bearer test_api_key",
            "http://localhost:4318/v1/traces",
            "otlp_http",
            id="prioritize http if both endpoints are set",
        ),
        pytest.param(
            {
                "PHOENIX_COLLECTOR_ENDPOINT": "https://localhost:6006",
                "PHOENIX_API_KEY": "test_api_key",
            },
            "Authorization=Bearer test_api_key",
            "https://localhost:6006/v1/traces",
            "otlp_http",
            id="custom https endpoint treated as http",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_ENDPOINT": "https://localhost:6006"},
            None,
            "https://localhost:6006/v1/traces",
            "otlp_http",
            id="custom https endpoint with no auth treated as http",
        ),
        pytest.param(
            {
                "PHOENIX_COLLECTOR_ENDPOINT": "grpc://localhost:6006",
                "PHOENIX_API_KEY": "test_api_key",
            },
            "Authorization=Bearer test_api_key",
            "grpc://localhost:6006",
            "otlp_grpc",
            id="explicit grpc endpoint with grpc:// prefix",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_ENDPOINT": "http://localhost:4317"},
            None,
            "http://localhost:4317",
            "otlp_grpc",
            id="grpc endpoint with standard grpc port 4317",
        ),
        pytest.param(
            {
                "PHOENIX_COLLECTOR_HTTP_ENDPOINT": "https://localhost:6006",
                "PHOENIX_API_KEY": "test_api_key",
            },
            "Authorization=Bearer test_api_key",
            "https://localhost:6006/v1/traces",
            "otlp_http",
            id="custom http endpoint",
        ),
    ],
)
def test_get_arize_phoenix_config(
    monkeypatch, env_vars, expected_headers, expected_endpoint, expected_protocol
):
    # Clear all Phoenix-related env vars first to ensure clean state
    for key in [
        "PHOENIX_API_KEY",
        "PHOENIX_COLLECTOR_ENDPOINT",
        "PHOENIX_COLLECTOR_HTTP_ENDPOINT",
    ]:
        monkeypatch.delenv(key, raising=False)

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    config = ArizePhoenixLogger.get_arize_phoenix_config()

    assert isinstance(config, ArizePhoenixConfig)
    assert config.otlp_auth_headers == expected_headers
    assert config.endpoint == expected_endpoint
    assert config.protocol == expected_protocol


@pytest.mark.parametrize(
    "env_vars",
    [
        pytest.param(
            {"PHOENIX_COLLECTOR_ENDPOINT": "https://app.phoenix.arize.com/v1/traces"},
            id="missing api_key with explicit Arize Phoenix Cloud endpoint",
        ),
        pytest.param(
            {
                "PHOENIX_COLLECTOR_HTTP_ENDPOINT": "https://app.phoenix.arize.com/v1/traces"
            },
            id="missing api_key with HTTP Arize Phoenix Cloud endpoint",
        ),
    ],
)
def test_get_arize_phoenix_config_expection_on_missing_api_key(monkeypatch, env_vars):
    # Clear all Phoenix-related env vars first to ensure clean state
    for key in [
        "PHOENIX_API_KEY",
        "PHOENIX_COLLECTOR_ENDPOINT",
        "PHOENIX_COLLECTOR_HTTP_ENDPOINT",
    ]:
        monkeypatch.delenv(key, raising=False)

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(
        ValueError, match="PHOENIX_API_KEY must be set when using Phoenix Cloud"
    ):
        ArizePhoenixLogger.get_arize_phoenix_config()


# ---------------------------------------------------------------------------
# Dynamic project naming from metadata
# ---------------------------------------------------------------------------


class TestGetDynamicProjectName:
    """Tests for _get_dynamic_project_name extraction logic."""

    def test_extracts_from_standard_logging_object_metadata(self):
        kwargs = {
            "standard_logging_object": {
                "metadata": {"phoenix_project_name": "my-project"},
            }
        }
        assert ArizePhoenixLogger._get_dynamic_project_name(kwargs) == "my-project"

    def test_extracts_from_litellm_params_metadata(self):
        kwargs = {
            "litellm_params": {
                "metadata": {"phoenix_project_name": "sdk-project"},
            }
        }
        assert ArizePhoenixLogger._get_dynamic_project_name(kwargs) == "sdk-project"

    def test_returns_none_when_no_metadata(self):
        assert ArizePhoenixLogger._get_dynamic_project_name({}) is None

    def test_non_dict_standard_logging_object_does_not_raise(self):
        """isinstance(dict) guard prevents AttributeError on non-dict payloads."""
        kwargs = {"standard_logging_object": "not-a-dict"}
        assert ArizePhoenixLogger._get_dynamic_project_name(kwargs) is None


class TestResolveProjectName:
    """_resolve_project_name returns the correct project for various kwargs shapes."""

    def test_arize_override_takes_highest_priority(self):
        kwargs = {
            "standard_logging_object": {
                "metadata": {
                    "arize_project_name_override": "arize-proj",
                    "phoenix_project_name": "phoenix-proj",
                },
            }
        }
        assert ArizePhoenixLogger._resolve_project_name(kwargs) == "arize-proj"

    def test_team_metadata_override_via_user_api_key_team_metadata(self):
        """Team metadata is nested under user_api_key_team_metadata — must be found."""
        kwargs = {
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_team_metadata": {
                        "arize_project_name_override": "claude-code",
                    },
                },
            }
        }
        assert ArizePhoenixLogger._resolve_project_name(kwargs) == "claude-code"

    def test_team_metadata_phoenix_project_name_via_user_api_key_team_metadata(self):
        kwargs = {
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_team_metadata": {
                        "phoenix_project_name": "team-phoenix",
                    },
                },
            }
        }
        assert ArizePhoenixLogger._resolve_project_name(kwargs) == "team-phoenix"

    def test_key_metadata_override_via_user_api_key_metadata(self):
        """Key metadata is nested under user_api_key_metadata — must be found."""
        kwargs = {
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_metadata": {
                        "arize_project_name_override": "key-proj",
                    },
                },
            }
        }
        assert ArizePhoenixLogger._resolve_project_name(kwargs) == "key-proj"

    def test_request_metadata_beats_team_and_key_metadata(self):
        """Per-request override beats both team and key metadata."""
        kwargs = {
            "standard_logging_object": {
                "metadata": {
                    "arize_project_name_override": "request-proj",
                    "user_api_key_team_metadata": {
                        "arize_project_name_override": "team-proj",
                    },
                    "user_api_key_metadata": {
                        "arize_project_name_override": "key-proj",
                    },
                },
            }
        }
        assert ArizePhoenixLogger._resolve_project_name(kwargs) == "request-proj"

    def test_key_metadata_beats_team_metadata(self):
        """Key metadata wins over team metadata when no per-request override."""
        kwargs = {
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_metadata": {
                        "arize_project_name_override": "key-proj",
                    },
                    "user_api_key_team_metadata": {
                        "arize_project_name_override": "team-proj",
                    },
                },
            }
        }
        assert ArizePhoenixLogger._resolve_project_name(kwargs) == "key-proj"

    def test_phoenix_project_name_beats_env(self):
        kwargs = {
            "standard_logging_object": {
                "metadata": {"phoenix_project_name": "meta-proj"},
            }
        }
        with patch.dict("os.environ", {"PHOENIX_PROJECT_NAME": "env-proj"}):
            assert ArizePhoenixLogger._resolve_project_name(kwargs) == "meta-proj"

    @patch.dict("os.environ", {"PHOENIX_PROJECT_NAME": "env-phoenix"}, clear=False)
    def test_falls_back_to_phoenix_env_var(self):
        assert ArizePhoenixLogger._resolve_project_name({}) == "env-phoenix"

    @patch.dict(
        "os.environ",
        {"ARIZE_PROJECT_NAME": "env-arize"},
        clear=False,
    )
    def test_falls_back_to_arize_env_var(self):
        for key in ("PHOENIX_PROJECT_NAME",):
            patch.dict("os.environ", {key: ""}, clear=False).__enter__()
        with patch.dict("os.environ", {"PHOENIX_PROJECT_NAME": ""}):
            # PHOENIX_PROJECT_NAME empty string is falsy — should fall to ARIZE_PROJECT_NAME
            assert ArizePhoenixLogger._resolve_project_name({}) == "env-arize"

    def test_ultimate_fallback_is_default(self):
        with patch.dict(
            "os.environ",
            {"PHOENIX_PROJECT_NAME": "", "ARIZE_PROJECT_NAME": ""},
        ):
            assert ArizePhoenixLogger._resolve_project_name({}) == "default"

    def test_litellm_params_metadata_also_checked(self):
        kwargs = {
            "litellm_params": {
                "metadata": {"phoenix_project_name": "sdk-proj"},
            }
        }
        assert ArizePhoenixLogger._resolve_project_name(kwargs) == "sdk-proj"


class TestPerProjectProviderCache:
    """Per-project TracerProvider registry behaviour."""

    def _make_logger(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from litellm.integrations.opentelemetry import OpenTelemetryConfig

        exporter = InMemorySpanExporter()
        config = OpenTelemetryConfig(exporter=exporter)
        return ArizePhoenixLogger(config=config, callback_name="arize_phoenix")

    def test_different_projects_get_different_tracers(self):
        logger = self._make_logger()
        tracer_a = logger._get_tracer_for("project-a")
        tracer_b = logger._get_tracer_for("project-b")
        assert tracer_a is not tracer_b

    def test_same_project_returns_cached_provider(self):
        logger = self._make_logger()
        logger._get_tracer_for("my-project")
        provider_first = logger._project_providers["my-project"]
        logger._get_tracer_for("my-project")
        provider_second = logger._project_providers["my-project"]
        # Same provider object — cache was hit, not rebuilt
        assert provider_first is provider_second

    def test_lru_eviction_at_max_capacity(self):
        from litellm.integrations.arize.arize_phoenix import _MAX_PROJECT_PROVIDERS

        logger = self._make_logger()
        # Fill cache to capacity
        for i in range(_MAX_PROJECT_PROVIDERS):
            logger._get_tracer_for(f"project-{i}")
        assert len(logger._project_providers) == _MAX_PROJECT_PROVIDERS

        # One more triggers eviction of the LRU entry (project-0)
        logger._get_tracer_for("overflow-project")
        assert len(logger._project_providers) == _MAX_PROJECT_PROVIDERS
        assert "project-0" not in logger._project_providers
        assert "overflow-project" in logger._project_providers

    def test_get_tracer_to_use_for_request_routes_by_project(self):
        logger = self._make_logger()
        kwargs_a = {
            "standard_logging_object": {
                "metadata": {"phoenix_project_name": "proj-a"},
            }
        }
        kwargs_b = {
            "standard_logging_object": {
                "metadata": {"phoenix_project_name": "proj-b"},
            }
        }
        tracer_a = logger.get_tracer_to_use_for_request(kwargs_a)
        tracer_b = logger.get_tracer_to_use_for_request(kwargs_b)
        assert tracer_a is not tracer_b

    def test_get_tracer_to_use_for_request_consistent_within_request(self):
        """Parent, child, guardrail spans for one request must share a provider."""
        logger = self._make_logger()
        kwargs = {
            "standard_logging_object": {
                "metadata": {"phoenix_project_name": "consistent-proj"},
            }
        }
        # Each call for the same project must resolve to the same cached provider
        logger.get_tracer_to_use_for_request(kwargs)
        provider_1 = logger._project_providers["consistent-proj"]
        logger.get_tracer_to_use_for_request(kwargs)
        provider_2 = logger._project_providers["consistent-proj"]
        assert provider_1 is provider_2

    def test_project_resource_carries_project_name(self):
        """The provider Resource must carry openinference.project.name."""
        logger = self._make_logger()
        logger._get_tracer_for("resource-check-proj")
        provider = logger._project_providers["resource-check-proj"]
        resource_attrs = provider.resource.attributes
        assert resource_attrs.get("openinference.project.name") == "resource-check-proj"
        assert resource_attrs.get("model_id") == "resource-check-proj"
        assert resource_attrs.get("service.name") == "resource-check-proj"

    def test_no_span_attribute_set_for_project_name(self):
        """set_arize_phoenix_attributes must NOT set openinference.project.name on the span."""
        span = MagicMock()
        kwargs = {
            "standard_logging_object": {
                "metadata": {"phoenix_project_name": "some-proj"},
            }
        }
        with patch("litellm.integrations.arize._utils.set_attributes"):
            ArizePhoenixLogger.set_arize_phoenix_attributes(
                span, kwargs, response_obj=None
            )

        for call in span.set_attribute.call_args_list:
            assert (
                call.args[0] != "openinference.project.name"
            ), "openinference.project.name must live in the Resource, not on the span"


if __name__ == "__main__":
    unittest.main()


# --- Security: SSRF via prompt_version_id path traversal ---


def test_arize_phoenix_client_sanitize_id_rejects_traversal():
    from litellm.integrations.arize.arize_phoenix_client import _sanitize_id

    # dotdot without slashes
    with pytest.raises(ValueError, match="path traversal"):
        _sanitize_id("..something")
    # full traversal (slash caught first)
    with pytest.raises(ValueError, match="disallowed characters"):
        _sanitize_id("../../projects")


def test_arize_phoenix_client_sanitize_id_rejects_slash():
    from litellm.integrations.arize.arize_phoenix_client import _sanitize_id

    with pytest.raises(ValueError, match="disallowed characters"):
        _sanitize_id("valid/extra")


def test_arize_phoenix_client_sanitize_id_rejects_fragment():
    from litellm.integrations.arize.arize_phoenix_client import _sanitize_id

    with pytest.raises(ValueError, match="disallowed characters"):
        _sanitize_id("abc#suffix")


def test_arize_phoenix_client_sanitize_id_rejects_query():
    from litellm.integrations.arize.arize_phoenix_client import _sanitize_id

    with pytest.raises(ValueError, match="disallowed characters"):
        _sanitize_id("abc?x=1")


def test_arize_phoenix_client_sanitize_id_allows_uuid():
    from litellm.integrations.arize.arize_phoenix_client import _sanitize_id

    uid = "550e8400-e29b-41d4-a716-446655440000"
    assert _sanitize_id(uid) == uid


def test_arize_phoenix_client_get_prompt_version_rejects_traversal():
    from litellm.integrations.arize.arize_phoenix_client import ArizePhoenixClient

    client = ArizePhoenixClient(
        api_key="test-key", api_base="https://app.phoenix.arize.com"
    )
    with pytest.raises(ValueError, match="disallowed characters"):
        client.get_prompt_version("../../projects")
