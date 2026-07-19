import unittest
from unittest.mock import MagicMock, patch

import pytest

from litellm.integrations.arize.arize_phoenix import (
    ArizePhoenixConfig,
    ArizePhoenixLogger,
)


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
# Per-project routing via Resource (not span attributes)
# ---------------------------------------------------------------------------


class TestResolveProjectName:
    """Tests for _resolve_project_name priority chain."""

    def test_extracts_phoenix_name_from_standard_logging_object_metadata(self):
        kwargs = {
            "standard_logging_object": {
                "metadata": {"phoenix_project_name": "my-project"},
            }
        }
        assert ArizePhoenixLogger._resolve_project_name(kwargs) == "my-project"

    def test_extracts_phoenix_name_from_litellm_params_metadata(self):
        kwargs = {
            "litellm_params": {
                "metadata": {"phoenix_project_name": "sdk-project"},
            }
        }
        assert ArizePhoenixLogger._resolve_project_name(kwargs) == "sdk-project"

    @patch.dict("os.environ", {"PHOENIX_PROJECT_NAME": "env-project"}, clear=False)
    def test_falls_back_to_phoenix_env_when_no_metadata(self):
        assert ArizePhoenixLogger._resolve_project_name({}) == "env-project"

    @patch.dict(
        "os.environ",
        {"ARIZE_PROJECT_NAME": "arize-env", "PHOENIX_PROJECT_NAME": ""},
        clear=False,
    )
    def test_falls_back_to_arize_env_when_phoenix_unset(self):
        assert ArizePhoenixLogger._resolve_project_name({}) == "arize-env"

    @patch.dict("os.environ", {}, clear=True)
    def test_falls_back_to_default_when_no_metadata_or_env(self):
        assert ArizePhoenixLogger._resolve_project_name({}) == "default"

    def test_phoenix_override_beats_phoenix_metadata(self):
        kwargs = {
            "standard_logging_object": {
                "metadata": {
                    "phoenix_project_name_override": "override-proj",
                    "phoenix_project_name": "phoenix-proj",
                },
            }
        }
        assert ArizePhoenixLogger._resolve_project_name(kwargs) == "override-proj"

    def test_whitespace_only_metadata_falls_through_to_default(self):
        kwargs = {
            "standard_logging_object": {
                "metadata": {"phoenix_project_name_override": "   "},
            }
        }
        with patch.dict("os.environ", {}, clear=True):
            assert ArizePhoenixLogger._resolve_project_name(kwargs) == "default"

    def test_strips_whitespace_from_project_name(self):
        kwargs = {
            "standard_logging_object": {
                "metadata": {"phoenix_project_name": "  trimmed  "},
            }
        }
        assert ArizePhoenixLogger._resolve_project_name(kwargs) == "trimmed"

    def test_non_dict_standard_logging_object_does_not_raise(self):
        kwargs = {"standard_logging_object": "not-a-dict"}
        with patch.dict("os.environ", {}, clear=True):
            assert ArizePhoenixLogger._resolve_project_name(kwargs) == "default"

    def test_resolves_override_from_user_api_key_auth_metadata(self):
        kwargs = {
            "litellm_params": {
                "metadata": {
                    "user_api_key_auth_metadata": {
                        "phoenix_project_name_override": "claude-code",
                    },
                },
            },
        }
        with patch.dict("os.environ", {}, clear=True):
            assert ArizePhoenixLogger._resolve_project_name(kwargs) == "claude-code"

    def test_resolves_phoenix_name_from_user_api_key_auth_metadata(self):
        kwargs = {
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_auth_metadata": {
                        "phoenix_project_name": "team-project",
                    },
                },
            },
        }
        with patch.dict("os.environ", {}, clear=True):
            assert ArizePhoenixLogger._resolve_project_name(kwargs) == "team-project"

    def test_proxy_ignores_client_metadata_when_auth_metadata_set(self):
        kwargs = {
            "litellm_params": {
                "proxy_server_request": {
                    "url": "/v1/chat/completions",
                    "method": "POST",
                    "headers": {},
                },
                "metadata": {
                    "phoenix_project_name_override": "attacker-project",
                    "user_api_key_auth_metadata": {
                        "phoenix_project_name_override": "team-project",
                    },
                },
            },
        }
        with patch.dict("os.environ", {}, clear=True):
            assert ArizePhoenixLogger._resolve_project_name(kwargs) == "team-project"

    def test_proxy_without_auth_metadata_falls_back_to_env(self):
        kwargs = {
            "litellm_params": {
                "proxy_server_request": {
                    "url": "/v1/chat/completions",
                    "method": "POST",
                    "headers": {},
                },
                "metadata": {"phoenix_project_name": "attacker-project"},
            },
        }
        with patch.dict(
            "os.environ", {"PHOENIX_PROJECT_NAME": "env-project"}, clear=True
        ):
            assert ArizePhoenixLogger._resolve_project_name(kwargs) == "env-project"


class TestProjectNameNotOnSpan:
    """Project routing uses Resource on TracerProvider, not span attributes."""

    @patch("litellm.integrations.arize._utils.set_attributes")
    def test_set_arize_phoenix_attributes_does_not_set_project_on_span(
        self, _mock_set_attrs
    ):
        span = MagicMock()
        kwargs = {
            "standard_logging_object": {
                "metadata": {"phoenix_project_name": "dynamic-proj"},
            }
        }
        ArizePhoenixLogger.set_arize_phoenix_attributes(span, kwargs, response_obj=None)

        for call in span.set_attribute.call_args_list:
            assert call[0][0] != "openinference.project.name"


class TestPerProjectTracerProviderCache:
    """Spans for different projects use different Resources on export."""

    def test_different_metadata_routes_to_different_resource(self):
        from datetime import datetime

        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from litellm.integrations.opentelemetry import OpenTelemetryConfig

        exporter = InMemorySpanExporter()
        logger = ArizePhoenixLogger(
            config=OpenTelemetryConfig(exporter=exporter),
            callback_name="arize_phoenix",
        )

        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 1)

        logger._handle_success(
            {
                "standard_logging_object": {
                    "metadata": {"phoenix_project_name": "project-a"},
                },
            },
            response_obj={},
            start_time=start,
            end_time=end,
        )
        logger._handle_success(
            {
                "standard_logging_object": {
                    "metadata": {"phoenix_project_name": "project-b"},
                },
            },
            response_obj={},
            start_time=start,
            end_time=end,
        )

        spans = exporter.get_finished_spans()
        project_names = {
            s.resource.attributes.get("openinference.project.name") for s in spans
        }
        assert "project-a" in project_names
        assert "project-b" in project_names

    def test_shared_span_processor_created_once_at_init(self):
        from litellm.integrations.opentelemetry import (
            OpenTelemetry,
            OpenTelemetryConfig,
        )

        mock_processor = MagicMock()
        with patch.object(
            OpenTelemetry, "_get_span_processor", return_value=mock_processor
        ) as mock_get_processor:
            logger = ArizePhoenixLogger(
                config=OpenTelemetryConfig(exporter=MagicMock()),
                callback_name="arize_phoenix",
            )
            assert mock_get_processor.call_count == 1
            assert logger._shared_span_processor is mock_processor

            logger._project_providers.clear()
            logger._get_tracer_for("project-a")
            logger._get_tracer_for("project-b")
            assert mock_get_processor.call_count == 1

    def test_lru_eviction_does_not_shutdown_provider(self):
        from litellm.integrations.opentelemetry import OpenTelemetryConfig

        logger = ArizePhoenixLogger(
            config=OpenTelemetryConfig(exporter=MagicMock()),
            callback_name="arize_phoenix",
        )
        logger._project_providers.clear()

        logger._get_tracer_for("project-0")
        evicted_provider = logger._project_providers["project-0"]
        shutdown_mock = MagicMock()
        evicted_provider.shutdown = shutdown_mock  # type: ignore[method-assign]

        for i in range(1, 65):
            logger._get_tracer_for(f"project-{i}")

        assert len(logger._project_providers) == 64
        assert "project-0" not in logger._project_providers
        assert "project-64" in logger._project_providers
        shutdown_mock.assert_not_called()

    def test_flush_tracer_providers_force_flushes_shared_processor(self):
        from litellm.integrations.opentelemetry import OpenTelemetryConfig

        logger = ArizePhoenixLogger(
            config=OpenTelemetryConfig(exporter=MagicMock()),
            callback_name="arize_phoenix",
        )
        mock_processor = MagicMock()
        logger._shared_span_processor = mock_processor
        mock_provider = MagicMock()
        logger._project_providers["proj"] = mock_provider

        logger.flush_tracer_providers()

        mock_processor.force_flush.assert_called_once()
        mock_provider.force_flush.assert_called_once()


class TestGetLitellmResourceForProject:
    """Resource attrs used by Phoenix OSS and Arize AX for project routing."""

    def test_project_attrs_win_over_otel_resource_attributes_env(self):
        from litellm.integrations.opentelemetry import OpenTelemetryConfig

        logger = ArizePhoenixLogger(
            config=OpenTelemetryConfig(exporter=MagicMock()),
            callback_name="arize_phoenix",
        )

        with patch.dict(
            "os.environ",
            {
                "OTEL_RESOURCE_ATTRIBUTES": "openinference.project.name=env-pinned,model_id=env-model"
            },
            clear=False,
        ):
            resource = logger._get_litellm_resource_for_project("dynamic-proj")

        assert resource.attributes["openinference.project.name"] == "dynamic-proj"
        assert resource.attributes["model_id"] == "dynamic-proj"
        assert resource.attributes["service.name"] == "dynamic-proj"

    @patch.dict("os.environ", {"OTEL_DEPLOYMENT_ENVIRONMENT": "staging"}, clear=False)
    def test_preserves_deployment_environment_from_config(self):
        from litellm.integrations.opentelemetry import OpenTelemetryConfig

        logger = ArizePhoenixLogger(
            config=OpenTelemetryConfig(
                exporter=MagicMock(), deployment_environment="staging"
            ),
            callback_name="arize_phoenix",
        )
        resource = logger._get_litellm_resource_for_project("my-proj")
        assert resource.attributes.get("deployment.environment") == "staging"


class TestTracerResolutionAndCache:
    """_resolve_tracer_for_kwargs, get_tracer_to_use_for_request, provider cache."""

    def test_get_tracer_to_use_for_request_matches_resolve_tracer(self):
        from litellm.integrations.opentelemetry import OpenTelemetryConfig

        logger = ArizePhoenixLogger(
            config=OpenTelemetryConfig(exporter=MagicMock()),
            callback_name="arize_phoenix",
        )
        kwargs = {
            "standard_logging_object": {
                "metadata": {"phoenix_project_name": "same-proj"},
            }
        }
        project_name, _ = logger._resolve_tracer_for_kwargs(kwargs)
        tracer_from_request = logger.get_tracer_to_use_for_request(kwargs)
        assert project_name == "same-proj"
        assert "same-proj" in logger._project_providers
        assert logger._resolve_project_name(kwargs) == project_name
        assert tracer_from_request is not None

    def test_cache_reuses_provider_for_same_project(self):
        from litellm.integrations.opentelemetry import OpenTelemetryConfig

        logger = ArizePhoenixLogger(
            config=OpenTelemetryConfig(exporter=MagicMock()),
            callback_name="arize_phoenix",
        )
        logger._project_providers.clear()

        logger._get_tracer_for("cached-proj")
        provider_first = logger._project_providers["cached-proj"]

        logger._get_tracer_for("cached-proj")
        provider_second = logger._project_providers["cached-proj"]

        assert provider_first is provider_second
        assert len(logger._project_providers) == 1

    def test_parallel_cache_miss_for_same_project_inserts_once(self):
        import threading

        from litellm.integrations.opentelemetry import OpenTelemetryConfig

        logger = ArizePhoenixLogger(
            config=OpenTelemetryConfig(exporter=MagicMock()),
            callback_name="arize_phoenix",
        )
        logger._project_providers.clear()

        build_calls: list[str] = []
        real_build = logger._build_tracer_provider_for_project

        def tracking_build(project_name: str):
            build_calls.append(project_name)
            return real_build(project_name)

        barrier = threading.Barrier(10)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                barrier.wait()
                logger._get_tracer_for("race-proj")
            except Exception as exc:
                errors.append(exc)

        with patch.object(
            logger,
            "_build_tracer_provider_for_project",
            side_effect=tracking_build,
        ):
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert not errors
        assert len(logger._project_providers) == 1
        assert "race-proj" in logger._project_providers
        assert len(build_calls) >= 1

    def test_injected_tracer_provider_bypasses_project_cache(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from litellm.integrations.opentelemetry import OpenTelemetryConfig

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        logger = ArizePhoenixLogger(
            config=OpenTelemetryConfig(exporter=exporter),
            callback_name="arize_phoenix",
            tracer_provider=provider,
        )

        assert getattr(logger, "_use_injected_tracer_provider", False) is True
        assert not hasattr(logger, "_project_providers") or not getattr(
            logger, "_project_providers", None
        )

        tracer_a = logger._get_tracer_for("any-project")
        tracer_b = logger.get_tracer_to_use_for_request(
            {"standard_logging_object": {"metadata": {"phoenix_project_name": "x"}}}
        )
        assert tracer_a is logger.tracer
        assert tracer_b is logger.tracer

    def test_flush_tracer_providers_noop_for_injected_provider(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from litellm.integrations.opentelemetry import OpenTelemetryConfig

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        logger = ArizePhoenixLogger(
            config=OpenTelemetryConfig(exporter=exporter),
            callback_name="arize_phoenix",
            tracer_provider=provider,
        )
        logger.flush_tracer_providers()
        exporter.shutdown()

    def test_standard_logging_metadata_wins_over_litellm_params(self):
        kwargs = {
            "standard_logging_object": {
                "metadata": {"phoenix_project_name_override": "from-logging"},
            },
            "litellm_params": {
                "metadata": {"phoenix_project_name_override": "from-params"},
            },
        }
        assert ArizePhoenixLogger._resolve_project_name(kwargs) == "from-logging"


class TestPhoenixTraceHandling:
    """_handle_success / _handle_failure span export behavior."""

    def test_handle_failure_sets_error_status_on_request_span(self):
        from datetime import datetime

        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
        from opentelemetry.trace import StatusCode

        from litellm.integrations.opentelemetry import (
            LITELLM_REQUEST_SPAN_NAME,
            OpenTelemetryConfig,
        )

        exporter = InMemorySpanExporter()
        logger = ArizePhoenixLogger(
            config=OpenTelemetryConfig(exporter=exporter),
            callback_name="arize_phoenix",
        )

        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 1)

        logger._handle_failure(
            {
                "standard_logging_object": {
                    "metadata": {"phoenix_project_name": "fail-proj"},
                },
                "exception": Exception("boom"),
            },
            response_obj=None,
            start_time=start,
            end_time=end,
        )

        spans = exporter.get_finished_spans()
        request_spans = [s for s in spans if s.name == LITELLM_REQUEST_SPAN_NAME]
        assert len(request_spans) == 1
        assert request_spans[0].status.status_code == StatusCode.ERROR
        assert (
            request_spans[0].resource.attributes.get("openinference.project.name")
            == "fail-proj"
        )

    def test_proxy_mode_parent_and_child_share_trace_id(self):
        from datetime import datetime

        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from litellm.integrations.opentelemetry import (
            LITELLM_REQUEST_SPAN_NAME,
            OpenTelemetryConfig,
        )

        exporter = InMemorySpanExporter()
        logger = ArizePhoenixLogger(
            config=OpenTelemetryConfig(exporter=exporter),
            callback_name="arize_phoenix",
        )

        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 1)

        logger._handle_success(
            {
                "litellm_params": {
                    "proxy_server_request": {
                        "url": "/chat/completions",
                        "method": "POST",
                        "headers": {},
                    },
                    "metadata": {
                        "user_api_key_auth_metadata": {
                            "phoenix_project_name_override": "proxy-proj",
                        },
                    },
                },
            },
            response_obj={},
            start_time=start,
            end_time=end,
        )

        spans = exporter.get_finished_spans()
        span_names = {s.name for s in spans}
        assert "litellm_proxy_request" in span_names
        assert LITELLM_REQUEST_SPAN_NAME in span_names

        trace_ids = {s.context.trace_id for s in spans}
        assert len(trace_ids) == 1
        for span in spans:
            assert (
                span.resource.attributes.get("openinference.project.name")
                == "proxy-proj"
            )

    def test_override_routes_all_spans_to_one_project_in_single_request(self):
        from datetime import datetime

        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from litellm.integrations.opentelemetry import OpenTelemetryConfig

        exporter = InMemorySpanExporter()
        logger = ArizePhoenixLogger(
            config=OpenTelemetryConfig(exporter=exporter),
            callback_name="arize_phoenix",
        )

        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 1)

        logger._handle_success(
            {
                "standard_logging_object": {
                    "metadata": {
                        "user_api_key_auth_metadata": {
                            "phoenix_project_name_override": "unified-proj",
                        },
                    },
                },
                "litellm_params": {
                    "proxy_server_request": {
                        "url": "/v1/chat/completions",
                        "method": "POST",
                        "headers": {},
                    },
                },
            },
            response_obj={"id": "resp-1"},
            start_time=start,
            end_time=end,
        )

        for span in exporter.get_finished_spans():
            assert (
                span.resource.attributes.get("openinference.project.name")
                == "unified-proj"
            )
            assert span.resource.attributes.get("model_id") == "unified-proj"


class TestGetArizePhoenixConfigProjectName:
    @patch.dict(
        "os.environ", {"PHOENIX_PROJECT_NAME": "phoenix-config-proj"}, clear=True
    )
    def test_project_name_from_phoenix_env(self):
        config = ArizePhoenixLogger.get_arize_phoenix_config()
        assert config.project_name == "phoenix-config-proj"

    @patch.dict("os.environ", {}, clear=True)
    def test_project_name_defaults_when_env_unset(self):
        config = ArizePhoenixLogger.get_arize_phoenix_config()
        assert config.project_name == "default"


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
