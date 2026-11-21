"""LangWatch preset tests: env parsing, cloud and self-hosted endpoints, auth
header, registration, and the V2 / legacy factory paths."""

import contextvars
import os

import pytest

pytest.importorskip("opentelemetry")

import litellm  # noqa: E402
from litellm.integrations.otel.model.config import (  # noqa: E402
    ExporterOwner,
    ExporterSpec,
    OpenTelemetryV2Config,
    is_otel_v2_enabled,
)
from litellm.integrations.otel.presets import (  # noqa: E402
    DYNAMIC_HEADERS_BY_CALLBACK,
    PRESET_BY_CALLBACK,
)
from litellm.integrations.otel.presets.langwatch import (  # noqa: E402
    get_langwatch_otel_config,
    langwatch_otel_traces_endpoint,
    langwatch_preset,
)

_CLOUD_TRACES: str = "https://app.langwatch.ai/api/otel/v1/traces"


@pytest.fixture
def langwatch_env(monkeypatch):
    monkeypatch.setenv("LANGWATCH_API_KEY", "lw-test-key")
    monkeypatch.delenv("LANGWATCH_ENDPOINT", raising=False)
    # Isolate from any operator-level OTel exporter config in the environment.
    for name in [name for name in os.environ if name.startswith("OTEL_")]:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _langwatch_spec(cfg: OpenTelemetryV2Config) -> ExporterSpec:
    specs = [spec for spec in cfg.exporters if spec.owner == ExporterOwner.LANGWATCH]
    assert len(specs) == 1
    return specs[0]


def test_preset_exports_to_langwatch_cloud_with_bearer_key(langwatch_env):
    cfg = langwatch_preset()

    spec = _langwatch_spec(cfg)
    assert spec.kind == "otlp_http"
    assert spec.endpoint == _CLOUD_TRACES
    assert spec.headers == "Authorization=Bearer lw-test-key"
    # Canonical GenAI vocabulary only; no vendor-specific mapper is layered on.
    assert cfg.mapper_names[0] == "genai"
    assert set(cfg.mapper_names) <= {"genai", "legacy"}


def test_preset_drops_the_unconfigured_console_placeholder(langwatch_env):
    # With nothing else configured, spans must go to LangWatch only, not also to stdout.
    cfg = langwatch_preset()

    assert [spec.owner for spec in cfg.exporters] == [ExporterOwner.LANGWATCH]


def test_preset_keeps_an_exporter_the_operator_configured(langwatch_env):
    base = OpenTelemetryV2Config(exporter="otlp_http", endpoint="http://collector.local:4318")

    cfg = langwatch_preset(config_overrides=base)

    assert [spec.endpoint for spec in cfg.exporters] == ["http://collector.local:4318", _CLOUD_TRACES]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("https://langwatch.internal.example.com", "https://langwatch.internal.example.com/api/otel/v1/traces"),
        ("https://langwatch.internal.example.com/", "https://langwatch.internal.example.com/api/otel/v1/traces"),
        ("http://localhost:5560", "http://localhost:5560/api/otel/v1/traces"),
        ("langwatch.internal.example.com", "https://langwatch.internal.example.com/api/otel/v1/traces"),
        (
            "https://langwatch.internal.example.com/api/otel/v1/traces",
            "https://langwatch.internal.example.com/api/otel/v1/traces",
        ),
    ],
)
def test_self_hosted_endpoint_gets_the_otel_traces_path(langwatch_env, configured, expected):
    langwatch_env.setenv("LANGWATCH_ENDPOINT", configured)

    assert _langwatch_spec(langwatch_preset()).endpoint == expected


@pytest.mark.parametrize("unset", [None, "", "   "])
def test_endpoint_defaults_to_langwatch_cloud(unset):
    assert langwatch_otel_traces_endpoint(unset) == _CLOUD_TRACES


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("LANGWATCH_API_KEY", raising=False)

    with pytest.raises(ValueError, match="LANGWATCH_API_KEY"):
        get_langwatch_otel_config()
    with pytest.raises(ValueError, match="LANGWATCH_API_KEY"):
        langwatch_preset()


def test_langwatch_is_a_registered_callback():
    assert PRESET_BY_CALLBACK["langwatch"] is langwatch_preset
    assert ExporterOwner("langwatch") is ExporterOwner.LANGWATCH
    assert "langwatch" in litellm._known_custom_logger_compatible_callbacks
    # No per-request credentials in this version: every request uses the operator's key.
    assert "langwatch" not in DYNAMIC_HEADERS_BY_CALLBACK


def _build_callback(monkeypatch, *, v2: bool):
    from litellm.litellm_core_utils import litellm_logging

    monkeypatch.setenv("LITELLM_OTEL_V2", "true" if v2 else "false")
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [])
    is_otel_v2_enabled.cache_clear()
    try:
        return contextvars.copy_context().run(
            litellm_logging._init_custom_logger_compatible_class,
            "langwatch",
            None,
            None,
        )
    finally:
        is_otel_v2_enabled.cache_clear()


def test_v2_factory_builds_a_langwatch_otel_v2_logger(langwatch_env):
    from litellm.integrations.otel.logger import OpenTelemetryV2

    logger = _build_callback(langwatch_env, v2=True)

    assert isinstance(logger, OpenTelemetryV2)
    assert logger.callback_name == "langwatch"
    assert _langwatch_spec(logger.config).endpoint == _CLOUD_TRACES


def test_legacy_factory_exports_to_langwatch_when_v2_is_off(langwatch_env):
    from litellm.integrations.opentelemetry import OpenTelemetry

    logger = _build_callback(langwatch_env, v2=False)

    assert type(logger) is OpenTelemetry
    assert logger.callback_name == "langwatch"
    assert logger.OTEL_EXPORTER == "otlp_http"
    assert logger.OTEL_ENDPOINT == _CLOUD_TRACES
    assert logger.OTEL_HEADERS == "Authorization=Bearer lw-test-key"
