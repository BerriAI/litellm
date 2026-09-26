from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import pytest
import yaml
from integration._support.otlp_sink import (
    ConnectSink,
    GrpcSink,
    SpanSinks,
    owned_connect_sink,
    owned_grpc_sink,
    owned_sinks,
)
from pydantic import JsonValue

AuditConfigWriter = Callable[[Path, Mapping[str, JsonValue]], Path]


@pytest.fixture(scope="module")
def audit_sinks(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SpanSinks]:
    directory: Final = tmp_path_factory.mktemp("otel-audit-sinks")
    with owned_sinks(directory) as sinks:
        yield sinks


@pytest.fixture(scope="module")
def newrelic_sink(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ConnectSink]:
    directory: Final = tmp_path_factory.mktemp("otel-audit-connect")
    with owned_connect_sink(directory) as sink:
        yield sink


@pytest.fixture(scope="module")
def arize_grpc_sink(tmp_path_factory: pytest.TempPathFactory) -> Iterator[GrpcSink]:
    directory: Final = tmp_path_factory.mktemp("otel-audit-grpc")
    with owned_grpc_sink(directory) as sink:
        yield sink


@pytest.fixture(scope="module")
def otel_audit_config(audit_sinks: SpanSinks) -> AuditConfigWriter:
    tenant_host: Final = urlparse(audit_sinks.tenant).netloc

    def write(directory: Path, litellm_settings: Mapping[str, JsonValue] = {}) -> Path:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["litellm_settings"] = {
            **config.get("litellm_settings", {}),
            "callbacks": ["otel"],
            "provider_url_destination_allowed_hosts": [tenant_host],
            **dict(litellm_settings),
        }
        config["callback_settings"] = {
            "otel": {"exporter": "http/json", "endpoint": audit_sinks.operator, "use_simple_processor": True}
        }
        config["general_settings"] = {**config.get("general_settings", {}), "user_api_key_cache_ttl": 2}
        path: Final = directory / f"otel-audit-{uuid.uuid4().hex}.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    return write


@pytest.fixture(scope="module")
def langfuse_vars(audit_sinks: SpanSinks) -> dict[str, JsonValue]:
    return {
        "langfuse_public_key": "pk-lf-audit",
        "langfuse_secret_key": "sk-lf-audit",
        "langfuse_host": audit_sinks.tenant,
    }
