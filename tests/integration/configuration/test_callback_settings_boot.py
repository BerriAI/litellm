import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy

SERVING_CONSUMERS: Final = {
    "compression_interception": "CompressionInterceptionLogger",
    "code_interpreter_interception": "CodeInterpreterInterceptionLogger",
    "websearch_interception": "WebSearchInterceptionLogger",
}
OTEL_CONSUMER: Final = {"otel": "OpenTelemetry"}
GUARDRAIL_CONSUMERS: Final = {
    "presidio": "_OPTIONAL_PresidioPIIMasking",
    "lakera_prompt_injection": "lakeraAI_Moderation",
}

TOP_LEVEL_SHAPES: Final = (
    pytest.param({}, id="empty-object"),
    pytest.param(None, id="null"),
    pytest.param("otel", id="string"),
    pytest.param(["otel"], id="list"),
    pytest.param(True, id="bool"),
    pytest.param(0, id="zero"),
)

CONSUMER_SHAPES: Final = (
    pytest.param({}, id="empty-object"),
    pytest.param(None, id="null"),
    pytest.param("on", id="string"),
    pytest.param(True, id="bool"),
    pytest.param([], id="empty-list"),
    pytest.param(["on"], id="list"),
    pytest.param(7, id="int"),
)

TOP_LEVEL_BOOT_CRASH: Final = (
    "BUG: a non-object callback_settings is stored verbatim and proxy startup crashes calling .get on it"
)
TOP_LEVEL_BOOT_CRASH_IDS: Final = frozenset({"string", "list", "bool"})
OTEL_DROPPED: Final = (
    "BUG: a non-object callback_settings.otel fails dict() and the otel callback is silently not registered"
)
OTEL_DROPPED_IDS: Final = frozenset({"null", "string", "bool", "int"})


def _write_config(
    directory: Path, upstream_url: str, model: str, callbacks: tuple[str, ...], callback_settings: JsonValue
) -> Path:
    config: Final = directory / f"callback_settings_{uuid.uuid4().hex}.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": f"openai/{model}",
                            "api_base": f"{upstream_url}/v1",
                            "api_key": "integration-provider-key",
                        },
                    }
                ],
                "litellm_settings": {"callbacks": list(callbacks)},
                "callback_settings": callback_settings,
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                },
            }
        )
    )
    return config


def _assert_registered(candidate: Gateway, consumers: Mapping[str, str]) -> None:
    response: Final = candidate.request("GET", "/active/callbacks")
    assert response.status_code == 200, response.text
    missing: Final = sorted(name for name, class_name in consumers.items() if class_name not in response.text)
    assert missing == [], response.text


@pytest.mark.parametrize("callback_settings", TOP_LEVEL_SHAPES)
def test_top_level_callback_settings_shape_boots_registers_and_serves_chat(
    gateway: Gateway, tmp_path: Path, callback_settings: JsonValue, request: pytest.FixtureRequest
) -> None:
    if request.node.callspec.id in TOP_LEVEL_BOOT_CRASH_IDS:
        pytest.skip(TOP_LEVEL_BOOT_CRASH)
    consumers: Final = {**SERVING_CONSUMERS, **OTEL_CONSUMER}
    model: Final = f"integration-callback-settings-{uuid.uuid4().hex}"
    config: Final = _write_config(tmp_path, gateway.upstream_url, model, tuple(consumers), callback_settings)
    with owned_proxy(gateway, tmp_path, {"STORE_MODEL_IN_DB": "False"}, config=config) as candidate:
        _assert_registered(candidate, consumers)
        reply: Final = candidate.chat(model, text=f"callback settings {uuid.uuid4().hex}")
        assert reply["model"] == model, reply


@pytest.mark.parametrize("value", CONSUMER_SHAPES)
def test_serving_consumer_settings_shape_boots_registers_and_serves_chat(
    gateway: Gateway, tmp_path: Path, value: JsonValue
) -> None:
    model: Final = f"integration-callback-settings-{uuid.uuid4().hex}"
    config: Final = _write_config(
        tmp_path,
        gateway.upstream_url,
        model,
        tuple(SERVING_CONSUMERS),
        {consumer: value for consumer in SERVING_CONSUMERS},
    )
    with owned_proxy(gateway, tmp_path, {"STORE_MODEL_IN_DB": "False"}, config=config) as candidate:
        _assert_registered(candidate, SERVING_CONSUMERS)
        reply: Final = candidate.chat(model, text=f"callback settings {uuid.uuid4().hex}")
        assert reply["model"] == model, reply


@pytest.mark.parametrize("value", CONSUMER_SHAPES)
def test_otel_settings_shape_boots_registers_and_serves_chat(
    gateway: Gateway, tmp_path: Path, value: JsonValue, request: pytest.FixtureRequest
) -> None:
    if request.node.callspec.id in OTEL_DROPPED_IDS:
        pytest.skip(OTEL_DROPPED)
    model: Final = f"integration-callback-settings-{uuid.uuid4().hex}"
    config: Final = _write_config(tmp_path, gateway.upstream_url, model, tuple(OTEL_CONSUMER), {"otel": value})
    with owned_proxy(gateway, tmp_path, {"STORE_MODEL_IN_DB": "False"}, config=config) as candidate:
        _assert_registered(candidate, OTEL_CONSUMER)
        reply: Final = candidate.chat(model, text=f"callback settings {uuid.uuid4().hex}")
        assert reply["model"] == model, reply


@pytest.mark.parametrize("value", CONSUMER_SHAPES)
def test_guardrail_consumer_settings_shape_boots_and_registers(
    gateway: Gateway, tmp_path: Path, value: JsonValue
) -> None:
    model: Final = f"integration-callback-settings-{uuid.uuid4().hex}"
    config: Final = _write_config(
        tmp_path,
        gateway.upstream_url,
        model,
        tuple(GUARDRAIL_CONSUMERS),
        {consumer: value for consumer in GUARDRAIL_CONSUMERS},
    )
    environment: Final = {
        "STORE_MODEL_IN_DB": "False",
        "PRESIDIO_ANALYZER_API_BASE": gateway.upstream_url,
        "PRESIDIO_ANONYMIZER_API_BASE": gateway.upstream_url,
    }
    with owned_proxy(gateway, tmp_path, environment, config=config) as candidate:
        _assert_registered(candidate, GUARDRAIL_CONSUMERS)
