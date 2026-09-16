from __future__ import annotations

from importlib import import_module
from typing import Final, cast

from ..models import TraceSuite


def _suite(module: str) -> TraceSuite:
    loaded: Final = import_module(module)
    candidate: Final = cast(object, getattr(loaded, "TRACE_SUITE"))
    assert isinstance(candidate, TraceSuite)
    return candidate


def test_core_sdk_scenario_matrix_keeps_distinct_migration_paths() -> None:
    chat: Final = _suite("tests.rust-python-harness.strategies.trace_parity.sdk.chat_completions.case")
    messages: Final = _suite("tests.rust-python-harness.strategies.trace_parity.sdk.messages.case")
    ocr: Final = _suite("tests.rust-python-harness.strategies.trace_parity.sdk.ocr.case")
    responses: Final = _suite("tests.rust-python-harness.strategies.trace_parity.sdk.responses.case")

    assert {(scenario.name, scenario.asynchronous) for scenario in chat.scenarios} >= {
        ("sync-anthropic", False),
        ("async-anthropic", True),
        ("sync-anthropic-stream", False),
        ("async-anthropic-stream", True),
        ("async-anthropic-provider-error", True),
        ("async-anthropic-stream-error", True),
        ("sync-bedrock", False),
        ("async-bedrock", True),
        ("sync-bedrock-event-stream", False),
        ("async-bedrock-event-stream", True),
    }
    assert {(scenario.name, scenario.asynchronous) for scenario in messages.scenarios} >= {
        ("async-anthropic-stream", True),
        ("async-azure-ai-stream", True),
        ("async-bedrock-event-stream", True),
        ("async-bedrock-event-stream-error", True),
        ("async-bedrock-invalid-thinking-retry", True),
        ("sync-unsupported", False),
    }
    assert {(scenario.name, scenario.asynchronous) for scenario in ocr.scenarios} >= {
        ("async-cohere", True),
        ("sync-public-rust-dispatch", False),
        ("async-public-rust-dispatch", True),
    }
    assert {(scenario.name, scenario.asynchronous) for scenario in responses.scenarios} >= {
        ("sync-openai", False),
        ("async-openai", True),
        ("sync-openai-stream", False),
        ("async-openai-stream", True),
        ("async-openai-provider-error", True),
        ("async-openai-stream-failed", True),
        ("async-azure", True),
        ("async-anthropic-chat-bridge", True),
        ("async-anthropic-chat-bridge-stream", True),
    }


def test_core_gateway_matrix_keeps_downstream_streams_separate() -> None:
    modules: Final = (
        "tests.rust-python-harness.strategies.trace_parity.gateway.chat_completions.case",
        "tests.rust-python-harness.strategies.trace_parity.gateway.messages.case",
        "tests.rust-python-harness.strategies.trace_parity.gateway.responses.case",
    )

    for module in modules:
        suite = _suite(module)
        assert any("downstream-stream" in scenario.name for scenario in suite.scenarios)
