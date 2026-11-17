import importlib.util
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Final
from unittest.mock import MagicMock

import pytest
import respx

import litellm
from litellm import completion


def test_wandb_logging():
    try:
        response = completion(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hi 👋 - i'm claude"}],
            max_tokens=10,
            temperature=0.2,
        )
        print(response)
    except litellm.Timeout as e:
        pass
    except Exception as e:
        print(e)


@dataclass(frozen=True)
class _TraceResult:
    inputs: Mapping[str, object]
    outputs: Mapping[str, object]


@dataclass(frozen=True)
class _TraceSpan:
    name: str
    attributes: Mapping[str, object]
    start_time_ms: int
    end_time_ms: int
    span_kind: str
    results: Sequence[_TraceResult]


@dataclass(frozen=True)
class _TraceTree:
    root_span: _TraceSpan
    model_dict: Mapping[str, object]


def _weights_biases_with_stub_trace_tree(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    data_types: Final = ModuleType("wandb.sdk.data_types")
    data_types.trace_tree = SimpleNamespace(
        Result=_TraceResult, Span=_TraceSpan, WBTraceTree=_TraceTree, SpanKind=SimpleNamespace(LLM="LLM")
    )
    for name in ("wandb", "wandb.sdk"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "wandb.sdk.data_types", data_types)
    spec: Final = importlib.util.find_spec("litellm.integrations.weights_biases")
    assert spec is not None and spec.loader is not None
    module: Final = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_wandb_chat_completion_trace_lists_each_request_message_with_its_role(monkeypatch: pytest.MonkeyPatch) -> None:
    weights_biases: Final = _weights_biases_with_stub_trace_tree(monkeypatch)
    request: Final = {
        "model": "gpt-4o",
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
    }
    response: Final = {
        "object": "chat.completion",
        "model": "gpt-4o",
        "created": 1767225600,
        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
    }

    trace: Final = weights_biases.OpenAIRequestResponseResolver()(request, response, 1.5)

    assert trace.root_span.results == [
        _TraceResult(
            inputs={"request": "\n\n**system**: Be brief.\n\n\n**user**: Hi\n"},
            outputs={"response": "\n\n**assistant**: Hello!\n"},
        )
    ]
    assert trace.root_span.end_time_ms - trace.root_span.start_time_ms == 1500
