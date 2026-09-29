from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from _openinference_support import (
    CHAT_TOOLS,
    _assert_chat_request,
    _chat_caller_response,
    _chat_response,
    _json_messages,
    _json_object,
    _matching_genai_marker_span,
    _matching_marker_span,
    _rig,
)
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from pydantic import JsonValue

_GENAI_B3_KEYS_WITHOUT_BAGGAGE: Final = frozenset(
    {
        "gen_ai.input.messages",
        "gen_ai.operation.name",
        "gen_ai.output.messages",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.response.finish_reasons",
        "gen_ai.response.id",
        "gen_ai.response.model",
        "gen_ai.system",
        "gen_ai.tool.0.description",
        "gen_ai.tool.0.name",
        "gen_ai.tool.0.parameters",
        "gen_ai.usage.completion_tokens",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.prompt_tokens",
        "gen_ai.usage.total_tokens",
        "litellm.api_key.hash",
        "litellm.call_id",
        "litellm.call_type",
        "litellm.cost.discount_amount",
        "litellm.cost.discount_percent",
        "litellm.cost.input",
        "litellm.cost.margin_fixed_amount",
        "litellm.cost.margin_percent",
        "litellm.cost.margin_total_amount",
        "litellm.cost.original",
        "litellm.cost.output",
        "litellm.cost.tool_usage",
        "litellm.cost.total",
        "litellm.provider.model",
        "litellm.request.route",
        "litellm.request.tools.declared",
        "llm.request.functions.0.description",
        "llm.request.functions.0.name",
        "llm.request.functions.0.parameters",
        "server.address",
        "server.port",
    }
)
_GENAI_B3_KEYS_WITH_BAGGAGE: Final = _GENAI_B3_KEYS_WITHOUT_BAGGAGE | frozenset({"litellm.metadata.trace_marker"})
_LANGFUSE_B3_KEYS: Final = frozenset(
    {
        "gen_ai.input.messages",
        "gen_ai.operation.name",
        "gen_ai.output.messages",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.response.finish_reasons",
        "gen_ai.response.id",
        "gen_ai.response.model",
        "gen_ai.system",
        "gen_ai.tool.0.description",
        "gen_ai.tool.0.name",
        "gen_ai.tool.0.parameters",
        "gen_ai.usage.completion_tokens",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.prompt_tokens",
        "gen_ai.usage.total_tokens",
        "langfuse.observation.cost_details",
        "langfuse.observation.id",
        "langfuse.observation.input",
        "langfuse.observation.metadata.provider",
        "langfuse.observation.model.name",
        "langfuse.observation.output",
        "langfuse.observation.type",
        "langfuse.observation.usage_details",
        "litellm.api_key.hash",
        "litellm.call_id",
        "litellm.call_type",
        "litellm.cost.discount_amount",
        "litellm.cost.discount_percent",
        "litellm.cost.input",
        "litellm.cost.margin_fixed_amount",
        "litellm.cost.margin_percent",
        "litellm.cost.margin_total_amount",
        "litellm.cost.original",
        "litellm.cost.output",
        "litellm.cost.tool_usage",
        "litellm.cost.total",
        "litellm.provider.model",
        "litellm.request.route",
        "litellm.request.tools.declared",
        "llm.request.functions.0.description",
        "llm.request.functions.0.name",
        "llm.request.functions.0.parameters",
        "server.address",
        "server.port",
    }
)
_B3_ATTRIBUTE_KEYS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "langfuse_otel": _LANGFUSE_B3_KEYS,
        "langtrace": _GENAI_B3_KEYS_WITHOUT_BAGGAGE,
        "signoz": _GENAI_B3_KEYS_WITH_BAGGAGE,
        "newrelic": _GENAI_B3_KEYS_WITH_BAGGAGE,
        "levo": _GENAI_B3_KEYS_WITH_BAGGAGE,
        "agentops": _GENAI_B3_KEYS_WITH_BAGGAGE,
        "otel": _GENAI_B3_KEYS_WITH_BAGGAGE,
    }
)
_B3_BAGGAGE_CALLBACKS: Final = frozenset({"signoz", "newrelic", "levo", "agentops", "otel"})
_B4_ATTRIBUTE_KEYS: Final = frozenset(
    {
        "input.value",
        "litellm.trace_id",
        "llm.cost.total",
        "llm.input_messages.0.message.content",
        "llm.input_messages.0.message.role",
        "llm.invocation_parameters",
        "llm.is_streaming",
        "llm.model_name",
        "llm.output_messages.0.message.content",
        "llm.output_messages.0.message.role",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name",
        "llm.output_messages.0.message.tool_calls.0.tool_call.id",
        "llm.provider",
        "llm.request.type",
        "llm.response.cost",
        "llm.response.id",
        "llm.response.model",
        "llm.token_count.completion",
        "llm.token_count.prompt",
        "llm.token_count.total",
        "llm.tools.0.description",
        "llm.tools.0.name",
        "llm.tools.0.parameters",
        "metadata",
        "openinference.span.kind",
        "output.value",
        "user.id",
    }
)


def _request(proxy: Gateway, model: str, marker: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": CHAT_TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
            "metadata": {"trace_marker": marker},
            "cache": {"no-cache": True},
        },
    )


def _assert_openinference(attributes: dict[str, str], marker: str) -> None:
    assert attributes["llm.output_messages.0.message.tool_calls.0.tool_call.id"] == f"call_{marker}", attributes
    assert attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == "lookup_weather", (
        attributes
    )
    assert (
        attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"] == '{"city": "Paris"}'
    ), attributes
    assert _json_object(attributes["metadata"].encode()) == {"trace_marker": marker}, attributes
    assert attributes["litellm.metadata.trace_marker"] == marker, attributes
    assert _json_messages(attributes["output.value"]) == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{marker}",
                    "type": "function",
                    "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        }
    ], attributes


def test_arize_otel_v2_b1_phoenix_openinference(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "b1-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])
        return _chat_response(marker)

    with _rig(
        gateway,
        tmp_path,
        upstream,
        callbacks=("arize_phoenix",),
        callback_settings={"otel": {"exporter": "http/protobuf", "endpoint": "unused"}},
        environment={"PHOENIX_PROJECT_NAME": "integration"},
    ) as rig:
        response: Final = _request(rig.proxy, rig.model, marker)
        assert response.status_code == 200, response.text
        assert _json_object(response.content) == _chat_caller_response(_chat_response(marker), rig.model), response.text
        _assert_openinference(_matching_marker_span(rig.destination, marker), marker)


def test_arize_otel_v2_b2_weave_openinference(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "b2-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])
        return _chat_response(marker)

    with _rig(
        gateway,
        tmp_path,
        upstream,
        callbacks=("weave_otel",),
        callback_settings={"otel": {"exporter": "http/protobuf", "endpoint": "unused"}},
        environment={"WANDB_BASE_URL": "http://127.0.0.1"},
    ) as rig:
        response: Final = _request(rig.proxy, rig.model, marker)
        assert response.status_code == 200, response.text
        assert _json_object(response.content) == _chat_caller_response(_chat_response(marker), rig.model), response.text
        _assert_openinference(_matching_marker_span(rig.destination, marker), marker)


@pytest.mark.parametrize(
    "callback",
    ("langfuse_otel", "langtrace", "signoz", "newrelic", "levo", "agentops", "otel"),
)
def test_arize_otel_v2_b3_non_openinference_callback_family(callback: str, gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"b3-{callback}-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        body: Final = _json_object(request.body)
        assert body == {
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "model": "gpt-4o-mini",
            "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
            "tools": CHAT_TOOLS,
        }, body
        return _chat_response(marker)

    callback_settings: Final[dict[str, JsonValue]] = {
        "otel": {"exporter": "http/protobuf", "endpoint": "unused", "mapper_names": ["genai"]}
    }
    with _rig(
        gateway,
        tmp_path,
        upstream,
        callbacks=(callback,),
        callback_settings=callback_settings,
        environment={
            "LANGFUSE_HOST": "http://127.0.0.1",
            **(
                {
                    "HTTPS_PROXY": "http://127.0.0.1:0",
                    "https_proxy": "http://127.0.0.1:0",
                    "NO_PROXY": "",
                    "no_proxy": "",
                }
                if callback == "agentops"
                else {}
            ),
        },
        remove_environment=(
            ("NEW_RELIC_LICENSE_KEY",)
            if callback == "newrelic"
            else ("AGENTOPS_API_KEY",)
            if callback == "agentops"
            else ()
        ),
    ) as rig:
        response: Final = _request(rig.proxy, rig.model, marker)
        assert response.status_code == 200, response.text
        assert _json_object(response.content) == _chat_caller_response(_chat_response(marker), rig.model), response.text
        attributes: Final = _matching_genai_marker_span(rig.destination, marker)
        assert frozenset(attributes) == _B3_ATTRIBUTE_KEYS[callback], attributes
        assert "metadata" not in attributes, attributes
        assert not any(".tool_calls." in key for key in attributes), attributes
        baggage: Final = tuple(
            sorted((key, value) for key, value in attributes.items() if key.startswith("litellm.metadata."))
        )
        expected_baggage: Final = (
            (("litellm.metadata.trace_marker", marker),) if callback in _B3_BAGGAGE_CALLBACKS else ()
        )
        assert baggage == expected_baggage, attributes


def test_arize_otel_v2_b4_legacy_otel_is_unchanged(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "b4-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        body: Final = _json_object(request.body)
        assert body == {
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "model": "gpt-4o-mini",
            "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
            "tools": CHAT_TOOLS,
        }, body
        return _chat_response(marker)

    with _rig(
        gateway,
        tmp_path,
        upstream,
        callbacks=("arize",),
        callback_settings={"otel": {"exporter": "http/protobuf", "endpoint": "unused"}},
        remove_environment=("LITELLM_OTEL_V2",),
        disabled_environment=("LITELLM_OTEL_V2",),
    ) as rig:
        response: Final = _request(rig.proxy, rig.model, marker)
        assert response.status_code == 200, response.text
        assert _json_object(response.content) == _chat_caller_response(_chat_response(marker), rig.model), response.text
        attributes: Final = _matching_marker_span(rig.destination, marker)
        assert frozenset(attributes) == _B4_ATTRIBUTE_KEYS, attributes
