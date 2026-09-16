"""Wire-format e2e: one scripted upstream per provider wire, answering with a
usage payload where every token kind the wire can report is nonzero. The spend
row's gross input cost must equal fresh tokens at the input rate plus each cache
and audio component at its own rate -- proving the wire's usage shape landed the
cached tokens inside the total (OpenAI/Gemini) or as separate fields
(Anthropic), and that the biller subtracted them before billing fresh tokens.

Also covers the Responses API wire (an openai/gpt-5.5-pro deployment bridged by
the proxy to POST /responses) and a streamed Anthropic-messages case.
"""

from __future__ import annotations

import pytest
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from conftest import CostCalcClient, cost_rows, register_scenario_deployment
from cost_matrix import (
    FRONTIER_MODELS,
    Case,
    FrontierModel,
    expected_breakdown,
    expected_token_columns,
)
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatStreamOptions, ChatTool, ChatToolFunction
from scripted_provider import ScriptedUsage

pytestmark: Final = [pytest.mark.e2e, pytest.mark.cost_map_stack]  # mutable-ok: pytest only accepts a list for pytestmark

_MODELS: Final[Mapping[str, FrontierModel]] = MappingProxyType(
    {model.map_key: model for model in FRONTIER_MODELS}
)

# One scripted usage per wire, every reportable token kind nonzero.
_WIRE_USAGE: Final[Mapping[str, tuple[str, ScriptedUsage]]] = MappingProxyType({
    "openai_chat": (
        "gpt-5.6",
        ScriptedUsage(
            fresh_input_tokens=80,
            cache_read_tokens=40,
            cache_write_5m_tokens=20,
            cache_write_1h_tokens=10,
            output_tokens=25,
            reasoning_tokens=15,
            audio_input_tokens=5,
            audio_output_tokens=3,
        ),
    ),
    "openai_responses": (
        "gpt-5.5-pro",
        ScriptedUsage(
            fresh_input_tokens=80, cache_read_tokens=40, output_tokens=25, reasoning_tokens=15
        ),
    ),
    "anthropic_messages": (
        "claude-sonnet-5",
        ScriptedUsage(
            fresh_input_tokens=80,
            cache_read_tokens=40,
            cache_write_5m_tokens=20,
            cache_write_1h_tokens=10,
            output_tokens=25,
        ),
    ),
    "gemini_generate": (
        "gemini/gemini-3.8-flash",
        ScriptedUsage(
            fresh_input_tokens=80,
            cache_read_tokens=40,
            output_tokens=25,
            reasoning_tokens=15,
            audio_input_tokens=5,
            audio_output_tokens=3,
        ),
    ),
    "together_chat": (
        "together_ai/moonshotai/Kimi-K3",
        ScriptedUsage(
            fresh_input_tokens=80,
            cache_read_tokens=40,
            cache_write_5m_tokens=20,
            cache_write_1h_tokens=10,
            output_tokens=25,
            reasoning_tokens=15,
            audio_input_tokens=5,
            audio_output_tokens=3,
        ),
    ),
    "fireworks_chat": (
        "fireworks_ai/kimi-k3",
        ScriptedUsage(fresh_input_tokens=80, cache_read_tokens=40, output_tokens=25),
    ),
    "azure_chat": (
        "azure/gpt-5.6",
        ScriptedUsage(
            fresh_input_tokens=80,
            cache_read_tokens=40,
            cache_write_5m_tokens=20,
            cache_write_1h_tokens=10,
            output_tokens=25,
            reasoning_tokens=15,
            audio_input_tokens=5,
            audio_output_tokens=3,
        ),
    ),
    "bedrock_converse": (
        "anthropic.claude-sonnet-5-v1:0",
        ScriptedUsage(
            fresh_input_tokens=80,
            cache_read_tokens=40,
            cache_write_5m_tokens=20,
            cache_write_1h_tokens=10,
            output_tokens=25,
        ),
    ),
    "vertex_generate": (
        "gemini-3.8-flash",
        ScriptedUsage(
            fresh_input_tokens=80,
            cache_read_tokens=40,
            output_tokens=25,
            reasoning_tokens=15,
            audio_input_tokens=5,
            audio_output_tokens=3,
        ),
    ),
})

_SHAPE_USAGE: Final = ScriptedUsage(fresh_input_tokens=80, output_tokens=25)

# Renderer-level shapes the pricing matrix gates per cap, pinned here once per
# wire so the sidecar emits prove they survive the proxy end to end.
_SHAPES: Final[tuple[tuple[str, str, Case], ...]] = (
    *(
        (
            f"tool_call_{'stream' if stream else 'sync'}",
            wire,
            Case(name="tool_call", usage=_SHAPE_USAGE, stream=stream, tool_call=True),
        )
        for wire in _WIRE_USAGE
        for stream in (False, True)
    ),
    (
        "responses_incomplete",
        "openai_responses",
        Case(name="stream_no_usage_incomplete", usage=_SHAPE_USAGE, stream=True, terminal="incomplete"),
    ),
    (
        "responses_unvalidated",
        "openai_responses",
        Case(name="stream_unvalidated", usage=_SHAPE_USAGE, stream=True, terminal="unvalidated"),
    ),
    (
        "gemini_prompt_blocked",
        "gemini_generate",
        Case(
            name="prompt_blocked",
            usage=ScriptedUsage(fresh_input_tokens=1000, output_tokens=0),
            terminal="prompt_blocked",
            response_model_override=True,
        ),
    ),
    (
        "gemini_prompt_blocked_stream",
        "gemini_generate",
        Case(
            name="stream_prompt_blocked",
            usage=ScriptedUsage(fresh_input_tokens=1000, output_tokens=0),
            stream=True,
            terminal="prompt_blocked",
            response_model_override=True,
        ),
    ),
    (
        "vertex_prompt_blocked",
        "vertex_generate",
        Case(
            name="prompt_blocked",
            usage=ScriptedUsage(fresh_input_tokens=1000, output_tokens=0),
            terminal="prompt_blocked",
            response_model_override=True,
        ),
    ),
    (
        "vertex_prompt_blocked_stream",
        "vertex_generate",
        Case(
            name="stream_prompt_blocked",
            usage=ScriptedUsage(fresh_input_tokens=1000, output_tokens=0),
            stream=True,
            terminal="prompt_blocked",
            response_model_override=True,
        ),
    ),
    (
        "azure_served_model_override",
        "azure_chat",
        Case(
            name="response_model_override",
            usage=_SHAPE_USAGE,
            response_model_override=True,
        ),
    ),
)


def _shape_id(entry: tuple[str, str, Case]) -> str:
    return entry[0]


class TestWireFormats:
    @pytest.mark.parametrize("wire", tuple(_WIRE_USAGE))
    @pytest.mark.covers("quota_management.spend_tracking.scripted_wire.logs_cost")
    def test_wire_usage_shape_bills_each_component(
        self,
        client: CostCalcClient,
        resources: ResourceManager,
        scoped_key: str,
        wire: str,
    ) -> None:
        map_key, usage = _WIRE_USAGE[wire]
        model: Final = _MODELS[map_key]
        case: Final = Case(name="basic", usage=usage)
        marker: Final = unique_marker()
        model_name, _handle = register_scenario_deployment(client, resources, model, case, marker)
        response: Final = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(scoped_key),
            json=ChatBody(
                model=model_name,
                messages=(ChatMessage(role="user", content=f"{marker} scripted wire call"),),
            ),
        )
        assert response.ok, f"{wire}: proxy returned {response.status_code}: {response.body[:400]}"

        expected: Final = expected_breakdown(model, case)
        row: Final = cost_rows.poll_cost_row_where(
            client.proxy,
            scoped_key,
            lambda r: r.metadata is not None and r.metadata.cost_breakdown is not None,
        )
        assert row is not None, f"{wire}: no spend row landed"
        assert row.spend is not None and cost_rows.approx_equal(row.spend, expected.total), (
            f"{wire}: spend {row.spend} != expected {expected.total} "
            f"(breakdown {row.breakdown.model_dump()})"
        )
        breakdown: Final = row.breakdown
        assert breakdown.input_cost is not None and cost_rows.approx_equal(
            breakdown.input_cost, expected.input_cost
        ), (
            f"{wire}: gross input_cost {breakdown.input_cost} != expected {expected.input_cost}; "
            "cached/written tokens billed at the input rate"
        )
        assert breakdown.output_cost is not None and cost_rows.approx_equal(
            breakdown.output_cost, expected.output_cost
        ), f"{wire}: output_cost {breakdown.output_cost} != expected {expected.output_cost}"

        prompt_tokens, completion_tokens = expected_token_columns(model, case)
        assert row.prompt_tokens == prompt_tokens, (
            f"{wire}: prompt_tokens {row.prompt_tokens} != {prompt_tokens}"
        )
        assert row.completion_tokens == completion_tokens, (
            f"{wire}: completion_tokens {row.completion_tokens} != {completion_tokens}"
        )
        cost_rows.assert_total_is_sum_of_components(row)

    @pytest.mark.covers("quota_management.spend_tracking.scripted_wire.logs_cost")
    def test_anthropic_streamed_usage_bills_each_component(
        self, client: CostCalcClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        map_key, usage = _WIRE_USAGE["anthropic_messages"]
        model: Final = _MODELS[map_key]
        case: Final = Case(name="stream", usage=usage, stream=True)
        marker: Final = unique_marker()
        model_name, _handle = register_scenario_deployment(client, resources, model, case, marker)
        response: Final = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(scoped_key),
            json=ChatBody(
                model=model_name,
                messages=(ChatMessage(role="user", content=f"{marker} scripted anthropic stream"),),
                stream=True,
                stream_options=ChatStreamOptions(include_usage=True),
            ),
            stream=True,
        )
        assert response.ok, f"anthropic stream: proxy returned {response.status_code}: {response.body[:400]}"
        assert response.stream_done, "anthropic stream did not reach its terminal event"
        assert response.stream_error is None, f"stream carried an error event: {response.stream_error}"

        expected: Final = expected_breakdown(model, case)
        row: Final = cost_rows.poll_cost_row_where(
            client.proxy,
            scoped_key,
            lambda r: r.metadata is not None and r.metadata.cost_breakdown is not None,
        )
        assert row is not None, "anthropic stream: no spend row landed"
        assert row.spend is not None and cost_rows.approx_equal(row.spend, expected.total), (
            f"anthropic stream: spend {row.spend} != expected {expected.total} "
            f"(breakdown {row.breakdown.model_dump()})"
        )
        cost_rows.assert_total_is_sum_of_components(row)

    @pytest.mark.parametrize("shape_wire_case", _SHAPES, ids=_shape_id)
    @pytest.mark.covers("quota_management.spend_tracking.scripted_wire.logs_cost")
    def test_response_shape_bills_reported_usage(
        self,
        client: CostCalcClient,
        resources: ResourceManager,
        scoped_key: str,
        shape_wire_case: tuple[str, str, Case],
    ) -> None:
        shape, wire, case = shape_wire_case
        map_key, _usage = _WIRE_USAGE[wire]
        model: Final = _MODELS[map_key]
        marker: Final = unique_marker()
        model_name, _handle = register_scenario_deployment(client, resources, model, case, marker)
        response: Final = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(scoped_key),
            json=ChatBody(
                model=model_name,
                messages=(ChatMessage(role="user", content=f"{marker} scripted {shape}"),),
                stream=case.stream,
                stream_options=ChatStreamOptions(include_usage=True) if case.stream else None,
                tools=(
                    (
                        ChatTool(
                            function=ChatToolFunction(
                                name="get_weather",
                                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                            )
                        ),
                    )
                    if case.tool_call
                    else None
                ),
            ),
            stream=case.stream,
        )
        assert response.ok, f"{shape}: proxy returned {response.status_code}: {response.body[:400]}"
        if case.stream:
            assert response.stream_done, f"{shape}: stream did not reach its terminal event"
        assert response.stream_error is None, f"{shape}: stream error: {response.stream_error}"

        expected: Final = expected_breakdown(model, case)
        row: Final = cost_rows.poll_cost_row_where(
            client.proxy,
            scoped_key,
            lambda r: r.metadata is not None and r.metadata.cost_breakdown is not None,
        )
        assert row is not None, f"{shape}: no spend row landed"
        assert row.spend is not None and cost_rows.approx_equal(row.spend, expected.total), (
            f"{shape}: spend {row.spend} != expected {expected.total} "
            f"(breakdown {row.breakdown.model_dump()})"
        )
        prompt_tokens, completion_tokens = expected_token_columns(model, case)
        assert row.prompt_tokens == prompt_tokens, (
            f"{shape}: prompt_tokens {row.prompt_tokens} != {prompt_tokens}"
        )
        assert row.completion_tokens == completion_tokens, (
            f"{shape}: completion_tokens {row.completion_tokens} != {completion_tokens}"
        )
        cost_rows.assert_total_is_sum_of_components(row)
