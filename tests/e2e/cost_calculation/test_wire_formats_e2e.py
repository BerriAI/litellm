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
from models import ChatBody, ChatMessage, ChatStreamOptions
from scripted_provider import ScriptedUsage

pytestmark = [pytest.mark.e2e, pytest.mark.cost_map_stack]

_MODELS: dict[str, FrontierModel] = {model.map_key: model for model in FRONTIER_MODELS}

# One scripted usage per wire, every reportable token kind nonzero.
_WIRE_USAGE: dict[str, tuple[str, ScriptedUsage]] = {
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
}


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
        model = _MODELS[map_key]
        case = Case(name="basic", usage=usage)
        marker = unique_marker()
        model_name, _handle = register_scenario_deployment(client, resources, model, case, marker)
        response = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(scoped_key),
            json=ChatBody(
                model=model_name,
                messages=[ChatMessage(role="user", content=f"{marker} scripted wire call")],
            ),
        )
        assert response.ok, f"{wire}: proxy returned {response.status_code}: {response.body[:400]}"

        expected = expected_breakdown(model, case)
        row = cost_rows.poll_cost_row_where(
            client.proxy,
            scoped_key,
            lambda r: r.metadata is not None and r.metadata.cost_breakdown is not None,
        )
        assert row is not None, f"{wire}: no spend row landed"
        assert row.spend is not None and cost_rows.approx_equal(row.spend, expected.total), (
            f"{wire}: spend {row.spend} != expected {expected.total} "
            f"(breakdown {row.breakdown.model_dump()})"
        )
        breakdown = row.breakdown
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
        model = _MODELS[map_key]
        case = Case(name="stream", usage=usage, stream=True)
        marker = unique_marker()
        model_name, _handle = register_scenario_deployment(client, resources, model, case, marker)
        response = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(scoped_key),
            json=ChatBody(
                model=model_name,
                messages=[ChatMessage(role="user", content=f"{marker} scripted anthropic stream")],
                stream=True,
                stream_options=ChatStreamOptions(include_usage=True),
            ),
            stream=True,
        )
        assert response.ok, f"anthropic stream: proxy returned {response.status_code}: {response.body[:400]}"
        assert response.stream_done, "anthropic stream did not reach its terminal event"
        assert response.stream_error is None, f"stream carried an error event: {response.stream_error}"

        expected = expected_breakdown(model, case)
        row = cost_rows.poll_cost_row_where(
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
