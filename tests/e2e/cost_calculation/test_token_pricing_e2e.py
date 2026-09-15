"""Token-pricing e2e: every (frontier model, pricing-component case) cell runs a
scripted-usage call through a deployment registered on the cost-map proxy, and
the spend row plus response-cost header must equal literal arithmetic on the
test map's rates.

Nothing here touches a real provider or the bundled cost map: the proxy's
upstream is the scripted-provider sidecar and its entire cost map is
tests/e2e/cost_map.json.
"""

from __future__ import annotations

import pytest
from typing import Final

from conftest import CostCalcClient, cost_rows, register_scenario_deployment
from cost_matrix import (
    FRONTIER_MODELS,
    Case,
    FrontierModel,
    cases_for,
    expected_cost,
    expected_token_columns,
)
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatStreamOptions

pytestmark: Final = [pytest.mark.e2e, pytest.mark.cost_map_stack]  # mutable-ok: pytest only accepts a list for pytestmark

_MATRIX: Final[tuple[tuple[FrontierModel, Case], ...]] = tuple(
    (model, case) for model in FRONTIER_MODELS for case in cases_for(model)
)


def _case_id(param: tuple[FrontierModel, Case]) -> str:
    model, case = param
    return f"{model.map_key.replace('/', '-')}-{case.name}"


def _chat_body(model_name: str, marker: str, case: Case) -> ChatBody:
    return ChatBody(
        model=model_name,
        messages=(ChatMessage(role="user", content=f"{marker} scripted pricing call"),),
        stream=case.stream,
        stream_options=ChatStreamOptions(include_usage=True) if case.stream else None,
        service_tier=case.service_tier,
    )


class TestTokenPricing:
    @pytest.mark.parametrize("model_case", _MATRIX, ids=_case_id)
    @pytest.mark.covers("quota_management.spend_tracking.cost_matrix.logs_cost")
    def test_scripted_usage_bills_at_map_rates(
        self,
        client: CostCalcClient,
        resources: ResourceManager,
        scoped_key: str,
        model_case: tuple[FrontierModel, Case],
    ) -> None:
        model, case = model_case
        marker: Final = unique_marker()
        model_name, _handle = register_scenario_deployment(client, resources, model, case, marker)
        response: Final = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(scoped_key),
            json=_chat_body(model_name, marker, case),
            stream=case.stream,
        )
        assert response.ok, (
            f"{model.map_key}/{case.name}: proxy returned {response.status_code}: {response.body[:400]}"
        )
        assert response.stream_error is None, f"stream carried an error event: {response.stream_error}"

        expected: Final = expected_cost(model, case)
        if case.exact_spend and not case.stream:
            # Streamed responses commit headers before the bill is computed, so
            # the x-litellm-response-cost header is asserted only on non-stream
            # calls.
            assert response.response_cost is not None and cost_rows.approx_equal(
                response.response_cost, expected
            ), (
                f"x-litellm-response-cost {response.response_cost} != expected {expected}"
            )

        row: Final = cost_rows.poll_cost_row_where(
            client.proxy,
            scoped_key,
            lambda r: r.metadata is not None and r.metadata.cost_breakdown is not None,
        )
        assert row is not None, f"no spend row with a cost breakdown landed for {model.map_key}/{case.name}"

        if not case.exact_spend and case.expect_zero_bill:
            # The provider reported no usage and this wire has no proxy-side
            # recount, so the bill is exactly zero.
            assert row.spend is not None and row.spend == 0, f"no-usage stream billed {row.spend}: {row}"
            return
        if not case.exact_spend:
            # stream_usage=absent: the provider reported no usage, so the row's
            # token counts are the proxy's own recount; only assert a bill landed.
            assert row.spend is not None and row.spend > 0, f"no-usage stream billed nothing: {row}"
            return

        assert row.spend is not None and cost_rows.approx_equal(row.spend, expected), (
            f"{model.map_key}/{case.name}: spend {row.spend} != expected {expected} "
            f"(breakdown {row.breakdown.model_dump()})"
        )

        prompt_tokens, completion_tokens = expected_token_columns(model, case)
        assert row.prompt_tokens == prompt_tokens, (
            f"prompt_tokens {row.prompt_tokens} != {prompt_tokens}"
        )
        assert row.completion_tokens == completion_tokens, (
            f"completion_tokens {row.completion_tokens} != {completion_tokens}"
        )
        cost_rows.assert_total_is_sum_of_components(row)
