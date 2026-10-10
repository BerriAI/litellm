"""Live e2e: every Decisions provider writes a correctly priced spend row from both routes.

/v1/decisions (the OpenAI Decisions body) and /v1/systemone (the System One body)
share one handler and one biller, and a decision model from any provider answers on
either route. Each cell registers a deployment on the real provider through
/model/new, asks it one yes/no question, and checks the SpendLogs row against the
response it got back: call_type names the route, the provider and model are
attributed, the token counts are the response's usage, the row's spend is that
usage priced at the rates /model/info reports for the deployment, and the
x-litellm-response-cost header agrees with the row. The rates are read back from
the proxy rather than pinned, so a cost-map price change never fails this test; a
biller that stops pricing the row, drops tokens, or logs the wrong call_type does.

OpenRouter has no cost-map row for its Decisions models and bills the cost
OpenRouter itself reports in usage, so its cells check that the header and the row
agree on a positive spend instead of the tokens-times-rates formula.
"""

from dataclasses import dataclass
from typing import Final, Literal

import pytest
from cost_rows import approx_equal
from e2e_config import unique_marker
from e2e_http import StreamingResponse
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import (
    CustomPricing,
    DecisionPredicateQuestion,
    DecisionsBody,
    DecisionsResponse,
    LiteLLMParamsBody,
    SpendLogRow,
    SystemOneBody,
    SystemOneNoulQuestion,
)
from proxy_client import ProxyClient
from spend_e2e_client import SpendClient

pytestmark = pytest.mark.e2e

INPUT: Final = "My order arrived broken and I want a refund today."
QUESTION: Final = "Is the customer asking for a refund?"


@dataclass(frozen=True, slots=True)
class DecisionProvider:
    provider: str
    model: str
    api_key_env: str
    billed_by: Literal["cost_map", "provider"]


PROVIDERS: Final = (
    DecisionProvider("openai", "openai/gpt-6-luna", "OPENAI_API_KEY", "cost_map"),
    DecisionProvider("typesafe", "typesafe/jev-latest", "TYPESAFE_API_KEY", "cost_map"),
    DecisionProvider("perplexity", "perplexity/pplx-decider-v1-27b", "PERPLEXITY_API_KEY", "cost_map"),
    DecisionProvider("cloudflare", "cloudflare/@cf/cloudflare/clef-flash", "CLOUDFLARE_API_KEY", "cost_map"),
    DecisionProvider("databricks", "databricks/databricks-openjev-qwen35-4b", "DATABRICKS_API_KEY", "cost_map"),
    DecisionProvider("openrouter", "openrouter/typesafe/jev-1.13", "OPENROUTER_API_KEY", "provider"),
)

provider_cases: Final = pytest.mark.parametrize(
    "decision_provider", PROVIDERS, ids=tuple(p.provider for p in PROVIDERS)
)
SUBJECT: Final = Subject(
    domain=Domain.SPEND_BUDGETS,
    route=Route.SPEND_REPORTING,
    providers=tuple(Provider(p.provider) for p in PROVIDERS),
    models=tuple(p.model for p in PROVIDERS),
    mode=Mode.NONSTREAM,
)


def _register(proxy: ProxyClient, resources: ResourceManager, decision_provider: DecisionProvider) -> str:
    model_name: Final = f"decisions-{decision_provider.provider}-{unique_marker()}"
    model_id: Final = proxy.create_model(
        model_name,
        LiteLLMParamsBody(model=decision_provider.model, api_key=f"os.environ/{decision_provider.api_key_env}"),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model_name


def _resolved_rates(proxy: ProxyClient, model_name: str) -> CustomPricing:
    rates: Final = next((entry.model_info for entry in proxy.model_info() if entry.model_name == model_name), None)
    assert rates is not None, f"{model_name} absent from /model/info"
    return rates


def _spend_row(proxy: ProxyClient, response: StreamingResponse) -> SpendLogRow:
    assert response.call_id, f"no x-litellm-call-id header on the response: {response.headers}"
    rows: Final = proxy.poll_logs_for_request_id(response.call_id)
    assert len(rows) == 1, f"expected one spend row for call {response.call_id}, got {len(rows)}"
    return rows[0]


def _header_cost(response: StreamingResponse) -> float:
    """The proxy drops the x-litellm-response-cost header on a call it priced at zero
    (the component headers still read 0.0), so a missing total is the proxy saying $0."""
    return 0.0 if response.response_cost is None else response.response_cost


def _assert_priced_row(
    proxy: ProxyClient,
    decision_provider: DecisionProvider,
    model_name: str,
    response: StreamingResponse,
    call_type: str,
) -> None:
    assert response.status_code == 200, f"{call_type} failed ({response.status_code}): {response.body[:300]}"
    usage: Final = DecisionsResponse.model_validate_json(response.body).usage
    header_cost: Final = _header_cost(response)

    row: Final = _spend_row(proxy, response)
    assert row.call_type == call_type, f"call_type {row.call_type!r}, expected {call_type!r}"
    assert row.custom_llm_provider == decision_provider.provider, f"provider attributed as {row.custom_llm_provider!r}"
    assert row.model == decision_provider.model, f"model attributed as {row.model!r}"
    assert (row.prompt_tokens, row.completion_tokens) == (usage.input_tokens, usage.output_tokens), (
        f"row tokens {row.prompt_tokens}/{row.completion_tokens} differ from the response usage "
        f"{usage.input_tokens}/{usage.output_tokens}"
    )
    assert row.spend is not None and approx_equal(row.spend, header_cost), (
        f"row spend {row.spend} differs from the x-litellm-response-cost header {response.response_cost}"
    )

    match decision_provider.billed_by:
        case "cost_map":
            rates: Final = _resolved_rates(proxy, model_name)
            expected: Final = rates.token_cost(usage.input_tokens, usage.output_tokens)
            assert approx_equal(row.spend, expected), (
                f"row spend {row.spend} is not {usage.input_tokens} input tokens x {rates.input_cost_per_token} + "
                f"{usage.output_tokens} output tokens x {rates.output_cost_per_token} = {expected}"
            )
        case "provider":
            assert row.spend > 0, f"provider-reported cost did not reach the row: spend {row.spend}"


class TestDecisionsSpend:
    @provider_cases
    @pytest.mark.covers("quota_management.spend_tracking.decisions.logs_cost")
    @meta(SUBJECT)
    def test_decisions_route_logs_usage_priced_at_deployment_rates(
        self,
        client: SpendClient,
        resources: ResourceManager,
        scoped_key: str,
        decision_provider: DecisionProvider,
    ) -> None:
        model_name: Final = _register(client.proxy, resources, decision_provider)
        response: Final = client.proxy.decisions(
            scoped_key,
            DecisionsBody(
                model=model_name,
                input=INPUT,
                questions=[DecisionPredicateQuestion(name="is_refund_request", instructions=QUESTION)],
            ),
        )
        _assert_priced_row(client.proxy, decision_provider, model_name, response, "adecisions")

    @provider_cases
    @pytest.mark.covers("quota_management.spend_tracking.systemone.logs_cost")
    @meta(SUBJECT)
    def test_systemone_route_logs_usage_priced_at_deployment_rates(
        self,
        client: SpendClient,
        resources: ResourceManager,
        scoped_key: str,
        decision_provider: DecisionProvider,
    ) -> None:
        model_name: Final = _register(client.proxy, resources, decision_provider)
        response: Final = client.proxy.systemone(
            scoped_key,
            SystemOneBody(
                model=model_name,
                state=INPUT,
                questions={"is_refund_request": SystemOneNoulQuestion(instructions=QUESTION)},
            ),
        )
        _assert_priced_row(client.proxy, decision_provider, model_name, response, "asystemone")
