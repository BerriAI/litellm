"""Live e2e: a virtual key's per-model `budget_fallbacks` reroutes `/v1/messages`
transparently from an exhausted Anthropic model to an OpenAI model, instead of
blocking the caller with a `budget_exceeded` error. Coverage for the
budget_fallbacks feature in litellm/proxy/hooks/model_max_budget_limiter.py.
"""

import time

import pytest

from budget_client import BudgetClient, model_budget
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import AnthropicMessagesResponse

pytestmark = pytest.mark.e2e

PRIMARY_MODEL = "claude-haiku-4-5"
FALLBACK_MODEL = "gpt-5.5"


@pytest.mark.covers("quota_management.budget.fallback.routes_to_fallback")
def test_budget_fallback_reroutes_anthropic_messages_to_openai(
    client: BudgetClient, resources: ResourceManager
) -> None:
    key = client.generate_key(
        model_max_budget=model_budget(PRIMARY_MODEL, 1e-6),
        budget_fallbacks={PRIMARY_MODEL: [FALLBACK_MODEL]},
    )
    resources.defer(lambda: client.delete_key(key))

    # Exhaust the primary model's near-zero budget. Once exceeded, every
    # subsequent /v1/messages call for this key must reroute to the fallback
    # instead of surfacing a budget_exceeded block.
    served_by = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = client.messages(
            key, PRIMARY_MODEL, f"hi {unique_marker()}", max_tokens=16
        )
        if not result.ok:
            pytest.fail(
                "budget_fallbacks must reroute transparently, never surface a "
                f"block; status={result.status_code} body={result.body[:300]}"
            )
        served_by = AnthropicMessagesResponse.model_validate_json(result.body).model
        if served_by is not None and FALLBACK_MODEL in served_by:
            break
        time.sleep(1)
    assert served_by is not None and FALLBACK_MODEL in served_by, (
        f"{PRIMARY_MODEL}'s budget_fallbacks never rerouted to {FALLBACK_MODEL}"
    )

    # The rerouted call must be recorded under the fallback model, not the
    # exhausted primary - proving spend tracking followed the reroute.
    rows = client.proxy.poll_logs_for_key(
        key, predicate=lambda rows: any(FALLBACK_MODEL in (r.model or "") for r in rows)
    )
    assert any(FALLBACK_MODEL in (r.model or "") for r in rows), (
        f"no spend log recorded against {FALLBACK_MODEL} after the reroute"
    )
