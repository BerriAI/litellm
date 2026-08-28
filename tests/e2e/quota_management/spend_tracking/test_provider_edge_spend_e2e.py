"""The provider-edge demonstrator: one spend-tracking flow wired through the
record/replay edge (LIT-5745).

This is the reference for wiring a suite to the edge: register a deployment
whose ``api_base`` comes from ``e2e_config.provider_edge_base``, then exercise
the proxy exactly as a live test would. In live mode the base is None and the
deployment talks to the real provider; in record mode it talks through the
local edge, which forwards to the provider and captures the exchange; in
replay mode the same test drives the REAL proxy and REAL database on the
recorded provider traffic alone, so key auth, routing, and the spend-log
write path are all still under test with zero provider calls.
"""

import pytest

from e2e_config import CHEAP_OPENAI_MODEL, provider_edge_base
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from spend_e2e_client import SpendClient, unique_marker, unwrap

pytestmark = [pytest.mark.e2e, pytest.mark.replayable]


@pytest.mark.covers("quota_management.spend_tracking.chat_completions.logs_cost")
def test_edge_wired_chat_writes_nonzero_spend_row(
    client: SpendClient, resources: ResourceManager, scoped_key: str
) -> None:
    base = provider_edge_base("openai")
    model = f"e2e-edge-openai-{unique_marker()}"
    model_id = client.proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=f"openai/{CHEAP_OPENAI_MODEL}",
            api_key="os.environ/OPENAI_API_KEY",
            api_base=None if base is None else f"{base}/v1",
        ),
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))

    chat = unwrap(
        client.chat(scoped_key, model, f"reply with one word {unique_marker()}", max_tokens=16)
    )
    assert chat.id

    rows = client.poll_logs_for_key(
        scoped_key, predicate=lambda rs: any((r.spend or 0) > 0 for r in rs)
    )
    matching = [row for row in rows if row.request_id == chat.id]
    assert matching, f"no SpendLogs row for request_id {chat.id}; saw {len(rows)} row(s)"
    assert (matching[0].spend or 0) > 0, f"spend row for {chat.id} has zero spend"
