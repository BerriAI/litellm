import json
import uuid
from typing import Final

import pytest

from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows


@pytest.mark.covers("other.routing.auto_router.semantic_traffic_lands_in_session_rollup_and_benchmarks")
def test_semantic_auto_router_traffic_is_counted_by_the_benchmarks_endpoint(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        default_model: Final = scenario.model(model="openai/gpt-4o-mini", input_cost_per_token=0.001, output_cost_per_token=0.002)
        embedding_model: Final = scenario.model(model="openai/text-embedding-3-small", mode="embedding")
        routes: Final = json.dumps({"routes": [{"name": default_model, "utterances": ["write a poem about the sea"], "score_threshold": 0.3}]})
        router_alias: Final = scenario.model(
            model="auto_router/integration-" + uuid.uuid4().hex,
            auto_router_config=routes,
            auto_router_default_model=default_model,
            auto_router_embedding_model=embedding_model,
        )
        session_id: Final = "semantic-session-" + uuid.uuid4().hex

        for turn in range(2):
            body: Final = gateway.post(
                "/v1/chat/completions",
                {"model": router_alias, "messages": [{"role": "user", "content": f"semantic turn {turn}"}], "litellm_session_id": session_id},
            )
            assert body["model"] is not None, body

        rows: Final = eventually(
            lambda: read_rows(
                'SELECT router_name, router_type, turns FROM "LiteLLM_AutoRouterSession" WHERE session_id=%s',
                (session_id,),
            ),
            lambda values: len(values) == 1 and values[0]["turns"] == 2,
            seconds=70,
        )
        assert rows[0]["router_name"] == router_alias and rows[0]["router_type"] == "semantic", rows

        groups: Final = gateway.get("/auto_router/benchmarks")["groups"]
        assert isinstance(groups, list)
        group: Final = next(object_value(entry) for entry in groups if object_value(entry)["router_name"] == router_alias)
        assert group["router_type"] == "semantic", group
        assert (group["sessions"], group["turns"]) == (1, 2), group
        assert float(str(group["spend"])) > 0, group
