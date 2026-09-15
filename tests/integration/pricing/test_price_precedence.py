import json
import uuid
from typing import Final

import pytest
from hypothesis import Phase, example, given, settings, strategies as st

from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows


@pytest.mark.covers("quota_management.spend_tracking.price_precedence.zero_and_default_rates")
@pytest.mark.timeout(180)
def test_generated_zero_null_and_omitted_prices_follow_independent_arithmetic(gateway: Gateway) -> None:
    @settings(max_examples=20, deadline=None, database=None, phases=(Phase.explicit, Phase.generate, Phase.shrink))
    @example(rates=(0, 0))
    @example(rates=(1, 2))
    @example(rates=("null", "null"))
    @given(
        rates=st.one_of(
            st.sampled_from((("omitted", "omitted"), ("null", "null"))),
            st.tuples(st.integers(0, 25), st.integers(0, 25)),
        )
    )
    def check(rates: tuple[str | int, str | int]) -> None:
        defaults: Final = rates[0] in ("omitted", "null")
        assert defaults or (isinstance(rates[0], int) and isinstance(rates[1], int))
        input_rate, output_rate = (
            (0.00000015, 0.0000006) if defaults else (float(rates[0]) / 1_000_000, float(rates[1]) / 1_000_000)
        )
        parameters: Final = (
            {}
            if rates[0] == "omitted"
            else {
                "input_cost_per_token": None if rates[0] == "null" else input_rate,
                "output_cost_per_token": None if rates[0] == "null" else output_rate,
            }
        )
        with gateway.scenario() as scenario:
            model: Final = scenario.model(**parameters)
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": f"independent price {uuid.uuid4().hex}"}],
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["usage"] == {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40}
            expected: Final = 20 * input_rate + 20 * output_rate
            if expected:
                assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected, rel=1e-6)
            else:
                assert response.headers.get("x-litellm-response-cost") in (None, "0", "0.0")
            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT spend, metadata, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                    (response.json()["id"],),
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
            assert rows[0]["prompt_tokens"] == 20 and rows[0]["completion_tokens"] == 20
            assert float(rows[0]["spend"]) == pytest.approx(expected, rel=1e-6)
            metadata: Final = rows[0]["metadata"]
            parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
            breakdown: Final = object_value(parsed["cost_breakdown"])
            assert float(breakdown["input_cost"]) == pytest.approx(20 * input_rate, rel=1e-6)
            assert float(breakdown["output_cost"]) == pytest.approx(20 * output_rate, rel=1e-6)

    check()


@pytest.mark.covers("quota_management.spend_tracking.alias_prices.remain_independent_on_reload")
def test_same_upstream_aliases_keep_distinct_prices_after_reload(gateway: Gateway) -> None:
    for order in (("free", "paid"), ("paid", "free")):
        with gateway.scenario() as scenario:
            rates: Final = {
                "free": {"input_cost_per_token": 0, "output_cost_per_token": 0},
                "paid": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.003},
            }
            aliases: Final = {kind: scenario.model(**rates[kind]) for kind in order}
            for generation in range(2):
                for kind in order if generation == 0 else reversed(order):
                    model: Final = aliases[kind]
                    cost: Final = 0.08 if kind == "paid" else 0.0
                    response: Final = gateway.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": model,
                            "messages": [{"role": "user", "content": f"alias price {model} {generation}"}],
                        },
                    )
                    assert response.status_code == 200, response.text
                    assert response.json()["usage"]["total_tokens"] == 40
                    if cost:
                        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(cost)
                    rows: Final = eventually(
                        lambda response=response: read_rows(
                            'SELECT spend, metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                            (response.json()["id"],),
                        ),
                        lambda values: len(values) == 1,
                        seconds=70,
                    )
                    assert float(rows[0]["spend"]) == pytest.approx(cost)
                    metadata: Final = rows[0]["metadata"]
                    parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
                    breakdown: Final = object_value(parsed["cost_breakdown"])
                    assert float(breakdown["input_cost"]) == pytest.approx(20 * rates[kind]["input_cost_per_token"])
                    assert float(breakdown["output_cost"]) == pytest.approx(20 * rates[kind]["output_cost_per_token"])
                if generation == 0:
                    entries: Final = gateway.get("/model/info")["data"]
                    target: Final = next(entry for entry in entries if entry["model_name"] == aliases["paid"])
                    changed: Final = gateway.request(
                        "PATCH",
                        f"/model/{target['model_info']['id']}/update",
                        {"model_info": {"description": "price reload"}},
                    )
                    assert changed.status_code == 200, changed.text
