from collections.abc import Iterator, Mapping
from typing import Final
from pathlib import Path
import uuid

import pytest
import yaml

from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows


@pytest.mark.covers("quota_management.spend_tracking.custom_price.matches_input_rates")
def test_custom_price_is_reported_and_charged(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "price control"}]}
        )
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(20 * 0.001 + 20 * 0.002)
        entries: Final = gateway.get("/model/info")["data"]
        assert isinstance(entries, list)
        matching: Final = tuple(object_value(entry) for entry in entries if object_value(entry)["model_name"] == model)
        assert len(matching) == 1
        params: Final = object_value(matching[0]["litellm_params"])
        assert params["input_cost_per_token"] == 0.001
        assert params["output_cost_per_token"] == 0.002


@pytest.mark.covers("quota_management.spend_tracking.default_prices.survive_nullable_sibling_reload")
def test_default_prices_survive_nullable_sibling_and_reload(gateway: Gateway) -> None:
    for registration_order in (("custom", "omitted", "nullable"), ("nullable", "omitted", "custom")):
        with gateway.scenario() as scenario:
            configured: Final = {
                "custom": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002},
                "omitted": {},
                "nullable": {"input_cost_per_token": None, "output_cost_per_token": None},
            }
            rates: Final = {
                "custom": (0.001, 0.002),
                "omitted": (0.00000015, 0.0000006),
                "nullable": (0.00000015, 0.0000006),
            }
            models: Final = {kind: scenario.model(**configured[kind]) for kind in registration_order}

            def observe_requests(
                registration_order: tuple[str, ...],
                models: Mapping[str, str],
                rates: Mapping[str, tuple[float, float]],
            ) -> Iterator[tuple[str, float]]:
                for generation in range(2):
                    entries: Final = gateway.get("/model/info")["data"]
                    assert isinstance(entries, list)
                    kinds: Final = tuple(reversed(registration_order)) if generation else registration_order
                    for index, kind in enumerate(kinds):
                        model: Final = models[kind]
                        target: Final = next(
                            object_value(entry) for entry in entries if object_value(entry)["model_name"] == model
                        )
                        info: Final = object_value(target["model_info"])
                        assert info["input_cost_per_token"] == rates[kind][0]
                        assert info["output_cost_per_token"] == rates[kind][1]
                        response: Final = gateway.request(
                            "POST",
                            "/v1/chat/completions",
                            {
                                "model": model,
                                "messages": [
                                    {"role": "user", "content": f"price {generation * len(registration_order) + index}"}
                                ],
                            },
                        )
                        assert response.status_code == 200, response.text
                        expected: Final = 20 * rates[kind][0] + 20 * rates[kind][1]
                        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected, rel=1e-6)
                        request_id: Final = string_value(object_value(response.json())["id"])
                        yield request_id, expected
                    target: Final = next(
                        object_value(entry)
                        for entry in entries
                        if object_value(entry)["model_name"] == models["nullable"]
                    )
                    identity: Final = string_value(object_value(target["model_info"])["id"])
                    updated: Final = gateway.request(
                        "PATCH", f"/model/{identity}/update", {"model_info": {"description": "reload price contract"}}
                    )
                    assert updated.status_code == 200, updated.text

            observations: Final = tuple(observe_requests(registration_order, models, rates))
            for request_id, expected in observations:
                rows: Final = eventually(
                    lambda request_id=request_id: read_rows(
                        'SELECT request_id, spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" '
                        "WHERE request_id = %s",
                        (request_id,),
                    ),
                    lambda values: len(values) == 1,
                    seconds=70,
                )
                assert rows[0]["prompt_tokens"] == 20
                assert rows[0]["completion_tokens"] == 20
                assert float(rows[0]["spend"]) == pytest.approx(expected, rel=1e-6)


@pytest.mark.covers("quota_management.spend_tracking.default_prices.loaded_router_preserves_cached_defaults")
def test_loaded_router_preserves_cached_defaults_during_real_requests(gateway: Gateway, tmp_path: Path) -> None:
    from litellm import Router

    aliases: Final = tuple(f"pricing-{uuid.uuid4().hex}" for _ in range(3))
    path: Final = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    {
                        "model_name": alias,
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                            "api_key": "integration-provider-key",
                            "api_base": f"{gateway.upstream_url}/v1",
                        },
                        "model_info": {"id": alias, **pricing},
                    }
                    for alias, pricing in zip(
                        aliases, ({}, {"input_cost_per_token": None, "output_cost_per_token": None}, {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}), strict=True
                    )
                ]
            }
        )
    )
    for reverse in (False, True):
        configured: Final = yaml.safe_load(path.read_text())["model_list"]
        router: Final = Router(model_list=list(reversed(configured)) if reverse else configured, num_retries=0)
        try:
            for alias in (*aliases, *reversed(aliases)):
                result: Final = router.completion(
                    model=alias, messages=[{"role": "user", "content": "router price control"}]
                )
                assert result.usage.prompt_tokens == 20
                assert result.usage.completion_tokens == 20
                expected_cost: Final = 0.0 if alias == aliases[2] else 20 * 0.00000015 + 20 * 0.0000006
                assert result._hidden_params["response_cost"] == pytest.approx(expected_cost, rel=1e-6)
                deployment: Final = router.get_deployment(model_id=alias)
                assert deployment is not None
                info: Final = router.get_router_model_info(deployment=deployment, received_model_name=alias)
                assert info["input_cost_per_token"] == (0.0 if alias == aliases[2] else 0.00000015)
                assert info["output_cost_per_token"] == (0.0 if alias == aliases[2] else 0.0000006)
        finally:
            router.reset()
