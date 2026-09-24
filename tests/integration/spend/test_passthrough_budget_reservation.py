import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse
from pydantic import JsonValue

INPUT_COST_PER_TOKEN: Final = 0.000001
OUTPUT_COST_PER_TOKEN: Final = 0.001
PROMPT_TOKENS: Final = 10
CANDIDATE_TOKENS: Final = 5
COST_PER_CALL: Final = PROMPT_TOKENS * INPUT_COST_PER_TOKEN + CANDIDATE_TOKENS * OUTPUT_COST_PER_TOKEN
MAX_BUDGET: Final = 0.02
CALLS_WITHIN_BUDGET: Final = 4


def _key_spend(digest: str) -> float:
    rows: Final = read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,))
    assert len(rows) == 1, rows
    return float(rows[0]["spend"])


def _generate_content_request(model: str) -> dict[str, JsonValue]:
    return {"contents": [{"role": "user", "parts": [{"text": f"budget {model}"}]}]}


def _generate_content_response(model: str) -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "candidates": [
                {
                    "content": {"parts": [{"text": f"scripted answer {model}"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": PROMPT_TOKENS,
                "candidatesTokenCount": CANDIDATE_TOKENS,
                "totalTokenCount": PROMPT_TOKENS + CANDIDATE_TOKENS,
            },
            "modelVersion": model,
        },
    )


def _served_call(gateway: Gateway, model: str, key: str, scenario_id: str, call: int) -> None:
    digest: Final = sha256(key.encode()).hexdigest()
    spend_before: Final = _key_spend(digest)
    assert spend_before == pytest.approx((call - 1) * COST_PER_CALL) and spend_before < MAX_BUDGET
    response: Final = gateway.request(
        "POST",
        f"/gemini/v1beta/models/{model}:generateContent",
        _generate_content_request(model),
        headers={"x-goog-api-key": key, "x-pass-x-scripted-scenario": scenario_id},
    )
    assert response.status_code == 200, f"call {call} with key spend {spend_before}: {response.text}"
    assert response.json() == _generate_content_response(model).body, response.text
    eventually(lambda: _key_spend(digest), lambda spend: spend >= call * COST_PER_CALL - 1e-9, seconds=70)


@pytest.mark.covers("spend.budget_reservation.gemini_passthrough_success_releases_reservation_from_spend_counter")
def test_repeated_gemini_passthrough_calls_stay_served_while_key_spend_is_below_max_budget(
    gateway: Gateway, tmp_path: Path
) -> None:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["environment_variables"] = {
        "GEMINI_API_BASE": gateway.upstream_url,
        "GEMINI_API_KEY": "scripted",
    }
    path: Final = tmp_path / "gemini-passthrough.yaml"
    path.write_text(yaml.safe_dump(config))
    with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
        model: Final = f"gemini-passthrough-{uuid.uuid4().hex}"
        created: Final = candidate.post(
            "/model/new",
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gemini/gemini-2.5-flash",
                    "api_key": "scripted",
                    "api_base": gateway.upstream_url,
                    "input_cost_per_token": INPUT_COST_PER_TOKEN,
                    "output_cost_per_token": OUTPUT_COST_PER_TOKEN,
                },
                "model_info": {"id": model, "max_output_tokens": 10},
            },
        )
        scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
        handle: Final = register_scenario(f"sc-{model}", _generate_content_response(model))
        scenario.cleanups.callback(delete_scenario, handle)
        key: Final = scenario.key(models=[model], max_budget=MAX_BUDGET)
        for call in range(1, CALLS_WITHIN_BUDGET + 1):
            _served_call(candidate, model, key, handle.scenario_id, call)
        assert _key_spend(sha256(key.encode()).hexdigest()) == pytest.approx(CALLS_WITHIN_BUDGET * COST_PER_CALL)
        denied: Final = candidate.request(
            "POST",
            f"/gemini/v1beta/models/{model}:generateContent",
            _generate_content_request(model),
            headers={"x-goog-api-key": key, "x-pass-x-scripted-scenario": handle.scenario_id},
        )
        assert denied.status_code == 422 and denied.json()["error"]["type"] == "budget_exceeded", denied.text
