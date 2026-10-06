from collections.abc import Mapping
from hashlib import sha256
from typing import Final
from uuid import uuid4

import pytest
from integration._support.client import JSON_OBJECT, Gateway, object_value, string_value
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.case import CostTrackingTestCase
from integration.cost_calculation.chat_completions.bases.anthropic import CLAUDE_SONNET_5_TEST_CASE
from integration.cost_calculation.chat_completions.bases.openai import GPT_5_6_TEST_CASE
from integration.cost_calculation.conftest import poll_rollups

CASES: Final[tuple[CostTrackingTestCase, ...]] = (CLAUDE_SONNET_5_TEST_CASE, GPT_5_6_TEST_CASE)
EXPECTED_ROLLUPS: Final[Mapping[str, tuple[float, int, int]]] = {
    "anthropic/claude-sonnet-5-basic": (0.0351, 5520, 1236),
    "openai/gpt-5.6-basic": (0.026964, 5520, 1236),
}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_spend_rollups(case: CostTrackingTestCase, gateway: Gateway) -> None:
    expected_spend, expected_prompt_tokens, expected_completion_tokens = EXPECTED_ROLLUPS[case.id]
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        user_id: Final = scenario.user(team_id=team_id)
        key: Final = scenario.key(team_id=team_id, user_id=user_id)
        end_user_id: Final = f"end-user-{uuid4()}"
        case_slug: Final = case.id.replace("/", "-")
        run_id: Final = sha256(key.encode()).hexdigest()[:12]
        upstream: Final = register_scenario(f"sc-{case_slug}-{run_id}", case.mock_provider_response)
        scenario.cleanups.callback(delete_scenario, upstream)
        model_name: Final = f"cost-{case_slug}-{run_id}"
        created: Final = gateway.post(
            "/model/new",
            {"model_name": model_name, "litellm_params": {**case.deployment, "api_base": upstream.api_base()}},
        )
        model_id: Final = string_value(object_value(created["model_info"])["id"])
        scenario.cleanups.callback(scenario.delete_model, model_id)
        request_body: Final = JSON_OBJECT.validate_python(
            {**case.litellm_request, "model": model_name, "user": end_user_id, "cache": {"no-cache": True}}
        )
        responses: Final = tuple(
            gateway.request("POST", case.litellm_endpoint, request_body, key=key) for _ in range(3)
        )
        assert tuple(response.status_code for response in responses) == (200, 200, 200), tuple(
            response.text for response in responses
        )
        expected_response_cost: Final = case.expected_response_cost_header
        assert expected_response_cost is not None
        assert tuple(float(response.headers["x-litellm-response-cost"]) for response in responses) == pytest.approx(
            (expected_response_cost, expected_response_cost, expected_response_cost), rel=1e-6
        )
        rollups: Final = poll_rollups(key, team_id, user_id, end_user_id, expected_spend, 3)
        assert (
            rollups.key_spend,
            rollups.team_spend,
            rollups.user_spend,
            rollups.end_user_spend,
            rollups.daily_user.spend,
            rollups.daily_team.spend,
        ) == pytest.approx((expected_spend,) * 6, rel=1e-6)
        assert (
            rollups.daily_user.prompt_tokens,
            rollups.daily_user.completion_tokens,
            rollups.daily_user.api_requests,
            rollups.daily_team.prompt_tokens,
            rollups.daily_team.completion_tokens,
            rollups.daily_team.api_requests,
        ) == (
            expected_prompt_tokens,
            expected_completion_tokens,
            3,
            expected_prompt_tokens,
            expected_completion_tokens,
            3,
        )
