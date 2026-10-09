from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration.cost_calculation.assertions import assert_stream_has_no_error
from integration.cost_calculation.case import CostTrackingTestCase
from integration.cost_calculation.conftest import register_scenario_deployment


def assert_cost_tracking(case: CostTrackingTestCase, gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        deployment: Final = register_scenario_deployment(scenario, case, "chat-completions", key)
        response: Final = gateway.request(
            "POST", case.litellm_endpoint, {**case.litellm_request, "model": deployment.model_name}, key=key
        )
        assert response.status_code == case.expected_litellm_status_code, response.text
        if case.mock_provider_response.content_type == "text/event-stream":
            assert_stream_has_no_error(response.text)
        if case.expected_response_cost_header is None:
            assert "x-litellm-response-cost" not in response.headers
        else:
            assert float(string_value(response.headers["x-litellm-response-cost"])) == pytest.approx(
                case.expected_response_cost_header, rel=1e-6
            )

        key_hash: Final = sha256(key.encode()).hexdigest()
        rows: Final = eventually(
            lambda: read_rows(
                "SELECT spend, status, prompt_tokens, completion_tokens, metadata "
                'FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
                (key_hash,),
            ),
            lambda values: len(values) == 1,
            seconds=60,
        )
        row: Final = JSON_OBJECT.validate_python(rows[0])
        assert row["status"] == "success"
        metadata_value: Final = row["metadata"]
        metadata: Final = (
            JSON_OBJECT.validate_json(metadata_value)
            if isinstance(metadata_value, str)
            else JSON_OBJECT.validate_python(metadata_value)
        )
        breakdown: Final = object_value(metadata["cost_breakdown"])
        actual: Final = {
            "spend": row["spend"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            **breakdown,
        }
        assert actual == pytest.approx(dict(case.expected_spend_log), rel=1e-6)
