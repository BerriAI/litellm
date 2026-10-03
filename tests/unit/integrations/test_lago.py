import json
from datetime import datetime
from typing import Final

import pytest
import respx

from litellm.integrations.lago import LagoLogger
from litellm.litellm_core_utils.internal_call_metadata import (
    EVALUATION_BILLING_OWNER_KEY,
    EvaluationBillingOwner,
    project_evaluation_billing_kwargs,
)


@pytest.mark.parametrize("charge_by", ("user_id", "team_id", "end_user_id"))
@pytest.mark.parametrize("evaluation", (False, True))
def test_evaluation_receipts_charge_the_creator_regardless_of_lago_charge_by(
    monkeypatch: pytest.MonkeyPatch, charge_by: str, evaluation: bool
) -> None:
    for name, value in {
        "LAGO_API_KEY": "test",
        "LAGO_API_BASE": "https://lago.invalid",
        "LAGO_API_EVENT_CODE": "usage",
        "LAGO_API_CHARGE_BY": charge_by,
    }.items():
        monkeypatch.setenv(name, value)
    receipt: Final = project_evaluation_billing_kwargs(
        {
            EVALUATION_BILLING_OWNER_KEY: EvaluationBillingOwner("creator") if evaluation else None,
            "response_cost": 0.25,
            "litellm_params": {
                "metadata": {"user_api_key_user_id": "user_id", "user_api_key_team_id": "team_id"},
                "proxy_server_request": {"body": {"user": "end_user_id"}},
            },
        }
    )
    with respx.mock(assert_all_called=True) as transport:
        endpoint: Final = transport.post("https://lago.invalid/api/v1/events").respond(200)
        LagoLogger().log_success_event(receipt, {}, datetime(2026, 1, 1), datetime(2026, 1, 1))
        sent: Final = json.loads(endpoint.calls.last.request.content)["event"]
    assert sent["external_subscription_id"] == ("creator" if evaluation else charge_by)
    assert sent["properties"]["response_cost"] == 0.25
