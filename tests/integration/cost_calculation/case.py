from collections.abc import Mapping
from dataclasses import dataclass

from integration.cost_calculation.cost_tracking_case import StoredResponse
from pydantic import JsonValue


@dataclass(frozen=True, slots=True, kw_only=True)
class CostTrackingTestCase:
    """One request through the proxy to a deployment registered for the test: the deployment, what the test sends
    to LiteLLM, the fake provider's reply, and the cost LiteLLM must report and log."""

    scenario: str
    deployment: Mapping[str, JsonValue]
    litellm_endpoint: str
    litellm_request: Mapping[str, JsonValue]
    mock_provider_response: StoredResponse
    expected_litellm_status_code: int = 200
    expected_response_cost_header: float | None
    expected_spend_log: Mapping[str, JsonValue]

    @property
    def id(self) -> str:
        return f"{self.deployment['model']}-{self.scenario}"
