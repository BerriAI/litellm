from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import JsonValue


@dataclass(frozen=True, slots=True, kw_only=True)
class TranslationTestCase:
    """One request through the proxy to a deployment in `proxy_config.yaml`: what the test sends to LiteLLM, the
    exact request the provider must receive, the fake provider's status and reply, and the exact response LiteLLM must
    return."""

    scenario: str
    litellm_endpoint: str
    litellm_request: Mapping[str, JsonValue]
    expected_provider_endpoint: str
    expected_provider_headers: Mapping[str, str]
    expected_provider_request: Mapping[str, JsonValue]
    mock_provider_status_code: int = 200
    mock_provider_response: Mapping[str, JsonValue]
    expected_litellm_status_code: int = 200
    expected_litellm_response: Mapping[str, JsonValue]

    @property
    def id(self) -> str:
        return f"{self.litellm_request['model']}-{self.scenario}"
