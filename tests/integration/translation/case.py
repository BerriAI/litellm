from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import JsonValue


@dataclass(frozen=True, slots=True, kw_only=True)
class TranslationTestCase:
    """One client request through the proxy to a deployment in `proxy_config.yaml`: what the client sends, the
    exact request the provider must receive, the provider's reply, and the exact response the client must get."""

    scenario: str
    litellm_endpoint: str
    litellm_request: Mapping[str, JsonValue]
    provider_endpoint: str
    provider_headers: Mapping[str, str]
    provider_request: Mapping[str, JsonValue]
    provider_response: Mapping[str, JsonValue]
    litellm_status_code: int = 200
    litellm_response: Mapping[str, JsonValue]

    @property
    def id(self) -> str:
        return f"{self.litellm_request['model']}-{self.scenario}"
