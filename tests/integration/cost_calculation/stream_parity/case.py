from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue

MODEL: Final = "$MODEL"
RESPONSE_ID: Final = "$ID"


@dataclass(frozen=True, slots=True, kw_only=True)
class StreamParityTestCase:
    """One request sent twice through the same deployment and key, once plain and once with `stream_parameters` merged
    in. The fake provider answers the plain request with `mock_provider_response` and the streamed one with
    `mock_provider_stream`, and both LiteLLM_SpendLogs rows must equal `expected_spend_row`.

    `$MODEL` in a request or expected row is the deployment the runner registers from `deployment` (its name is
    generated per run). `$ID` in a mock is replaced by a fresh id per call because `request_id` is the SpendLogs
    primary key, so a repeated provider id would drop the second row.
    """

    scenario: str
    litellm_endpoint: str
    deployment: Mapping[str, JsonValue]
    litellm_request: Mapping[str, JsonValue]
    stream_parameters: Mapping[str, JsonValue] = MappingProxyType({"stream": True})
    expected_provider_endpoint: str
    mock_provider_response: Mapping[str, JsonValue]
    mock_provider_stream: tuple[Mapping[str, JsonValue], ...]
    expected_spend_row: Mapping[str, JsonValue]

    @property
    def id(self) -> str:
        return f"{self.deployment['model']}-{self.scenario}"
