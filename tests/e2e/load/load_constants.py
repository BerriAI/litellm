from __future__ import annotations

from models import LiteLLMParamsBody

# Backend id is never called: mock_response short-circuits before OpenAI.
# model_name on the proxy is unique per session (see conftest) so a stale
# stage deployment without mock_response cannot be reused across runs.
LOAD_MOCK_PARAMS = LiteLLMParamsBody(
    model="openai/load-mock",
    mock_response="This is a mock response for the throughput load test.",
)

LOAD_MOCK_BODY_SNIPPET = "mock response for the throughput load test"
