"""Live e2e: POST /v1/responses returns a real completion across providers.

Registers a deployment per provider at runtime, drives the Responses API through
the gateway, and asserts real output text came back (not just a 200). GH #28991
broke /responses for most models on some releases, so this covers OpenAI plus a
second provider (Gemini, exercising the non-OpenAI responses-translation path);
a regression that empties the response for either provider fails that row.
Migrated from litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import require_successful_call
from endpoints_client import EndpointsClient, ResponsesResult
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

RESPONSES_PROVIDERS: tuple[tuple[str, LiteLLMParamsBody], ...] = (
    ("openai", LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY")),
    ("gemini", LiteLLMParamsBody(model="gemini/gemini-2.5-flash", api_key="os.environ/GEMINI_API_KEY")),
)


class TestResponses:
    @pytest.mark.parametrize(
        ("route", "params"),
        RESPONSES_PROVIDERS,
        ids=[route for route, _ in RESPONSES_PROVIDERS],
    )
    @pytest.mark.covers("llm.responses.provider.basic.nonstream.works", exercised_on=[])
    def test_responses_returns_completion(
        self,
        endpoints_client: EndpointsClient,
        resources: ResourceManager,
        route: str,
        params: LiteLLMParamsBody,
    ) -> None:
        model = f"e2e-responses-{route}-{unique_marker()}"
        model_id = endpoints_client.create_model(model, params)
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.responses(key, model, "reply with one word")
        require_successful_call(result)
        parsed = ResponsesResult.model_validate_json(result.body)
        assert parsed.text.strip(), (
            f"/responses ({route}) returned no output text: {result.body[:300]}"
        )
