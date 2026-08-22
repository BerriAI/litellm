"""Live e2e: POST /v1/completions returns a real text completion.

The legacy text-completion endpoint (prompt-style, non-chat) is the second-busiest
route in production yet was previously uncovered; the rest of the "completions"
surface is chat only. Registers an OpenAI instruct deployment at runtime (deleted
on teardown), drives /v1/completions through the gateway, and asserts real
generated text came back so a regression that empties the completion fails here.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import require_successful_call
from endpoints_client import CompletionsResult, EndpointsClient
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e


class TestCompletionsEndpoint:
    @pytest.mark.covers("llm.completions.openai.basic.nonstream.works")
    def test_text_completion_returns_text(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-completions-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="text-completion-openai/gpt-3.5-turbo-instruct",
                api_key="os.environ/OPENAI_API_KEY",
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.text_completions(
            key, model, "Finish this sentence in a few words: the capital of France is"
        )
        require_successful_call(result)
        parsed = CompletionsResult.model_validate_json(result.body)
        assert parsed.choices, f"/v1/completions returned no choices: {result.body[:300]}"
        completion = (parsed.choices[0].text or "").strip()
        assert completion, f"/v1/completions returned an empty completion: {result.body[:300]}"
