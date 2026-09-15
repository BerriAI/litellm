"""Live e2e: POST /v1/completions returns a real text completion.

The legacy text-completion endpoint (prompt-style, non-chat) is the second-busiest
route in production yet was previously uncovered; the rest of the "completions"
surface is chat only. Registers an OpenAI instruct deployment at runtime (deleted
on teardown), drives /v1/completions through the gateway with the real OpenAI SDK
(LIT-4577), and asserts real generated text came back so a regression that empties
the completion fails here.
"""

from __future__ import annotations

import pytest
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e


class TestCompletionsEndpoint:
    @pytest.mark.covers("llm.completions.openai.basic.nonstream.works")
    def test_text_completion_returns_text(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = f"e2e-completions-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(
                model="text-completion-openai/gpt-3.5-turbo-instruct",
                api_key="os.environ/OPENAI_API_KEY",
            ),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        client = sdk.openai(resources.key())

        completion = client.completions.create(
            model=model,
            prompt="Finish this sentence in a few words: the capital of France is",
            max_tokens=32,
        )
        assert completion.choices, f"/v1/completions returned no choices: {completion!r}"
        text = (completion.choices[0].text or "").strip()
        assert text, f"/v1/completions returned an empty completion: {completion!r}"
