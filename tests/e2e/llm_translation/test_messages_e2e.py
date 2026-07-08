"""Live e2e: POST /v1/messages (Anthropic Messages API) returns a real completion.

Registers an Anthropic deployment at runtime, drives the Messages endpoint through
the gateway, and asserts an assistant message with text came back. Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import require_successful_call
from endpoints_client import EndpointsClient, MessagesResult
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e


class TestAnthropicMessages:
    def test_messages_returns_completion(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-messages-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="anthropic/claude-haiku-4-5", api_key="os.environ/ANTHROPIC_API_KEY"
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.messages(key, model, "reply with one word")
        require_successful_call(result)
        parsed = MessagesResult.model_validate_json(result.body)
        assert parsed.role == "assistant", f"unexpected role: {result.body[:300]}"
        assert parsed.text.strip(), f"/v1/messages returned no text: {result.body[:300]}"
