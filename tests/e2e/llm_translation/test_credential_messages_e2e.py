"""Live e2e: stored credentials resolve into a deployment serving /v1/messages."""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import require_successful_call
from endpoints_client import EndpointsClient, MessagesResult
from lifecycle import ResourceManager
from models import CredentialCreateBody, LiteLLMParamsBody

pytestmark = pytest.mark.e2e


class TestCredentialBackedMessages:
    @pytest.mark.covers("mgmt.credential.new.serves_request")
    def test_credential_backed_messages(self, endpoints_client: EndpointsClient, resources: ResourceManager) -> None:
        marker = unique_marker()
        credential_name = f"e2e-cred-{marker}"
        model = f"e2e-cred-messages-{marker}"

        endpoints_client.proxy.create_credential(
            CredentialCreateBody(
                credential_name=credential_name,
                credential_values={"api_key": "os.environ/ANTHROPIC_API_KEY"},
            )
        )
        resources.defer(lambda: endpoints_client.proxy.delete_credential(credential_name))

        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="anthropic/claude-haiku-4-5",
                litellm_credential_name=credential_name,
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))

        key = resources.key()
        result = endpoints_client.messages(key, model, "reply with one word")
        require_successful_call(result)
        parsed = MessagesResult.model_validate_json(result.body)
        assert parsed.role == "assistant", f"unexpected role: {result.body[:300]}"
        assert parsed.text.strip(), f"/v1/messages returned no text: {result.body[:300]}"
