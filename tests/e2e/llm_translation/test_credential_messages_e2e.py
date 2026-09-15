"""Live e2e: stored credentials resolve into a deployment serving /v1/messages."""

from __future__ import annotations

import os

import pytest

from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import CredentialCreateBody, LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e


class TestCredentialBackedMessages:
    @pytest.mark.covers("mgmt.credential.new.serves_request")
    def test_credential_backed_messages(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        marker = unique_marker()
        credential_name = f"e2e-cred-{marker}"
        model = f"e2e-cred-messages-{marker}"
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        assert anthropic_api_key, "ANTHROPIC_API_KEY must be set for this live e2e test"

        proxy.create_credential(
            CredentialCreateBody(
                credential_name=credential_name,
                credential_values={"api_key": anthropic_api_key},
            )
        )
        resources.defer(lambda: proxy.delete_credential(credential_name))

        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(
                model="anthropic/claude-haiku-4-5",
                litellm_credential_name=credential_name,
            ),
        )
        resources.defer(lambda: proxy.delete_model(model_id))

        client = sdk.anthropic(resources.key())
        message = client.messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": "reply with one word"}],
        )
        assert message.role == "assistant", f"unexpected role: {message.role!r}"
        text = "".join(block.text for block in message.content if block.type == "text")
        assert text.strip(), f"/v1/messages returned no text: {message.content!r}"
