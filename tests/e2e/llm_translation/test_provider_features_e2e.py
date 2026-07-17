"""Live e2e for model-specific request features: service_tier.

Each case asserts the feature took effect, not just a 200.

service_tier is an OpenAI concept. The proxy forwards it and the provider echoes
the tier back on the response, so sending a non-default tier ("priority") and
reading it back off ``service_tier`` proves the param was honored end to end;
litellm's own default injection (and service_tier="auto") both report "default",
so a "priority" echo can only come from the request being forwarded. "flex" is
avoided here because it is capacity-constrained and returns a transient 429 when
flex resources are unavailable. Bedrock and Vertex do not accept service_tier, so
that cell is OpenAI-only by design.

Prompt caching lives in test_cache_control.py.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

SERVICE_TIER = "priority"


class TestServiceTier:
    @pytest.mark.covers(
        "llm.chat_completions.openai.service_tier.nonstream.works", exercised_on=[]
    )
    def test_openai_service_tier_is_echoed(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-service-tier-{unique_marker()}"
        model_id = client.gateway.create_model(
            model,
            LiteLLMParamsBody(
                model="openai/gpt-5.5", api_key="os.environ/OPENAI_API_KEY"
            ),
        )
        resources.defer(lambda: client.gateway.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            client.gateway.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content="reply with one word")],
                    max_tokens=64,
                    service_tier=SERVICE_TIER,
                ),
            )
        )
        assert response.service_tier == SERVICE_TIER, (
            f"service_tier not honored: sent {SERVICE_TIER!r}, response reported "
            f"{response.service_tier!r} ({response})"
        )
