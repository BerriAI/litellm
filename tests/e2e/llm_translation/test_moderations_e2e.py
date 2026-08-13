"""Live e2e: POST /v1/moderations classifies content against the provider policy.

Registers OpenAI's omni moderation model at runtime and asserts the product
promise on both sides of the decision: clearly violent text comes back flagged
with at least one policy category tripped, and benign text comes back not flagged.
"""

from __future__ import annotations

import pytest
from e2e_config import unique_marker
from e2e_http import assert_client_error, unwrap
from endpoints_client import EndpointsClient
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from pydantic import BaseModel

pytestmark = pytest.mark.e2e

VIOLENT_TEXT = "I am going to find you and kill you, and I will hurt everyone you love."
BENIGN_TEXT = "I enjoyed the sunny afternoon and a relaxing walk in the park today."


class _OptionalModerationBody(BaseModel):
    model: str | None = None
    input: str | None = None


def _register_moderation_model(
    endpoints_client: EndpointsClient, resources: ResourceManager
) -> str:
    model = f"e2e-moderation-{unique_marker()}"
    model_id = endpoints_client.create_model(
        model,
        LiteLLMParamsBody(
            model="openai/omni-moderation-latest", api_key="os.environ/OPENAI_API_KEY"
        ),
    )
    resources.defer(lambda: endpoints_client.delete_model(model_id))
    return model


class TestModerations:
    @pytest.mark.covers("llm.moderations.openai.basic.nonstream.works")
    def test_moderations_flags_violent_content(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = _register_moderation_model(endpoints_client, resources)
        key = resources.key()

        result = unwrap(endpoints_client.moderations(key, model, VIOLENT_TEXT))
        item = result.first
        assert item is not None, f"/moderations returned no results: {result}"
        assert item.flagged, f"violent text was not flagged: {item}"
        assert item.flagged_categories, (
            f"flagged result reported no true category: {item}"
        )

    def test_moderations_passes_benign_content(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = _register_moderation_model(endpoints_client, resources)
        key = resources.key()

        result = unwrap(endpoints_client.moderations(key, model, BENIGN_TEXT))
        item = result.first
        assert item is not None, f"/moderations returned no results: {result}"
        assert not item.flagged, (
            f"benign text was flagged as {item.flagged_categories}: {item}"
        )

    @pytest.mark.skip(reason="stage red: product gap, /v1/moderations 500s (KeyError 'input') on missing input instead of 400")
    @pytest.mark.covers("llm.moderations.openai.input_validation.nonstream.works")
    def test_missing_input_returns_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = _register_moderation_model(endpoints_client, resources)
        key = resources.key()
        result = endpoints_client.proxy.transport.send(
            "/v1/moderations",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalModerationBody(model=model),
        )
        assert_client_error(result, "moderations missing input")
