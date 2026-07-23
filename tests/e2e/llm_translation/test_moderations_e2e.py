"""Live e2e: POST /v1/moderations classifies content against the provider policy.

Registers OpenAI's omni moderation model at runtime, drives it through the real
OpenAI SDK (LIT-4577), and asserts the product promise on both sides of the
decision: clearly violent text comes back flagged with at least one policy
category tripped, and benign text comes back not flagged.
"""

from __future__ import annotations

import pytest
from openai.types import Moderation
from pydantic import TypeAdapter

from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e

VIOLENT_TEXT = "I am going to find you and kill you, and I will hurt everyone you love."
BENIGN_TEXT = "I enjoyed the sunny afternoon and a relaxing walk in the park today."


def _register_moderation_model(proxy: ProxyClient, resources: ResourceManager) -> str:
    model = f"e2e-moderation-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(
            model="openai/omni-moderation-latest", api_key="os.environ/OPENAI_API_KEY"
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


_CATEGORY_FLAGS = TypeAdapter(dict[str, bool | None])


def _flagged_categories(item: Moderation) -> tuple[str, ...]:
    flags = _CATEGORY_FLAGS.validate_python(item.categories.model_dump())
    return tuple(name for name, hit in flags.items() if hit)


class TestModerations:
    @pytest.mark.covers("llm.moderations.openai.basic.nonstream.works")
    def test_moderations_flags_violent_content(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register_moderation_model(proxy, resources)
        client = sdk.openai(resources.key())

        moderation = client.moderations.create(model=model, input=VIOLENT_TEXT)
        assert moderation.results, f"/moderations returned no results: {moderation!r}"
        item = moderation.results[0]
        assert item.flagged, f"violent text was not flagged: {item!r}"
        assert _flagged_categories(item), f"flagged result reported no true category: {item!r}"

    def test_moderations_passes_benign_content(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register_moderation_model(proxy, resources)
        client = sdk.openai(resources.key())

        moderation = client.moderations.create(model=model, input=BENIGN_TEXT)
        assert moderation.results, f"/moderations returned no results: {moderation!r}"
        item = moderation.results[0]
        assert not item.flagged, (
            f"benign text was flagged as {_flagged_categories(item)}: {item!r}"
        )
