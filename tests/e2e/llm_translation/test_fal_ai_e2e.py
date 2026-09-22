"""Live e2e for fal_ai deployments (LIT-8340).

A fal image deployment registered with an `api_base` must send its provider
request to that base and nowhere else: the test points `api_base` at a local
edge that forwards to fal.run and asserts the edge saw the prompt marker, so a
proxy that egresses to fal.run directly fails on the observation, not on the
image. A fal chat deployment given a `reasoning_effort` that is not a string
must answer 4xx: the same deployment first serves a well formed request so the
rejection is attributable to the malformed field.
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from e2e_config import (
    FIXTURE_DIR,
    FIXTURE_MODE_RAW,
    PROVIDER_EDGE_ADVERTISE_HOST,
    PROVIDER_EDGE_BIND_HOST,
    REQUEST_TIMEOUT,
    unique_marker,
)
from e2e_http import Success, assert_client_error
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ImageContentPart, ImageUrl, LiteLLMParamsBody, TextContentPart
from provider_edge import ProviderRequestObservation, observed_provider_edge
from proxy_client import ProxyClient
from pydantic import BaseModel
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e

FAL_IMAGE_BACKEND: Final = "fal_ai/fal-ai/flux/schnell"
FAL_CHAT_BACKEND: Final = "fal_ai/fal-ai/moondream3-preview/query"
FAL_EDGE_MOUNT: Final = "fal_ai"
CAT_IMAGE: Final = Path(__file__).parent / "fixtures" / "cat.jpg"


class _UnhashableReasoningChatBody(BaseModel):
    model: str
    messages: list[ChatMessage]
    reasoning_effort: dict[str, str]


def _register(proxy: ProxyClient, resources: ResourceManager, prefix: str, params: LiteLLMParamsBody) -> str:
    model: Final = f"{prefix}-{unique_marker()}"
    model_id: Final = proxy.create_model(model, params)
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


def _vision_messages(marker: str) -> list[ChatMessage]:
    image: Final = "data:image/jpeg;base64," + base64.b64encode(CAT_IMAGE.read_bytes()).decode()
    return [
        ChatMessage(
            role="user",
            content=[
                TextContentPart(text=f"What animal is this? One word. Marker {marker}"),
                ImageContentPart(image_url=ImageUrl(url=image)),
            ],
        )
    ]


class TestFalAI:
    @pytest.mark.provider_edge_host
    @pytest.mark.covers("llm.images_generations.fal_ai.custom_api_base.nonstream.works")
    def test_image_generation_egresses_through_deployment_api_base(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        marker: Final = unique_marker()
        observation: Final = ProviderRequestObservation(marker)
        with observed_provider_edge(
            observation,
            mode_raw=FIXTURE_MODE_RAW,
            bundle_dir=FIXTURE_DIR,
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
            forward_timeout=REQUEST_TIMEOUT,
            mounts=MappingProxyType({FAL_EDGE_MOUNT: "https://fal.run"}),
        ) as edge:
            model: Final = _register(
                proxy,
                resources,
                "e2e-fal-image",
                LiteLLMParamsBody(
                    model=FAL_IMAGE_BACKEND,
                    api_key="os.environ/FAL_AI_API_KEY",
                    api_base=edge.api_base(FAL_EDGE_MOUNT),
                ),
            )
            images: Final = sdk.openai(resources.key()).images.generate(
                model=model, prompt=f"a blue lantern, marker {marker}", n=1
            )
            data: Final = images.data or []
            assert data and (data[0].url or data[0].b64_json), f"fal returned no image: {images!r}"
            assert observation.count == 1, (
                f"fal request bypassed the deployment api_base: edge saw {observation.count} requests carrying {marker}"
            )

    @pytest.mark.covers("llm.chat_completions.fal_ai.input_validation.nonstream.works")
    def test_unhashable_reasoning_effort_returns_client_error(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model: Final = _register(
            proxy,
            resources,
            "e2e-fal-chat",
            LiteLLMParamsBody(model=FAL_CHAT_BACKEND, api_key="os.environ/FAL_AI_API_KEY"),
        )
        key: Final = resources.key()
        messages: Final = _vision_messages(unique_marker())

        well_formed: Final = proxy.chat(key, ChatBody(model=model, messages=messages, reasoning_effort="low"))
        assert isinstance(well_formed, Success), (
            f"precondition: fal chat deployment must serve a valid request: {well_formed!r}"
        )
        first: Final = well_formed.data.choices[0].message
        assert first is not None and first.content, f"fal chat returned no content: {well_formed.data!r}"

        malformed: Final = proxy.transport.send(
            "/v1/chat/completions",
            headers=proxy.transport.bearer(key),
            json=_UnhashableReasoningChatBody(model=model, messages=messages, reasoning_effort={"level": "low"}),
        )
        assert_client_error(malformed, "fal chat with object reasoning_effort")
