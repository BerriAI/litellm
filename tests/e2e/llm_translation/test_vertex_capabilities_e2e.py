from __future__ import annotations

from typing import Final, TypedDict, cast

import pytest
from typing_extensions import NotRequired, ReadOnly

from e2e_config import unique_marker
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = [pytest.mark.e2e, pytest.mark.provider_live]

GEMINI_MODEL: Final = "vertex_ai/gemini-3-flash-preview"


class GoogleSearchTool(TypedDict):
    googleSearch: ReadOnly[dict[str, str]]


class VertexGroundingBody(TypedDict):
    tools: ReadOnly[tuple[GoogleSearchTool, ...]]


class GoogleMapsOptions(TypedDict):
    enableWidget: ReadOnly[bool]
    latitude: NotRequired[ReadOnly[float]]
    longitude: NotRequired[ReadOnly[float]]
    languageCode: NotRequired[ReadOnly[str]]


class GoogleMapsTool(TypedDict):
    googleMaps: ReadOnly[GoogleMapsOptions]


class VertexMapsBody(TypedDict):
    tools: ReadOnly[tuple[GoogleMapsTool, ...]]


def _register(proxy: ProxyClient, resources: ResourceManager, base: str, model: str, location: str) -> tuple[str, str]:
    model_name: Final = f"{base}-{unique_marker()}"
    model_id: Final = proxy.create_model(
        model_name,
        LiteLLMParamsBody(
            model=model,
            vertex_project="os.environ/VERTEXAI_PROJECT",
            vertex_location=location,
            vertex_credentials="os.environ/VERTEXAI_CREDENTIALS",
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model_name, resources.key()


class TestVertexCapabilities:
    @pytest.mark.covers("llm.chat_completions.vertex.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.VERTEX_AI,),
            models=(GEMINI_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_google_maps_tool_accepts_widget_and_location(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, "e2e-vertex-maps", GEMINI_MODEL, "global")
        options: Final[tuple[GoogleMapsOptions, ...]] = (
            {"enableWidget": True},
            {"enableWidget": True, "latitude": 37.7749, "longitude": -122.4194, "languageCode": "en_US"},
        )
        client: Final = sdk.openai(key)
        bodies: Final[tuple[VertexMapsBody, ...]] = tuple(
            {"tools": ({"googleMaps": maps_options},)} for maps_options in options
        )
        responses: Final = tuple(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "What restaurants are nearby?"}],
                extra_body=body,
            )
            for body in bodies
        )
        assert all(response.choices[0].message.content for response in responses)

    @pytest.mark.covers("llm.chat_completions.vertex.web_search.stream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.VERTEX_AI,),
            models=(GEMINI_MODEL,),
            capabilities=(Capability.WEB_SEARCH,),
            mode=Mode.STREAM,
        )
    )
    def test_google_search_grounding_metadata_is_returned(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, "e2e-vertex-grounding", GEMINI_MODEL, "global")
        grounding_body: Final[VertexGroundingBody] = {"tools": ({"googleSearch": {}},)}
        chunks: Final = sdk.openai(key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is today's weather in San Francisco?"}],
            stream=True,
            extra_body=grounding_body,
        )
        metadata: Final = tuple(
            cast(object | None, getattr(chunk, "vertex_ai_grounding_metadata", None)) for chunk in chunks
        )
        assert any(value is not None for value in metadata)
