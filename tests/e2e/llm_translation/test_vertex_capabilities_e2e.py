from __future__ import annotations

from typing import Final, TypedDict, cast

import pytest
from typing_extensions import ReadOnly

from e2e_config import unique_marker
from e2e_http import unwrap
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import ChatMessage, ChatResponse, LiteLLMParamsBody
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict
from sdk_clients import SdkClients, response_header

pytestmark = [pytest.mark.e2e, pytest.mark.provider_live]

GEMINI_MODEL: Final = "vertex_ai/gemini-3-flash-preview"
LLAMA_MODEL: Final = "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas"


class GoogleSearchTool(TypedDict):
    googleSearch: ReadOnly[dict[str, str]]


class VertexGroundingBody(TypedDict):
    tools: ReadOnly[tuple[GoogleSearchTool, ...]]


class GoogleMapsOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    enableWidget: bool
    latitude: float | None = None
    longitude: float | None = None
    languageCode: str | None = None


class GoogleMapsTool(BaseModel):
    model_config = ConfigDict(frozen=True)

    googleMaps: GoogleMapsOptions


class VertexChatBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    messages: tuple[ChatMessage, ...]
    tools: tuple[GoogleMapsTool, ...]


def _register(
    proxy: ProxyClient, resources: ResourceManager, base: str, model: str, location: str
) -> tuple[str, str]:
    model_name = f"{base}-{unique_marker()}"
    model_id = proxy.create_model(
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
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        deployment: Final = _register(proxy, resources, "e2e-vertex-maps", GEMINI_MODEL, "global")
        model, key = deployment
        options: Final = (
            GoogleMapsOptions(enableWidget=True),
            GoogleMapsOptions(
                enableWidget=True, latitude=37.7749, longitude=-122.4194, languageCode="en_US"
            ),
        )
        responses: Final = tuple(
            unwrap(
                proxy.transport.post(
                    "/chat/completions",
                    headers=proxy.transport.bearer(key),
                    json=VertexChatBody(
                        model=model,
                        messages=(ChatMessage(role="user", content="What restaurants are nearby?"),),
                        tools=(GoogleMapsTool(googleMaps=maps_options),),
                    ),
                    response_type=ChatResponse,
                )
            )
            for maps_options in options
        )
        assert len(responses) == 2
        assert all(
            response.choices[0].message is not None and response.choices[0].message.content
            for response in responses
        )

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
        deployment: Final = _register(proxy, resources, "e2e-vertex-grounding", GEMINI_MODEL, "global")
        model, key = deployment
        grounding_body: Final[VertexGroundingBody] = {"tools": ({"googleSearch": {}},)}
        chunks: Final = sdk.openai(key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is today's weather in San Francisco?"}],
            stream=True,
            extra_body=grounding_body,
        )
        metadata = tuple(
            cast(object | None, getattr(chunk, "vertex_ai_grounding_metadata", None)) for chunk in chunks
        )
        assert any(value is not None for value in metadata)

    @pytest.mark.covers("llm.chat_completions.vertex.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.VERTEX_AI,),
            models=(LLAMA_MODEL,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_llama_tool_calling_returns_weather_tool_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        deployment: Final = _register(proxy, resources, "e2e-vertex-llama", LLAMA_MODEL, "us-east5")
        model, key = deployment
        raw_response: Final = sdk.openai(key).chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "What is the weather in Boston, MA today?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current temperature for a given location.",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        )
        response: Final = raw_response.parse()
        choice: Final = response.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.tool_calls
        tool_call: Final = choice.message.tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"
        cost: Final = response_header(raw_response.headers, "x-litellm-response-cost")
        assert cost is not None
        assert float(cost) > 0
