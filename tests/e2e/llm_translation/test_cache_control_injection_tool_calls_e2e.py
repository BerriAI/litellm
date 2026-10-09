from __future__ import annotations

from typing import Final, Literal, TypeAlias

import pytest
from e2e_config import unique_marker
from e2e_http import Result, unwrap
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import (
    CacheControl,
    CacheControlInjectionPoint,
    ChatAssistantTurn,
    ChatBody,
    ChatMessage,
    ChatResponse,
    ChatTool,
    ChatToolFunction,
    ChatToolResultTurn,
    LiteLLMParamsBody,
    TextContentPart,
    ToolCall,
    ToolCallFunction,
)
from passthrough_client import PassthroughClient

pytestmark: Final = pytest.mark.e2e

Backend: TypeAlias = Literal["azure_foundry", "vertex"]
AZURE_MODEL: Final[str] = "azure_ai/claude-haiku-4-5"
VERTEX_MODEL: Final[str] = "vertex_ai/claude-sonnet-5"
VERTEX_LOCATION: Final[str] = "global"


def _deployment_params(*, backend: Backend, inject_cache_control: bool) -> LiteLLMParamsBody:
    cache_control_injection_points: Final = (
        [
            CacheControlInjectionPoint(location="message", role="system"),
            CacheControlInjectionPoint(location="message", index=-1),
        ]
        if inject_cache_control
        else None
    )
    match backend:
        case "azure_foundry":
            return LiteLLMParamsBody(
                model=AZURE_MODEL,
                api_base="os.environ/AZURE_AI_API_BASE",
                api_key="os.environ/AZURE_AI_API_KEY",
                cache_control_injection_points=cache_control_injection_points,
            )
        case "vertex":
            return LiteLLMParamsBody(
                model=VERTEX_MODEL,
                vertex_project="os.environ/VERTEXAI_PROJECT",
                vertex_location=VERTEX_LOCATION,
                vertex_credentials="os.environ/VERTEXAI_CREDENTIALS",
                cache_control_injection_points=cache_control_injection_points,
            )


def _register_deployment(
    client: PassthroughClient,
    resources: ResourceManager,
    *,
    backend: Backend,
    marker: str,
    inject_cache_control: bool,
) -> str:
    model_name: Final[str] = f"e2e-cache-control-tool-calls-{backend}-{marker}"
    model_id: Final[str] = client.proxy.create_model(
        model_name,
        _deployment_params(backend=backend, inject_cache_control=inject_cache_control),
        provider_live=True,
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model_name


def _request(model: str, marker: str) -> ChatBody:
    return ChatBody(
        model=model,
        messages=[
            ChatMessage(role="system", content="Use the provided tool results to answer the user."),
            ChatMessage(
                role="user",
                content=[
                    TextContentPart(
                        text="Look up the weather in London, Paris, and Tokyo.",
                        cache_control=CacheControl(),
                    )
                ],
            ),
            ChatAssistantTurn(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_weather_london",
                        type="function",
                        function=ToolCallFunction(name="lookup_weather", arguments='{"city":"London"}'),
                        cache_control=CacheControl(),
                    ),
                    ToolCall(
                        id="call_weather_paris",
                        type="function",
                        function=ToolCallFunction(name="lookup_weather", arguments='{"city":"Paris"}'),
                        cache_control=CacheControl(),
                    ),
                    ToolCall(
                        id="call_weather_tokyo",
                        type="function",
                        function=ToolCallFunction(name="lookup_weather", arguments='{"city":"Tokyo"}'),
                        cache_control=CacheControl(),
                    ),
                ],
            ),
            ChatToolResultTurn(tool_call_id="call_weather_london", content="London is sunny."),
            ChatToolResultTurn(tool_call_id="call_weather_paris", content="Paris is cloudy."),
            ChatToolResultTurn(tool_call_id="call_weather_tokyo", content="Tokyo is rainy."),
            ChatMessage(
                role="user",
                content=f"Summarize the results in one word and do not call another tool. {marker}",
            ),
        ],
        tools=[
            ChatTool(
                function=ChatToolFunction(
                    name="lookup_weather",
                    description="Look up the weather in a city.",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                )
            )
        ],
        max_tokens=64,
    )


def _post_chat(client: PassthroughClient, key: str, body: ChatBody) -> Result[ChatResponse]:
    return client.proxy.transport.post(
        "/v1/chat/completions",
        headers=client.proxy.transport.bearer(key),
        json=body,
        response_type=ChatResponse,
    )


def _assert_normal_completion(response: ChatResponse, model_name: str) -> None:
    assert response.choices, f"{model_name}: chat completion returned no choices: {response}"
    completion: Final = response.choices[0]
    assert completion.finish_reason in ("stop", "length"), (
        f"{model_name}: unexpected finish reason: {completion.finish_reason}"
    )
    assert (
        completion.message is not None
        and completion.message.content is not None
        and completion.message.content.strip()
    ), f"{model_name}: chat completion returned no text: {completion.message}"


@pytest.mark.parametrize(
    "backend",
    (
        pytest.param(
            "azure_foundry",
            id="azure-foundry",
            marks=meta(
                Subject(
                    domain=Domain.LLM_TRANSLATION,
                    route=Route.CHAT_COMPLETIONS,
                    providers=(Provider.AZURE_AI,),
                    models=(AZURE_MODEL,),
                    capabilities=(Capability.FUNCTION_CALLING, Capability.PROMPT_CACHING),
                    mode=Mode.NONSTREAM,
                )
            ),
        ),
        pytest.param(
            "vertex",
            id="vertex",
            marks=meta(
                Subject(
                    domain=Domain.LLM_TRANSLATION,
                    route=Route.CHAT_COMPLETIONS,
                    providers=(Provider.VERTEX_AI,),
                    models=(VERTEX_MODEL,),
                    capabilities=(Capability.FUNCTION_CALLING, Capability.PROMPT_CACHING),
                    mode=Mode.NONSTREAM,
                )
            ),
        ),
    ),
)
@pytest.mark.provider_live
@pytest.mark.covers("llm.chat_completions.azure_foundry.basic.nonstream.works")
@pytest.mark.covers("llm.chat_completions.vertex.basic.nonstream.works")
class TestCacheControlInjectionToolCalls:
    def test_injection_points_respect_cap_with_tool_call_marks(
        self, client: PassthroughClient, resources: ResourceManager, backend: Backend
    ) -> None:
        marker: Final[str] = unique_marker()
        model_name: Final[str] = _register_deployment(
            client,
            resources,
            backend=backend,
            marker=marker,
            inject_cache_control=True,
        )
        key: Final[str] = resources.key()
        response: Final[ChatResponse] = unwrap(_post_chat(client, key, _request(model_name, marker)))

        _assert_normal_completion(response, model_name)

    def test_client_tool_call_marks_work_without_injection_points(
        self, client: PassthroughClient, resources: ResourceManager, backend: Backend
    ) -> None:
        marker: Final[str] = unique_marker()
        model_name: Final[str] = _register_deployment(
            client,
            resources,
            backend=backend,
            marker=marker,
            inject_cache_control=False,
        )
        key: Final[str] = resources.key()
        response: Final[ChatResponse] = unwrap(_post_chat(client, key, _request(model_name, marker)))

        _assert_normal_completion(response, model_name)
