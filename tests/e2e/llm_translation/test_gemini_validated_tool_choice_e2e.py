from __future__ import annotations

from itertools import chain
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import unwrap_status
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatTool, ChatToolFunction, LiteLLMParamsBody
from openai.types.chat import ChatCompletionChunk
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

GEMINI: Final = "gemini/gemini-3.8-flash"
VERTEX: Final = "vertex_ai/gemini-3.8-flash"
WEATHER: Final = ChatTool(
    function=ChatToolFunction(
        name="get_weather",
        description="Return the weather for a city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
)


def _subject(provider: Provider, backend: str, mode: Mode) -> Subject:
    return Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.CHAT_COMPLETIONS,
        providers=(provider,),
        models=(backend,),
        mode=mode,
        capabilities=(Capability.FUNCTION_CALLING, Capability.TOOL_CHOICE),
    )


def _completion(client: PassthroughClient, key: str, request: ChatBody) -> tuple[str, tuple[str | None, ...]]:
    if not request.stream:
        response: Final = unwrap_status(client.proxy.chat(key, request), 200)
        assert response.choices, f"completion returned no choices: {response}"
        message: Final = response.choices[0].message
        assert message is not None
        return message.content or "", tuple(call.function.name for call in message.tool_calls or ())

    stream: Final = client.proxy.chat_stream(key, request)
    assert stream.status_code == 200 and stream.is_streaming, f"stream was not established: {stream}"
    assert stream.stream_error is None, f"stream carried an error: {stream.stream_error}"
    assert stream.stream_done, "stream must end with [DONE]"
    chunks: Final = tuple(ChatCompletionChunk.model_validate_json(event) for event in stream.stream_events)
    deltas: Final = tuple(choice.delta for choice in chain.from_iterable(chunk.choices for chunk in chunks))
    text: Final = "".join(delta.content or "" for delta in deltas)
    names: Final = tuple(
        call.function.name
        for call in chain.from_iterable(delta.tool_calls or () for delta in deltas)
        if call.function is not None and call.function.name is not None
    )
    return text, names


@pytest.mark.parametrize(
    ("backend", "stream"),
    [
        pytest.param(
            GEMINI,
            False,
            marks=(
                meta(_subject(Provider.GEMINI, GEMINI, Mode.NONSTREAM)),
                pytest.mark.covers(
                    "llm.chat_completions.gemini.tool_use.nonstream.works", exercised_on=["chat_completions"]
                ),
            ),
            id="gemini-nonstream",
        ),
        pytest.param(
            GEMINI,
            True,
            marks=(
                meta(_subject(Provider.GEMINI, GEMINI, Mode.STREAM)),
                pytest.mark.covers(
                    "llm.chat_completions.gemini.tool_use.stream.works", exercised_on=["chat_completions"]
                ),
            ),
            id="gemini-stream",
        ),
        pytest.param(
            VERTEX,
            False,
            marks=(
                meta(_subject(Provider.VERTEX_AI, VERTEX, Mode.NONSTREAM)),
                pytest.mark.covers(
                    "llm.chat_completions.vertex.tool_use.nonstream.works", exercised_on=["chat_completions"]
                ),
            ),
            id="vertex-nonstream",
        ),
        pytest.param(
            VERTEX,
            True,
            marks=meta(_subject(Provider.VERTEX_AI, VERTEX, Mode.STREAM)),
            id="vertex-stream",
        ),
    ],
)
def test_gemini_validated_tool_choice_permits_text_and_required_calls_a_tool(
    client: PassthroughClient, resources: ResourceManager, backend: str, stream: bool
) -> None:
    marker: Final = unique_marker()
    model: Final = f"e2e-gemini-validated-{marker}"
    params: Final = (
        LiteLLMParamsBody(model=backend, api_key="os.environ/GEMINI_API_KEY", drop_params=True)
        if backend == GEMINI
        else LiteLLMParamsBody(
            model=backend, vertex_project="os.environ/VERTEXAI_PROJECT", vertex_location="global", drop_params=True
        )
    )
    model_id: Final = client.proxy.create_model(model, params)
    resources.defer(lambda: client.proxy.delete_model(model_id))
    key: Final = resources.key(models=[model])
    request: Final = ChatBody(
        model=model,
        messages=[ChatMessage(role="user", content=f"Reply with {marker}. No weather information is needed.")],
        tools=[WEATHER],
        tool_choice="validated",
        stream=stream,
    )

    text, calls = _completion(client, key, request)
    assert marker in text and not calls, f"validated choice should permit a text answer: {text!r}, {calls}"

    _, required_calls = _completion(client, key, request.model_copy(update={"tool_choice": "required"}))
    assert WEATHER.function.name in required_calls, f"required choice returned tool calls {required_calls}"
