from __future__ import annotations

from itertools import chain
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionToolParam
from passthrough_client import PassthroughClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

GEMINI: Final = "gemini/gemini-3.8-flash"
VERTEX: Final = "vertex_ai/gemini-3.8-flash"
WEATHER: Final[ChatCompletionToolParam] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Return the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _subject(provider: Provider, backend: str, mode: Mode) -> Subject:
    return Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.CHAT_COMPLETIONS,
        providers=(provider,),
        models=(backend,),
        mode=mode,
        capabilities=(Capability.FUNCTION_CALLING, Capability.TOOL_CHOICE),
    )


def _completion(client: OpenAI, model: str, prompt: str, stream: bool, tool_choice: str) -> tuple[str, tuple[str, ...]]:
    completions: Final = client.chat.completions
    extra_body: Final = {**NO_PROXY_CACHE, "tool_choice": tool_choice}
    if not stream:
        raw: Final = completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": prompt}], tools=[WEATHER], extra_body=extra_body
        )
        assert raw.status_code == 200
        response: Final = raw.parse()
        assert response.choices, f"completion returned no choices: {response}"
        message: Final = response.choices[0].message
        calls: Final = tuple(call for call in message.tool_calls or () if call.type == "function")
        return message.content or "", tuple(call.function.name for call in calls)

    with completions.with_streaming_response.create(
        model=model, messages=[{"role": "user", "content": prompt}], tools=[WEATHER], extra_body=extra_body, stream=True
    ) as response_stream:
        assert response_stream.status_code == 200
        assert "text/event-stream" in response_stream.headers.get("content-type", "")
        events: Final = tuple(
            line.removeprefix("data: ") for line in response_stream.iter_lines() if line.startswith("data: ")
        )
    assert events and events[-1] == "[DONE]", "stream must end with [DONE]"
    chunks: Final = tuple(ChatCompletionChunk.model_validate_json(event) for event in events[:-1])
    deltas: Final = tuple(choice.delta for choice in chain.from_iterable(chunk.choices for chunk in chunks))
    text: Final = "".join(delta.content or "" for delta in deltas)
    names: Final = tuple(
        call.function.name
        for call in chain.from_iterable(delta.tool_calls or () for delta in deltas)
        if call.function is not None and call.function.name is not None
    )
    return text, names


class TestGeminiValidatedToolChoice:
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
        self, client: PassthroughClient, resources: ResourceManager, sdk: SdkClients, backend: str, stream: bool
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
        openai: Final = sdk.openai(resources.key(models=[model]))
        prompt: Final = f"Reply with {marker}. No weather information is needed."

        text, calls = _completion(openai, model, prompt, stream, "validated")
        assert marker in text and not calls, f"validated choice should permit a text answer: {text!r}, {calls}"

        _, required_calls = _completion(openai, model, prompt, stream, "required")
        assert WEATHER["function"]["name"] in required_calls, f"required choice returned tool calls {required_calls}"
