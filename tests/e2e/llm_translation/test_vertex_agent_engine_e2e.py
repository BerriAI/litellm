"""/chat/completions through the gateway to a hosted Vertex AI Agent Engine (Reasoning Engine)."""

from __future__ import annotations

import os
from typing import Final

import pytest
from e2e_config import VERTEX_AGENT_ENGINE_OPT_IN_ENV, unique_marker
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from openai.types.chat import ChatCompletionMessageParam
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = [pytest.mark.e2e, pytest.mark.vertex_agent_engine]


def _agent_engine_deployment(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    resource: Final = os.environ[VERTEX_AGENT_ENGINE_OPT_IN_ENV]
    location: Final = resource.split("/")[3]
    model: Final = f"e2e-agent-engine-{unique_marker()}"
    model_id: Final = proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=f"vertex_ai/agent_engine/{resource}",
            vertex_location=location,
            vertex_credentials="os.environ/VERTEXAI_CREDENTIALS",
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _greeting() -> list[ChatCompletionMessageParam]:
    return [{"role": "user", "content": f"Hello! What can you do? ({unique_marker()})"}]


class TestVertexAgentEngine:
    @pytest.mark.covers("llm.chat_completions.vertex_agent_engine.basic.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.VERTEX_AI,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_agent_replies(self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients) -> None:
        model, key = _agent_engine_deployment(proxy, resources)

        response: Final = sdk.openai(key).chat.completions.create(model=model, messages=_greeting())

        choice: Final = response.choices[0]
        assert choice.message.content, f"{model}: the agent returned no text: {response}"
        assert choice.finish_reason == "stop", f"{model}: unexpected finish_reason: {response}"

    @pytest.mark.covers("llm.chat_completions.vertex_agent_engine.basic.stream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.VERTEX_AI,),
            mode=Mode.STREAM,
        )
    )
    def test_agent_streams_a_reply(self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients) -> None:
        model, key = _agent_engine_deployment(proxy, resources)

        chunks: Final = tuple(sdk.openai(key).chat.completions.create(model=model, messages=_greeting(), stream=True))

        text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
        finish_reasons: Final = tuple(chunk.choices[0].finish_reason for chunk in chunks if chunk.choices)
        assert text.strip(), f"{model}: the streamed deltas carried no text across {len(chunks)} chunks"
        assert finish_reasons and finish_reasons[-1] == "stop", (
            f"{model}: stream did not end with stop: {finish_reasons}"
        )
