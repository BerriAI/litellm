import asyncio
import json
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Final
from unittest.mock import patch

import pytest

import litellm
from litellm.fusion_router import FUSION_TOOL_NAME
from litellm.llms.litellm.adapters import (
    adispatch_anthropic_messages,
    adispatch_completion,
    adispatch_responses,
)
from litellm.llms.litellm.fusion import FusionLiteLLMModel, FusionSDKConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper


def _response(
    content: str | None,
    *,
    model: str = "concrete-model",
    tool_calls: list[dict[str, object]] | None = None,
) -> ModelResponse:
    return ModelResponse(
        model=model,
        choices=[
            {
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
            }
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _fusion_call() -> ModelResponse:
    return _response(
        None,
        model="judge-model",
        tool_calls=[
            {
                "id": "fusion-call",
                "type": "function",
                "function": {"name": FUSION_TOOL_NAME, "arguments": '{"query":"compare approaches"}'},
            }
        ],
    )


def _analysis() -> str:
    return json.dumps(
        {
            "consensus": ["shared conclusion"],
            "contradictions": [],
            "partial_coverage": ["one gap"],
            "unique_insights": [{"model": "panel-b", "insight": "unique detail"}],
            "blind_spots": ["missing measurement"],
        }
    )


class RecordingCompletion:
    def __init__(self, responses: Mapping[str, Sequence[ModelResponse]]) -> None:
        self.responses: Final = {model: deque(model_responses) for model, model_responses in responses.items()}
        self.calls: Final[list[dict[str, object]]] = []
        self.active_panel_calls = 0
        self.max_active_panel_calls = 0

    async def __call__(
        self,
        *,
        model: str,
        messages: list[AllMessageValues],
        stream: bool,
        **kwargs: object,
    ) -> ModelResponse | CustomStreamWrapper:
        self.calls.append({"model": model, "messages": messages, "stream": stream, **kwargs})
        if model.startswith("panel-"):
            self.active_panel_calls += 1
            self.max_active_panel_calls = max(self.max_active_panel_calls, self.active_panel_calls)
            await asyncio.sleep(0.01)
            self.active_panel_calls -= 1
        return self.responses[model].popleft()


class StaticLiteLLMModel:
    def __init__(self) -> None:
        self.requests: Final[list[tuple[list[AllMessageValues], Mapping[str, object]]]] = []

    async def acompletion(
        self,
        *,
        messages: list[AllMessageValues],
        stream: bool,
        request_kwargs: Mapping[str, object],
    ) -> ModelResponse | CustomStreamWrapper:
        assert stream is False
        self.requests.append((messages, request_kwargs))
        response: Final = _response("shared answer")
        response._hidden_params["fusion"] = {
            "invoked": True,
            "panel_successes": 2,
            "panel_failures": 0,
            "analysis_available": True,
        }
        return response


@pytest.mark.asyncio
async def test_fusion_sdk_model_always_deliberates_with_configured_panel_and_judge() -> None:
    completion = RecordingCompletion(
        {
            "judge-model": [_fusion_call(), _response(_analysis()), _response("final", model="judge-model")],
            "panel-a": [_response("candidate a", model="panel-a")],
            "panel-b": [_response("candidate b", model="panel-b")],
        }
    )

    response = await FusionLiteLLMModel(completion=completion).acompletion(
        messages=[{"role": "user", "content": "question"}],
        stream=False,
        request_kwargs={
            "web_search_options": None,
            "fusion": {
                "models": ["panel-a", "panel-b"],
                "judge": {"model": "judge-model", "criteria": "Prefer verifiable evidence"},
                "max_completion_tokens": 321,
                "reasoning": "low",
                "temperature": 0.2,
            },
        },
    )

    assert isinstance(response, ModelResponse)
    assert response.model == "judge-model"
    assert response.choices[0].message.content == "final"
    assert [call["model"] for call in completion.calls] == [
        "judge-model",
        "panel-a",
        "panel-b",
        "judge-model",
        "judge-model",
    ]
    assert completion.max_active_panel_calls == 2
    analyst_call = completion.calls[3]
    assert "Prefer verifiable evidence" in str(analyst_call["messages"])
    panel_calls = completion.calls[1:3]
    assert all(call["max_completion_tokens"] == 321 for call in panel_calls)
    assert all(call["reasoning_effort"] == "low" for call in panel_calls)
    assert all(call["temperature"] == 0.2 for call in panel_calls)
    assert all("web_search_options" not in call for call in completion.calls)


def test_fusion_sdk_config_rejects_recursion_and_more_than_eight_models() -> None:
    with pytest.raises(ValueError, match="cannot be a panel or judge model"):
        FusionSDKConfig.model_validate({"models": ["litellm/fusion-1"]})
    with pytest.raises(ValueError, match="at most 8"):
        FusionSDKConfig.model_validate({"models": [f"model-{index}" for index in range(9)]})


def test_fusion_1_is_registered_as_a_litellm_model() -> None:
    assert litellm.LlmProviders.LITELLM.value == "litellm"
    assert "litellm/fusion-1" in litellm.models_by_provider["litellm"]
    assert litellm.model_cost["litellm/fusion-1"] == {
        "litellm_provider": "litellm",
        "mode": "chat",
        "supported_endpoints": ["/v1/chat/completions", "/v1/messages", "/v1/responses"],
    }


@pytest.mark.asyncio
async def test_shared_model_adapts_to_chat_responses_and_anthropic_messages() -> None:
    chat_model = StaticLiteLLMModel()
    chat_response = await adispatch_completion(
        model="litellm/fusion-1",
        messages=[{"role": "user", "content": "chat question"}],
        stream=False,
        request_kwargs={},
        litellm_model=chat_model,
    )
    assert isinstance(chat_response, ModelResponse)
    assert chat_response.model == "concrete-model"
    assert chat_response._hidden_params["router"] == "litellm/fusion-1"
    assert chat_response._hidden_params["fusion"]["panel_successes"] == 2

    responses_model = StaticLiteLLMModel()
    responses_response = await adispatch_responses(
        model="litellm/fusion-1",
        input="responses question",
        stream=False,
        request_kwargs={},
        litellm_model=responses_model,
    )
    assert responses_response.model == "concrete-model"
    assert responses_response.output_text == "shared answer"
    assert responses_response._hidden_params["router"] == "litellm/fusion-1"
    assert responses_response._hidden_params["fusion"]["panel_successes"] == 2
    assert responses_model.requests[0][0][-1]["content"] == "responses question"

    messages_model = StaticLiteLLMModel()
    messages_response = await adispatch_anthropic_messages(
        model="litellm/fusion-1",
        messages=[{"role": "user", "content": "messages question"}],
        max_tokens=128,
        metadata=None,
        stop_sequences=None,
        stream=False,
        system="Be concise",
        temperature=None,
        thinking=None,
        tool_choice=None,
        tools=None,
        top_k=None,
        top_p=None,
        request_kwargs={},
        litellm_model=messages_model,
    )
    assert messages_response["model"] == "concrete-model"
    assert messages_response["content"][0]["text"] == "shared answer"
    assert messages_response["_hidden_params"]["router"] == "litellm/fusion-1"
    assert messages_response["_hidden_params"]["fusion"]["panel_successes"] == 2
    assert messages_model.requests[0][0][0]["role"] == "system"


def test_public_sdk_surfaces_dispatch_through_the_shared_model() -> None:
    shared_model = StaticLiteLLMModel()
    with patch("litellm.llms.litellm.adapters.get_litellm_model", return_value=shared_model):
        chat_response = litellm.completion(
            model="litellm/fusion-1",
            messages=[{"role": "user", "content": "chat question"}],
        )
        responses_response = litellm.responses(
            model="litellm/fusion-1",
            input="responses question",
        )
        messages_response = litellm.anthropic.messages.create(
            model="litellm/fusion-1",
            messages=[{"role": "user", "content": "messages question"}],
            max_tokens=128,
        )

    assert chat_response.choices[0].message.content == "shared answer"
    assert responses_response.output_text == "shared answer"
    assert messages_response["content"][0]["text"] == "shared answer"
    assert len(shared_model.requests) == 3
