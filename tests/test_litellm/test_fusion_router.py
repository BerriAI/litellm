import asyncio
import json
from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.fusion_router import (
    FusionRouterConfig,
    build_fusion_router,
    fusion_router_dependencies,
    validate_fusion_router_write,
)
from litellm.router import Router
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import CustomStreamWrapper, ModelResponse


def _response(content: str | None, tool_calls: list[dict[str, object]] | None = None) -> ModelResponse:
    return ModelResponse(
        choices=[
            {
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
            }
        ]
    )


class RecordingCompletion:
    def __init__(self, responses: Mapping[str, ModelResponse | Exception | CustomStreamWrapper]) -> None:
        self.responses: Final = responses
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
        response: Final = self.responses[model]
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
async def test_panel_runs_in_parallel_and_aggregator_synthesizes_from_canonical_history() -> None:
    completion = RecordingCompletion(
        {
            "panel-a": _response("First approach"),
            "panel-b": _response("Second approach"),
            "aggregator": _response("Synthesized answer"),
        }
    )
    router = build_fusion_router(
        model_name="fusion/coding",
        raw_config={"panel_models": ["panel-a", "panel-b"], "aggregator_model": "aggregator"},
        completion=completion,
    )
    messages: list[AllMessageValues] = [
        {"role": "system", "content": "Be accurate"},
        {"role": "user", "content": "Fix the bug"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "tool_call_id": "call-1", "content": "traceback"},
        {"role": "user", "content": "Continue"},
    ]

    response = await router.acompletion(messages=messages, stream=False, request_kwargs={})

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Synthesized answer"
    assert completion.max_active_panel_calls == 2
    panel_calls = completion.calls[:2]
    assert {call["model"] for call in panel_calls} == {"panel-a", "panel-b"}
    assert all(call["messages"] == messages for call in panel_calls)
    aggregator_messages = completion.calls[-1]["messages"]
    assert isinstance(aggregator_messages, list)
    assert aggregator_messages[0] == messages[0]
    assert aggregator_messages[1]["role"] == "developer"
    assert aggregator_messages[2:] == messages[1:]
    candidate_payload = str(aggregator_messages[1]["content"]).split("Candidate responses:\n", 1)[1]
    candidates = json.loads(candidate_payload)
    assert [candidate["content"] for candidate in candidates] == ["First approach", "Second approach"]


@pytest.mark.asyncio
async def test_panel_gets_only_function_schemas_and_aggregator_owns_tool_call() -> None:
    completion = RecordingCompletion(
        {
            "panel-a": _response(
                None,
                [
                    {
                        "id": "panel-call-id",
                        "type": "function",
                        "function": {"name": "send_email", "arguments": '{"to":"a@example.com"}'},
                    }
                ],
            ),
            "panel-b": _response("Ask before sending"),
            "aggregator": _response(
                None,
                [
                    {
                        "id": "authoritative-call-id",
                        "type": "function",
                        "function": {"name": "send_email", "arguments": '{"to":"a@example.com"}'},
                    }
                ],
            ),
        }
    )
    router = build_fusion_router(
        model_name="fusion/actions",
        raw_config={"panel_models": ["panel-a", "panel-b"], "aggregator_model": "aggregator"},
        completion=completion,
    )
    function_tool: Final = {
        "type": "function",
        "function": {"name": "send_email", "parameters": {"type": "object"}},
    }
    hosted_tool: Final = {"type": "web_search_preview"}
    hosted_tool_choice: Final = {"type": "web_search_preview"}

    response = await router.acompletion(
        messages=[{"role": "user", "content": "Send the update"}],
        stream=False,
        request_kwargs={
            "tools": [function_tool, hosted_tool],
            "tool_choice": hosted_tool_choice,
            "litellm_metadata": {
                "user_api_key_budget_reservation": {"id": "must-not-propagate"},
                "user_api_key_user_id": "u-1",
            },
        },
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.tool_calls[0].id == "authoritative-call-id"
    for panel_call in completion.calls[:2]:
        assert panel_call["tools"] == [function_tool]
        assert "tool_choice" not in panel_call
        assert panel_call["metadata"]["internal_call_origin"] == "fusion_panel"
        assert "user_api_key_budget_reservation" not in panel_call["metadata"]
    aggregator_call = completion.calls[-1]
    assert aggregator_call["tools"] == [function_tool, hosted_tool]
    assert aggregator_call["tool_choice"] == hosted_tool_choice
    aggregator_messages = aggregator_call["messages"]
    assert isinstance(aggregator_messages, list)
    instruction = str(aggregator_messages[0]["content"])
    assert "panel-call-id" not in instruction
    assert "send_email" in instruction


@pytest.mark.asyncio
async def test_quorum_failure_modes_and_candidate_bound() -> None:
    failing_completion = RecordingCompletion(
        {
            "panel-a": _response("x" * 2000),
            "panel-b": RuntimeError("provider down"),
            "aggregator": _response("fallback"),
        }
    )
    fail_router = build_fusion_router(
        model_name="fusion/quality",
        raw_config={"panel_models": ["panel-a", "panel-b"], "aggregator_model": "aggregator"},
        completion=failing_completion,
    )
    with pytest.raises(litellm.ServiceUnavailableError, match="quorum"):
        await fail_router.acompletion(messages=[{"role": "user", "content": "Answer"}], stream=False, request_kwargs={})
    assert [call["model"] for call in failing_completion.calls] == ["panel-a", "panel-b"]

    resilient_completion = RecordingCompletion(failing_completion.responses)
    resilient_router = build_fusion_router(
        model_name="fusion/resilient",
        raw_config={
            "panel_models": ["panel-a", "panel-b"],
            "aggregator_model": "aggregator",
            "on_quorum_failure": "aggregator_only",
            "max_candidate_chars": 1000,
        },
        completion=resilient_completion,
    )
    response = await resilient_router.acompletion(
        messages=[{"role": "user", "content": "Answer"}], stream=False, request_kwargs={}
    )
    assert isinstance(response, ModelResponse)
    assert resilient_completion.calls[-1]["messages"] == [{"role": "user", "content": "Answer"}]

    bounded_completion = RecordingCompletion(
        {
            "panel-a": _response("x" * 2000),
            "panel-b": _response("second"),
            "aggregator": _response("bounded"),
        }
    )
    bounded_router = build_fusion_router(
        model_name="fusion/bounded",
        raw_config={
            "panel_models": ["panel-a", "panel-b"],
            "aggregator_model": "aggregator",
            "max_candidate_chars": 1000,
        },
        completion=bounded_completion,
    )
    await bounded_router.acompletion(messages=[{"role": "user", "content": "Answer"}], stream=False, request_kwargs={})
    instruction = str(bounded_completion.calls[-1]["messages"][0]["content"])
    payload = json.loads(instruction.split("Candidate responses:\n", 1)[1])
    assert len(payload[0]["content"]) == 1000


@pytest.mark.asyncio
async def test_n_greater_than_one_is_rejected_before_any_child_call() -> None:
    completion = RecordingCompletion({})
    router = build_fusion_router(
        model_name="fusion/test",
        raw_config={"panel_models": ["panel-a", "panel-b"], "aggregator_model": "aggregator"},
        completion=completion,
    )
    with pytest.raises(litellm.BadRequestError, match="n=1"):
        await router.acompletion(
            messages=[{"role": "user", "content": "Answer"}], stream=False, request_kwargs={"n": 2}
        )
    assert completion.calls == []


def test_config_write_validation_and_dependencies() -> None:
    assert validate_fusion_router_write("openai/gpt-4o", {"panel_models": [], "aggregator_model": "x"}) is not None
    assert validate_fusion_router_write("fusion_router", None) is not None
    assert (
        validate_fusion_router_write(
            "fusion_router",
            {"panel_models": ["same", "same"], "aggregator_model": "aggregator"},
        )
        is not None
    )
    params: Final = {
        "model": "fusion_router",
        "fusion_router_config": {
            "panel_models": ["panel-a", "panel-b", "aggregator"],
            "aggregator_model": "aggregator",
        },
    }
    assert validate_fusion_router_write("fusion_router", params["fusion_router_config"]) is None
    assert [(dependency.model_name, dependency.role) for dependency in fusion_router_dependencies(params)] == [
        ("panel-a", "panel"),
        ("panel-b", "panel"),
        ("aggregator", "panel"),
        ("aggregator", "aggregator"),
    ]


def test_config_is_frozen_and_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        FusionRouterConfig.model_validate(
            {"panel_models": ["panel-a", "panel-b"], "aggregator_model": "aggregator", "cadence": "automatic"}
        )


def _router_model_list() -> list[dict[str, object]]:
    return [
        {
            "model_name": "panel-a",
            "litellm_params": {"model": "openai/test", "api_key": "fake", "mock_response": "Panel A"},
        },
        {
            "model_name": "panel-b",
            "litellm_params": {"model": "openai/test", "api_key": "fake", "mock_response": "Panel B"},
        },
        {
            "model_name": "aggregator",
            "litellm_params": {"model": "openai/test", "api_key": "fake", "mock_response": "Final"},
        },
        {
            "model_name": "fusion/test",
            "litellm_params": {
                "model": "fusion_router",
                "fusion_router_config": {
                    "panel_models": ["panel-a", "panel-b"],
                    "aggregator_model": "aggregator",
                },
            },
        },
    ]


@pytest.mark.asyncio
async def test_router_registers_and_executes_fusion_deployment() -> None:
    router = Router(model_list=_router_model_list())

    response = await router.acompletion(model="fusion/test", messages=[{"role": "user", "content": "Answer"}])

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Final"
    deployment = router.get_deployment(model_id=router.model_list[-1]["model_info"]["id"])
    assert deployment is not None
    router.delete_deployment(id=deployment.model_info.id)
    assert "fusion/test" not in router.fusion_routers


def test_router_upsert_replaces_fusion_config_and_restores_after_invalid_update() -> None:
    router = Router(model_list=_router_model_list(), ignore_invalid_deployments=True)
    model_id = router.model_list[-1]["model_info"]["id"]
    deployment = router.get_deployment(model_id=model_id)
    assert deployment is not None
    updated = deployment.model_copy(deep=True)
    updated.litellm_params.fusion_router_config = {
        "panel_models": ["panel-a", "panel-b"],
        "aggregator_model": "panel-a",
    }

    assert router.upsert_deployment(updated) is not None
    assert router.fusion_routers["fusion/test"].config.aggregator_model == "panel-a"

    stored = router.get_deployment(model_id=model_id)
    assert stored is not None
    invalid = stored.model_copy(deep=True)
    invalid.litellm_params.fusion_router_config = {
        "panel_models": ["panel-a"],
        "aggregator_model": "aggregator",
    }
    assert router.upsert_deployment(invalid) is None
    assert router.fusion_routers["fusion/test"].config.aggregator_model == "panel-a"
    assert router.get_deployment(model_id=model_id) is not None


@pytest.mark.asyncio
async def test_router_responses_api_bridges_through_the_same_fusion_model() -> None:
    router = Router(model_list=_router_model_list())

    response = await router.aresponses(model="fusion/test", input="Answer")

    assert response.output[0].content[0].text == "Final"
    with pytest.raises(litellm.BadRequestError, match="Background Responses"):
        await router.aresponses(model="fusion/test", input="Answer", background=True)


@pytest.mark.asyncio
async def test_router_anthropic_messages_bridges_through_the_same_fusion_model() -> None:
    router = Router(model_list=_router_model_list())

    response = await router.aanthropic_messages(
        model="fusion/test",
        messages=[{"role": "user", "content": "Answer"}],
        max_tokens=256,
    )
    alias_response = await router.anthropic_messages(
        model="fusion/test",
        messages=[{"role": "user", "content": "Answer"}],
        max_tokens=256,
    )

    assert response["content"][0]["text"] == "Final"
    assert alias_response["content"][0]["text"] == "Final"


def test_sync_responses_api_supports_nonstreaming_fusion() -> None:
    router = Router(model_list=_router_model_list())

    response = router.responses(model="fusion/test", input="Answer")

    assert response.output[0].content[0].text == "Final"
    with pytest.raises(litellm.BadRequestError, match="Synchronous Responses streaming"):
        router.responses(model="fusion/test", input="Answer", stream=True)


def test_sync_router_supports_nonstreaming_fusion_and_rejects_sync_streaming() -> None:
    router = Router(model_list=_router_model_list())

    response = router.completion(model="fusion/test", messages=[{"role": "user", "content": "Answer"}])

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Final"
    with pytest.raises(litellm.BadRequestError, match="Synchronous streaming"):
        router.completion(
            model="fusion/test",
            messages=[{"role": "user", "content": "Answer"}],
            stream=True,
        )


@pytest.mark.asyncio
async def test_router_rejects_recursive_fusion_members() -> None:
    model_list = _router_model_list()
    model_list.append(
        {
            "model_name": "fusion/recursive",
            "litellm_params": {
                "model": "fusion_router",
                "fusion_router_config": {
                    "panel_models": ["fusion/test", "panel-a"],
                    "aggregator_model": "aggregator",
                    "on_quorum_failure": "aggregator_only",
                },
            },
        }
    )
    router = Router(model_list=model_list)

    response = await router.acompletion(
        model="fusion/recursive",
        messages=[{"role": "user", "content": "Answer"}],
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Final"
