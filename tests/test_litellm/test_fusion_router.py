import asyncio
import inspect
import json
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Final
from unittest.mock import AsyncMock

import pytest

import litellm
from litellm.fusion_router import (
    FUSION_TOOL_NAME,
    FusionRouterConfig,
    _without_stream_tool_call_indexes,
    build_fusion_router,
    fusion_router_dependencies,
    validate_fusion_router_write,
)
from litellm.litellm_core_utils.internal_call_metadata import MODEL_ACCESS_GROUP_METADATA_KEY
from litellm.router import Router
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponseStream
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


def _fusion_call(query: str = "Investigate this") -> ModelResponse:
    return _response(
        None,
        [
            {
                "id": "fusion-call-1",
                "type": "function",
                "function": {"name": FUSION_TOOL_NAME, "arguments": json.dumps({"query": query})},
            }
        ],
    )


def _analysis() -> str:
    return json.dumps(
        {
            "consensus": ["Both approaches agree on the root cause."],
            "contradictions": [
                {
                    "topic": "rollout order",
                    "stances": [
                        {"model": "panel-a", "stance": "lock first"},
                        {"model": "panel-b", "stance": "idempotency first"},
                    ],
                }
            ],
            "partial_coverage": [],
            "unique_insights": [{"model": "panel-b", "insight": "identified an edge case"}],
            "blind_spots": ["Neither response measured latency."],
        }
    )


class RecordingCompletion:
    def __init__(self, responses: Mapping[str, Sequence[ModelResponse | Exception]]) -> None:
        self.responses: Final = {model: deque(values) for model, values in responses.items()}
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
        response = self.responses[model].popleft()
        if isinstance(response, Exception):
            raise response
        return response


def _router(
    completion: RecordingCompletion,
    search=None,
    **config: object,
):
    return build_fusion_router(
        model_name="fusion/test",
        raw_config={
            "outer_model": "outer",
            "panel_models": ["panel-a", "panel-b"],
            "analyst_model": "analyst",
            **config,
        },
        completion=completion,
        search=search,
    )


@pytest.mark.asyncio
async def test_outer_can_skip_fusion_and_answer_or_call_client_tools_directly() -> None:
    completion = RecordingCompletion({"outer": [_response("Hello!")]})
    router = _router(completion)
    client_tool = {
        "type": "function",
        "function": {"name": "send_email", "parameters": {"type": "object"}},
    }

    response = await router.acompletion(
        messages=[{"role": "user", "content": "Say hello"}],
        stream=False,
        request_kwargs={"tools": [client_tool], "tool_choice": "auto"},
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Hello!"
    assert [call["model"] for call in completion.calls] == ["outer"]
    initial = completion.calls[0]
    assert [tool["function"]["name"] for tool in initial["tools"]] == ["send_email", FUSION_TOOL_NAME]
    assert initial["tool_choice"] == "auto"
    assert initial["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response._hidden_params["fusion"]["invoked"] is False


@pytest.mark.asyncio
async def test_outer_client_tool_call_is_returned_without_running_panel_or_second_outer_call() -> None:
    client_call = {
        "id": "email-1",
        "type": "function",
        "function": {"name": "send_email", "arguments": '{"to":"user@example.com"}'},
    }
    completion = RecordingCompletion({"outer": [_response(None, [client_call])]})
    router = _router(completion)

    response = await router.acompletion(
        messages=[{"role": "user", "content": "Send the update"}],
        stream=False,
        request_kwargs={
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "send_email", "parameters": {"type": "object"}},
                }
            ]
        },
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.tool_calls[0].function.name == "send_email"
    assert [call["model"] for call in completion.calls] == ["outer"]


@pytest.mark.asyncio
async def test_mixed_fusion_and_client_tool_calls_return_only_executable_client_calls() -> None:
    client_call = {
        "id": "email-1",
        "type": "function",
        "function": {"name": "send_email", "arguments": '{"to":"user@example.com"}'},
    }
    mixed_response = _response(
        None,
        [
            {
                "id": "fusion-call-1",
                "type": "function",
                "function": {"name": FUSION_TOOL_NAME, "arguments": '{"query":"Investigate this"}'},
            },
            client_call,
        ],
    )
    completion = RecordingCompletion({"outer": [mixed_response]})

    response = await _router(completion).acompletion(
        messages=[{"role": "user", "content": "Research and send the update"}],
        stream=False,
        request_kwargs={
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "send_email", "parameters": {"type": "object"}},
                }
            ]
        },
    )

    assert isinstance(response, ModelResponse)
    assert [call.function.name for call in response.choices[0].message.tool_calls] == ["send_email"]
    assert [call["model"] for call in completion.calls] == ["outer"]
    assert response._hidden_params["fusion"]["invoked"] is False


def test_mixed_stream_removes_private_call_and_reindexes_client_call() -> None:
    chunks = [
        ModelResponseStream(
            choices=[
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "fusion-call-1",
                                "type": "function",
                                "function": {"name": FUSION_TOOL_NAME, "arguments": '{"query":"test"}'},
                            },
                            {
                                "index": 1,
                                "id": "email-1",
                                "type": "function",
                                "function": {"name": "send_email", "arguments": '{"to":"user@example.com"}'},
                            },
                        ]
                    }
                }
            ]
        )
    ]

    sanitized = _without_stream_tool_call_indexes(chunks, frozenset({0}))

    tool_calls = sanitized[0].choices[0].delta.tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0].index == 0
    assert tool_calls[0].function.name == "send_email"


@pytest.mark.asyncio
async def test_responses_only_kwargs_never_reach_fusion_chat_calls() -> None:
    completion = RecordingCompletion({"outer": [_response("Final")]})

    await _router(completion).acompletion(
        messages=[{"role": "system", "content": "Follow instructions"}, {"role": "user", "content": "Answer"}],
        stream=False,
        request_kwargs={
            "input": "raw Responses input",
            "instructions": "Follow instructions",
            "previous_response_id": "resp-1",
            "include": ["reasoning.encrypted_content"],
            "text": {"format": {"type": "text"}},
        },
    )

    outer_call = completion.calls[0]
    assert not {"input", "instructions", "previous_response_id", "include", "text"} & outer_call.keys()


@pytest.mark.asyncio
async def test_forced_fusion_runs_parallel_panel_then_analyst_then_outer() -> None:
    completion = RecordingCompletion(
        {
            "outer": [_fusion_call("Find and fix the race"), _response("Final answer")],
            "panel-a": [_response("Use a lock")],
            "panel-b": [_response("Use idempotency")],
            "analyst": [_response(_analysis())],
        }
    )
    router = _router(completion, invocation="required")
    messages: list[AllMessageValues] = [
        {"role": "system", "content": "Be accurate"},
        {"role": "user", "content": "Fix the bug"},
    ]
    client_tool = {
        "type": "function",
        "function": {"name": "apply_patch", "parameters": {"type": "object"}},
    }

    response = await router.acompletion(
        messages=messages,
        stream=False,
        request_kwargs={
            "tools": [client_tool],
            "tool_choice": "required",
            "litellm_metadata": {
                "user_api_key_user_id": "u-1",
                "user_api_key_budget_reservation": {"id": "parent-only"},
            },
        },
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Final answer"
    assert completion.max_active_panel_calls == 2
    assert [call["model"] for call in completion.calls] == ["outer", "panel-a", "panel-b", "analyst", "outer"]
    panel_calls = completion.calls[1:3]
    assert [tool["function"]["name"] for tool in completion.calls[0]["tools"]] == [
        "apply_patch",
        FUSION_TOOL_NAME,
    ]
    assert completion.calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": FUSION_TOOL_NAME},
    }
    assert all("tools" not in call for call in panel_calls)
    assert all(call["messages"][-1] == {"role": "user", "content": "Find and fix the race"} for call in panel_calls)
    assert all(call["reasoning_effort"] == "none" for call in panel_calls)
    assert all(call["metadata"]["internal_call_origin"] == "fusion_panel" for call in panel_calls)
    reservation = completion.calls[0]["metadata"]["user_api_key_budget_reservation"]
    assert completion.calls[0]["metadata"]["internal_call_origin"] == "fusion_initial"
    assert all(call["metadata"]["user_api_key_budget_reservation"] is reservation for call in panel_calls)
    analyst = completion.calls[3]
    assert analyst["temperature"] == 0
    assert analyst["response_format"] == {"type": "json_object"}
    assert analyst["metadata"]["internal_call_origin"] == "fusion_analyst"
    assert analyst["metadata"]["user_api_key_budget_reservation"] is reservation
    final = completion.calls[4]
    assert final["tools"] == [client_tool]
    assert final["tool_choice"] == "auto"
    continuation = final["messages"]
    assert continuation[0] == messages[0]
    assert continuation[1]["role"] == "developer"
    assert "untrusted evidence" in continuation[1]["content"]
    assert continuation[2] == messages[1]
    assert continuation[3]["tool_calls"][0]["function"]["name"] == FUSION_TOOL_NAME
    payload = json.loads(continuation[4]["content"])
    assert payload["analysis"]["consensus"][0].startswith("Both approaches")
    assert [item["content"] for item in payload["responses"]] == ["Use a lock", "Use idempotency"]
    assert final["metadata"]["internal_call_origin"] == "fusion_continuation"
    assert final["metadata"]["user_api_key_budget_reservation"] is reservation
    assert response._hidden_params["fusion"] == {
        "invoked": True,
        "protocol": "fusion-tool-v1",
        "panel_successes": 2,
        "panel_failures": 0,
        "analysis_available": True,
    }


@pytest.mark.asyncio
async def test_partial_panel_and_invalid_analyst_degrade_to_raw_responses() -> None:
    completion = RecordingCompletion(
        {
            "outer": [_fusion_call(), _response("Recovered")],
            "panel-a": [_response("Useful evidence")],
            "panel-b": [RuntimeError("down")],
            "analyst": [_response("not-json")],
        }
    )

    response = await _router(completion).acompletion(
        messages=[{"role": "user", "content": "Hard question"}],
        stream=False,
        request_kwargs={},
    )

    assert isinstance(response, ModelResponse)
    payload = json.loads(completion.calls[-1]["messages"][-1]["content"])
    assert payload["status"] == "ok"
    assert "analysis" not in payload
    assert payload["responses"][0]["content"] == "Useful evidence"
    assert payload["failed_models"] == [
        {"model": "panel-b", "error_type": "RuntimeError", "failure_reason": "unexpected_error"}
    ]


@pytest.mark.asyncio
async def test_analyst_timeout_degrades_to_raw_panel_responses() -> None:
    class HangingAnalystCompletion(RecordingCompletion):
        def __init__(self) -> None:
            super().__init__(
                {
                    "outer": [_fusion_call(), _response("Final")],
                    "panel-a": [_response("Panel A")],
                    "panel-b": [_response("Panel B")],
                }
            )
            self.analyst_started = asyncio.Event()
            self.analyst_cancelled = asyncio.Event()

        async def __call__(
            self,
            *,
            model: str,
            messages: list[AllMessageValues],
            stream: bool,
            **kwargs: object,
        ) -> ModelResponse | CustomStreamWrapper:
            if model != "analyst":
                return await super().__call__(model=model, messages=messages, stream=stream, **kwargs)
            self.calls.append({"model": model, "messages": messages, "stream": stream, **kwargs})
            self.analyst_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.analyst_cancelled.set()
                raise
            raise AssertionError("unreachable")

    completion = HangingAnalystCompletion()
    response = await asyncio.wait_for(
        _router(completion, panel_timeout_seconds=0.2).acompletion(
            messages=[{"role": "user", "content": "Hard question"}],
            stream=False,
            request_kwargs={},
        ),
        timeout=2,
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Final"
    assert completion.analyst_started.is_set()
    assert completion.analyst_cancelled.is_set()
    assert response._hidden_params["fusion"]["analysis_available"] is False
    payload = json.loads(completion.calls[-1]["messages"][-1]["content"])
    assert [item["content"] for item in payload["responses"]] == ["Panel A", "Panel B"]


@pytest.mark.asyncio
async def test_all_panel_failures_are_a_typed_tool_result_the_outer_can_recover_from() -> None:
    completion = RecordingCompletion(
        {
            "outer": [_fusion_call(), _response("Answered without the panel")],
            "panel-a": [litellm.RateLimitError("slow", "openai", "panel-a")],
            "panel-b": [litellm.RateLimitError("slow", "openai", "panel-b")],
            "analyst": [],
        }
    )

    response = await _router(completion).acompletion(
        messages=[{"role": "user", "content": "Hard question"}],
        stream=False,
        request_kwargs={"tool_choice": "required"},
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Answered without the panel"
    assert [call["model"] for call in completion.calls] == ["outer", "panel-a", "panel-b", "outer"]
    assert completion.calls[0]["tool_choice"] == "required"
    assert "tool_choice" not in completion.calls[-1]
    payload = json.loads(completion.calls[-1]["messages"][-1]["content"])
    assert payload["status"] == "error"
    assert payload["failure_reason"] == "rate_limited"


@pytest.mark.asyncio
async def test_invalid_fusion_arguments_continue_with_typed_error_and_mark_invocation() -> None:
    invalid_fusion_call = _response(
        None,
        [
            {
                "id": "fusion-call-1",
                "type": "function",
                "function": {"name": FUSION_TOOL_NAME, "arguments": "not-json"},
            }
        ],
    )
    completion = RecordingCompletion({"outer": [invalid_fusion_call, _response("Recovered")]})

    response = await _router(completion).acompletion(
        messages=[{"role": "user", "content": "Hard question"}],
        stream=False,
        request_kwargs={},
    )

    assert isinstance(response, ModelResponse)
    assert [call["model"] for call in completion.calls] == ["outer", "outer"]
    payload = json.loads(completion.calls[-1]["messages"][-1]["content"])
    assert payload["status"] == "error"
    assert payload["failure_reason"] == "invalid_tool_arguments"
    assert response._hidden_params["fusion"] == {
        "invoked": True,
        "protocol": "fusion-tool-v1",
        "panel_successes": 0,
        "panel_failures": 0,
        "analysis_available": False,
    }


@pytest.mark.asyncio
async def test_configured_search_tool_is_private_to_panel_and_analyst() -> None:
    search_calls: list[dict[str, object]] = []

    async def search(**kwargs: object) -> object:
        search_calls.append(dict(kwargs))
        return {"results": [{"title": "Source", "url": "https://example.com", "snippet": "Evidence"}]}

    research_call = _response(
        None,
        [
            {
                "id": "search-1",
                "type": "function",
                "function": {"name": "litellm_fusion_search", "arguments": '{"query":"current evidence"}'},
            }
        ],
    )
    completion = RecordingCompletion(
        {
            "outer": [_fusion_call(), _response("Final")],
            "panel-a": [research_call, _response("Evidence-backed answer")],
            "panel-b": [_response("Independent answer")],
            "analyst": [_response(_analysis())],
        }
    )

    reservation = {"id": "shared-reservation"}
    response = await _router(completion, search=search, search_tool_name="web-search", max_tool_calls=4).acompletion(
        messages=[{"role": "user", "content": "Research this"}],
        stream=False,
        request_kwargs={
            "litellm_metadata": {"user_api_key_budget_reservation": reservation},
            "proxy_server_request": {"body": {}},
        },
    )

    assert isinstance(response, ModelResponse)
    assert search_calls[0]["model"] == "web-search"
    assert search_calls[0]["query"] == "current evidence"
    assert search_calls[0]["_fusion_proxy_auth_required"] is True
    assert search_calls[0]["litellm_metadata"]["internal_call_origin"] == "fusion_research"
    assert search_calls[0]["litellm_metadata"]["user_api_key_budget_reservation"] is reservation
    second_panel_call = [call for call in completion.calls if call["model"] == "panel-a"][1]
    assert second_panel_call["messages"][-1]["role"] == "tool"
    assert completion.calls[-1].get("tools") is None


@pytest.mark.asyncio
async def test_search_continuation_drops_provider_prose_and_bounds_arguments() -> None:
    search_queries: list[str] = []

    async def search(*, query: str, **_: object) -> object:
        search_queries.append(query)
        return {"results": [{"snippet": "evidence"}]}

    oversized_query = '\\"' * 2000
    research_call = _response(
        "unneeded provider prose" * 1000,
        [
            {
                "id": "provider-controlled-id" * 1000,
                "type": "function",
                "function": {
                    "name": "litellm_fusion_search",
                    "arguments": json.dumps({"query": oversized_query}),
                },
            }
        ],
    )
    completion = RecordingCompletion(
        {
            "outer": [_fusion_call(), _response("Final")],
            "panel-a": [research_call, _response("Evidence-backed answer")],
            "panel-b": [_response("Independent answer")],
            "analyst": [_response(_analysis())],
        }
    )

    await _router(
        completion,
        search=search,
        search_tool_name="web-search",
        max_tool_calls=1,
        max_candidate_chars=1000,
    ).acompletion(
        messages=[{"role": "user", "content": "Research this"}],
        stream=False,
        request_kwargs={},
    )

    second_panel_call = [call for call in completion.calls if call["model"] == "panel-a"][1]
    assistant_message = second_panel_call["messages"][-2]
    bounded_tool_call = assistant_message["tool_calls"][0]
    assert "content" not in assistant_message
    assert bounded_tool_call["id"] == "fusion-search-0"
    assert len(bounded_tool_call["function"]["arguments"]) <= 1000
    assert len(search_queries[0]) < len(oversized_query)
    assert second_panel_call["messages"][-1]["tool_call_id"] == "fusion-search-0"


@pytest.mark.asyncio
async def test_reserved_tool_name_and_multiple_choices_are_rejected_before_calls() -> None:
    completion = RecordingCompletion({})
    router = _router(completion)
    with pytest.raises(litellm.BadRequestError, match="n=1"):
        await router.acompletion(
            messages=[{"role": "user", "content": "Answer"}], stream=False, request_kwargs={"n": 2}
        )
    with pytest.raises(litellm.BadRequestError, match="reserved"):
        await router.acompletion(
            messages=[{"role": "user", "content": "Answer"}],
            stream=False,
            request_kwargs={
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": FUSION_TOOL_NAME, "parameters": {"type": "object"}},
                    }
                ]
            },
        )
    assert completion.calls == []


def test_config_validation_and_dependencies() -> None:
    assert validate_fusion_router_write("fusion_router", None) is not None
    assert validate_fusion_router_write(
        "fusion_router", {"outer_model": "outer", "panel_models": [f"p-{i}" for i in range(9)]}
    )
    params: Final = {
        "model": "fusion_router",
        "fusion_router_config": {
            "outer_model": "outer",
            "panel_models": ["panel-a", "panel-b", "outer"],
        },
    }
    assert validate_fusion_router_write("fusion_router", params["fusion_router_config"]) is None
    assert [(dependency.model_name, dependency.role) for dependency in fusion_router_dependencies(params)] == [
        ("panel-a", "panel"),
        ("panel-b", "panel"),
        ("outer", "panel"),
        ("outer", "analyst"),
        ("outer", "outer"),
    ]
    with pytest.raises(ValueError, match="Extra inputs"):
        FusionRouterConfig.model_validate(
            {"outer_model": "outer", "panel_models": ["panel"], "aggregator_model": "old-shape"}
        )
    with pytest.raises(ValueError, match="outer_model must not be empty"):
        FusionRouterConfig.model_validate({"outer_model": " ", "panel_models": ["panel"]})


def _router_model_list() -> list[dict[str, object]]:
    return [
        {
            "model_name": "outer",
            "litellm_params": {"model": "openai/test", "api_key": "fake", "mock_response": "Final"},
        },
        {
            "model_name": "panel-a",
            "litellm_params": {"model": "openai/test", "api_key": "fake", "mock_response": "Panel A"},
        },
        {
            "model_name": "fusion/test",
            "litellm_params": {
                "model": "fusion_router",
                "fusion_router_config": {"outer_model": "outer", "panel_models": ["panel-a"]},
            },
        },
    ]


@pytest.mark.asyncio
async def test_router_registers_and_executes_fusion_deployment() -> None:
    router = Router(model_list=_router_model_list())
    assert router.get_configured_mode("fusion/test") is None
    response = await router.acompletion(model="fusion/test", messages=[{"role": "user", "content": "Answer"}])
    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Final"
    deployment = router.get_deployment(model_id=router.model_list[-1]["model_info"]["id"])
    assert deployment is not None
    router._unregister_fusion_router_for_deployment(deployment)  # pyright: ignore[reportPrivateUsage]
    assert "fusion/test" not in router.fusion_routers
    router.init_fusion_router_deployment(deployment)
    router.delete_deployment(id=deployment.model_info.id)
    assert "fusion/test" not in router.fusion_routers


@pytest.mark.asyncio
async def test_proxy_fusion_authorizes_every_hidden_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth import auth_checks

    model_list = _router_model_list()
    model_list.insert(
        -1,
        {
            "model_name": "analyst",
            "litellm_params": {"model": "openai/test", "api_key": "fake", "mock_response": "Analysis"},
        },
    )
    model_list[-1]["litellm_params"]["fusion_router_config"]["analyst_model"] = "analyst"
    router = Router(model_list=model_list)
    authorize = AsyncMock(side_effect=[("outer-budget",), ("panel-budget",), ("analyst-budget",)])
    monkeypatch.setattr(auth_checks, "can_key_call_resolved_model", authorize)

    auth = UserAPIKeyAuth(models=["*"], matched_model_access_groups=["fusion-budget"])
    metadata: dict[str, object] = {"user_api_key_auth": auth}
    await router._authorize_fusion_dependencies(  # pyright: ignore[reportPrivateUsage]
        fusion_router=router.fusion_routers["fusion/test"],
        request_kwargs={
            "metadata": metadata,
            "proxy_server_request": {"body": {"model": "fusion/test"}},
        },
    )

    assert [call.kwargs["model"] for call in authorize.await_args_list] == ["outer", "panel-a", "analyst"]
    assert metadata[MODEL_ACCESS_GROUP_METADATA_KEY] == [
        "fusion-budget",
        "analyst-budget",
        "outer-budget",
        "panel-budget",
    ]
    assert auth.matched_model_access_groups == [
        "fusion-budget",
        "analyst-budget",
        "outer-budget",
        "panel-budget",
    ]


@pytest.mark.asyncio
async def test_proxy_fusion_denies_hidden_model_before_any_provider_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
    from litellm.proxy.auth import auth_checks

    router = Router(model_list=_router_model_list())
    denial = ProxyException(
        message="key not allowed to access model",
        type=ProxyErrorTypes.key_model_access_denied,
        param="model",
        code=403,
    )
    monkeypatch.setattr(auth_checks, "can_key_call_resolved_model", AsyncMock(side_effect=denial))
    fusion_completion = AsyncMock()
    monkeypatch.setattr(router.fusion_routers["fusion/test"], "acompletion", fusion_completion)

    with pytest.raises(ProxyException, match="key not allowed"):
        await router.acompletion(
            model="fusion/test",
            messages=[{"role": "user", "content": "Answer"}],
            metadata={"user_api_key_auth": UserAPIKeyAuth(models=["fusion/test"])},
            proxy_server_request={"body": {"model": "fusion/test"}},
        )

    fusion_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_proxy_fusion_fails_closed_without_authorization_context() -> None:
    from litellm.proxy._types import ProxyException

    router = Router(model_list=_router_model_list())

    with pytest.raises(ProxyException, match="authorization context is missing or invalid"):
        await router.acompletion(
            model="fusion/test",
            messages=[{"role": "user", "content": "Answer"}],
            proxy_server_request={"body": {"model": "fusion/test"}},
        )


@pytest.mark.asyncio
async def test_router_replays_direct_outer_response_as_an_async_stream() -> None:
    router = Router(model_list=_router_model_list())
    response = await router.acompletion(
        model="fusion/test",
        messages=[{"role": "user", "content": "Answer"}],
        stream=True,
    )

    assert isinstance(response, CustomStreamWrapper)
    chunks = [chunk async for chunk in response]
    rebuilt = litellm.stream_chunk_builder(chunks=chunks)
    assert isinstance(rebuilt, ModelResponse)
    assert rebuilt.choices[0].message.content == "Final"
    assert response._hidden_params["fusion"]["invoked"] is False


@pytest.mark.asyncio
async def test_invoked_fusion_closes_suppressed_initial_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    completion = RecordingCompletion(
        {
            "outer": [_response("Final")],
            "panel-a": [_response("Panel A")],
            "panel-b": [_response("Panel B")],
            "analyst": [_response(_analysis())],
        }
    )
    router = _router(completion)
    replay_stream = AsyncMock()
    monkeypatch.setattr(
        router,
        "_initial_outer_call",
        AsyncMock(return_value=(_fusion_call(), replay_stream)),
    )

    response = await router.acompletion(
        messages=[{"role": "user", "content": "Answer"}],
        stream=True,
        request_kwargs={},
    )

    assert isinstance(response, ModelResponse)
    replay_stream.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_fusion_search_checks_proxy_permissions_before_router_search(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
    from litellm.proxy._types import LiteLLM_TeamTable, ProxyException, UserAPIKeyAuth
    from litellm.proxy.auth import auth_checks

    router = Router(model_list=[])
    user_api_key_auth = UserAPIKeyAuth(team_id="restricted-team")
    team = LiteLLM_TeamTable(
        team_id="restricted-team",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="team-permissions",
            search_tools=["allowed-search"],
        ),
    )
    get_team_object = AsyncMock(return_value=team)
    raw_search = AsyncMock(return_value={"results": []})
    monkeypatch.setattr(auth_checks, "get_team_object", get_team_object)
    monkeypatch.setattr(router, "asearch", raw_search)

    with pytest.raises(ProxyException, match="Team not allowed to access search tool"):
        await router._fusion_asearch(  # pyright: ignore[reportPrivateUsage]
            model="restricted-search",
            query="evidence",
            litellm_metadata={"user_api_key_auth": user_api_key_auth.model_dump()},
        )

    get_team_object.assert_awaited_once()
    raw_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_fusion_search_fails_closed_when_proxy_auth_context_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy._types import ProxyException

    router = Router(model_list=[])
    raw_search = AsyncMock(return_value={"results": []})
    monkeypatch.setattr(router, "asearch", raw_search)

    with pytest.raises(ProxyException, match="authorization context is missing or invalid"):
        await router._fusion_asearch(  # pyright: ignore[reportPrivateUsage]
            model="restricted-search",
            query="evidence",
            litellm_metadata={},
            _fusion_proxy_auth_required=True,
        )

    raw_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_router_responses_and_anthropic_adapters_use_same_fusion_model() -> None:
    router = Router(model_list=_router_model_list())
    responses_result = await router._fusion_aware_aresponses(model="fusion/test", input="Answer")
    assert responses_result.output[0].content[0].text == "Final"
    assert inspect.iscoroutinefunction(router.aanthropic_messages)
    anthropic_result = await router._fusion_aware_aanthropic_messages(
        model="fusion/test", messages=[{"role": "user", "content": "Answer"}], max_tokens=256
    )
    assert anthropic_result["content"][0]["text"] == "Final"


@pytest.mark.asyncio
async def test_router_responses_and_anthropic_adapters_stream_direct_outer_response() -> None:
    router = Router(model_list=_router_model_list())

    responses_stream = await router.aresponses(model="fusion/test", input="Answer", stream=True)
    response_events = [event async for event in responses_stream]
    assert any(str(getattr(event, "type", "")).endswith("RESPONSE_COMPLETED") for event in response_events)

    anthropic_stream = await router.aanthropic_messages(
        model="fusion/test",
        messages=[{"role": "user", "content": "Answer"}],
        max_tokens=256,
        stream=True,
    )
    anthropic_events = [event async for event in anthropic_stream]
    assert anthropic_events
    assert all(isinstance(event, bytes) for event in anthropic_events)


def test_sync_router_and_responses_support_nonstreaming_fusion() -> None:
    router = Router(model_list=_router_model_list())
    response = router.completion(model="fusion/test", messages=[{"role": "user", "content": "Answer"}])
    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "Final"
    responses_result = router._fusion_aware_responses(model="fusion/test", input="Answer")
    assert responses_result.output[0].content[0].text == "Final"
