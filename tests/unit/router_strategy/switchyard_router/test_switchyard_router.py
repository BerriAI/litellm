import asyncio
import json
from itertools import chain
from collections.abc import AsyncIterator, Mapping
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_api import ChangeSet
from litellm.router_strategy.switchyard_router.config import SwitchyardRouterConfig, build_algorithm
from litellm.router_strategy.switchyard_router.switchyard_router import SwitchyardRouter
from litellm.types.decisions import OpenAIDecisionResponse
from litellm.types.router import Deployment
from litellm.types.utils import ModelResponse

libsy = pytest.importorskip("switchyard_rust.libsy")

HELLO: Final = [{"role": "user", "content": "Hello"}]


async def unexpected_judge(request: Mapping[str, object]) -> ModelResponse:
    raise AssertionError("This route must not call an LLM judge")


async def unexpected_decision(request: Mapping[str, object]) -> OpenAIDecisionResponse:
    raise AssertionError("This route must not call a decision model")


def config(
    *, classifier: Mapping[str, object] | None = None, picker: str = "efficient_first", threshold: float | None = None
) -> SwitchyardRouterConfig:
    return SwitchyardRouterConfig.model_validate(
        {
            "models": {
                "capable": "strong",
                "efficient": "fast",
                **({"judge": "judge"} if classifier is not None else {}),
            },
            "options": {"picker": picker, **({"confidence_threshold": threshold} if threshold is not None else {})},
            **({"classifier": classifier} if classifier is not None else {}),
        }
    )


def router(
    route_config: SwitchyardRouterConfig,
    *,
    completion=unexpected_judge,
    decide=unexpected_decision,
    algorithm: object = None,
) -> SwitchyardRouter:
    return SwitchyardRouter("route", route_config, algorithm or build_algorithm(route_config), completion, decide)


async def route(
    route_config: SwitchyardRouterConfig,
    messages: object,
    *,
    call_type: str | None = "acompletion",
    request_kwargs: Mapping[str, object] | None = None,
    **router_kwargs: object,
) -> tuple[str, Mapping[str, object]]:
    result: Final = await router(route_config, **router_kwargs).async_pre_routing_hook(  # pyright: ignore[reportArgumentType]  # test doubles
        "route", request_kwargs or {}, messages, call_type=call_type
    )
    assert result is not None
    assert result.routing_decision is not None
    return result.model, result.routing_decision


def anthropic_turn(call_id: str, tool: str, result: str, marked: bool) -> tuple[dict[str, object], ...]:
    error: Final = {"is_error": True} if marked else {}
    return (
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Next step", "signature": "sig"},
                {"type": "tool_use", "id": call_id, "name": tool, "input": {"file_path": "a.py"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": result, **error}]},
    )


def responses_turn(call_id: str, tool: str, result: str, marked: bool) -> tuple[dict[str, object], ...]:
    return (
        {"type": "function_call", "call_id": call_id, "name": tool, "arguments": '{"file_path":"a.py"}'},
        {"type": "function_call_output", "call_id": call_id, "output": result},
    )


def chat_turn(call_id: str, tool: str, result: str, marked: bool) -> tuple[dict[str, object], ...]:
    call: Final = {"id": call_id, "type": "function", "function": {"name": tool, "arguments": '{"file_path":"a.py"}'}}
    return (
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    )


TURN_FORMATS: Final = {"anthropic_messages": anthropic_turn, "aresponses": responses_turn, "acompletion": chat_turn}


def transcript(endpoint: str, tool: str, results: tuple[str, ...], *, marked: bool = False) -> list[dict[str, object]]:
    turn: Final = TURN_FORMATS[endpoint]
    turns: Final = chain.from_iterable(
        turn(f"call{index}", tool, result, marked) for index, result in enumerate(results)
    )
    return [{"role": "user", "content": "Fix the helper"}, *turns]


def llm_router(
    *, route_config: Mapping[str, object], judge_fails: bool = False, num_retries: int = 0
) -> litellm.Router:
    def deployment(name: str, reply: object) -> Mapping[str, object]:
        return {
            "model_name": name,
            "litellm_params": {"model": f"openai/{name}", "api_key": "k", "mock_response": reply},
        }

    return litellm.Router(
        num_retries=num_retries,
        model_list=[
            deployment("strong", "strong"),
            deployment("fast", "fast"),
            *((deployment("judge", Exception("judge down")),) if judge_fails else ()),
            switchyard_deployment(route_config),
        ],
    )


def switchyard_deployment(route_config: Mapping[str, object]) -> dict[str, object]:
    return {
        "model_name": "route",
        "litellm_params": {"model": "auto_router/switchyard", "switchyard_router_config": route_config},
        "model_info": {"id": "route-id"},
    }


@pytest.mark.asyncio
async def test_critical_tool_error_selects_capable_without_rewriting_messages() -> None:
    messages: Final = transcript("acompletion", "bash", ("CUDA out of memory",))
    result: Final = await router(config()).async_pre_routing_hook("route", {}, messages, call_type="acompletion")
    assert result is not None
    assert result.model == "strong"
    assert result.messages == messages


async def routed_reply(proxy: litellm.Router, endpoint: str, edits: tuple[str, ...]) -> object:
    match endpoint:
        case "anthropic_messages":
            message: Final = await proxy.aanthropic_messages(
                model="route", messages=transcript(endpoint, "Edit", edits), max_tokens=16
            )
            return message["content"][0]["text"]  # pyright: ignore[reportIndexIssue]  # non-streaming response
        case "aresponses":
            response: Final = await proxy.aresponses(model="route", input=transcript(endpoint, "Edit", edits))
            return response.output[0].content[0].text  # pyright: ignore[reportAttributeAccessIssue]  # non-streaming response
        case _:
            chat: Final = await proxy.acompletion(model="route", messages=transcript(endpoint, "Edit", edits))
            return chat.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]  # non-streaming response


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["acompletion", "anthropic_messages", "aresponses"])
async def test_router_routes_agent_traffic_from_every_endpoint(endpoint: str) -> None:
    proxy: Final = llm_router(route_config={"models": {"capable": "strong", "efficient": "fast"}})
    assert await routed_reply(proxy, endpoint, ("Edit applied",) * 3) == "fast"


@pytest.mark.asyncio
async def test_router_follows_an_edited_switchyard_router_and_forgets_a_deleted_one() -> None:
    proxy: Final = llm_router(route_config={"models": {"capable": "strong", "efficient": "fast"}})
    assert await routed_reply(proxy, "acompletion", ("Edit applied",) * 3) == "fast"

    swapped: Final = switchyard_deployment({"models": {"capable": "fast", "efficient": "strong"}})
    _ = proxy.apply(ChangeSet(upsert=[Deployment(**swapped)]))
    assert await routed_reply(proxy, "acompletion", ("Edit applied",) * 3) == "strong"

    _ = proxy.apply(ChangeSet(delete=["route-id"]))
    with pytest.raises(litellm.BadRequestError):
        await proxy.acompletion(model="route", messages=HELLO)


@pytest.mark.asyncio
@pytest.mark.parametrize(("marked", "expected"), [(True, "strong"), (False, "fast")])
async def test_anthropic_tool_failure_marker_reaches_switchyard(marked: bool, expected: str) -> None:
    failures: Final = transcript("anthropic_messages", "Edit", ("String to replace not found.",) * 4, marked=marked)
    model, decision = await route(config(), failures, call_type="anthropic_messages")
    assert model == expected
    assert "switchyard_fallback_reason" not in decision


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_kwargs", "expected"),
    [
        ({"litellm_session_id": "s1"}, ["strong", "strong", "strong", "fast"]),
        (
            {"proxy_server_request": {"headers": {"X-Session-Id": "s1", "session-id": "s1"}}},
            ["strong", "fast", "fast", "fast"],
        ),
        ({}, ["strong", "fast", "fast", "fast"]),
    ],
)
async def test_capable_hold_follows_the_session(request_kwargs: Mapping[str, object], expected: list[str]) -> None:
    switchyard: Final = router(config())
    turns: Final = (("CUDA out of memory",), *((("All tests passed",) * count) for count in (2, 3, 4)))
    picks: Final = [await pick(switchyard, request_kwargs, results) for results in turns]
    assert picks == expected


async def pick(switchyard: SwitchyardRouter, request_kwargs: Mapping[str, object], results: tuple[str, ...]) -> str:
    result: Final = await switchyard.async_pre_routing_hook(
        "route", request_kwargs, transcript("acompletion", "bash", results), call_type="acompletion"
    )
    assert result is not None
    return result.model


@pytest.mark.asyncio
async def test_a_session_hold_never_crosses_api_keys() -> None:
    switchyard: Final = router(config())
    owner: Final = {"litellm_session_id": "shared", "metadata": {"user_api_key_hash": "owner"}}
    other: Final = {"litellm_session_id": "shared", "metadata": {"user_api_key_hash": "other"}}
    assert await pick(switchyard, owner, ("CUDA out of memory",)) == "strong"
    assert await pick(switchyard, other, ("All tests passed",) * 2) == "fast"
    assert await pick(switchyard, owner, ("All tests passed",) * 2) == "strong"


@pytest.mark.asyncio
async def test_a_client_metadata_key_hash_cannot_join_another_keys_session() -> None:
    switchyard: Final = router(config())
    owner: Final = {"litellm_session_id": "shared", "litellm_metadata": {"user_api_key_hash": "owner"}}
    spoofer: Final = {
        "litellm_session_id": "shared",
        "metadata": {"user_api_key_hash": "owner"},
        "litellm_metadata": {"user_api_key_hash": "spoofer"},
    }
    assert await pick(switchyard, owner, ("CUDA out of memory",)) == "strong"
    assert await pick(switchyard, spoofer, ("All tests passed",) * 2) == "fast"


@pytest.mark.asyncio
@pytest.mark.parametrize(("p_solve", "expected"), [(0.0, "strong"), (1.0, "fast")])
async def test_llm_judge_verdict_selects_the_target(p_solve: float, expected: str) -> None:
    async def judge(request: Mapping[str, object]) -> ModelResponse:
        assert "autorouter_classifier" in str(request["metadata"])
        verdict: Final = {"crux": "c", "primary_rule": "none", "capability_boundary": "unmatched", "p_solve": p_solve}
        return ModelResponse(choices=[{"message": {"role": "assistant", "content": json.dumps(verdict)}}])

    model, _ = await route(config(classifier={}), HELLO, completion=judge)
    assert model == expected


def decision_answer(advantage: float) -> OpenAIDecisionResponse:
    return OpenAIDecisionResponse.model_validate(
        {
            "model": "jev",
            "answers": [
                {
                    "type": "choice",
                    "name": "route",
                    "choice": "advantage",
                    "probabilities": [
                        {"value": "advantage", "probability": advantage},
                        {"value": "no_advantage", "probability": 1 - advantage},
                    ],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_bucket", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("classifier", [{}, {"kind": "decision"}])
async def test_judge_calls_carry_the_callers_tags_and_origin_where_their_endpoint_reads_them(
    parent_bucket: str, classifier: Mapping[str, object]
) -> None:
    seen: Final[list[Mapping[str, object]]] = []  # mutable-ok: collects the judge request

    async def judge(request: Mapping[str, object]) -> ModelResponse:
        seen.append(request)
        verdict: Final = {"crux": "c", "primary_rule": "none", "capability_boundary": "unmatched", "p_solve": 1.0}
        return ModelResponse(choices=[{"message": {"role": "assistant", "content": json.dumps(verdict)}}])

    async def decide(request: Mapping[str, object]) -> OpenAIDecisionResponse:
        seen.append(request)
        return decision_answer(0.0)

    caller: Final = {parent_bucket: {"tags": ["eu-only"], "inherited_tags": ["team-eu"]}}
    await route(
        config(classifier=classifier, threshold=1.0), HELLO, completion=judge, decide=decide, request_kwargs=caller
    )
    (request,) = seen
    read_by_endpoint: Final = "litellm_metadata" if classifier else "metadata"
    assert set(request) & {"metadata", "litellm_metadata"} == {read_by_endpoint}
    metadata: Final = request[read_by_endpoint]
    assert metadata["tags"] == ["eu-only"]  # pyright: ignore[reportIndexIssue]  # the judge request's metadata dict
    assert metadata["inherited_tags"] == ["team-eu"]  # pyright: ignore[reportIndexIssue]  # same
    assert metadata["internal_call_origin"] == "autorouter_classifier"  # pyright: ignore[reportIndexIssue]  # same


@pytest.mark.asyncio
@pytest.mark.parametrize(("advantage", "expected"), [(0.61, "strong"), (0.6, "fast")])
async def test_decision_model_advantage_must_exceed_the_cutoff(advantage: float, expected: str) -> None:
    async def decide(request: Mapping[str, object]) -> OpenAIDecisionResponse:
        (question,) = request["questions"]  # pyright: ignore[reportGeneralTypeIssues]  # the decisions request shape
        assert {choice["value"] for choice in question["choices"]} == {"advantage", "no_advantage"}
        return decision_answer(advantage)

    model, decision = await route(
        config(classifier={"kind": "decision", "cutoff": 0.6}, threshold=1.0), HELLO, decide=decide
    )
    assert model == expected
    assert decision["switchyard_evidence"]["score"] == advantage


async def judge_down(request: Mapping[str, object]) -> ModelResponse:
    raise ConnectionError("judge down")


async def decision_refused(request: Mapping[str, object]) -> OpenAIDecisionResponse:
    return OpenAIDecisionResponse.model_validate(
        {
            "model": "jev",
            "answers": [{"type": "refusal", "name": "route"}],
            "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("classifier", "router_kwargs", "picker", "expected"),
    [
        ({}, {"completion": judge_down}, "efficient_first", "fast"),
        ({}, {"completion": judge_down}, "capable_first", "strong"),
        ({"kind": "decision"}, {"decide": decision_refused}, "efficient_first", "fast"),
    ],
)
async def test_classifier_failure_falls_open_to_the_picker_default(
    classifier: Mapping[str, object], router_kwargs: Mapping[str, object], picker: str, expected: str
) -> None:
    model, decision = await route(config(classifier=classifier, picker=picker, threshold=1.0), HELLO, **router_kwargs)
    assert model == expected
    assert "switchyard_fallback_reason" not in decision


class EndsWithoutDecision:
    async def _steps(self) -> AsyncIterator[object]:
        for step in ():
            yield step

    def run_stream(self, *args: object, **kwargs: object) -> AsyncIterator[object]:
        return self._steps()


class RaisesLibsyError:
    async def _steps(self) -> AsyncIterator[object]:
        raise libsy.LibsyError("algorithm failed")
        yield

    def run_stream(self, *args: object, **kwargs: object) -> AsyncIterator[object]:
        return self._steps()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_config", "messages", "call_type", "algorithm", "reason"),
    [
        (
            config(),
            [{"role": "user", "content": [{"type": "input_audio"}]}],
            "acompletion",
            None,
            "unsupported_content",
        ),
        (config(), HELLO, None, None, "unsupported_request_type"),
        (config(), HELLO, "acompletion", EndsWithoutDecision(), "no_decision"),
        (config(), HELLO, "acompletion", RaisesLibsyError(), "libsy_error"),
        (
            config(classifier={}, threshold=1.0),
            HELLO,
            "acompletion",
            build_algorithm(config(classifier={"kind": "decision"}, threshold=1.0)),
            "unexpected_decision_call",
        ),
        (
            config(classifier={"kind": "decision"}, threshold=1.0),
            HELLO,
            "acompletion",
            build_algorithm(config(classifier={}, threshold=1.0)),
            "unexpected_judge_call",
        ),
    ],
)
async def test_anything_switchyard_cannot_route_goes_to_capable(
    route_config: SwitchyardRouterConfig, messages: object, call_type: str | None, algorithm: object, reason: str
) -> None:
    model, decision = await route(route_config, messages, call_type=call_type, algorithm=algorithm)
    assert model == "strong"
    assert decision["switchyard_fallback_reason"] == reason


@pytest.mark.asyncio
async def test_failing_judge_is_called_once_even_when_the_router_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: Final[list[str]] = []

    class CountAttempts(CustomLogger):
        def log_pre_api_call(self, model: str, messages: object, kwargs: dict[str, object]) -> None:
            attempts.append(model)

    monkeypatch.setattr(litellm, "callbacks", [CountAttempts()])
    proxy: Final = llm_router(
        route_config={
            "models": {"capable": "strong", "efficient": "fast", "judge": "judge"},
            "classifier": {},
            "options": {"confidence_threshold": 1.0},
        },
        judge_fails=True,
        num_retries=3,
    )
    await proxy.acompletion(model="route", messages=HELLO)
    assert attempts.count("judge") == 1


@pytest.mark.asyncio
async def test_judge_cancellation_propagates() -> None:
    async def cancelled(request: Mapping[str, object]) -> ModelResponse:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await route(config(classifier={}), HELLO, completion=cancelled)
