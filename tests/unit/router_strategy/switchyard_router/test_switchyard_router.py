import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Final

import pytest

from litellm.router_strategy.switchyard_router.config import SwitchyardRouterConfig, build_algorithm
from litellm.router_strategy.switchyard_router.switchyard_router import SwitchyardRouter
from litellm.types.decisions import OpenAIDecisionResponse
from litellm.types.utils import ModelResponse

libsy = pytest.importorskip("switchyard_rust.libsy")


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


def tool_history(*results: str) -> list[dict[str, object]]:
    turns: Final = (
        message
        for index, result in enumerate(results)
        for message in (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call{index}",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"pytest"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": f"call{index}", "content": result},
        )
    )
    return [{"role": "user", "content": "Apply the edit and run the tests"}, *turns]


def decision_answer(advantage: float) -> OpenAIDecisionResponse:
    return OpenAIDecisionResponse.model_validate(
        {
            "model": "jev-1.14.0",
            "answers": [
                {
                    "type": "choice",
                    "name": "route",
                    "choice": "advantage" if advantage > 0.5 else "no_advantage",
                    "probabilities": [
                        {"value": "advantage", "probability": advantage},
                        {"value": "no_advantage", "probability": 1 - advantage},
                    ],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
        }
    )


@pytest.mark.asyncio
async def test_critical_tool_error_selects_capable_without_rewriting_messages() -> None:
    messages: Final = tool_history("CUDA out of memory")
    result: Final = await router(config()).async_pre_routing_hook("route", {}, messages)
    assert result is not None
    assert result.model == "strong"
    assert result.messages == messages
    assert result.routing_decision is not None
    assert result.routing_decision["router_type"] == "switchyard"
    assert result.routing_decision["switchyard_evidence"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Edit applied successfully", "All tests passed", "Hello"])
async def test_heuristic_decision_is_a_configured_target_with_evidence(text: str) -> None:
    result: Final = await router(config()).async_pre_routing_hook("route", {}, tool_history(text))
    assert result is not None
    assert result.model in {"strong", "fast"}
    assert result.routing_decision is not None
    assert result.routing_decision["routed_model"] == result.model
    assert result.routing_decision["switchyard_evidence"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("session_kwargs", "expected"),
    [
        ({"litellm_session_id": "session-1"}, ["strong", "strong", "strong", "fast"]),
        ({"proxy_server_request": {"headers": {"X-Session-Id": "session-1"}}}, ["strong", "strong", "strong", "fast"]),
        ({}, ["strong", "fast", "fast", "fast"]),
    ],
)
async def test_capable_hold_follows_the_session_without_a_classifier(
    session_kwargs: Mapping[str, object], expected: list[str]
) -> None:
    route: Final = router(config())
    transcripts: Final = (
        tool_history("CUDA out of memory"),
        tool_history("All tests passed", "All tests passed"),
        tool_history("All tests passed", "All tests passed", "All tests passed"),
        tool_history("All tests passed", "All tests passed", "All tests passed", "All tests passed"),
    )
    picks: Final = [
        result.model
        for transcript in transcripts
        if (result := await route.async_pre_routing_hook("route", session_kwargs, transcript)) is not None
    ]
    assert picks == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("probability", "expected"), [(0.0, "strong"), (1.0, "fast")])
async def test_llm_judge_answer_selects_target_and_carries_classifier_origin(probability: float, expected: str) -> None:
    async def judge(request: Mapping[str, object]) -> ModelResponse:
        assert request["model"] == "judge"
        assert request["stream"] is False
        assert "autorouter_classifier" in str(request["metadata"])
        assert request["response_format"] is not None
        return ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "crux": "The supplied task contract",
                                "primary_rule": "none",
                                "capability_boundary": "unmatched",
                                "p_solve": probability,
                            }
                        ),
                    }
                }
            ]
        )

    messages: Final[list[dict[str, object]]] = [{"role": "user", "content": "Hello"}]
    result: Final = await router(config(classifier={}), completion=judge).async_pre_routing_hook("route", {}, messages)
    assert result is not None
    assert result.model == expected
    assert result.messages == messages
    assert result.routing_decision is not None
    assert "switchyard_fallback_reason" not in result.routing_decision


@pytest.mark.asyncio
@pytest.mark.parametrize(("advantage", "expected"), [(0.61, "strong"), (0.6, "fast"), (0.1, "fast")])
async def test_decision_model_advantage_is_compared_strictly_against_the_cutoff(
    advantage: float, expected: str
) -> None:
    requests: Final[list[Mapping[str, object]]] = []

    async def decide(request: Mapping[str, object]) -> OpenAIDecisionResponse:
        requests.append(request)
        return decision_answer(advantage)

    route_config: Final = config(classifier={"kind": "decision", "cutoff": 0.6}, threshold=1.0)
    result: Final = await router(route_config, decide=decide).async_pre_routing_hook(
        "route", {"metadata": {"user_api_key_hash": "hash"}}, [{"role": "user", "content": "Refactor the parser"}]
    )
    assert result is not None
    assert result.model == expected
    assert result.routing_decision is not None
    assert result.routing_decision["switchyard_evidence"] == {
        "source": "decision_classifier",
        "verdict": "relative_advantage",
        "score": advantage,
        "threshold": 0.6,
    }
    (request,) = requests
    assert request["model"] == "judge"
    assert "Refactor the parser" in str(request["input"])
    (question,) = request["questions"]
    assert {choice["value"] for choice in question["choices"]} == {"advantage", "no_advantage"}
    assert "autorouter_classifier" in str(request["metadata"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        ConnectionError("Decision model unavailable"),
        OpenAIDecisionResponse.model_validate(
            {
                "model": "jev-1.14.0",
                "answers": [{"type": "refusal", "name": "route"}],
                "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
            }
        ),
    ],
)
async def test_decision_model_failure_falls_open_to_the_picker_default(outcome: object) -> None:
    async def decide(request: Mapping[str, object]) -> OpenAIDecisionResponse:
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, OpenAIDecisionResponse)
        return outcome

    route_config: Final = config(classifier={"kind": "decision"}, threshold=1.0)
    result: Final = await router(route_config, decide=decide).async_pre_routing_hook(
        "route", {}, [{"role": "user", "content": "Hello"}]
    )
    assert result is not None
    assert result.model == "fast"
    assert result.routing_decision is not None
    assert "switchyard_fallback_reason" not in result.routing_decision
    assert result.routing_decision["switchyard_evidence"]["source"] == "fail_open"


@pytest.mark.asyncio
@pytest.mark.parametrize(("picker", "expected"), [("efficient_first", "fast"), ("capable_first", "strong")])
async def test_llm_judge_failure_returns_library_default_decision(picker: str, expected: str) -> None:
    async def failing_judge(request: Mapping[str, object]) -> ModelResponse:
        raise ConnectionError("Judge unavailable")

    result: Final = await router(config(classifier={}, picker=picker), completion=failing_judge).async_pre_routing_hook(
        "route", {}, [{"role": "user", "content": "Hello"}]
    )
    assert result is not None
    assert result.model == expected
    assert result.routing_decision is not None
    assert "switchyard_fallback_reason" not in result.routing_decision
    assert result.routing_decision["switchyard_evidence"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}],
        [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}],
    ],
)
@pytest.mark.parametrize("picker", ["efficient_first", "capable_first"])
async def test_unsupported_content_falls_back_to_capable_and_preserves_messages(content: object, picker: str) -> None:
    messages: Final[list[dict[str, object]]] = [{"role": "user", "content": content}]
    result: Final = await router(config(picker=picker)).async_pre_routing_hook("route", {}, messages)
    assert result is not None
    assert result.model == "strong"
    assert result.messages == messages
    assert result.routing_decision is not None
    assert result.routing_decision["switchyard_fallback_reason"] == "unsupported_content"


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
    ("route_config", "algorithm", "reason"),
    [
        (config(), EndsWithoutDecision(), "no_decision"),
        (config(), RaisesLibsyError(), "libsy_error"),
        (
            config(classifier={}, threshold=1.0),
            build_algorithm(config(classifier={"kind": "decision"}, threshold=1.0)),
            "unexpected_decision_call",
        ),
        (
            config(classifier={"kind": "decision"}, threshold=1.0),
            build_algorithm(config(classifier={}, threshold=1.0)),
            "unexpected_judge_call",
        ),
    ],
)
async def test_algorithm_misbehavior_routes_to_capable_instead_of_failing_the_request(
    route_config: SwitchyardRouterConfig, algorithm: object, reason: str
) -> None:
    result: Final = await router(route_config, algorithm=algorithm).async_pre_routing_hook(
        "route", {}, [{"role": "user", "content": "Hello"}]
    )
    assert result is not None
    assert result.model == "strong"
    assert result.routing_decision is not None
    assert result.routing_decision["switchyard_fallback_reason"] == reason


@pytest.mark.asyncio
async def test_judge_cancellation_propagates() -> None:
    async def cancelled_judge(request: Mapping[str, object]) -> ModelResponse:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await router(config(classifier={}), completion=cancelled_judge).async_pre_routing_hook(
            "route", {}, [{"role": "user", "content": "Hello"}]
        )
