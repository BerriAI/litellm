import asyncio
import json
from collections.abc import Mapping
from typing import Final

import pytest

from litellm.router_strategy.switchyard_router.config import SwitchyardRouterConfig
from litellm.router_strategy.switchyard_router.switchyard_router import SwitchyardRouter
from litellm.types.utils import ModelResponse

pytest.importorskip("switchyard.libsy")


async def unexpected_judge(request: Mapping[str, object]) -> ModelResponse:
    raise AssertionError("A classifier-free route must not call a judge")


def config(*, classifier: bool = False, picker: str = "efficient_first") -> SwitchyardRouterConfig:
    return SwitchyardRouterConfig.model_validate(
        {
            "models": {"capable": "strong", "efficient": "fast", **({"judge": "judge"} if classifier else {})},
            "options": {"picker": picker},
            **({"classifier": {}} if classifier else {}),
        }
    )


def tool_history(result: str) -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "Apply the edit and run the tests"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"pytest"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call1", "content": result},
    ]


@pytest.mark.asyncio
async def test_critical_tool_error_selects_capable_without_rewriting_messages() -> None:
    router: Final = SwitchyardRouter("route", config(), unexpected_judge)
    messages: Final = tool_history("CUDA out of memory")
    result: Final = await router.async_pre_routing_hook("route", {}, messages)
    assert result is not None
    assert result.model == "strong"
    assert result.messages == messages
    assert result.routing_decision is not None
    assert result.routing_decision["router_type"] == "switchyard"
    assert result.routing_decision["switchyard_evidence"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Edit applied successfully", "All tests passed", "Hello"])
async def test_heuristic_decision_is_a_configured_target_with_evidence(text: str) -> None:
    router: Final = SwitchyardRouter("route", config(), unexpected_judge)
    result: Final = await router.async_pre_routing_hook("route", {}, tool_history(text))
    assert result is not None
    assert result.model in {"strong", "fast"}
    assert result.routing_decision is not None
    assert result.routing_decision["routed_model"] == result.model
    assert result.routing_decision["switchyard_evidence"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(("probability", "expected"), [(0.0, "strong"), (1.0, "fast")])
async def test_judge_answer_selects_target_and_carries_classifier_origin(probability: float, expected: str) -> None:
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

    router: Final = SwitchyardRouter("route", config(classifier=True), judge)
    messages: Final[list[dict[str, object]]] = [{"role": "user", "content": "Hello"}]
    result: Final = await router.async_pre_routing_hook("route", {}, messages)
    assert result is not None
    assert result.model == expected
    assert result.messages == messages
    assert result.routing_decision is not None
    assert "switchyard_fallback_reason" not in result.routing_decision


@pytest.mark.asyncio
@pytest.mark.parametrize(("picker", "expected"), [("efficient_first", "fast"), ("capable_first", "strong")])
async def test_judge_failure_returns_library_default_decision(picker: str, expected: str) -> None:
    async def failing_judge(request: Mapping[str, object]) -> ModelResponse:
        raise ConnectionError("Judge unavailable")

    router: Final = SwitchyardRouter("route", config(classifier=True, picker=picker), failing_judge)
    result: Final = await router.async_pre_routing_hook("route", {}, [{"role": "user", "content": "Hello"}])
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
async def test_unsupported_content_falls_back_and_preserves_original_messages(content: object) -> None:
    router: Final = SwitchyardRouter("route", config(), unexpected_judge)
    messages: Final[list[dict[str, object]]] = [{"role": "user", "content": content}]
    result: Final = await router.async_pre_routing_hook("route", {}, messages)
    assert result is not None
    assert result.model == "fast"
    assert result.messages == messages
    assert result.routing_decision is not None
    assert result.routing_decision["switchyard_fallback_reason"] == "unsupported_content"


@pytest.mark.asyncio
async def test_judge_cancellation_propagates() -> None:
    async def cancelled_judge(request: Mapping[str, object]) -> ModelResponse:
        raise asyncio.CancelledError()

    router: Final = SwitchyardRouter("route", config(classifier=True), cancelled_judge)
    with pytest.raises(asyncio.CancelledError):
        await router.async_pre_routing_hook("route", {}, [{"role": "user", "content": "Hello"}])
