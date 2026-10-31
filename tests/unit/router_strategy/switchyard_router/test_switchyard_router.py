import asyncio
from collections.abc import Mapping
from typing import Final

import pytest

from litellm.exceptions import APIError, AuthenticationError, ServiceUnavailableError
from litellm.router_strategy.switchyard_router import SwitchyardAutoRouter, SwitchyardRouterConfig
from litellm.types.utils import ModelResponse


def response(model: str, text: str) -> ModelResponse:
    return ModelResponse(
        model=model,
        choices=[{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7, "provider_extra_usage": {"count": 2}},
    )


class ScriptedProvider:
    def __init__(self, replies: Mapping[str, ModelResponse | Exception]):
        self.replies = replies
        self.calls: tuple[tuple[str, Mapping[str, object]], ...] = ()

    async def __call__(self, model: str, request: Mapping[str, object]) -> ModelResponse:
        self.calls = (*self.calls, (model, request))
        reply: Final = self.replies[model]
        if isinstance(reply, Exception):
            raise reply
        return reply


def classifier_config(algorithm: str, **extra: object) -> SwitchyardRouterConfig:
    return SwitchyardRouterConfig.model_validate(
        {
            "algorithm": algorithm,
            "models": {"efficient": ["fast"], "capable": ["strong"], "judge": ["judge"]},
            **extra,
        }
    )


@pytest.mark.asyncio
async def test_weighted_routing_executes_only_selected_target_and_preserves_response() -> None:
    answer: Final = response("fast", "done")
    provider: Final = ScriptedProvider({"fast": answer})
    router: Final = SwitchyardAutoRouter(
        SwitchyardRouterConfig.model_validate(
            {
                "algorithm": "random",
                "models": {"any": ["fast", "slow"]},
                "weights": [1, 0],
                "seed": 42,
            }
        )
    )
    actual: Final = await router.execute({"messages": [{"role": "user", "content": "hello"}]}, provider)
    assert actual is answer
    assert tuple(model for model, _ in provider.calls) == ("fast",)


@pytest.mark.asyncio
@pytest.mark.parametrize(("probability", "selected"), [(0.9, "fast"), (0.1, "strong")])
async def test_capability_judge_selects_model_and_does_not_receive_user_stop(probability: float, selected: str) -> None:
    answer: Final = response(selected, "answer")
    provider: Final = ScriptedProvider(
        {
            "judge": response(
                "judge",
                '{"p_solve":'
                + str(probability)
                + ',"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported"}',
            ),
            selected: answer,
        }
    )
    router: Final = SwitchyardAutoRouter(classifier_config("capability"))
    actual: Final = await router.execute({"messages": [{"role": "user", "content": "hello"}], "stop": ["}"]}, provider)
    assert actual is answer
    assert tuple(model for model, _ in provider.calls) == ("judge", selected)
    assert "stop" not in provider.calls[0][1]
    assert provider.calls[1][1]["stop"] == ["}"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verdict", "selected", "calls"),
    [
        ('{"escalate":false,"category":"none","new_evidence":false,"reason":"progress"}', "fast", ("fast", "judge")),
        (
            '{"escalate":true,"category":"repetition","new_evidence":true,"reason":"stuck"}',
            "strong",
            ("fast", "judge", "strong"),
        ),
    ],
)
async def test_escalation_reuses_accepted_answer_or_executes_capable_model(
    verdict: str, selected: str, calls: tuple[str, ...]
) -> None:
    efficient: Final = response("fast", "draft")
    capable: Final = response("strong", "answer")
    provider: Final = ScriptedProvider({"judge": response("judge", verdict), "fast": efficient, "strong": capable})
    router: Final = SwitchyardAutoRouter(classifier_config("escalation", escalation={"confirmations": 1}))
    actual: Final = await router.execute({"messages": [{"role": "user", "content": "hello"}]}, provider)
    assert actual is (efficient if selected == "fast" else capable)
    assert tuple(model for model, _ in provider.calls) == calls
    assert actual.usage.model_dump()["provider_extra_usage"] == {"count": 2}


@pytest.mark.asyncio
async def test_custom_classifier_selects_runtime_group() -> None:
    answer: Final = response("code", "result")
    provider: Final = ScriptedProvider({"judge": response("judge", '{"target":"coding"}'), "code": answer})
    router: Final = SwitchyardAutoRouter(
        SwitchyardRouterConfig.model_validate(
            {
                "algorithm": "custom",
                "models": {"judge": ["judge"], "coding": ["code"], "general": ["fast"]},
                "custom": {
                    "prompt": "Choose target",
                    "selector": "/target",
                    "default_target": "general",
                    "response_schema": {
                        "type": "object",
                        "required": ["target"],
                        "properties": {"target": {"type": "string", "enum": ["coding", "general"]}},
                    },
                },
            }
        )
    )
    actual: Final = await router.execute({"messages": [{"role": "user", "content": "write code"}]}, provider)
    assert actual is answer
    assert tuple(model for model, _ in provider.calls) == ("judge", "code")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [APIError(503, "unavailable", "fake", "fast"), ServiceUnavailableError("unavailable", "fake", "fast")]
)
async def test_service_failure_uses_configured_fallback(error: Exception) -> None:
    answer: Final = response("backup", "answer")
    provider: Final = ScriptedProvider({"fast": error, "backup": answer})
    router: Final = SwitchyardAutoRouter(
        classifier_config(
            "stage",
            models={"efficient": ["fast", "backup"], "capable": ["strong"]},
        )
    )
    actual: Final = await router.execute({"messages": [{"role": "user", "content": "hello"}]}, provider)
    assert actual is answer
    assert tuple(model for model, _ in provider.calls) == ("fast", "backup")


@pytest.mark.asyncio
async def test_authentication_failure_is_not_retried_on_another_model() -> None:
    error: Final = AuthenticationError("denied", "fake", "fast")
    provider: Final = ScriptedProvider({"fast": error})
    router: Final = SwitchyardAutoRouter(
        classifier_config(
            "stage",
            models={"efficient": ["fast", "backup"], "capable": ["strong"]},
        )
    )
    with pytest.raises(AuthenticationError) as raised:
        await router.execute({"messages": [{"role": "user", "content": "hello"}]}, provider)
    assert raised.value is error
    assert tuple(model for model, _ in provider.calls) == ("fast",)


@pytest.mark.asyncio
async def test_intermediate_authentication_error_preserves_public_error_type() -> None:
    error: Final = AuthenticationError("denied", "fake", "fast")
    provider: Final = ScriptedProvider({"fast": error})
    router: Final = SwitchyardAutoRouter(classifier_config("escalation", escalation={"confirmations": 1}))
    with pytest.raises(AuthenticationError) as raised:
        await router.execute({"messages": [{"role": "user", "content": "hello"}]}, provider)
    assert raised.value is error
    assert tuple(model for model, _ in provider.calls) == ("fast",)


@pytest.mark.asyncio
async def test_escalation_confirmations_are_isolated_between_sessions() -> None:
    efficient: Final = response("fast", "draft")
    capable: Final = response("strong", "answer")
    provider: Final = ScriptedProvider(
        {
            "fast": efficient,
            "strong": capable,
            "judge": response(
                "judge", '{"escalate":true,"category":"repetition","new_evidence":true,"reason":"stuck"}'
            ),
        }
    )
    router: Final = SwitchyardAutoRouter(classifier_config("escalation", escalation={"confirmations": 2}))
    first: Final = await router.execute(
        {"messages": [{"role": "user", "content": "first"}]}, provider, headers={"x-switchyard-session-id": "session-a"}
    )
    separate: Final = await router.execute(
        {"messages": [{"role": "user", "content": "second"}]},
        provider,
        headers={"x-switchyard-session-id": "session-b"},
    )
    confirmed: Final = await router.execute(
        {"messages": [{"role": "user", "content": "third"}]}, provider, headers={"x-switchyard-session-id": "session-a"}
    )
    assert (first is efficient, separate is efficient, confirmed is capable) == (True, True, True)
    assert tuple(model for model, _ in provider.calls) == ("fast", "judge", "fast", "judge", "fast", "judge", "strong")


@pytest.mark.asyncio
async def test_escalation_buffers_intermediate_calls_for_stream_requests() -> None:
    answer: Final = response("fast", "draft")
    provider: Final = ScriptedProvider(
        {
            "fast": answer,
            "judge": response("judge", '{"escalate":false,"category":"none","new_evidence":false,"reason":"progress"}'),
        }
    )
    router: Final = SwitchyardAutoRouter(classifier_config("escalation", escalation={"confirmations": 1}))
    actual: Final = await router.execute({"messages": [{"role": "user", "content": "hello"}], "stream": True}, provider)
    assert actual is answer
    assert tuple((model, payload["stream"]) for model, payload in provider.calls) == (("fast", False), ("judge", False))


@pytest.mark.asyncio
async def test_cancelling_judge_cancels_provider_work() -> None:
    started: Final = asyncio.Event()
    cancelled: Final = asyncio.Event()
    never: Final = asyncio.Event()

    async def pending_judge(model: str, request: Mapping[str, object]) -> ModelResponse:
        started.set()
        try:
            await never.wait()
        finally:
            cancelled.set()
        return response(model, "unreachable")

    router: Final = SwitchyardAutoRouter(classifier_config("capability"))
    task: Final = asyncio.create_task(
        router.execute({"messages": [{"role": "user", "content": "hello"}]}, pending_judge)
    )
    entered: Final = asyncio.create_task(started.wait())
    done, _ = await asyncio.wait((task, entered), return_when=asyncio.FIRST_COMPLETED)
    if task in done:
        entered.cancel()
        await task
        pytest.fail("Routing finished before the judge executed")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
