import asyncio
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Final

import httpx
import pytest

import litellm
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig, JevClassifierConfig
from litellm.router_strategy.complexity_router.jev_classifier import (
    DEFAULT_JEV_INSTRUCTIONS,
    HttpJevClassifierClient,
    JevChoiceAnswer,
    JevSystemOneResponse,
    JevUsage,
    build_jev_request,
    jev_classifier_cost,
)
from litellm.types.utils import AUTOROUTER_CLASSIFIER_CALL_ORIGIN


class _UsageRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.calls: tuple[Mapping[str, object], ...] = ()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        if str(kwargs.get("model", "")).removeprefix("typesafe/") != "jev-accounting":
            return
        self.calls = (*self.calls, kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["SIMPLE", "UNAVAILABLE", "malformed"])
@pytest.mark.parametrize("private", [False, True])
async def test_jev_accounts_once_with_parent_identity_even_when_the_verdict_fails(
    monkeypatch: pytest.MonkeyPatch, answer: str, private: bool
) -> None:
    recorder: Final = _UsageRecorder()
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    monkeypatch.setitem(
        litellm.model_cost,
        "typesafe/jev-accounting",
        {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002},
    )

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "jev-accounting",
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "answers": {"tier": {"type": "choice", "choice": answer, "confidence": 1, "probabilities": {answer: 1}}}
                if answer != "malformed"
                else "invalid",
            },
        )

    handler: Final = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    provider: Final = HttpJevClassifierClient("test", "https://typesafe.test", handler)
    router: Final = ComplexityRouter(
        "jev-router",
        litellm.Router(model_list=[]),
        {"classifier_type": "jev", "jev_classifier_config": {}, "tiers": {"SIMPLE": "cheap"}},
        jev_client=provider,
        derive_savings_baseline=False,
    )
    metadata: Final = {
        "user_api_key": "hashed-test-key",
        "user_api_key_user_id": "user-a",
        "user_api_key_team_id": "team-a",
        "user_api_key_project_id": "project-a",
        "user_api_key_org_id": "org-a",
        "user_api_key_budget_reservation": {"reservation_id": "parent-reservation"},
        "user_api_key_auth": {"budget_reservation": {"reservation_id": "parent-reservation"}},
    }
    outcome: Final = await router.aclassify(
        "private current ask",
        request_kwargs={
            "metadata": metadata,
            "litellm_session_id": "session-a",
            "litellm_trace_id": "trace-a",
            "turn_off_message_logging": private,
        },
    )
    await GLOBAL_LOGGING_WORKER.flush()
    await handler.client.aclose()

    assert (outcome.cause == "jev_classifier") is (answer == "SIMPLE")
    assert len(recorder.calls) == 1
    event: Final = recorder.calls[0]
    assert event["response_cost"] == pytest.approx(0.007)
    assert event["model"] == "typesafe/jev-accounting"
    params: Final = event["litellm_params"]
    assert isinstance(params, Mapping)
    logged_metadata: Final = params["metadata"]
    assert isinstance(logged_metadata, Mapping)
    assert logged_metadata[INTERNAL_CALL_ORIGIN_METADATA_KEY] == AUTOROUTER_CLASSIFIER_CALL_ORIGIN
    assert logged_metadata["user_api_key_team_id"] == "team-a"
    assert logged_metadata["user_api_key_user_id"] == "user-a"
    assert logged_metadata["user_api_key_project_id"] == "project-a"
    assert logged_metadata["user_api_key_org_id"] == "org-a"
    assert logged_metadata["user_api_key"] == "hashed-test-key"
    assert "user_api_key_budget_reservation" not in logged_metadata
    assert logged_metadata["user_api_key_auth"] == {}
    assert metadata["user_api_key_budget_reservation"] == {"reservation_id": "parent-reservation"}
    assert params["litellm_session_id"] == "session-a"
    assert event["litellm_trace_id"] == "trace-a"
    assert ("private current ask" in str(event["messages"])) is not private
    standard: Final = event["standard_logging_object"]
    assert isinstance(standard, Mapping)
    assert (standard["prompt_tokens"], standard["completion_tokens"], standard["total_tokens"]) == (3, 2, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("include_assistant", [False, True])
async def test_jev_uses_bounded_history_and_separates_operator_instructions(include_assistant: bool) -> None:
    captured: list[Mapping[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"answers": {"tier": _answer().model_dump()}})

    handler: Final = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    router: Final = ComplexityRouter(
        "jev-context",
        litellm.Router(model_list=[]),
        {
            "classifier_type": "jev",
            "jev_classifier_config": {"instructions": "operator-only rubric"},
            "tiers": {"SIMPLE": "cheap"},
            "classifier_context_window_size": 2 if include_assistant else 1,
            "classifier_context_per_turn_chars": 100,
            "classifier_context_budget_chars": 120,
            "classifier_context_include_assistant_turns": include_assistant,
        },
        jev_client=HttpJevClassifierClient("test", "https://typesafe.test", handler),
        derive_savings_baseline=False,
    )
    await router.aclassify(
        "current real ask",
        system_prompt="caller constraints",
        messages=[
            {"role": "user", "content": "old discarded conversation"},
            {"role": "user", "content": "recent question " + "x" * 300},
            {"role": "assistant", "content": "assistant context"},
            {"role": "tool", "content": "untrusted tool output"},
            {"role": "user", "content": "<system-reminder>hidden reminder</system-reminder>current real ask"},
        ],
    )
    await GLOBAL_LOGGING_WORKER.flush()
    await handler.client.aclose()
    assert len(captured) == 1
    state: Final = str(captured[0]["state"])
    assert "current real ask" in state
    assert "caller constraints" in state
    assert "recent question" in state
    assert "x" * 101 not in state
    assert "old discarded conversation" not in state
    assert "hidden reminder" not in state
    assert "untrusted tool output" not in state
    assert ("assistant context" in state) is include_assistant
    assert "operator-only rubric" not in state
    assert "operator-only rubric" in str(captured[0]["questions"])


@pytest.mark.asyncio
async def test_jev_cancellation_propagates_without_opening_timeout_breaker() -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise asyncio.CancelledError
        return httpx.Response(200, json={"answers": {"tier": _answer().model_dump()}})

    handler: Final = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    router: Final = ComplexityRouter(
        "jev-cancellation",
        litellm.Router(model_list=[]),
        {"classifier_type": "jev", "jev_classifier_config": {}, "tiers": {"SIMPLE": "cheap"}},
        jev_client=HttpJevClassifierClient("test", "https://typesafe.test", handler),
        derive_savings_baseline=False,
    )
    with pytest.raises(asyncio.CancelledError):
        await router.aclassify("cancel this")
    outcome: Final = await router.aclassify("still available")
    await GLOBAL_LOGGING_WORKER.flush()
    await handler.client.aclose()
    assert outcome.cause == "jev_classifier"
    assert len(calls) == 2


def _answer(choice: str = "SIMPLE") -> JevChoiceAnswer:
    return JevChoiceAnswer(
        type="choice",
        choice=choice,
        probabilities={choice: 0.9},
        confidence=0.9,
    )


def test_jev_config_requires_classifier_config() -> None:
    with pytest.raises(ValueError, match="jev_classifier_config is required"):
        ComplexityRouterConfig.model_validate({"classifier_type": "jev"})


def test_jev_config_is_rejected_for_other_classifier_types() -> None:
    with pytest.raises(ValueError, match="has no effect"):
        ComplexityRouterConfig.model_validate(
            {
                "jev_classifier_config": {},
            }
        )


def test_jev_instructions_reject_blank_values() -> None:
    with pytest.raises(ValueError, match="instructions must be non-empty"):
        JevClassifierConfig(instructions=" \t")


@pytest.mark.parametrize(
    ("missing_key", "rejection"),
    [
        ({}, r"api_base requires jev_classifier_config\.api_key"),
        ({"api_key": ""}, r"api_key must be non-empty"),
        ({"api_key": "   "}, r"api_key must be non-empty"),
    ],
)
def test_jev_api_base_without_its_own_key_is_rejected_so_the_environment_key_stays_home(
    missing_key: Mapping[str, str], rejection: str
) -> None:
    with pytest.raises(ValueError, match=rejection):
        ComplexityRouterConfig.model_validate(
            {
                "classifier_type": "jev",
                "jev_classifier_config": {"api_base": "https://collector.invalid", **missing_key},
            }
        )
    paired: Final = JevClassifierConfig(api_base="https://eu.typesafe.invalid", api_key="sk-own")
    assert (paired.api_base, paired.api_key) == ("https://eu.typesafe.invalid", "sk-own")
    assert JevClassifierConfig(api_key="sk-own").api_base is None


@pytest.mark.parametrize(
    ("probabilities", "confidence"),
    [
        ({"SIMPLE": -0.1}, 0.9),
        ({"SIMPLE": 1.1}, 0.9),
        ({"SIMPLE": 0.9}, -0.1),
        ({"SIMPLE": 0.9}, 1.1),
        ({"SIMPLE": float("inf")}, 0.9),
        ({"SIMPLE": 0.9}, float("nan")),
    ],
)
def test_jev_answer_rejects_invalid_probability_values(probabilities: dict[str, float], confidence: float) -> None:
    with pytest.raises(ValueError, match=r"(greater than or equal to|less than or equal to|finite)"):
        JevChoiceAnswer(type="choice", choice="SIMPLE", probabilities=probabilities, confidence=confidence)


def test_build_jev_request_includes_system_prompt_and_criteria() -> None:
    criteria: Final[Mapping[str, str]] = {
        "Budget": "Short factual answers",
        "Premium": "Deep technical analysis",
    }
    request: Final = build_jev_request(
        prompt="Explain the failure",
        system_prompt="Answer as an engineer",
        model="jev-latest",
        instructions=DEFAULT_JEV_INSTRUCTIONS,
        criteria=criteria,
    )
    assert request.state == "System prompt:\nAnswer as an engineer\n\nRequest:\nExplain the failure"
    assert request.model == "jev-latest"
    assert request.questions["tier"].type == "choice"
    assert request.questions["tier"].instructions == DEFAULT_JEV_INSTRUCTIONS
    assert request.questions["tier"].criteria == criteria


def test_jev_classifier_cost_uses_registry_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "typesafe/jev-1.13.0",
        {"input_cost_per_token": 0.0001, "output_cost_per_token": 0.0002},
    )
    response: Final = JevSystemOneResponse(
        model="jev-1.13.0",
        answers={"tier": _answer()},
        usage=JevUsage(input_tokens=3, output_tokens=4),
    )
    assert jev_classifier_cost(response, "jev-latest") == pytest.approx(0.0011)


def test_jev_classifier_cost_is_none_without_registry_pricing() -> None:
    assert "typesafe/jev-unpriced" not in litellm.model_cost
    response: Final = JevSystemOneResponse(
        answers={"tier": _answer()},
        usage=JevUsage(input_tokens=3, output_tokens=4),
    )
    assert jev_classifier_cost(response, "jev-unpriced") is None


@pytest.mark.asyncio
async def test_http_jev_classifier_client_posts_to_system_one() -> None:
    captured: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["content_type"] = request.headers["Content-Type"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "tier": {
                        "type": "choice",
                        "choice": "SIMPLE",
                        "probabilities": {"SIMPLE": 1.0},
                        "confidence": 1.0,
                    }
                },
            },
        )

    handler: Final = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client: Final = HttpJevClassifierClient("secret", "https://typesafe.test", handler)
    request: Final = build_jev_request("Hello", None, "jev-latest", DEFAULT_JEV_INSTRUCTIONS, {"SIMPLE": "facts"})
    response: Final = await client.evaluate(request, 1.0)

    assert captured["url"] == "https://typesafe.test/v1/systemone"
    assert captured["authorization"] == "Bearer secret"
    assert captured["content_type"] == "application/json"
    assert captured["body"] == request.model_dump(mode="json")
    assert response.model == "jev-1.13.0"
