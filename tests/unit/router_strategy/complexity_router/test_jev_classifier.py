import asyncio
import json
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Final, NoReturn
from unittest.mock import create_autospec

import httpx
import pytest
import respx

import litellm
from litellm._logging import verbose_router_logger
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
    def __init__(self, model_key: str = "typesafe/jev-accounting") -> None:
        super().__init__()
        self.model_key = model_key
        self.calls: tuple[Mapping[str, object], ...] = ()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        if str(kwargs.get("model", "")) != self.model_key:
            return
        self.calls = (*self.calls, kwargs)


class _UncopyableAuth:
    budget_reservation: Final = "parent-reservation"

    def __init__(self, error: Exception) -> None:
        self.error = error

    def model_copy(self, *, update: Mapping[str, object]) -> NoReturn:
        raise self.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "error_name"),
    [
        ({1: "private-metadata"}, "ValidationError"),
        ({"user_api_key_auth": _UncopyableAuth(RuntimeError("private-metadata"))}, "RuntimeError"),
        ({"user_api_key_auth": _UncopyableAuth(TimeoutError("private-metadata"))}, "TimeoutError"),
    ],
)
async def test_jev_logging_failure_preserves_verdict_and_keeps_circuit_closed(
    caplog: pytest.LogCaptureFixture, metadata: Mapping[object, object], error_name: str
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "answers": {"tier": _answer().model_dump()},
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )

    handler: Final = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    router: Final = ComplexityRouter(
        "jev-logging-failure",
        litellm.Router(model_list=[]),
        {"classifier_type": "jev", "jev_classifier_config": {}, "tiers": {"SIMPLE": "cheap"}},
        jev_client=HttpJevClassifierClient("test", "https://typesafe.test", handler),
        derive_savings_baseline=False,
    )
    with caplog.at_level("WARNING", logger=verbose_router_logger.name):
        outcomes: Final = tuple(
            [await router.aclassify("choose a tier", request_kwargs={"metadata": metadata}) for _ in range(2)]
        )
    await handler.client.aclose()

    assert tuple(
        (outcome.cause, outcome.jev_verdict.label if outcome.jev_verdict else None) for outcome in outcomes
    ) == (
        ("jev_classifier", "SIMPLE"),
        ("jev_classifier", "SIMPLE"),
    )
    assert len(requests) == 2
    assert caplog.messages == [f"JEV response logging failed ({error_name})"] * 2
    assert "private-metadata" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 429, 500, 503])
async def test_jev_http_errors_do_not_dispatch_successful_usage(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    recorder: Final = _UsageRecorder()
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    handler: Final = create_autospec(AsyncHTTPHandler, instance=True)
    handler.post.return_value = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://typesafe.test/v1/systemone"),
        json={
            "model": "jev-accounting",
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "answers": {"tier": _answer().model_dump()},
        },
    )
    provider: Final = HttpJevClassifierClient("test", "https://typesafe.test", handler)
    request: Final = build_jev_request(
        "choose a tier", None, "jev-accounting", DEFAULT_JEV_INSTRUCTIONS, {"SIMPLE": "cheap"}
    )

    with pytest.raises(httpx.HTTPStatusError) as error:
        await provider.evaluate(request, timeout_s=3)
    await GLOBAL_LOGGING_WORKER.flush()

    assert error.value.response.status_code == status_code
    handler.post.assert_awaited_once()
    assert recorder.calls == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("tokens", [-1, True, 1.5, "3"])
async def test_jev_invalid_usage_never_reaches_spend_callbacks(
    monkeypatch: pytest.MonkeyPatch, field: str, tokens: object
) -> None:
    recorder: Final = _UsageRecorder()
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    handler: Final = create_autospec(AsyncHTTPHandler, instance=True)
    handler.post.return_value = httpx.Response(
        200,
        request=httpx.Request("POST", "https://typesafe.test/v1/systemone"),
        json={
            "model": "jev-accounting",
            "usage": {"input_tokens": 3, "output_tokens": 2, field: tokens},
            "answers": {"tier": _answer().model_dump()},
        },
    )
    provider: Final = HttpJevClassifierClient("test", "https://typesafe.test", handler)
    request: Final = build_jev_request(
        "choose a tier", None, "jev-accounting", DEFAULT_JEV_INSTRUCTIONS, {"SIMPLE": "cheap"}
    )

    with pytest.raises(ValueError, match=field):
        await provider.evaluate(request, timeout_s=3)
    await GLOBAL_LOGGING_WORKER.flush()

    handler.post.assert_awaited_once()
    assert recorder.calls == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["SIMPLE", "UNAVAILABLE", "malformed"])
@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
async def test_jev_accounts_once_with_parent_identity_even_when_the_verdict_fails(
    monkeypatch: pytest.MonkeyPatch, answer: str, private: bool, legacy: bool
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
        {
            "classifier_type": "jev" if legacy else "oss_classifier",
            "jev_classifier_config" if legacy else "opensource_classifier_config": {
                "provider": "typesafe" if legacy else "jev",
            },
            "tiers": {"SIMPLE": "cheap"},
            "session_affinity": False,
            "deployment_affinity": False,
        },
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
    result: Final = await router.async_pre_routing_hook(
        model="jev-router",
        messages=[{"role": "user", "content": "private current ask"}],
        request_kwargs={
            "metadata": metadata,
            "litellm_session_id": "session-a",
            "litellm_trace_id": "trace-a",
            "turn_off_message_logging": private,
        },
    )
    await GLOBAL_LOGGING_WORKER.flush()
    await handler.client.aclose()

    assert result is not None and result.model == "cheap"
    assert result.routing_decision is not None
    decision: Final = result.routing_decision
    assert (decision["cause"] == "jev_classifier") is (answer == "SIMPLE")
    if answer == "SIMPLE":
        assert decision["classifier_model"] == "typesafe/jev-accounting"
        assert decision["classifier_cost"] == pytest.approx(0.007)
        assert "jev-classifier:SIMPLE" in decision["signals"]
        assert "jev-confidence=1.000000" in decision["signals"]
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
@pytest.mark.parametrize(
    ("fallback", "expected_model", "expected_cause"),
    (
        (
            {"tier_definitions": [{"name": "SIMPLE"}, {"name": "REASONING"}], "fallback_tier": "REASONING"},
            "deep",
            "classifier_fallback",
        ),
        ({"classifier_fallback": "default_model", "default_model": "deep"}, "deep", "default_model_fallback"),
        ({"classifier_fallback": "heuristic"}, "cheap", "heuristic_scorer"),
    ),
)
async def test_jev_encrypted_task_skips_provider_without_disabling_plaintext_classification(
    fallback: Mapping[str, object], expected_model: str, expected_cause: str
) -> None:
    transport: Final = create_autospec(httpx.AsyncBaseTransport, instance=True)
    transport.handle_async_request.return_value = httpx.Response(
        200, json={"answers": {"tier": _answer().model_dump()}}
    )
    handler: Final = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=transport)
    router: Final = ComplexityRouter(
        "jev-encrypted",
        litellm.Router(model_list=[]),
        {
            "classifier_type": "jev",
            "jev_classifier_config": {},
            "tiers": {"SIMPLE": "cheap", "REASONING": "deep"},
            "session_affinity": False,
            "deployment_affinity": False,
            **fallback,
        },
        jev_client=HttpJevClassifierClient("test", "https://typesafe.test", handler),
        derive_savings_baseline=False,
    )
    request: Final = {
        "input": [
            {
                "type": "agent_message",
                "author": "/root",
                "recipient": "/root/child",
                "content": [
                    {"type": "input_text", "text": "Message Type: NEW_TASK\nPayload:\nHello"},
                    {"type": "encrypted_content", "encrypted_content": "opaque-task"},
                ],
            },
            {"role": "user", "content": "<environment_context>cwd=/repo</environment_context>"},
        ],
        "metadata": {"user_agent": "codex-tui"},
    }
    original: Final = deepcopy(request)
    try:
        result: Final = await router.async_pre_routing_hook(model="jev-encrypted", request_kwargs=request)
        assert result is not None and result.model == expected_model
        assert result.routing_decision is not None
        assert result.routing_decision["cause"] == expected_cause
        assert result.routing_decision.get("classifier_cost") is None
        assert result.messages is None
        assert request == original
        transport.handle_async_request.assert_not_awaited()

        plaintext: Final = await router.async_pre_routing_hook(
            model="jev-encrypted",
            request_kwargs={**request, "input": [*request["input"], {"role": "user", "content": "Say hello again"}]},
        )
        assert plaintext is not None and plaintext.model == "cheap"
        assert plaintext.routing_decision is not None
        assert plaintext.routing_decision["cause"] == "jev_classifier"
        transport.handle_async_request.assert_awaited_once()
        sent: Final = transport.handle_async_request.call_args.args[0]
        assert isinstance(sent, httpx.Request)
        assert "Say hello again" in sent.content.decode()
    finally:
        await GLOBAL_LOGGING_WORKER.flush()
        await handler.client.aclose()


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
    with pytest.raises(ValueError, match="opensource_classifier_config is required"):
        ComplexityRouterConfig.model_validate({"classifier_type": "jev"})


@pytest.mark.parametrize(
    ("classifier_type", "config_key"),
    [
        ("oss_classifier", "opensource_classifier_config"),
        ("jev", "jev_classifier_config"),
        ("oss_classifier", "jev_classifier_config"),
        ("jev", "opensource_classifier_config"),
    ],
)
@pytest.mark.parametrize(
    ("provider", "model", "canonical_provider"),
    [
        (None, "jev-latest", "jev"),
        ("typesafe", "jev-latest", "jev"),
        ("jev", "jev-latest", "jev"),
        ("laya", "english", "laya"),
        ("bespoke", "nimble-latest", "bespoke"),
        ("databricks", "databricks-openjev-qwen35-4b", "databricks"),
        ("databricks", "ai_decide", "databricks"),
    ],
)
def test_classifier_aliases_load_and_serialize_one_canonical_config(
    classifier_type: str, config_key: str, provider: str | None, model: str, canonical_provider: str
) -> None:
    incoming: Final = {
        "classifier_type": classifier_type,
        config_key: {"model": model, "api_key": None, **({"provider": provider} if provider is not None else {})},
    }
    original: Final = deepcopy(incoming)
    config: Final = ComplexityRouterConfig.model_validate(incoming)
    assert config.classifier_type == "oss_classifier"
    assert config.opensource_classifier_config is not None
    assert config.opensource_classifier_config.provider == canonical_provider
    assert config.opensource_classifier_config.model == model
    assert config.opensource_classifier_config.api_key is None
    assert "api_key" in config.opensource_classifier_config.model_fields_set
    assert "api_base" not in config.opensource_classifier_config.model_fields_set
    assert "jev_classifier_config" not in config.model_dump()
    assert config.jev_classifier_config is config.opensource_classifier_config
    assert incoming == original


@pytest.mark.parametrize("provider", ["laya", "bespoke"])
@pytest.mark.parametrize("model", [None, " "])
def test_oss_requires_its_own_checkpoint(provider: str, model: str | None) -> None:
    with pytest.raises(ValueError, match=f"{provider} model must be"):
        JevClassifierConfig.model_validate({"provider": provider, **({"model": model} if model is not None else {})})


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,model", [("laya", "english"), ("bespoke", "nimble-latest")])
@pytest.mark.parametrize("custom_base", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
async def test_oss_routes_with_its_own_credentials_and_accounts_the_checkpoint(
    monkeypatch: pytest.MonkeyPatch, custom_base: bool, legacy: bool, provider: str, model: str
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "never-send-typesafe-key")
    monkeypatch.setenv(f"{provider.upper()}_API_BASE", f"https://{provider}.test")
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "oss-env-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setitem(litellm.model_cost, f"{provider}/{model}", {"input_cost_per_token": 0.01})
    recorder: Final = _UsageRecorder(f"{provider}/{model}")
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    router: Final = ComplexityRouter(
        f"{provider}-route",
        litellm.Router(model_list=[]),
        {
            "classifier_type": "jev" if legacy else "oss_classifier",
            "jev_classifier_config" if legacy else "opensource_classifier_config": {
                "provider": provider,
                "model": model,
                **({"api_base": f"https://{provider}.test"} if custom_base else {}),
            },
            "tiers": {"SIMPLE": "cheap"},
        },
        derive_savings_baseline=False,
    )
    with respx.mock(assert_all_called=True) as upstream:
        route: Final = upstream.post(f"https://{provider}.test/v1/systemone").respond(
            200,
            json={
                "model": "laya-rl-agent" if provider == "laya" else model,
                **({"routing": {"model": model}} if provider == "laya" else {}),
                "answers": {"tier": _answer().model_dump()},
                "usage": {"input_tokens": 31, "output_tokens": 0},
            },
        )
        outcome: Final = await router.aclassify("choose a tier")
        await GLOBAL_LOGGING_WORKER.flush()

    assert outcome.cause == "jev_classifier"
    assert outcome.jev_verdict is not None
    assert (outcome.jev_verdict.provider, outcome.jev_verdict.model) == (provider, model)
    assert outcome.classifier_cost == pytest.approx(0.31)
    sent: Final = route.calls.last.request
    assert sent.headers.get("authorization") == (None if custom_base else "Bearer oss-env-key")
    assert json.loads(sent.content)["model"] == model
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["response_cost"] == pytest.approx(0.31)


@pytest.mark.parametrize("model", ["ai_decide", "my-openjev", "databricks-openjev-qwen35-4b"])
def test_databricks_accepts_ai_decide_or_a_bare_serving_endpoint_name(model: str) -> None:
    assert JevClassifierConfig.model_validate({"provider": "databricks", "model": model}).model == model


def test_databricks_requires_a_model_that_cannot_rewrite_the_url() -> None:
    with pytest.raises(ValueError, match="model is required for provider 'databricks'"):
        JevClassifierConfig.model_validate({"provider": "databricks"})
    with pytest.raises(ValueError, match="bare serving endpoint name"):
        JevClassifierConfig.model_validate({"provider": "databricks", "model": "serving-endpoints/my-openjev"})
    with pytest.raises(ValueError, match="bare serving endpoint name"):
        JevClassifierConfig.model_validate({"provider": "databricks", "model": ".."})


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_key", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
async def test_databricks_routes_through_the_serving_endpoint_and_accounts_the_endpoint_model(
    monkeypatch: pytest.MonkeyPatch, configured_key: bool, legacy: bool
) -> None:
    endpoint: Final = "databricks-openjev-qwen35-4b"
    monkeypatch.setenv("TYPESAFE_API_KEY", "never-send-typesafe-key")
    monkeypatch.setenv("DATABRICKS_API_BASE", "https://workspace.test/serving-endpoints")
    monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-env-token")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setitem(litellm.model_cost, f"databricks/{endpoint}", {"input_cost_per_token": 0.01})
    recorder: Final = _UsageRecorder(f"databricks/{endpoint}")
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    router: Final = ComplexityRouter(
        "databricks-route",
        litellm.Router(model_list=[]),
        {
            "classifier_type": "jev" if legacy else "oss_classifier",
            "jev_classifier_config" if legacy else "opensource_classifier_config": {
                "provider": "databricks",
                "model": endpoint,
                **({"api_key": "dapi-configured", "api_base": "https://workspace.test/serving-endpoints"} if configured_key else {}),
            },
            "tiers": {"SIMPLE": "cheap"},
        },
        derive_savings_baseline=False,
    )
    with respx.mock(assert_all_called=True) as upstream:
        route: Final = upstream.post(f"https://workspace.test/serving-endpoints/{endpoint}/invocations").respond(
            200,
            json={
                "model": "/mosaicml/local_model",
                "answers": {"tier": _answer().model_dump()},
                "usage": {"input_tokens": 31, "output_tokens": 0},
            },
        )
        outcome: Final = await router.aclassify("choose a tier")
        await GLOBAL_LOGGING_WORKER.flush()

    assert outcome.cause == "jev_classifier"
    assert outcome.jev_verdict is not None
    assert (outcome.jev_verdict.provider, outcome.jev_verdict.model) == ("databricks", endpoint)
    assert outcome.classifier_cost == pytest.approx(0.31)
    sent: Final = route.calls.last.request
    assert sent.headers.get("authorization") == ("Bearer dapi-configured" if configured_key else "Bearer dapi-env-token")
    assert json.loads(sent.content)["model"] == endpoint
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["response_cost"] == pytest.approx(0.31)


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_key", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("api_base", ["https://workspace.test", "https://workspace.test/serving-endpoints/"])
async def test_databricks_routes_through_ai_decide_without_a_model_field_and_accounts_ai_decide(
    monkeypatch: pytest.MonkeyPatch, configured_key: bool, legacy: bool, api_base: str
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "never-send-typesafe-key")
    monkeypatch.setenv("DATABRICKS_API_BASE", api_base)
    monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-env-token")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    recorder: Final = _UsageRecorder("databricks/ai_decide")
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    router: Final = ComplexityRouter(
        "databricks-route",
        litellm.Router(model_list=[]),
        {
            "classifier_type": "jev" if legacy else "oss_classifier",
            "jev_classifier_config" if legacy else "opensource_classifier_config": {
                "provider": "databricks",
                "model": "ai_decide",
                **({"api_key": "dapi-configured", "api_base": api_base} if configured_key else {}),
            },
            "tiers": {"SIMPLE": "cheap"},
        },
        derive_savings_baseline=False,
    )
    with respx.mock(assert_all_called=True) as upstream:
        route: Final = upstream.post("https://workspace.test/api/2.0/ai-functions/ai-decide").respond(
            200,
            json={"response": {"answers": {"tier": _answer().model_dump()}}, "metadata": {"version": "1.0"}},
        )
        outcome: Final = await router.aclassify("choose a tier")
        await GLOBAL_LOGGING_WORKER.flush()

    assert outcome.cause == "jev_classifier"
    assert outcome.jev_verdict is not None
    assert (outcome.jev_verdict.provider, outcome.jev_verdict.model) == ("databricks", "ai_decide")
    assert outcome.jev_verdict.label == "SIMPLE"
    assert outcome.classifier_cost is None
    sent: Final = route.calls.last.request
    assert sent.headers.get("authorization") == ("Bearer dapi-configured" if configured_key else "Bearer dapi-env-token")
    assert set(json.loads(sent.content)) == {"state", "questions"}
    assert len(recorder.calls) == 1


def test_databricks_without_a_base_or_key_fails_at_client_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_API_BASE", raising=False)
    monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    with pytest.raises(ValueError, match="DATABRICKS_API_BASE"):
        ComplexityRouter(
            "databricks-route",
            litellm.Router(model_list=[]),
            {
                "classifier_type": "oss_classifier",
                "opensource_classifier_config": {"provider": "databricks", "model": "ai_decide"},
                "tiers": {"SIMPLE": "cheap"},
            },
            derive_savings_baseline=False,
        )


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
        ({}, r"api_base requires opensource_classifier_config\.api_key"),
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


def test_databricks_api_base_without_its_own_key_is_rejected_so_the_workspace_token_stays_home() -> None:
    with pytest.raises(ValueError, match="DATABRICKS_API_KEY or DATABRICKS_TOKEN is only sent to DATABRICKS_API_BASE"):
        JevClassifierConfig.model_validate(
            {"provider": "databricks", "model": "ai_decide", "api_base": "https://collector.invalid"}
        )
    paired: Final = JevClassifierConfig.model_validate(
        {"provider": "databricks", "model": "ai_decide", "api_base": "https://workspace.invalid", "api_key": "dapi-own"}
    )
    assert (paired.api_base, paired.api_key) == ("https://workspace.invalid", "dapi-own")


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
