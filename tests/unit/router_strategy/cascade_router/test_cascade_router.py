"""Cascade router: cheap deployment first, verify the answer, escalate to the strong deployment."""

import asyncio
import datetime
import re
import uuid

import pytest

import litellm
from litellm import Router
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.router_strategy.auto_router.auto_router_model_naming import classify_strategy_router_model
from litellm.router_strategy.cascade_router import CascadeRouter, CascadeRouterConfig, needs_escalation, parse_verdict
from litellm.router_strategy.cascade_router.cascade_router import inner_call_cost, sync_request_logging
from litellm.types.utils import ModelResponse, Usage


def test_cascade_router_marker_is_classified():
    assert classify_strategy_router_model("auto_router/cascade_router/support") == "cascade"
    assert classify_strategy_router_model("auto_router/complexity_router") == "complexity"


@pytest.mark.parametrize(
    "text,finish,expected",
    [
        ("", "stop", "empty"),
        ("Paris.", "stop", "short"),
        ("I'm sorry, but I can't help with that request.", "stop", "refusal"),
        ("The capital of France is Paris, on the Seine.", "length", "truncated"),
        ("The capital of France is Paris, on the Seine.", "stop", None),
    ],
)
def test_needs_escalation_heuristics(text, finish, expected):
    assert needs_escalation(text, finish, min_chars=20) == expected


@pytest.mark.parametrize(
    "raw,expected", [("YES", True), ("yes.", True), ("Y", True), ("NO", False), ("", False), ("maybe", False)]
)
def test_parse_verdict_accepts_only_yes(raw, expected):
    assert parse_verdict(raw) is expected


def test_config_rejects_strong_equal_to_cheap():
    with pytest.raises(ValueError, match="must differ"):
        CascadeRouterConfig(cheap_model="a", strong_model="a")


def _router(cheap_answer: str, verifier_answer: str = "YES", verifier: bool = True) -> Router:
    config = {"cheap_model": "cheap", "strong_model": "strong", "min_chars": 10}
    if verifier:
        config.update({"verifier_model": "judge", "verifier_kind": "chat"})
    return Router(
        model_list=[
            {
                "model_name": "cheap",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "x",
                    "mock_response": cheap_answer,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            },
            {
                "model_name": "strong",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "x",
                    "mock_response": "Paris is the capital of France, on the Seine.",
                },
            },
            {
                "model_name": "judge",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "x", "mock_response": verifier_answer},
            },
            {
                "model_name": "support-cascade",
                "litellm_params": {"model": "auto_router/cascade_router/support", "cascade_router_config": config},
            },
        ]
    )


MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]


async def test_cheap_answer_is_kept_when_verifier_says_yes():
    router = _router("Paris is the capital of France.", verifier_answer="YES")
    response = await router.acompletion(model="support-cascade", messages=MESSAGES)
    assert response.choices[0].message.content == "Paris is the capital of France."
    record = response._hidden_params["cascade"]
    assert record["path"] == "cheap" and record["reason"] is None
    assert [c["role"] for c in record["calls"]] == ["cheap", "verifier"]
    # billing follows the auto-router convention: the answer legs are billed (cheap: $0 here); the verifier
    # is the decision's classifier_cost and is included in the record's total, not in response_cost
    assert response._hidden_params["response_cost"] == pytest.approx(record["billed_cost"])
    assert record["billed_cost"] == pytest.approx(0.0)
    assert record["total_cost"] > record["billed_cost"]
    assert record["routing_decision"]["classifier_model"] == "judge"
    assert record["routing_decision"]["classifier_cost"] == pytest.approx(record["total_cost"] - record["billed_cost"])


async def test_heuristic_escalates_without_calling_the_verifier():
    router = _router("Paris.", verifier_answer="YES")
    response = await router.acompletion(model="support-cascade", messages=MESSAGES)
    assert response.choices[0].message.content == "Paris is the capital of France, on the Seine."
    record = response._hidden_params["cascade"]
    assert record["path"] == "strong" and record["reason"] == "short"
    assert [c["role"] for c in record["calls"]] == ["cheap", "strong"]
    # usage is the sum of the answers taken (cheap + strong); a verifier's tokens stay on its own record
    answers = [c for c in record["calls"] if c["role"] in ("cheap", "strong")]
    assert response.usage.prompt_tokens == sum(c["prompt_tokens"] for c in answers)


async def test_verifier_no_escalates_and_charges_all_three():
    router = _router("The capital of France is Lyon, obviously.", verifier_answer="NO")
    response = await router.acompletion(model="support-cascade", messages=MESSAGES)
    assert response.choices[0].message.content.startswith("Paris is the capital")
    record = response._hidden_params["cascade"]
    assert record["path"] == "strong" and record["reason"] == "verifier_no"
    assert [c["role"] for c in record["calls"]] == ["cheap", "verifier", "strong"]


async def test_no_verifier_means_heuristics_only():
    router = _router("Paris is the capital of France.", verifier=False)
    response = await router.acompletion(model="support-cascade", messages=MESSAGES)
    record = response._hidden_params["cascade"]
    assert record["path"] == "cheap" and [c["role"] for c in record["calls"]] == ["cheap"]


async def test_routing_decision_names_the_cascade():
    router = _router("Paris.", verifier_answer="YES")
    metadata: dict = {}
    response = await router.acompletion(model="support-cascade", messages=MESSAGES, metadata=metadata)
    decision = response._hidden_params["cascade"]["routing_decision"]
    assert decision["router_type"] == "cascade"
    assert decision["router_model_name"] == "support-cascade"
    assert decision["routed_model"] == "strong" and decision["escalated"] is True
    assert decision["cause"] == "cascade_heuristic_escalation" and decision["escalation_reason"] == "short"
    # the baseline is stamped as the savings calculator prices it: the provider model, not the group name
    assert decision["savings_baseline_model"] == "openai/gpt-4o-mini"
    assert (
        decision["savings_baseline_deployment_id"] == router.get_model_list(model_name="strong")[0]["model_info"]["id"]
    )


async def test_cascade_path_is_exposed_as_response_headers():
    """Operators without a spend-log database still need to see which path served a request."""
    router = _router("Paris.", verifier_answer="YES")
    response = await router.acompletion(model="support-cascade", messages=MESSAGES)
    headers = response._hidden_params["additional_headers"]
    assert headers["x-litellm-cascade-path"] == "strong"
    assert headers["x-litellm-cascade-reason"] == "short"
    assert headers["x-litellm-cascade-served-by"] == "strong"


def test_truncation_by_the_callers_own_max_tokens_does_not_escalate():
    """A stronger model with the same max_tokens cannot fix a cap the caller chose."""
    text = "The capital of France is Paris, on the Seine."
    assert needs_escalation(text, "length", min_chars=20, caller_capped=False) == "truncated"
    assert needs_escalation(text, "length", min_chars=20, caller_capped=True) is None


async def test_caller_max_tokens_passes_through_to_the_cascade():
    """A caller-set max_tokens reaches the cheap call and the verdict; the cheap answer stands when it passes."""
    router = _router("The capital of France is Paris, on the Seine.", verifier_answer="YES")
    response = await router.acompletion(model="support-cascade", messages=MESSAGES, max_tokens=12)
    record = response._hidden_params["cascade"]
    assert record["path"] == "cheap" and record["reason"] is None


def test_inner_call_cost_falls_back_to_completion_cost_with_deployment_pricing(monkeypatch):
    """Router responses carry response_cost only after the async logger ran; price them ourselves when missing."""
    import litellm as _litellm

    class Resp:
        _hidden_params = {"model_id": "dep-123"}

    seen = {}

    def fake_cost(completion_response=None, **kw):
        seen.update(kw)
        return 0.000123

    monkeypatch.setattr(_litellm, "completion_cost", fake_cost)
    assert inner_call_cost(Resp()) == pytest.approx(0.000123)
    assert seen["router_model_id"] == "dep-123" and seen["custom_pricing"] is True


def test_inner_call_cost_prefers_the_logged_value_and_tolerates_errors(monkeypatch):
    import litellm as _litellm

    class Priced:
        _hidden_params = {"response_cost": 0.5}

    assert inner_call_cost(Priced()) == 0.5

    class Unpriced:
        _hidden_params = {}

    def boom(**kw):
        raise ValueError("no pricing")

    monkeypatch.setattr(_litellm, "completion_cost", boom)
    assert inner_call_cost(Unpriced()) is None


class _FakeLogging:
    """The slice of litellm's Logging object the cascade touches."""

    def __init__(self, deferred):
        self.model_call_details = {"response_cost": 0.00000043}
        self.deferred_success_result = deferred
        self.prepared_cost = "stale"
        self.finalized_cost = "stale"
        self.defer_async_logging = deferred is not None
        self.fired = 0
        self.enqueue_deferred_logging = self._fire if deferred is not None else None
        self.sync_callbacks_result = None
        self.marked = 0

    def _fire(self):
        self.fired += 1

    def handle_sync_success_callbacks_for_async_calls(self, result, start_time, end_time):
        self.sync_callbacks_result = result

    def mark_async_success_log_enqueued(self):
        self.marked += 1


def test_sync_request_logging_points_the_deferred_log_at_the_final_response(monkeypatch):
    """The request's logging object was bound to the cheap call; after escalation its parked success log,
    its cost and its deferred result must all describe the final answer."""
    import datetime as dt
    import importlib

    import litellm.utils as litellm_utils

    streaming_iterator = importlib.import_module("litellm.responses.streaming_iterator")

    logged = {}

    async def fake_helper(logging_obj, result, start_time, end_time, is_completion_with_fallbacks):
        logged["result"] = result

    def fake_spawn(loop, coro, task_name):
        import asyncio

        asyncio.get_event_loop().run_until_complete(coro) if not loop.is_running() else loop.create_task(coro)

    monkeypatch.setattr(litellm_utils, "_client_async_logging_helper", fake_helper)
    monkeypatch.setattr(streaming_iterator, "spawn_logging_task", fake_spawn)

    cheap, final = object(), object()
    start = dt.datetime(2026, 10, 30, 12, 0, 0)
    log = _FakeLogging(deferred=(cheap, start, start))
    sync_request_logging(log, final=final, billed_cost=0.0000127, fire_deferred=False)
    assert log.deferred_success_result[0] is final and log.deferred_success_result[1] == start
    assert log.model_call_details["response_cost"] == pytest.approx(0.0000127)
    assert log.prepared_cost is None and log.finalized_cost is None
    assert log.fired == 0 and log.enqueue_deferred_logging is not None  # replaced, not fired: the proxy fires it
    # when the proxy fires the parked log, it logs the FINAL answer
    import asyncio

    asyncio.run(_fire(log))
    assert log.sync_callbacks_result is final and log.marked == 1 and logged["result"] is final


async def _fire(log):
    log.enqueue_deferred_logging()


def test_sync_request_logging_fires_when_the_cascade_owns_the_deferral():
    """SDK path: nobody else will fire the parked success log, so the cascade does, once, and clears the flag."""
    log = _FakeLogging(deferred=None)
    log.defer_async_logging = True
    log.enqueue_deferred_logging = log._fire
    sync_request_logging(log, final=object(), billed_cost=0.0, fire_deferred=True)
    assert log.fired == 1 and log.enqueue_deferred_logging is None and log.defer_async_logging is False


def test_sync_request_logging_tolerates_a_missing_logging_object():
    assert sync_request_logging(None, final=object(), billed_cost=1.0, fire_deferred=True) is None


async def test_only_the_verifier_is_an_internal_sub_call_the_strong_leg_continues_the_request(monkeypatch):
    """The verifier is an internal sub-call with a row of its own (the auto-router classifier convention). The
    strong leg is the request itself continued on another deployment: same logging object, same metadata dict,
    no internal-call marker, run as an internal sub-call so it logs no row of its own and the request's one row
    bills both legs."""
    import litellm as litellm_mod
    from litellm._internal_context import is_internal_call

    calls = []
    original = litellm_mod.acompletion

    async def spy(**kwargs):
        calls.append((dict(kwargs), is_internal_call.get()))
        return await original(**kwargs)

    monkeypatch.setattr(litellm_mod, "acompletion", spy)
    router = _router("The capital of France is Lyon, obviously.", verifier_answer="NO")
    await router.acompletion(model="support-cascade", messages=MESSAGES)
    (cheap, cheap_internal), (verifier, verifier_internal), (strong, strong_internal) = calls
    assert cheap["metadata"].get(INTERNAL_CALL_ORIGIN_METADATA_KEY) is None and cheap_internal is False
    assert verifier["metadata"][INTERNAL_CALL_ORIGIN_METADATA_KEY] == "cascade_verifier" and verifier_internal is False
    assert strong["metadata"] is cheap["metadata"] and strong["litellm_logging_obj"] is cheap["litellm_logging_obj"]
    assert INTERNAL_CALL_ORIGIN_METADATA_KEY not in strong["metadata"] and strong_internal is True


async def test_escalation_does_not_mutate_the_strong_calls_own_record():
    """The Router's own response objects (the cheap call's, whose log may already be queued on the SDK path, and
    the strong call's) stay as their calls left them; the caller gets a copy carrying the cascade's sums."""
    router = _router("Paris.", verifier_answer="YES")
    response = await router.acompletion(model="support-cascade", messages=MESSAGES)
    strong_resp = await router.acompletion(model="strong", messages=MESSAGES)  # same mock object as the inner call
    assert response.choices[0].message.content == strong_resp.choices[0].message.content
    record = response._hidden_params["cascade"]
    assert record["path"] == "strong"
    assert response.usage.prompt_tokens > strong_resp.usage.prompt_tokens  # summed vs own
    assert response._hidden_params["response_cost"] == pytest.approx(record["billed_cost"])
    assert "cascade" not in strong_resp._hidden_params  # the strong call's own record is untouched


# ---- review findings: decision verifier through the Router, unusable output, buckets, failures ----


def _decision_router(
    cheap_answer: str, choice: str, confidence: float, min_conf: float = 0.0, on_error: str = "escalate"
):
    from litellm.types.decisions import ChoiceAnswer, DecisionsResponse

    config = {
        "cheap_model": "cheap",
        "strong_model": "strong",
        "min_chars": 10,
        "verifier_model": "judge-decision",
        "verifier_kind": "decision",
        "verifier_min_confidence": min_conf,
        "on_verifier_error": on_error,
    }
    router = Router(
        model_list=[
            {
                "model_name": "cheap",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "x",
                    "mock_response": cheap_answer,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            },
            {
                "model_name": "strong",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "x",
                    "mock_response": "Paris is the capital of France, on the Seine.",
                },
            },
            {
                "model_name": "support-cascade",
                "litellm_params": {"model": "auto_router/cascade_router/support", "cascade_router_config": config},
            },
        ]
    )
    seen = {}

    async def fake_adecisions(model, **kwargs):
        seen["model"] = model
        seen["metadata"] = kwargs.get("metadata")
        seen["questions"] = kwargs.get("questions")
        answers = {
            "answered": ChoiceAnswer(
                type="choice", choice=choice, confidence=confidence, probabilities={choice: confidence}
            )
        }
        return DecisionsResponse(model=model, answers=answers)

    return router, fake_adecisions, seen


async def test_decision_verifier_goes_through_the_router_and_passes_on_yes():
    router, fake, seen = _decision_router("Paris is the capital of France.", "yes", 0.9)
    router.adecisions = fake  # bound per instance in Router.__init__
    response = await router.acompletion(
        model="support-cascade", messages=MESSAGES, metadata={"user_api_key": "sk-caller"}
    )
    record = response._hidden_params["cascade"]
    assert record["path"] == "cheap" and record["reason"] is None
    assert seen["model"] == "judge-decision"  # a model GROUP, resolved by the Router, not a provider id
    assert seen["questions"]["answered"]["type"] == "choice"
    assert seen["metadata"]["user_api_key"] == "sk-caller"  # caller identity forwarded to the sub-call
    assert seen["metadata"][INTERNAL_CALL_ORIGIN_METADATA_KEY] == "cascade_verifier"
    verifier = [c for c in record["calls"] if c["role"] == "verifier"][0]
    assert verifier["choice"] == "yes" and verifier["confidence"] == pytest.approx(0.9)


async def test_decision_verifier_low_confidence_yes_escalates():
    router, fake, _ = _decision_router("Paris is the capital of France.", "yes", 0.4, min_conf=0.6)
    router.adecisions = fake
    response = await router.acompletion(model="support-cascade", messages=MESSAGES)
    record = response._hidden_params["cascade"]
    assert record["path"] == "strong" and record["reason"] == "verifier_no"
    assert record["routing_decision"]["cause"] == "cascade_verifier_escalation"


async def test_decision_verifier_no_escalates():
    router, fake, _ = _decision_router("Paris is the capital of France.", "no", 0.95)
    router.adecisions = fake
    response = await router.acompletion(model="support-cascade", messages=MESSAGES)
    assert response._hidden_params["cascade"]["path"] == "strong"


@pytest.mark.parametrize(
    "on_error,expected_path,expected_cause",
    [("escalate", "strong", "cascade_verifier_error"), ("accept", "cheap", "cascade_pass")],
)
async def test_unusable_verifier_output_is_governed_by_on_verifier_error(
    monkeypatch, on_error, expected_path, expected_cause
):
    from litellm.types.decisions import DecisionsResponse

    router, _, _ = _decision_router("Paris is the capital of France.", "yes", 0.9, on_error=on_error)

    async def no_answers(model, **kwargs):
        return DecisionsResponse(model=model, answers={})

    router.adecisions = no_answers
    response = await router.acompletion(model="support-cascade", messages=MESSAGES)
    record = response._hidden_params["cascade"]
    assert record["path"] == expected_path and record["routing_decision"]["cause"] == expected_cause
    assert record["verifier_error"]  # the reason the verifier could not be used is recorded


@pytest.mark.parametrize("on_error,expected_path", [("escalate", "strong"), ("accept", "cheap")])
async def test_verifier_exception_is_governed_by_on_verifier_error(on_error, expected_path):
    config = {
        "cheap_model": "cheap",
        "strong_model": "strong",
        "min_chars": 10,
        "verifier_model": "judge",
        "verifier_kind": "chat",
        "on_verifier_error": on_error,
    }
    router = Router(
        model_list=[
            {
                "model_name": "cheap",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "x",
                    "mock_response": "Paris is the capital of France.",
                },
            },
            {
                "model_name": "strong",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "x",
                    "mock_response": "Paris is the capital of France, on the Seine.",
                },
            },
            {
                "model_name": "judge",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "x",
                    "mock_response": Exception("judge down"),
                },
            },
            {
                "model_name": "support-cascade",
                "litellm_params": {"model": "auto_router/cascade_router/support", "cascade_router_config": config},
            },
        ],
        num_retries=0,
    )
    response = await router.acompletion(model="support-cascade", messages=MESSAGES)
    assert response._hidden_params["cascade"]["path"] == expected_path


async def test_litellm_metadata_bucket_still_triggers_the_judge():
    """The proxy keeps internal metadata in `litellm_metadata` on some routes; the cascade must find its stamp there."""
    router = _router("Paris.", verifier_answer="YES")
    response = await router.acompletion(model="support-cascade", messages=MESSAGES, litellm_metadata={})
    assert response._hidden_params["cascade"]["path"] == "strong"


def _strong_down_router(**config_extra):
    config = {"cheap_model": "cheap", "strong_model": "strong", "min_chars": 10, **config_extra}
    return Router(
        model_list=[
            {
                "model_name": "cheap",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "x", "mock_response": "Paris."},
            },
            {
                "model_name": "strong",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "x",
                    "mock_response": Exception("strong down"),
                },
            },
            {
                "model_name": "support-cascade",
                "litellm_params": {"model": "auto_router/cascade_router/support", "cascade_router_config": config},
            },
        ],
        num_retries=0,
    )


async def test_strong_call_failure_propagates_and_is_not_the_cheap_deployments_fault():
    from litellm.router_utils.cooldown_handlers import is_failure_outside_selected_deployment

    router = _strong_down_router()
    strong_id = router.get_model_list(model_name="strong")[0]["model_info"]["id"]
    with pytest.raises(Exception, match="strong down") as excinfo:
        await router.acompletion(model="support-cascade", messages=MESSAGES)
    # the strong deployment raised and is the one on record; the cheap deployment answered and must not cool down
    assert getattr(excinfo.value, "failed_deployment_id", None) == strong_id
    assert is_failure_outside_selected_deployment(excinfo.value)


async def test_a_failed_escalation_is_not_retried_as_a_whole_cascade(monkeypatch):
    """The strong leg goes through its own routing and retries; when it still fails under `on_strong_error: raise`,
    the Router's retry loop does not run the cheap call and the verifier again (the request's retries are spent),
    and the request's fallbacks take over as they do for any failed attempt."""
    import litellm as litellm_mod

    calls: dict[str, int] = {}
    original = litellm_mod.acompletion

    async def capture(**kwargs):
        name = kwargs["metadata"].get("deployment_model_name")
        calls[name] = calls.get(name, 0) + 1
        return await original(**kwargs)

    monkeypatch.setattr(litellm_mod, "acompletion", capture)
    router = Router(
        model_list=[
            {
                "model_name": "cheap",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "x", "mock_response": "tiny"},
            },
            {
                "model_name": "judge",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "x", "mock_response": "NO"},
            },
            {
                "model_name": "strong",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "x", "mock_response": Exception("strong busy")},
            },
            {
                "model_name": "fb",
                "litellm_params": {"model": "openai/gpt-4.1", "api_key": "x", "mock_response": "Paris."},
            },
            {
                "model_name": "cascade",
                "litellm_params": {
                    "model": "auto_router/cascade_router",
                    "cascade_router_config": {
                        "cheap_model": "cheap",
                        "strong_model": "strong",
                        "min_chars": 2,
                        "verifier_model": "judge",
                        "verifier_kind": "chat",
                    },
                },
            },
        ],
        num_retries=2,
        retry_after=0,
        allowed_fails=100,
        fallbacks=[{"cascade": ["fb"]}],
    )
    response = await router.acompletion(model="cascade", messages=MESSAGES, num_retries=2)
    assert response.choices[0].message.content == "Paris."
    assert calls == {"cheap": 1, "judge": 1, "strong": 3, "fb": 1}  # the strong leg's own retries, nothing else twice


async def test_on_strong_error_cheap_serves_the_judged_answer_and_says_so():
    router = _strong_down_router(on_strong_error="cheap")
    response = await router.acompletion(model="support-cascade", messages=MESSAGES)
    record = response._hidden_params["cascade"]
    assert response.choices[0].message.content == "Paris."
    assert record["path"] == "cheap" and record["reason"] == "strong_error"
    assert record["strong_error"] == "InternalServerError"  # the error's class; its message could echo the prompt
    assert record["routing_decision"]["cause"] == "cascade_strong_error"
    assert record["routing_decision"]["escalation_reason"] == "short"  # the check that asked for the strong model
    assert record["routing_decision"]["escalated"] is False
    assert response._hidden_params["additional_headers"]["x-litellm-cascade-reason"] == "strong_error"


async def test_strong_call_gets_the_callers_parameters_and_the_requests_identity(monkeypatch):
    """The strong call is the same request to another model: the caller's parameters (not only the few
    OpenAI basics) are forwarded, and it carries the request's own metadata (identity, stamps); LiteLLM's
    own kwargs and the cascade's config are not forwarded."""
    import litellm as litellm_mod

    calls = []
    original = litellm_mod.acompletion

    async def spy(**kwargs):
        calls.append(kwargs)
        return await original(**kwargs)

    monkeypatch.setattr(litellm_mod, "acompletion", spy)
    router = _router("Paris.")  # heuristics only: too short -> strong
    await router.acompletion(
        model="support-cascade",
        messages=MESSAGES,
        n=1,
        logit_bias={"1": 1},
        extra_body={"a": 1},
        user="u1",
        timeout=30,
        metadata={"user_api_key": "sk-caller"},
    )
    forwarded = calls[-1]  # the strong leg, routed last
    assert forwarded["metadata"] is calls[0]["metadata"]  # the request's own bucket, not a copy
    assert forwarded["metadata"]["deployment_model_name"] == "strong"
    assert forwarded["n"] == 1 and forwarded["logit_bias"] == {"1": 1} and forwarded["extra_body"] == {"a": 1}
    assert forwarded["user"] == "u1" and forwarded["timeout"] == 30
    assert forwarded["metadata"]["user_api_key"] == "sk-caller"  # caller identity, for its spend row
    assert "cascade_router_config" not in forwarded
    # the shared bucket ends up carrying the request's final decision, and the strong answer was not judged again
    assert forwarded["metadata"]["routing_decision"]["tier"] == "strong"


async def test_escalation_restamps_the_request_for_the_deployment_that_answered(monkeypatch):
    import litellm as litellm_mod

    calls = []
    original = litellm_mod.acompletion

    async def spy(**kwargs):
        calls.append(kwargs)
        return await original(**kwargs)

    monkeypatch.setattr(litellm_mod, "acompletion", spy)
    router = _router("Paris.")
    strong_id = router.get_model_list(model_name="strong")[0]["model_info"]["id"]
    await router.acompletion(model="support-cascade", messages=MESSAGES)
    request_metadata = calls[0]["metadata"]  # the request's own bucket, stamped for the cheap call, then re-stamped
    assert request_metadata["deployment_model_name"] == "strong"
    assert request_metadata["model_info"]["id"] == strong_id
    assert request_metadata["deployment"] == "openai/gpt-4o-mini"
    assert request_metadata["model_group"] == "support-cascade"  # one request on the alias, whichever group answered
    assert request_metadata["routing_decision"]["savings_baseline_deployment_id"] == strong_id


async def test_escalated_response_keeps_the_requests_own_id(monkeypatch):
    """The caller's response keeps the id the request's logging started with (the cheap call's), whichever
    path answered, so the id a client correlates on never depends on the escalation."""
    import litellm as litellm_mod

    returned = []
    original = litellm_mod.acompletion

    async def spy(**kwargs):
        response = await original(**kwargs)
        returned.append(response)
        return response

    monkeypatch.setattr(litellm_mod, "acompletion", spy)
    router = _router("Paris.")  # shorter than min_chars: escalates without a verifier call
    response = await router.acompletion(model="support-cascade", messages=MESSAGES)
    cheap_resp, strong_resp = returned[0], returned[-1]
    assert response._hidden_params["cascade"]["path"] == "strong"
    assert cheap_resp.id != strong_resp.id
    assert response.id == cheap_resp.id
    assert response.id != strong_resp.id
    assert response.choices[0].message.content == strong_resp.choices[0].message.content


async def test_cheap_path_leaves_the_cheap_calls_own_response_untouched(monkeypatch):
    """SDK path: the cheap call's success log is already queued with its response object when the cascade
    runs; the cascade's record and sums go on a copy."""
    import litellm as litellm_mod

    returned = []
    original = litellm_mod.acompletion

    async def spy(**kwargs):
        response = await original(**kwargs)
        returned.append(response)
        return response

    monkeypatch.setattr(litellm_mod, "acompletion", spy)
    router = _router("Paris is the capital of France.", verifier_answer="YES")
    response = await router.acompletion(model="support-cascade", messages=MESSAGES)
    cheap_resp = returned[0]
    assert response is not cheap_resp
    assert "cascade" not in (cheap_resp._hidden_params or {})
    assert response._hidden_params["cascade"]["path"] == "cheap"


def test_usage_of_reads_a_decisions_response():
    from litellm.router_strategy.cascade_router.cascade_router import _usage_of
    from litellm.types.decisions import DecisionsResponse, DecisionsUsage

    assert _usage_of(
        DecisionsResponse(model="m", answers={}, usage=DecisionsUsage(input_tokens=120, output_tokens=3))
    ) == (120, 3)


async def test_streaming_request_is_served_by_the_cheap_group_unjudged():
    router = _router("Paris.", verifier_answer="YES")
    stream = await router.acompletion(model="support-cascade", messages=MESSAGES, stream=True)
    chunks = [c async for c in stream]
    assert chunks and "cascade" not in (getattr(stream, "_hidden_params", {}) or {})


async def test_tool_call_answers_are_not_treated_as_empty():
    """A cheap answer that is a tool call has no text by design; it must not escalate as 'empty'."""
    from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message, ModelResponse

    tool_call = ChatCompletionMessageToolCall(
        id="c1", type="function", function=Function(name="lookup", arguments="{}")
    )
    resp = ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": Message(content=None, role="assistant", tool_calls=[tool_call]),
            }
        ]
    )
    strategy = CascadeRouter(
        model_name="x",
        ctx=None,
        cascade_router_config={"cheap_model": "a", "strong_model": "b", "verifier_model": "j", "verifier_kind": "chat"},
    )
    verdict = await strategy.async_verify(MESSAGES, resp)
    assert verdict.reason is None and verdict.cause == "cascade_pass" and verdict.calls == ()


@pytest.mark.parametrize(
    "config,fragment",
    [
        ({"cheap_model": "a"}, "strong_model"),
        ({"strong_model": "b"}, "cheap_model"),
        ({"cheap_model": "a", "strong_model": "b", "verifier_min_confidence": 1.5}, "verifier_min_confidence"),
        ({"cheap_model": "a", "strong_model": "b", "verifier_kind": "oracle"}, "verifier_kind"),
        ({"cheap_model": "a", "strong_model": "b", "on_verifier_error": "shrug"}, "on_verifier_error"),
        ({"cheap_model": "a", "strong_model": "b", "on_strong_error": "shrug"}, "on_strong_error"),
    ],
)
def test_config_validation_rejects_bad_configs(config, fragment):
    with pytest.raises(ValueError, match=re.escape(fragment)):
        CascadeRouterConfig(**config)


def test_marker_deployment_requires_a_config_and_cannot_be_its_own_tier():
    with pytest.raises(ValueError, match="cascade_router_config is required"):
        Router(model_list=[{"model_name": "c", "litellm_params": {"model": "auto_router/cascade_router/c"}}])
    with pytest.raises(ValueError, match="cannot be its own"):
        CascadeRouter(
            model_name="c",
            ctx=None,
            cascade_router_config={"cheap_model": "c", "strong_model": "b"},
        )


def test_hot_reload_clears_the_cascade_registry():
    router = _router("Paris is the capital of France.")
    assert "support-cascade" in router.registry().plugins("cascade")
    router.set_model_list([{"model_name": "cheap", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "x"}}])
    assert "support-cascade" not in router.registry().plugins("cascade")


def test_last_user_text_reads_pydantic_messages_and_skips_other_objects():
    """The cascade sees the caller's raw list: dicts, LiteLLM `Message` objects and SDK message models all count;
    an object that is neither is skipped rather than failing the verifier step."""
    from litellm.router_strategy.cascade_router.cascade_router import last_user_text
    from litellm.types.utils import Message

    class NotAMessage:  # has role/content attributes but is neither a mapping nor a pydantic model
        role = "user"
        content = "not this one"

    assert last_user_text([{"role": "user", "content": "first"}, NotAMessage()]) == "first"
    assert last_user_text([NotAMessage()]) == ""
    assert last_user_text([{"role": "user", "content": "first"}, Message(role="user", content="latest")]) == "latest"


def test_inner_call_metadata_merges_both_buckets_and_keeps_the_proxys_identity_keys():
    """Identity keys the proxy stamped in `metadata` must reach the sub-calls when a hook planted `litellm_metadata`,
    and the identity keys win over a conflicting value there, as they do for the request's own spend row."""
    from litellm.router_strategy.cascade_router.cascade_router import parent_metadata_of

    merged = parent_metadata_of(
        {
            "metadata": {"user_api_key_team_id": "proxy-team", "litellm_received_at": 1.0, "x": 1},
            "litellm_metadata": {"user_api_key_team_id": "hook-team", "litellm_received_at": 2.0, "x": 2, "y": 3},
        }
    )
    assert merged == {"user_api_key_team_id": "proxy-team", "litellm_received_at": 2.0, "x": 2, "y": 3}
    assert parent_metadata_of({"metadata": {"litellm_received_at": 1.0}, "litellm_metadata": {"y": 3}}) == {
        "litellm_received_at": 1.0,
        "y": 3,
    }
    assert parent_metadata_of({"metadata": {"a": 1}}) == {"a": 1}
    assert parent_metadata_of({"litellm_metadata": {}, "metadata": {"a": 1}}) == {"a": 1}
    assert parent_metadata_of({}) is None


# ---- review findings: one row bills both legs, failure keeps the cheap leg, verifier question, baseline ----


class _Recorder(CustomLogger):
    """What the caller's callbacks see: one entry per logged call."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    def _record(self, status: str, kwargs: dict, response_obj: object) -> None:
        payload = kwargs.get("standard_logging_object") or {}
        metadata = (kwargs.get("litellm_params") or {}).get("metadata") or {}
        choices = getattr(response_obj, "choices", None)
        self.events.append(
            {
                "status": status,
                "origin": metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY),
                "cost": payload.get("response_cost", kwargs.get("response_cost")),
                "content": choices[0].message.content if choices else None,
                "breakdown": payload.get("cost_breakdown"),
                "tokens": (payload.get("prompt_tokens"), payload.get("completion_tokens")),
                "model_group": metadata.get("model_group"),
                "deployment_model_name": metadata.get("deployment_model_name"),
                "model": payload.get("model"),
                "model_id": payload.get("model_id"),
                "messages": payload.get("messages"),
                "decision": metadata.get("routing_decision"),
            }
        )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record("success", kwargs, response_obj)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record("failure", kwargs, response_obj)


def _priced_router(
    strong_answer: object = "The capital of France is Paris.",
    fallbacks: list | None = None,
    cheap_params: dict | None = None,
    strong_params: dict | None = None,
    fb_params: dict | None = None,
    extra_deployments: list | None = None,
    **router_or_config,
) -> Router:
    """Controlled prices: a mock answer is 10 prompt + 20 completion tokens, so cheap costs $0.05, strong $0.50 and
    the fallback group $5. The cheap and strong deployments are different provider models, so attribution shows."""

    def deployment(name: str, model: str, answer: object, input_cost: float, output_cost: float) -> dict:
        return {
            "model_name": name,
            "litellm_params": {
                "model": model,
                "api_key": "x",
                "mock_response": answer,
                "input_cost_per_token": input_cost,
                "output_cost_per_token": output_cost,
            },
            "model_info": {"id": name},
        }

    router_kwargs = {
        k: router_or_config.pop(k)
        for k in (
            "cache_responses",
            "caching_groups",
            "routing_strategy",
            "content_policy_fallbacks",
            "enable_pre_call_checks",
        )
        if k in router_or_config
    }
    config = {"cheap_model": "cheap", "strong_model": "strong", "min_chars": 20, **router_or_config}
    cheap = deployment("cheap", "openai/gpt-4o-mini", "tiny", 0.001, 0.002)
    cheap["litellm_params"].update(cheap_params or {})
    strong = deployment("strong", "openai/gpt-4o", strong_answer, 0.01, 0.02)
    strong["litellm_params"].update(strong_params or {})
    fb = deployment("fb", "openai/gpt-4.1", "Fallback says Paris is the capital of France.", 0.1, 0.2)
    fb["litellm_params"].update(fb_params or {})
    return Router(
        model_list=[
            cheap,
            strong,
            deployment("judge", "openai/gpt-4o-mini", "YES", 0.001, 0.002),
            fb,
            *(extra_deployments or []),
            {
                "model_name": "cascade",
                "litellm_params": {"model": "auto_router/cascade_router", "cascade_router_config": config},
            },
        ],
        num_retries=0,
        fallbacks=fallbacks,
        **router_kwargs,
    )


def _proxy_logging(recorder: _Recorder) -> Logging:
    """The logging object the proxy hands in: success logging deferred past its post-call hooks, callbacks bound."""
    log = Logging(
        model="cascade",
        messages=MESSAGES,
        stream=False,
        call_type="acompletion",
        start_time=datetime.datetime.now(),
        litellm_call_id=str(uuid.uuid4()),
        function_id="test",
        dynamic_async_success_callbacks=[recorder],
        dynamic_async_failure_callbacks=[recorder],
    )
    log.defer_async_logging = True
    return log


async def _flush_logging() -> None:
    for _ in range(4):
        await asyncio.sleep(0)
        await GLOBAL_LOGGING_WORKER.flush()


async def test_sdk_path_logs_the_request_once_billing_both_legs(monkeypatch):
    """SDK callers' callbacks see ONE record for an escalated request: the strong answer, cheap + strong cost
    ($0.05 + $0.50) and the two legs' breakdowns summed. The strong leg has no record of its own (it would have
    billed $0.50 again), and the rejected cheap answer is not what the trace shows."""
    recorder = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router = _priced_router()
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    await _flush_logging()
    assert response.choices[0].message.content == "The capital of France is Paris."
    assert response._hidden_params["cascade"]["billed_cost"] == pytest.approx(0.55)
    assert [(e["status"], e["origin"]) for e in recorder.events] == [("success", None)]
    (event,) = recorder.events
    assert event["content"] == "The capital of France is Paris." and event["cost"] == pytest.approx(0.55)
    assert event["breakdown"]["total_cost"] == pytest.approx(0.55)
    assert event["breakdown"]["input_cost"] == pytest.approx(0.11)
    assert event["breakdown"]["output_cost"] == pytest.approx(0.44)
    assert event["tokens"] == (20, 40)
    assert event["decision"]["tier"] == "strong" and event["decision"]["routed_model"] == "strong"
    assert event["model_group"] == "cascade" and event["deployment_model_name"] == "strong"
    assert event["model"] == "openai/gpt-4o"  # the row names the deployment that answered
    assert event["messages"] == MESSAGES  # the row carries the prompt, as a plain request's row does


async def test_tagged_cascades_escalate_through_the_configuration_pre_routing_selected():
    """Two markers share the alias, one per tag. A tag-b request routes to cheap-b; the judgment must read the same
    tagged configuration back, so the escalation goes to strong-b (not the first marker's strong-a)."""

    def deployment(name, answer):
        return {
            "model_name": name,
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "x", "mock_response": answer},
            "model_info": {"id": name},
        }

    def marker(cheap, strong, tag):
        return {
            "model_name": "cascade",
            "litellm_params": {
                "model": "auto_router/cascade_router",
                "cascade_router_config": {"cheap_model": cheap, "strong_model": strong, "min_chars": 20},
                "tags": [tag],
            },
            "model_info": {"id": f"cascade-{tag}"},
        }

    router = Router(
        model_list=[
            deployment("cheap-a", "tiny"),
            deployment("cheap-b", "tiny"),
            deployment("strong-a", "Answer from strong A, long enough."),
            deployment("strong-b", "Answer from strong B, long enough."),
            marker("cheap-a", "strong-a", "a"),
            marker("cheap-b", "strong-b", "b"),
        ],
        num_retries=0,
    )
    response = await router.acompletion(model="cascade", messages=MESSAGES, metadata={"tags": ["b"]})
    assert response.choices[0].message.content == "Answer from strong B, long enough."
    record = response._hidden_params["cascade"]
    assert record["served_by"] == "strong-b"
    assert [c["model"] for c in record["calls"]] == ["cheap-b", "strong-b"]


class _HoldFirstSuccess(CustomLogger):
    """Holds the first request's success callback until released, as a slow logging sink would."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.held = False

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if not self.held:
            self.held = True
            self.started.set()
            await self.release.wait()


async def test_budget_attribution_survives_a_reused_metadata_dict_while_the_first_callback_is_pending(monkeypatch):
    """Two escalated requests reuse one metadata dict; the second starts while the first's budget callback is still
    waiting on a slow sink. Each cheap leg ($0.05) stays with the cheap deployment and each strong leg ($0.50) with
    the strong one: the attribution is the request's own, not something the next request can overwrite."""
    gate = _HoldFirstSuccess()
    monkeypatch.setattr(litellm, "callbacks", [gate])
    router = _priced_router(
        cheap_params={"max_budget": 100, "budget_duration": "1h"},
        strong_params={"max_budget": 100, "budget_duration": "1h"},
    )
    shared_metadata: dict = {}
    await router.acompletion(model="cascade", messages=MESSAGES, metadata=shared_metadata)
    await asyncio.wait_for(gate.started.wait(), 10)
    try:
        await router.acompletion(model="cascade", messages=MESSAGES, metadata=shared_metadata)
    finally:
        gate.release.set()
    await _flush_logging()
    limiter = router.registry().plugins("budget_limiter")
    assert await limiter.dual_cache.async_get_cache("deployment_spend:cheap:1h") == pytest.approx(0.10)
    assert await limiter.dual_cache.async_get_cache("deployment_spend:strong:1h") == pytest.approx(1.00)


def test_the_strong_call_keeps_the_requests_policies_and_drops_its_credentials():
    """The caller's own parameters pass; LiteLLM-owned ones do not, except the request's guardrails, region
    restriction and message-logging opt-out, which must hold on the strong leg as they held on the cheap one."""
    from litellm.router_rust.steps.cascade import strong_call_kwargs

    request = {
        "model": "cheap",
        "messages": MESSAGES,
        "temperature": 0.2,
        "guardrails": ["pii-scrub"],
        "allowed_model_region": "eu",
        "turn_off_message_logging": True,
        "no-log": True,
        "cache": {"no-cache": True},
        "api_key": "cheap-secret",
        "mock_response": "tiny",
        "metadata": {"model_group": "cascade"},
        "litellm_call_id": "call-1",
        "stream": True,
    }
    kwargs = strong_call_kwargs(request, None, "strong", MESSAGES)
    assert kwargs["model"] == "strong" and kwargs["messages"] == MESSAGES and kwargs["temperature"] == 0.2
    assert kwargs["guardrails"] == ["pii-scrub"]
    assert kwargs["allowed_model_region"] == "eu"
    assert kwargs["turn_off_message_logging"] is True
    assert kwargs["no-log"] is True and kwargs["cache"] == {"no-cache": True}
    assert kwargs["metadata"] is request["metadata"] and kwargs["litellm_call_id"] == "call-1"
    assert "api_key" not in kwargs and "mock_response" not in kwargs and "stream" not in kwargs


async def test_a_no_log_request_stays_unlogged_after_an_escalation(monkeypatch):
    """`no-log` is LiteLLM-owned and would be dropped from the strong call, whose wrapper then rewrites the shared
    logging object's params without it; the request's one row must stay suppressed."""
    recorder = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router = _priced_router()
    response = await router.acompletion(model="cascade", messages=MESSAGES, **{"no-log": True})
    await _flush_logging()
    assert response._hidden_params["cascade"]["path"] == "strong"
    assert recorder.events == []


def _verifier_calls(monkeypatch) -> list:
    """Capture the kwargs of every verifier call (its metadata names the cascade-verifier origin)."""
    import litellm as litellm_mod

    seen: list = []
    original = litellm_mod.acompletion

    async def capture(**kwargs):
        if (kwargs.get("metadata") or {}).get(INTERNAL_CALL_ORIGIN_METADATA_KEY) == "cascade_verifier":
            seen.append(kwargs)
        return await original(**kwargs)

    monkeypatch.setattr(litellm_mod, "acompletion", capture)
    return seen


async def test_the_verifier_call_carries_the_requests_cache_and_log_policies(monkeypatch):
    seen = _verifier_calls(monkeypatch)
    router = _priced_router(verifier_model="judge", verifier_kind="chat", min_chars=2)
    response = await router.acompletion(
        model="cascade",
        messages=MESSAGES,
        cache={"no-cache": True},
        **{"no-log": True},
    )
    assert response._hidden_params["cascade"]["path"] == "cheap"
    (call,) = seen
    assert call["cache"] == {"no-cache": True}
    assert call["no-log"] is True


async def test_a_region_restricted_request_never_reaches_an_out_of_region_verifier(monkeypatch):
    """With `allowed_model_region: eu`, a verifier whose only deployment is in another region is not called at all;
    the verifier error is handled by `on_verifier_error`, as a direct call to that deployment would be refused."""
    seen = _verifier_calls(monkeypatch)
    router = _priced_router(
        verifier_model="judge-us",
        verifier_kind="chat",
        min_chars=2,
        on_verifier_error="accept",
        enable_pre_call_checks=True,  # the region filter is one of the router's pre-call checks
        cheap_params={"region_name": "eu"},
        strong_params={"region_name": "eu"},
        extra_deployments=[
            {
                "model_name": "judge-us",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "x",
                    "mock_response": "YES",
                    "region_name": "us",
                },
                "model_info": {"id": "judge-us"},
            }
        ],
    )
    response = await router.acompletion(model="cascade", messages=MESSAGES, allowed_model_region="eu")
    record = response._hidden_params["cascade"]
    assert record["path"] == "cheap" and record["verifier_error"] is not None
    assert seen == []


async def test_a_pre_call_guardrail_that_ran_on_the_request_still_scans_the_verifier_prompt(monkeypatch):
    """The proxy marks a pre-call guardrail as already run on the request, so the deployment hook does not run it a
    second time on the same messages. The verifier's prompt is new (the request and the cheap answer): its call
    carries none of that marker, and the guardrail scans it."""
    from litellm.integrations.custom_guardrail import CustomGuardrail

    class _Scan(CustomGuardrail):
        def __init__(self):
            super().__init__(guardrail_name="scan", event_hook="pre_call", default_on=False)
            self.scanned = []

        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            self.scanned.append(str(data["messages"]))
            return data

    guardrail = _Scan()
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    router = _priced_router(verifier_model="judge", verifier_kind="chat", min_chars=2)
    metadata = {}
    guardrail.mark_pre_call_hook_ran({"metadata": metadata})  # as the proxy records its pre-call run on the request
    response = await router.acompletion(model="cascade", messages=MESSAGES, guardrails=["scan"], metadata=metadata)
    assert response._hidden_params["cascade"]["path"] == "cheap"
    assert len(guardrail.scanned) == 1 and "tiny" in guardrail.scanned[0]  # the verifier's prompt, scanned once


async def test_the_verifier_call_carries_the_requests_tags_not_the_cheap_deployments(monkeypatch):
    """Tag routing and tag spend on the verifier's own call see the tags the request sent, not the cheap deployment's
    tags the Router merged into the request's metadata when it served the cheap leg."""
    seen = _verifier_calls(monkeypatch)
    router = _priced_router(
        verifier_model="judge", verifier_kind="chat", min_chars=2, cheap_params={"tags": ["cheap-tier"]}
    )
    await router.acompletion(model="cascade", messages=MESSAGES, metadata={"tags": ["team-x"]})
    (call,) = seen
    assert call["metadata"]["tags"] == ["team-x"]


async def test_a_verifier_errors_text_stays_out_of_the_decision_and_the_log(monkeypatch):
    """A provider error can echo the prompt or the answer: the decision records the error's class only (as the
    auto-router records its classifier's) and the warning names the class, so a message-logging opt-out holds."""
    from litellm.router_strategy.cascade_router import cascade_router as cascade_module

    logged = []
    monkeypatch.setattr(cascade_module.verbose_router_logger, "warning", lambda msg, *args: logged.append(msg % args))
    echoed = f"rejected input: {MESSAGES[-1]['content']} tiny"
    router = _priced_router(
        verifier_model="judge-down",
        verifier_kind="chat",
        min_chars=2,
        on_verifier_error="accept",
        extra_deployments=[
            {
                "model_name": "judge-down",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "x", "mock_response": Exception(echoed)},
                "model_info": {"id": "judge-down"},
            }
        ],
    )
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    record = response._hidden_params["cascade"]
    assert record["path"] == "cheap" and record["verifier_error"]
    assert "capital" not in record["verifier_error"] and "tiny" not in record["verifier_error"]
    assert record["routing_decision"]["classifier_error_type"] == record["verifier_error"]
    assert logged and all("capital" not in line and "tiny" not in line for line in logged)


async def test_a_no_log_request_logs_no_verifier_row_either(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router = _priced_router(verifier_model="judge", verifier_kind="chat", min_chars=2)
    response = await router.acompletion(model="cascade", messages=MESSAGES, **{"no-log": True})
    await _flush_logging()
    assert [c["role"] for c in response._hidden_params["cascade"]["calls"]] == ["cheap", "verifier"]
    assert recorder.events == []


async def test_the_verifier_keeps_the_callers_message_logging_opt_out(monkeypatch):
    import litellm as litellm_mod

    seen = []
    original = litellm_mod.acompletion

    async def capture(**kwargs):
        if (kwargs.get("metadata") or {}).get(INTERNAL_CALL_ORIGIN_METADATA_KEY) == "cascade_verifier":
            seen.append(kwargs.get("turn_off_message_logging"))
        return await original(**kwargs)

    monkeypatch.setattr(litellm_mod, "acompletion", capture)
    router = _priced_router(verifier_model="judge", verifier_kind="chat", min_chars=2)
    response = await router.acompletion(model="cascade", messages=MESSAGES, turn_off_message_logging=True)
    assert response._hidden_params["cascade"]["path"] == "cheap"
    assert seen == [True]


async def test_a_guardrails_own_model_call_inside_the_strong_leg_logs_and_pays_as_its_own(monkeypatch):
    """The strong leg's provider call runs as an internal sub-call of the request, but a deployment hook it runs (a
    content-filter guardrail asking its image model) is a call of its own: it logs its own row and is charged to its
    own deployment's budget, as it is when the strong deployment is called directly."""
    from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail

    recorder = _Recorder()
    side = {
        "model_name": "side",
        "litellm_params": {
            "model": "openai/gpt-4.1",
            "api_key": "x",
            "mock_response": "This is a harmless image.",
            "input_cost_per_token": 0.1,
            "output_cost_per_token": 0.2,
            "max_budget": 100,
            "budget_duration": "1h",
        },
        "model_info": {"id": "side"},
    }
    # callbacks before the Router, which registers its budget limiter into the same list
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router = _priced_router(strong_params={"guardrails": ["review-image"]}, extra_deployments=[side])
    guardrail = ContentFilterGuardrail(
        guardrail_name="review-image", image_model="side", llm_router=router, event_hook="pre_call"
    )
    litellm.callbacks.append(guardrail)
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": image}},
            ],
        }
    ]
    response = await router.acompletion(model="cascade", messages=messages)
    await _flush_logging()
    assert response._hidden_params["cascade"]["path"] == "strong"
    rows = sorted((e["status"], e["model"], round(e["cost"], 4)) for e in recorder.events)
    assert rows == [("success", "openai/gpt-4.1", 5.0), ("success", "openai/gpt-4o", 0.55)]
    limiter = router.registry().plugins("budget_limiter")
    assert await limiter.dual_cache.async_get_cache("deployment_spend:side:1h") == pytest.approx(5.0)


def _content_filtered(text: str) -> ModelResponse:
    return ModelResponse(
        choices=[{"index": 0, "finish_reason": "content_filter", "message": {"role": "assistant", "content": text}}],
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


async def test_a_content_filtered_strong_answer_is_served_not_raised_over(monkeypatch):
    """Only the cheap answer is judged for a content filter (before the cascade runs); the strong answer the cascade
    took is served as the strong deployment gave it, so a content-policy fallback is not called and the request's one
    row (both legs billed) stands."""
    recorder = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router = _priced_router(
        strong_answer=_content_filtered("A long enough answer that the filter nonetheless flagged."),
        # keyed by the group the attempt ran on (the cheap group the cascade routed to), as the router looks it up
        content_policy_fallbacks=[{"cheap": ["fb"]}, {"cascade": ["fb"]}],
    )
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    await _flush_logging()
    assert response._hidden_params["cascade"]["path"] == "strong"
    assert response.choices[0].finish_reason == "content_filter"
    assert [(e["status"], e["origin"], e["deployment_model_name"]) for e in recorder.events] == [
        ("success", None, "strong")
    ]
    assert recorder.events[0]["cost"] == pytest.approx(0.55)


async def test_a_judge_side_failure_logs_once_and_lets_a_fallback_log_its_own_answer(monkeypatch):
    """A failure inside the judgment, outside the strong leg's own handling, is logged as the request's failure
    exactly once and releases the parked cheap log, so the fallback's success is the fallback's row."""
    import litellm.router_rust.steps.cascade as cascade_step

    def boom(kwargs, decision):
        raise RuntimeError("judge boom")

    monkeypatch.setattr(cascade_step, "_record_decision", boom)
    recorder = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router = _priced_router(fallbacks=[{"cascade": ["fb"]}])
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    await _flush_logging()
    assert response.choices[0].message.content.startswith("Fallback says")
    assert [(e["status"], e["origin"], e["deployment_model_name"]) for e in recorder.events] == [
        ("failure", None, "strong"),
        ("success", None, "fb"),
    ]
    assert recorder.events[1]["cost"] == pytest.approx(5.0)


async def test_sdk_path_cheap_pass_is_logged_once_with_the_verifier_on_its_own_row(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router = _priced_router(verifier_model="judge", verifier_kind="chat", min_chars=2)
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    await _flush_logging()
    assert response._hidden_params["cascade"]["path"] == "cheap"
    by_origin = {e["origin"]: (e["status"], e["cost"]) for e in recorder.events}
    assert len(recorder.events) == 2 and set(by_origin) == {None, "cascade_verifier"}
    assert by_origin[None] == ("success", pytest.approx(0.05))  # the request's row: the cheap answer only
    assert by_origin["cascade_verifier"] == ("success", pytest.approx(0.05))  # the verifier's own row


async def test_proxy_path_bills_cheap_and_strong_once_with_a_summed_breakdown():
    """The proxy's deferred logging object: cost preparation and the one parked success log both describe the
    request as cheap + strong, with a breakdown that agrees with the billed amount."""
    recorder = _Recorder()
    log = _proxy_logging(recorder)
    router = _priced_router()
    response = await router.acompletion(model="cascade", messages=MESSAGES, litellm_logging_obj=log)
    prepared = log.prepare_response_cost(response, datetime.datetime.now())
    assert prepared is not None and prepared.amount == pytest.approx(0.55)
    assert prepared.payload["cost_breakdown"]["total_cost"] == pytest.approx(0.55)
    assert prepared.payload["cost_breakdown"]["input_cost"] == pytest.approx(0.11)
    assert prepared.usage.total_tokens == 60
    assert log.enqueue_deferred_logging is not None
    log.enqueue_deferred_logging()  # the proxy fires the parked log after its post-call hooks
    await _flush_logging()
    assert [(e["status"], e["origin"], e["content"]) for e in recorder.events] == [
        ("success", None, "The capital of France is Paris.")
    ]
    assert recorder.events[0]["cost"] == pytest.approx(0.55) and recorder.events[0]["tokens"] == (20, 40)


async def test_failed_escalation_keeps_the_cheap_leg_for_the_proxys_failure_log():
    """on_strong_error=raise: the cheap answer was paid for. The proxy's failure hook must find its usage and cost
    on the logging object instead of billing zero."""
    recorder = _Recorder()
    log = _proxy_logging(recorder)
    router = _priced_router(strong_answer=Exception("strong unavailable"))
    with pytest.raises(Exception, match="strong unavailable"):
        await router.acompletion(model="cascade", messages=MESSAGES, litellm_logging_obj=log)
    await _flush_logging()
    # the request's failure was logged (the proxy does not log a provider error itself), billing the cheap leg
    assert [(e["status"], e["origin"], e["cost"]) for e in recorder.events] == [("failure", None, pytest.approx(0.05))]
    assert recorder.events[0]["model_group"] == "cascade" and recorder.events[0]["deployment_model_name"] == "strong"
    # the parked cheap success log is gone, and the cheap leg rides as a failed attempt's usage for the proxy's
    # failure row and for any fallback that answers next
    assert log.deferred_success_result is None and log.enqueue_deferred_logging is None
    assert log.defer_async_logging is True  # the proxy's deferral is the proxy's
    ((usage, cost),) = log.model_call_details["failed_attempts_usage"]
    assert usage.total_tokens == 30 and cost == pytest.approx(0.05)
    assert "combined_usage_object" not in log.model_call_details


async def test_failed_escalation_logs_one_failure_billing_the_cheap_leg_on_the_sdk_path(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router = _priced_router(strong_answer=Exception("strong unavailable"))
    with pytest.raises(Exception, match="strong unavailable"):
        await router.acompletion(model="cascade", messages=MESSAGES)
    await _flush_logging()
    assert [(e["status"], e["origin"], e["cost"]) for e in recorder.events] == [("failure", None, pytest.approx(0.05))]


async def test_chat_verifier_is_asked_the_configured_question(monkeypatch):
    import litellm as litellm_mod

    calls = []
    original = litellm_mod.acompletion

    async def spy(**kwargs):
        calls.append(kwargs)
        return await original(**kwargs)

    monkeypatch.setattr(litellm_mod, "acompletion", spy)
    question = "Does the answer link to the official documentation?"
    router = _priced_router(verifier_model="judge", verifier_kind="chat", min_chars=2, verifier_question=question)
    await router.acompletion(model="cascade", messages=MESSAGES)
    (verifier,) = [c for c in calls if c["metadata"].get(INTERNAL_CALL_ORIGIN_METADATA_KEY) == "cascade_verifier"]
    assert question in verifier["messages"][0]["content"]
    assert "YES or NO" in verifier["messages"][0]["content"]


async def test_savings_baseline_is_stamped_as_the_priceable_provider_model():
    """The savings calculator prices `savings_baseline_model` by name; a group name such as `strong` prices to
    nothing. Stamp the resolved provider model and its deployment, as the complexity router does."""
    router = _priced_router()
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    decision = response._hidden_params["cascade"]["routing_decision"]
    baseline = router.registry().plugins("cascade")["cascade"][0].strategy.savings_baseline()
    assert baseline is not None and decision["savings_baseline_model"] == baseline.model == "openai/gpt-4o"
    assert decision["savings_baseline_deployment_id"] == "strong"


async def test_proxy_fallback_after_a_failed_escalation_logs_the_fallbacks_answer_not_the_cheap_one():
    """Review: 'cover subsequent fallback attempts'. The Router falls back to another group after the strong leg
    fails. The request's one success log must describe the fallback's answer and cost, not the parked cheap
    answer; the cheap leg is billed through the failed attempt's usage."""
    recorder = _Recorder()
    log = _proxy_logging(recorder)
    router = _priced_router(strong_answer=Exception("strong unavailable"), fallbacks=[{"cascade": ["fb"]}])
    response = await router.acompletion(model="cascade", messages=MESSAGES, litellm_logging_obj=log)
    assert response.choices[0].message.content.startswith("Fallback says")
    prepared = log.prepare_response_cost(response, datetime.datetime.now())
    assert prepared is not None and prepared.amount == pytest.approx(5.0)
    assert log.deferred_success_result is not None
    assert log.deferred_success_result[0].choices[0].message.content.startswith("Fallback says")
    ((usage, cost),) = log.model_call_details["failed_attempts_usage"]
    assert usage.total_tokens == 30 and cost == pytest.approx(0.05)
    log.enqueue_deferred_logging()
    await _flush_logging()
    assert [(e["status"], e["origin"], e["cost"]) for e in recorder.events] == [
        ("failure", None, pytest.approx(0.05)),
        ("success", None, pytest.approx(5.0)),
    ]
    assert recorder.events[1]["content"].startswith("Fallback says") and recorder.events[1]["model_group"] == "fb"


async def test_sdk_fallback_after_a_failed_escalation_logs_each_attempt(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router = _priced_router(strong_answer=Exception("strong unavailable"), fallbacks=[{"cascade": ["fb"]}])
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    await _flush_logging()
    assert response.choices[0].message.content.startswith("Fallback says")
    assert sorted((e["status"], e["cost"]) for e in recorder.events) == [
        ("failure", pytest.approx(0.05)),
        ("success", pytest.approx(5.0)),
    ]


async def test_on_strong_error_cheap_serves_the_cheap_answer_as_the_cheap_deployments():
    """The strong leg failed and the cheap answer is served: the request must be attributed to the cheap
    deployment again (model, deployment stamps, cost), not to the strong deployment that answered nothing."""
    recorder = _Recorder()
    log = _proxy_logging(recorder)
    router = _priced_router(strong_answer=Exception("strong unavailable"), on_strong_error="cheap")
    response = await router.acompletion(model="cascade", messages=MESSAGES, litellm_logging_obj=log)
    assert response.choices[0].message.content == "tiny"
    assert response._hidden_params["cascade"]["path"] == "cheap"
    assert response._hidden_params["cascade"]["reason"] == "strong_error"
    assert log.model_call_details["model"] == "gpt-4o-mini"  # the logging object's view is the cheap call's again
    prepared = log.prepare_response_cost(response, datetime.datetime.now())
    assert prepared is not None and prepared.amount == pytest.approx(0.05) and prepared.usage.total_tokens == 30
    log.enqueue_deferred_logging()
    await _flush_logging()
    (event,) = recorder.events
    assert event["status"] == "success" and event["cost"] == pytest.approx(0.05)
    assert event["model"] == "openai/gpt-4o-mini" and event["model_id"] == "cheap"
    assert event["deployment_model_name"] == "cheap" and event["model_group"] == "cascade"
    assert event["decision"]["cause"] == "cascade_strong_error" and event["decision"]["routed_model"] == "cheap"


# Usage is counted for deployments that declare a limit; the cascade's accounting is read back through the
# router's own tpm/rpm view, the one its rate-limit routing reads.
_TRACKED = {"tpm": 100_000, "rpm": 1_000}


async def _usage(router: Router, *groups: str) -> dict[str, tuple[int | None, int | None]]:
    """(tpm, rpm) counted this minute per group."""
    return {group: await router.health().usage(group) for group in groups}


async def test_escalation_counts_each_legs_usage_against_its_own_deployment():
    """Per-deployment usage (tpm/rpm routing): the strong leg's own call counts the strong deployment, the cheap
    attempt counts the cheap deployment, and the summed response is not added to either."""
    router = _priced_router(cheap_params=dict(_TRACKED), strong_params=dict(_TRACKED))
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    assert response._hidden_params["cascade"]["path"] == "strong"
    assert response.usage.total_tokens == 60
    assert await _usage(router, "cheap", "strong") == {"cheap": (30, 1), "strong": (30, 1)}


async def test_cheap_pass_counts_the_cheap_deployment_once():
    router = _priced_router(min_chars=2, cheap_params=dict(_TRACKED), strong_params=dict(_TRACKED))
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    assert response._hidden_params["cascade"]["path"] == "cheap"
    assert await _usage(router, "cheap", "strong") == {"cheap": (30, 1), "strong": (None, None)}


_RESERVATION = {
    "_litellm_itpm_reserved": 50,
    "_litellm_otpm_reserved": 100,
    "_litellm_itpm_cache_key": "cheap:itpm:00-00",
    "_litellm_otpm_cache_key": "cheap:otpm:00-00",
}


def _reconcile_spy(monkeypatch, settled: list):
    """Wrap the real settlement: record (answer, whether a reservation was found) and let it run."""
    from litellm.litellm_core_utils.internal_call_metadata import IO_TOKEN_RESERVATION_KEYS
    from litellm.router_utils.pre_call_checks import io_token_rate_limit_check as reconcile_module

    real = reconcile_module.async_io_token_reconcile_success

    async def spy(dual_cache, kwargs, response_obj, *, parent_otel_span=None):
        reserved = any(key in (kwargs.get("metadata") or {}) for key in IO_TOKEN_RESERVATION_KEYS)
        settled.append((response_obj.choices[0].message.content, reserved))
        await real(dual_cache, kwargs, response_obj, parent_otel_span=parent_otel_span)

    monkeypatch.setattr(reconcile_module, "async_io_token_reconcile_success", spy)


async def test_io_token_reservations_are_settled_per_leg(monkeypatch):
    """Each leg's itpm/otpm reservation is settled against that leg's usage (the cheap leg's before the strong leg's
    routing could refund it as a stale attempt's), and nothing is left for the request's summed success log."""
    import litellm as litellm_mod
    from litellm.litellm_core_utils.internal_call_metadata import IO_TOKEN_RESERVATION_KEYS

    settled = []
    _reconcile_spy(monkeypatch, settled)
    buckets = []
    original = litellm_mod.acompletion

    async def capture(**kwargs):
        buckets.append(kwargs["metadata"])
        if kwargs["metadata"].get("deployment_model_name") == "cheap":
            kwargs["metadata"].update(_RESERVATION)  # as the rate-limit pre-call check stashes it after the pick
        return await original(**kwargs)

    monkeypatch.setattr(litellm_mod, "acompletion", capture)
    router = _priced_router()
    await router.acompletion(model="cascade", messages=MESSAGES)
    assert settled == [("tiny", True), ("The capital of France is Paris.", False)]
    assert not any(key in buckets[0] for key in IO_TOKEN_RESERVATION_KEYS)


async def test_the_verifier_call_carries_no_copy_of_the_cheap_legs_reservation(monkeypatch):
    """The cheap deployment's itpm/otpm reservation is the request's, settled against the cheap usage by the legs
    themselves. The verifier's call must not carry a copy: the Router refunds any reservation it finds on a call's
    metadata at the deployment pick as a stale attempt's, and the settlement of the original would then count that
    refund a second time, leaving the cheap deployment's counters below its real usage."""
    import litellm as litellm_mod

    original = litellm_mod.acompletion

    async def capture(**kwargs):
        if kwargs["metadata"].get("deployment_model_name") == "cheap":
            kwargs["metadata"].update(_RESERVATION)  # stashed after the pick, as the pre-call check does
        return await original(**kwargs)

    monkeypatch.setattr(litellm_mod, "acompletion", capture)
    router = _priced_router(
        verifier_model="judge-no",
        verifier_kind="chat",
        min_chars=2,
        extra_deployments=[
            {
                "model_name": "judge-no",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "x", "mock_response": "NO"},
                "model_info": {"id": "judge-no"},
            }
        ],
    )
    # the counters as the pre-call check left them: the cheap leg's reservation (50 in, 100 out) is held
    counters = router.health().store
    counters.set_cache(_RESERVATION["_litellm_itpm_cache_key"], 50)
    counters.set_cache(_RESERVATION["_litellm_otpm_cache_key"], 100)
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    assert response._hidden_params["cascade"]["reason"] == "verifier_no"
    # settled once, to the cheap answer's real usage (10 in, 20 out); a replayed refund would leave -40 / -80
    assert counters.get_cache(_RESERVATION["_litellm_itpm_cache_key"]) == 10
    assert counters.get_cache(_RESERVATION["_litellm_otpm_cache_key"]) == 20


async def test_a_cheap_served_request_does_not_resurrect_the_settled_reservation(monkeypatch):
    """on_strong_error=cheap: putting the request back to the cheap call's state must not bring back the cheap
    deployment's already-settled reservation, or the success log would settle it a second time."""
    import litellm as litellm_mod
    from litellm.litellm_core_utils.internal_call_metadata import IO_TOKEN_RESERVATION_KEYS

    settled = []
    _reconcile_spy(monkeypatch, settled)
    buckets = []
    original = litellm_mod.acompletion

    async def capture(**kwargs):
        buckets.append(kwargs["metadata"])
        if kwargs["metadata"].get("deployment_model_name") == "cheap":
            kwargs["metadata"].update(_RESERVATION)  # stashed after the pick, as the pre-call check does
        return await original(**kwargs)

    monkeypatch.setattr(litellm_mod, "acompletion", capture)
    router = _priced_router(strong_answer=Exception("strong unavailable"), on_strong_error="cheap")
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    assert response._hidden_params["cascade"]["reason"] == "strong_error"
    assert settled == [("tiny", True)]
    assert not any(key in buckets[0] for key in IO_TOKEN_RESERVATION_KEYS)
    assert buckets[0]["deployment_model_name"] == "cheap"


async def test_a_cheap_served_request_keeps_what_was_written_meanwhile(monkeypatch):
    """Restoring the cheap call's state puts back what the strong leg overwrote; a record written into the request's
    metadata during the strong leg (a guardrail's, say) is kept."""
    import litellm as litellm_mod

    buckets = []
    original = litellm_mod.acompletion

    async def capture(**kwargs):
        buckets.append(kwargs["metadata"])
        if kwargs["metadata"].get("deployment_model_name") == "strong":
            kwargs["metadata"]["hook_record"] = ["written during the strong leg"]
        return await original(**kwargs)

    monkeypatch.setattr(litellm_mod, "acompletion", capture)
    router = _priced_router(strong_answer=Exception("strong unavailable"), on_strong_error="cheap")
    await router.acompletion(model="cascade", messages=MESSAGES)
    assert buckets[0]["hook_record"] == ["written during the strong leg"]
    assert buckets[0]["deployment_model_name"] == "cheap" and buckets[0]["model_info"]["id"] == "cheap"


async def test_an_explicit_none_logging_object_still_gets_one_on_the_sdk_path(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router = _priced_router()
    response = await router.acompletion(model="cascade", messages=MESSAGES, litellm_logging_obj=None)
    await _flush_logging()
    assert response._hidden_params["cascade"]["path"] == "strong"
    assert [(e["status"], e["cost"]) for e in recorder.events] == [("success", pytest.approx(0.55))]


# ---- third review: cached cheap leg, deployment budgets, usage on failure, the cheap deployment's slot ----


async def test_a_cached_cheap_answer_does_not_make_the_strong_call_free(monkeypatch):
    """The cheap leg came from the response cache (the shared logging object says cache_hit). The strong leg is a
    fresh call and must be priced as one; the request bills the strong leg only, since the cached leg cost nothing."""
    from litellm.caching.caching import Cache
    from litellm.caching.caching_handler import _PENDING_CACHE_WRITES

    recorder = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "cache", Cache(type="local"))
    router = _priced_router(cache_responses=True, caching_groups=[("cheap", "cascade")])
    await router.acompletion(model="cheap", messages=MESSAGES)  # warm the cache the cascade's cheap call will hit
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    await _flush_logging()
    recorder.events.clear()
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    await _flush_logging()
    record = response._hidden_params["cascade"]
    assert record["path"] == "strong"
    assert [(c["role"], c["cost"]) for c in record["calls"]] == [("cheap", 0.0), ("strong", pytest.approx(0.5))]
    assert response._hidden_params["response_cost"] == pytest.approx(0.5)
    assert [(e["status"], e["cost"]) for e in recorder.events] == [("success", pytest.approx(0.5))]


async def test_a_cached_cheap_answer_served_after_a_failed_strong_call_stays_free(monkeypatch):
    from litellm.caching.caching import Cache
    from litellm.caching.caching_handler import _PENDING_CACHE_WRITES

    recorder = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "cache", Cache(type="local"))
    router = _priced_router(
        strong_answer=Exception("strong unavailable"),
        on_strong_error="cheap",
        cache_responses=True,
        caching_groups=[("cheap", "cascade")],
    )
    await router.acompletion(model="cheap", messages=MESSAGES)
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    await _flush_logging()
    recorder.events.clear()
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    await _flush_logging()
    assert response._hidden_params["cascade"]["reason"] == "strong_error"
    assert response._hidden_params["response_cost"] == pytest.approx(0.0)
    assert [(e["status"], e["cost"]) for e in recorder.events] == [("success", pytest.approx(0.0))]


async def test_each_leg_is_charged_to_its_own_deployments_budget():
    """Deployment budgets: the cheap leg's spend lands on the cheap deployment (and would stop it), the strong leg's
    on the strong deployment; the request's one row does not charge the strong deployment for both."""
    router = _priced_router(
        cheap_params={"max_budget": 0.01, "budget_duration": "1h"},
        strong_params={"max_budget": 10, "budget_duration": "1h"},
    )
    await router.acompletion(model="cascade", messages=MESSAGES)
    await _flush_logging()
    limiter = router.registry().plugins("budget_limiter")
    assert limiter is not None
    assert await limiter.dual_cache.async_get_cache("deployment_spend:cheap:1h") == pytest.approx(0.05)
    assert await limiter.dual_cache.async_get_cache("deployment_spend:strong:1h") == pytest.approx(0.5)
    with pytest.raises(
        ValueError, match="crossed budget"
    ):  # the cheap deployment is over its budget: no cheap leg to route
        await router.acompletion(model="cascade", messages=MESSAGES)


async def test_a_failed_escalation_still_charges_and_counts_the_cheap_leg():
    """The cheap leg completed; its usage and spend belong to the cheap deployment whether or not the strong leg
    then fails."""
    router = _priced_router(
        strong_answer=Exception("strong unavailable"),
        cheap_params={"max_budget": 1, "budget_duration": "1h", **_TRACKED},
        strong_params=dict(_TRACKED),
    )
    with pytest.raises(Exception, match="strong unavailable"):
        await router.acompletion(model="cascade", messages=MESSAGES)
    await _flush_logging()
    usage = await _usage(router, "cheap", "strong")
    assert usage["cheap"] == (30, 1)
    assert usage["strong"][0] is None  # the failed strong call answered nothing to count
    assert await router.registry().plugins("budget_limiter").dual_cache.async_get_cache(
        "deployment_spend:cheap:1h"
    ) == pytest.approx(0.05)


async def test_a_cheap_answer_served_after_a_failed_strong_call_is_counted_and_charged_once():
    router = _priced_router(
        strong_answer=Exception("strong unavailable"),
        on_strong_error="cheap",
        cheap_params={"max_budget": 1, "budget_duration": "1h", **_TRACKED},
        strong_params=dict(_TRACKED),
    )
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    await _flush_logging()
    assert response._hidden_params["cascade"]["reason"] == "strong_error"
    usage = await _usage(router, "cheap", "strong")
    assert usage["cheap"] == (30, 1)
    assert usage["strong"][0] is None  # the failed strong call answered nothing to count
    assert await router.registry().plugins("budget_limiter").dual_cache.async_get_cache(
        "deployment_spend:cheap:1h"
    ) == pytest.approx(0.05)


async def test_the_cheap_deployments_slot_is_released_before_the_verifier_runs():
    """With the verifier on the cheap group and one concurrency slot, holding the completed cheap call's slot
    through verification would hand the verifier a 429 and escalate for nothing."""
    router = _priced_router(
        min_chars=0,
        verifier_model="cheap",
        verifier_kind="chat",
        cheap_params={"mock_response": "YES", "max_parallel_requests": 1},
    )
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    record = response._hidden_params["cascade"]
    assert record["path"] == "cheap" and record["verifier_error"] is None
    assert [c["role"] for c in record["calls"]] == ["cheap", "verifier"]


async def test_a_fallback_after_a_failed_escalation_pays_its_own_budget_in_full():
    """The cheap leg was charged to the cheap deployment directly; the attributed-spend marker must not ride into
    the fallback's row, or the fallback deployment would be charged its cost minus the cheap leg's."""
    router = _priced_router(
        strong_answer=Exception("strong unavailable"),
        fallbacks=[{"cascade": ["fb"]}],
        cheap_params={"max_budget": 1, "budget_duration": "1h"},
        fb_params={"max_budget": 100, "budget_duration": "1h"},
    )
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    await _flush_logging()
    assert response.choices[0].message.content.startswith("Fallback says")
    limiter = router.registry().plugins("budget_limiter")
    assert await limiter.dual_cache.async_get_cache("deployment_spend:cheap:1h") == pytest.approx(0.05)
    assert await limiter.dual_cache.async_get_cache("deployment_spend:fb:1h") == pytest.approx(5.0)


@pytest.mark.parametrize("leg", ["cheap_model", "strong_model", "verifier_model"])
@pytest.mark.parametrize("inner_first", [True, False])
@pytest.mark.parametrize("via", ["name", "alias", "team_public_name"])
def test_cascades_do_not_nest(leg, inner_first, via):
    """A cascade's legs may not be another cascade alias, named directly, through model_group_alias or as a team's
    public model name, whichever marker is registered first."""

    def deployment(name, model="openai/gpt-4o-mini"):
        return {"model_name": name, "litellm_params": {"model": model, "api_key": "x", "mock_response": "ok"}}

    inner = {
        "model_name": "inner",
        "litellm_params": {
            "model": "auto_router/cascade_router",
            "cascade_router_config": {"cheap_model": "a", "strong_model": "b"},
        },
        "model_info": {"team_public_model_name": "inner-public"} if via == "team_public_name" else {},
    }
    reached_as = {"name": "inner", "alias": "inner-alias", "team_public_name": "inner-public"}[via]
    outer_config = {"cheap_model": "c", "strong_model": "d", leg: reached_as}
    if leg == "verifier_model":
        outer_config["verifier_kind"] = "chat"
    outer = {
        "model_name": "outer",
        "litellm_params": {"model": "auto_router/cascade_router", "cascade_router_config": outer_config},
    }
    markers = [inner, outer] if inner_first else [outer, inner]
    with pytest.raises(ValueError, match="cascades do not nest"):
        Router(
            model_list=[deployment("a"), deployment("b"), deployment("c"), deployment("d"), *markers],
            model_group_alias={"inner-alias": "inner"} if via == "alias" else None,
        )


@pytest.mark.parametrize("on_strong_error", ["raise", "cheap"])
async def test_a_nested_cascade_that_registration_could_not_see_is_refused_at_request_time(
    monkeypatch, on_strong_error
):
    """An alias added at runtime can make a leg reach another cascade after registration; the inner cascade then
    refuses to run inside the outer's leg instead of sharing its logging object."""
    from litellm.router_rust.steps.prerouting import PreRoutingStrategies

    monkeypatch.setattr(PreRoutingStrategies, "_reject_nested_cascade", lambda self, deployment, cascade_router: None)

    def deployment(name, answer="ok, a long enough answer to pass the heuristics"):
        return {
            "model_name": name,
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "x", "mock_response": answer},
        }

    router = Router(
        model_list=[
            deployment("a", "short"),
            deployment("b"),
            deployment("c", "short"),
            {
                "model_name": "inner",
                "litellm_params": {
                    "model": "auto_router/cascade_router",
                    "cascade_router_config": {"cheap_model": "a", "strong_model": "b"},
                },
            },
            {
                "model_name": "outer",
                "litellm_params": {
                    "model": "auto_router/cascade_router",
                    "cascade_router_config": {
                        "cheap_model": "c",
                        "strong_model": "inner",
                        "on_strong_error": on_strong_error,
                    },
                },
            },
        ],
        num_retries=0,
    )
    if on_strong_error == "raise":
        with pytest.raises(ValueError, match="cascades do not nest"):
            await router.acompletion(model="outer", messages=MESSAGES)
    else:
        response = await router.acompletion(model="outer", messages=MESSAGES)
        record = response._hidden_params["cascade"]
        assert record["path"] == "cheap" and record["strong_error"] == "ValueError"  # the refusal's class


async def test_a_cascade_used_as_a_rag_generation_model_is_served_under_a_bare_internal_sub_call():
    """The RAG pipeline routes its generation call through the Router under `internal_sub_call()` (the parent folds
    the cost). That is not a cascade inside a cascade and must be served."""
    from litellm._internal_context import internal_sub_call

    router = _priced_router()
    with internal_sub_call():
        response = await router.acompletion(model="cascade", messages=MESSAGES)
    assert response.choices[0].message.content == "The capital of France is Paris."
    assert response._hidden_params["cascade"]["path"] == "strong"


def _side_deployment(answer: str = "The side model gives a long answer.") -> dict:
    """A $5 deployment ($0.1 / $0.2 per token on the 10 + 20 mock tokens) with its own budget."""
    return {
        "model_name": "side",
        "litellm_params": {
            "model": "openai/gpt-4.1",
            "api_key": "x",
            "mock_response": answer,
            "input_cost_per_token": 0.1,
            "output_cost_per_token": 0.2,
            "max_budget": 100,
            "budget_duration": "1h",
        },
        "model_info": {"id": "side"},
    }


async def test_a_silent_experiment_on_the_strong_deployment_still_logs_and_bills_its_own_call(monkeypatch):
    """Only the strong leg's own provider call is an internal sub-call. A silent experiment the strong deployment
    starts is an independent call: it logs its own row and charges its own deployment's budget."""
    recorder = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    router = _priced_router(strong_params={"silent_model": "side"}, extra_deployments=[_side_deployment()])
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    await asyncio.sleep(0.2)
    await _flush_logging()
    assert response._hidden_params["cascade"]["path"] == "strong"
    costs = sorted((e["model"], e["cost"]) for e in recorder.events if e["status"] == "success")
    assert costs == [("openai/gpt-4.1", pytest.approx(5.0)), ("openai/gpt-4o", pytest.approx(0.55))]
    assert await router.registry().plugins("budget_limiter").dual_cache.async_get_cache(
        "deployment_spend:side:1h"
    ) == pytest.approx(5.0)


async def test_a_classifier_inside_the_strong_group_still_logs_and_bills_its_own_call(monkeypatch):
    """The strong group is a complexity router with an LLM classifier: the classifier call, made by the leg's
    routing before its provider call, keeps its own row (classifier origin), cost and budget charge."""
    recorder = _Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    smart = {
        "model_name": "smart",
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_default_model": "strong",
            "complexity_router_config": {
                "classifier_type": "llm",
                "classifier_llm_config": {"model": "side"},
                "tiers": {tier: "strong" for tier in ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")},
            },
        },
    }
    router = _priced_router(strong_model="smart", extra_deployments=[_side_deployment('{"tier":"SIMPLE"}'), smart])
    response = await router.acompletion(model="cascade", messages=MESSAGES)
    await _flush_logging()
    assert response._hidden_params["cascade"]["served_by"] == "smart"
    classifier_events = [e for e in recorder.events if e["origin"] == "autorouter_classifier"]
    assert [e["cost"] for e in classifier_events] == [pytest.approx(5.0)]
    assert [e["cost"] for e in recorder.events if e["origin"] is None] == [pytest.approx(0.55)]
    assert await router.registry().plugins("budget_limiter").dual_cache.async_get_cache(
        "deployment_spend:side:1h"
    ) == pytest.approx(5.0)


async def test_reusing_the_callers_metadata_dict_does_not_carry_attributed_spend_across_requests():
    """An SDK caller may hand the same metadata dict to every request. The attributed-spend marker is bound to the
    request's call id and reset per request, so later requests (cascade or plain) are charged in full."""
    router = _priced_router(
        cheap_params={"max_budget": 100, "budget_duration": "1h"},
        strong_params={"max_budget": 100, "budget_duration": "1h"},
    )
    shared: dict = {}
    for _ in range(3):
        await router.acompletion(model="cascade", messages=MESSAGES, metadata=shared)
        await _flush_logging()
    limiter = router.registry().plugins("budget_limiter")
    assert await limiter.dual_cache.async_get_cache("deployment_spend:cheap:1h") == pytest.approx(0.15)
    assert await limiter.dual_cache.async_get_cache("deployment_spend:strong:1h") == pytest.approx(1.5)
    await router.acompletion(model="strong", messages=MESSAGES, metadata=shared)  # a plain request, same dict
    await _flush_logging()
    assert await limiter.dual_cache.async_get_cache("deployment_spend:strong:1h") == pytest.approx(2.0)


async def test_a_plain_request_reusing_the_dict_and_a_fixed_call_id_is_charged_in_full():
    """Even a caller that reuses its metadata dict AND a fixed litellm_call_id (a contract violation: it is the
    spend-log request id) gets the previous cascade request's marker cleared before its plain request is charged."""
    router = _priced_router(strong_params={"max_budget": 100, "budget_duration": "1h"})
    shared: dict = {}
    await router.acompletion(model="cascade", messages=MESSAGES, metadata=shared, litellm_call_id="fixed-id")
    await _flush_logging()
    await router.acompletion(model="strong", messages=MESSAGES, metadata=shared, litellm_call_id="fixed-id")
    await _flush_logging()
    limiter = router.registry().plugins("budget_limiter")
    assert await limiter.dual_cache.async_get_cache("deployment_spend:strong:1h") == pytest.approx(1.0)
