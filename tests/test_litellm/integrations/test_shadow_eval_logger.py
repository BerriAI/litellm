"""Unit tests for the shadow-eval logger: sampling, unmasking, the hook's skip chain,
the detached pipeline's single attempt-row write, and the cache-first job lookup."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.shadow_eval_logger import (
    _MAX_CONCURRENT_SHADOW_TASKS,
    _MAX_ERROR_CHARS,
    _MAX_JUDGE_PROMPT_CHARS,
    JUDGE_MAX_OUTPUT_TOKENS,
    PAIRWISE_JUDGE_RESPONSE_FORMAT,
    ActiveShadowEvalJob,
    ShadowEvalLogger,
    _failure_detail,
    _judge_user_prompt,
    _sample_hits,
    _unmask_preference,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import SHADOW_EVAL_JUDGE_CALL_ORIGIN, SHADOW_EVAL_ROUTER_CALL_ORIGIN, ModelResponse


def _job(**overrides) -> ActiveShadowEvalJob:
    defaults = dict(
        id="job-1",
        router_name="my-router",
        shadow_percentage=100.0,
        judge_model="judge-model",
        max_turns=200,
        ends_at=datetime.now(timezone.utc) + timedelta(days=1),
        attempts=0,
    )
    return ActiveShadowEvalJob(**{**defaults, **overrides})


def _prisma(jobs=(), attempt_counts=(), attempt_costs=()) -> MagicMock:
    costs = {job_id: {"judge_cost": judge, "shadow_cost": shadow} for job_id, judge, shadow in attempt_costs}
    prisma = MagicMock()
    prisma.db.litellm_shadowevaljob.find_many = AsyncMock(return_value=list(jobs))
    prisma.db.litellm_shadowevalattempt.group_by = AsyncMock(
        return_value=[
            {
                "job_id": job_id,
                "_count": {"_all": count},
                "_sum": costs.get(job_id, {"judge_cost": 0.0, "shadow_cost": 0.0}),
            }
            for job_id, count in attempt_counts
        ]
    )
    prisma.db.litellm_shadowevalattempt.create = AsyncMock()
    return prisma


def _job_record(job: ActiveShadowEvalJob, target_type="key", target_id="key-hash") -> MagicMock:
    record = MagicMock()
    for field, value in dict(
        id=job.id,
        target_type=target_type,
        target_id=target_id,
        router_name=job.router_name,
        router_names=job.router_names,
        direction=job.direction,
        baseline_model=job.baseline_model,
        shadow_percentage=job.shadow_percentage,
        judge_model=job.judge_model,
        max_turns=job.max_turns,
        max_budget=job.max_budget,
        ends_at=job.ends_at,
    ).items():
        setattr(record, field, value)
    return record


def _router(
    shadow_text="shadow answer",
    judge_json='{"preference": "A", "confidence": 0.9, "reasoning": "x"}',
    classifier_cost=None,
    sibling_router_texts=None,
):
    """One mock router serving the shadow call first, the judge call second, told apart by
    the internal-origin stamp rather than the model, since a reverse job's shadow arm names
    a plain model. Only the auto-router writes a routing decision back, and only a plain
    model reports the model it served on the response, which is how each direction learns
    which model answered."""
    router = MagicMock()
    router.model_group_alias = {}
    router.get_model_list = MagicMock(return_value=[{"litellm_params": {"model": "openai/gpt-4o-mini"}}])

    async def acompletion(**kwargs):
        if kwargs["metadata"].get(INTERNAL_CALL_ORIGIN_METADATA_KEY) != SHADOW_EVAL_ROUTER_CALL_ORIGIN:
            return {"choices": [{"message": {"content": judge_json}}]}
        if kwargs["model"] == "my-router":
            decision = {"tier_label": "SIMPLE", "routed_model": "cheap-model"}
            if classifier_cost is not None:
                decision["classifier_cost"] = classifier_cost
            kwargs["metadata"]["routing_decision"] = decision
            return {"choices": [{"message": {"content": shadow_text}}], "usage": {"completion_tokens": 5}}
        if sibling_router_texts and kwargs["model"] in sibling_router_texts:
            kwargs["metadata"]["routing_decision"] = {
                "tier_label": "MEDIUM",
                "routed_model": f"{kwargs['model']}-pick",
            }
            return {
                "choices": [{"message": {"content": sibling_router_texts[kwargs["model"]]}}],
                "usage": {"completion_tokens": 5},
            }
        return ModelResponse(
            model=kwargs["model"],
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": shadow_text}}],
        )

    router.acompletion = MagicMock(side_effect=acompletion)
    return router


def _spend_counter(store=None):
    """In-memory stand-in for the proxy's cross-pod spend counter: reads take the max of
    the counter and the caller's fallback, exactly like get_current_spend does for a key
    shape the reseed helpers do not know."""
    counter = store if store is not None else {}

    async def read(key, fallback_spend, max_budget):
        return max(counter.get(key, 0.0), fallback_spend)

    async def write(key, cost):
        counter[key] = counter.get(key, 0.0) + cost

    return counter, read, write


def _logger(router=None, prisma=None, jobs=(), counter_store=None, jobs_by_target=None) -> ShadowEvalLogger:
    cache = InMemoryCache(max_size_in_memory=4, default_ttl=60)
    counter, read, write = _spend_counter(counter_store)
    funnel_events = []
    logger = ShadowEvalLogger(
        router_provider=lambda: router,
        prisma_provider=lambda: prisma,
        jobs_cache=cache,
        job_spend_reader=read,
        job_spend_writer=write,
        funnel_recorder=lambda job_id, stage: funnel_events.append((job_id, stage)),
    )
    logger._test_counter = counter
    logger._test_funnel = funnel_events
    seeded = jobs_by_target if jobs_by_target is not None else ({("key", "key-hash"): tuple(jobs)} if jobs else None)
    if seeded is not None:
        cache.set_cache("shadow_eval:active_jobs", seeded)
    return logger


def _routed_by(router_name="my-router", tier="COMPLEX"):
    """Metadata as a pre-routing strategy leaves it on the request it served."""
    return {"routing_decision": {"router_model_name": router_name, "tier_label": tier, "routed_model": "router-pick"}}


def _success_kwargs(
    request_id="req-1",
    api_key_hash="key-hash",
    request_metadata=None,
    call_type="acompletion",
    model="claude-opus",
    response_cost=None,
    cache_hit=None,
):
    return {
        "standard_logging_object": {
            "id": request_id,
            "call_type": call_type,
            "model": model,
            "metadata": {"user_api_key_hash": api_key_hash},
            "model_parameters": {"temperature": 0.5, "stream": True},
            "response_cost": response_cost,
            "cache_hit": cache_hit,
        },
        "litellm_params": {"metadata": request_metadata or {}},
        "messages": [{"role": "user", "content": "what is 2+2"}],
    }


RESPONSE = {"choices": [{"message": {"content": "real answer"}}]}

RESPONSES_API_RESPONSE = {
    "id": "resp_1",
    "created_at": 1,
    "model": "gpt-5",
    "object": "response",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "real answer", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "metadata": None,
    "temperature": None,
    "tool_choice": "auto",
    "tools": [],
    "top_p": None,
    "status": "completed",
}


async def _drain(logger: ShadowEvalLogger, target: int = 0):
    for _ in range(100):
        if logger._inflight_shadow_tasks == target:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("shadow tasks never drained")


@pytest.mark.asyncio
class TestSurfaceNormalization:
    """/v1/messages and /v1/responses arms: the hook normalizes each surface's logged
    request through litellm's own transformations and judges only text-final turns."""

    async def _drive(self, hook_kwargs, response_obj):
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma, jobs=(_job(),))
        await logger.async_log_success_event(hook_kwargs, response_obj, None, None)
        await _drain(logger)
        return prisma, router

    async def test_anthropic_messages_arm_normalizes_blocks_and_system(self):
        hook_kwargs = _success_kwargs(call_type="anthropic_messages")
        hook_kwargs["messages"] = [{"role": "user", "content": [{"type": "text", "text": "what is 2+2"}]}]
        hook_kwargs["system"] = "you are terse"

        prisma, router = await self._drive(hook_kwargs, RESPONSE)

        shadow_messages = router.acompletion.call_args_list[0].kwargs["messages"]
        assert shadow_messages[0]["role"] == "system"
        assert shadow_messages[0]["content"] == "you are terse"
        assert shadow_messages[1]["role"] == "user"
        prisma.db.litellm_shadowevalattempt.create.assert_called_once()

    async def test_anthropic_bridge_path_recovers_system_from_proxy_wire_body(self):
        """On the openai-compatible bridge path kwargs carry no system (live-probed:
        kwargs["system"] is None and complete_input_dict is empty); the proxy's snapshot
        of the client's wire body is the only remaining source."""
        hook_kwargs = _success_kwargs(call_type="anthropic_messages")
        hook_kwargs["messages"] = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        hook_kwargs["litellm_params"]["proxy_server_request"] = {
            "body": {"model": "gpt-5", "max_tokens": 100, "system": "from the wire body", "messages": []}
        }

        _, router = await self._drive(hook_kwargs, RESPONSE)

        shadow_messages = router.acompletion.call_args_list[0].kwargs["messages"]
        assert shadow_messages[0] == {"role": "system", "content": "from the wire body"}

    async def test_anthropic_arm_translates_wire_body_params_not_logged_optional_params(self):
        """The wire body is the only surface-native param source on both provider paths
        (the bridge's inner completion rewrites the logged optional_params to chat
        shape); anthropic tools and stop_sequences reach the shadow call translated,
        transport and litellm keys never do."""
        hook_kwargs = _success_kwargs(call_type="anthropic_messages")
        hook_kwargs["messages"] = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        hook_kwargs["standard_logging_object"]["model_parameters"] = {"temperature": 0.9}
        hook_kwargs["litellm_params"]["proxy_server_request"] = {
            "body": {
                "model": "claude-x",
                "messages": [],
                "system": "you are terse",
                "max_tokens": 100,
                "temperature": 0.1,
                "top_k": 5,
                "stop_sequences": ["END"],
                "stream": True,
                "tools": [
                    {"name": "get_weather", "description": "d", "input_schema": {"type": "object", "properties": {}}}
                ],
                "litellm_metadata": {"user_api_key_hash": "key-hash"},
            }
        }

        _, router = await self._drive(hook_kwargs, RESPONSE)

        shadow_call = router.acompletion.call_args_list[0].kwargs
        assert shadow_call["max_tokens"] == 100
        assert shadow_call["temperature"] == 0.1
        assert shadow_call["top_k"] == 5
        assert shadow_call["stop"] == ["END"]
        assert shadow_call["tools"][0]["type"] == "function"
        assert shadow_call["tools"][0]["function"]["name"] == "get_weather"
        assert "stop_sequences" not in shadow_call
        assert "stream" not in shadow_call
        assert shadow_call["metadata"][INTERNAL_CALL_ORIGIN_METADATA_KEY] == SHADOW_EVAL_ROUTER_CALL_ORIGIN

    async def test_responses_arm_translates_wire_body_params_and_drops_surface_only_keys(self):
        from litellm.types.llms.openai import ResponsesAPIResponse

        hook_kwargs = _success_kwargs(call_type="aresponses")
        hook_kwargs["messages"] = "what is 8+8"
        hook_kwargs["litellm_params"]["proxy_server_request"] = {
            "body": {
                "model": "gpt-5",
                "input": "what is 8+8",
                "instructions": "you are terse",
                "max_output_tokens": 128,
                "temperature": 0.3,
                "previous_response_id": "resp_0",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "d",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            }
        }
        response = ResponsesAPIResponse.model_validate(RESPONSES_API_RESPONSE)

        _, router = await self._drive(hook_kwargs, response)

        shadow_call = router.acompletion.call_args_list[0].kwargs
        assert shadow_call["messages"][0] == {"role": "system", "content": "you are terse"}
        assert shadow_call["max_tokens"] == 128
        assert shadow_call["temperature"] == 0.3
        assert shadow_call["tools"][0]["function"]["name"] == "get_weather"
        assert "max_output_tokens" not in shadow_call
        assert "previous_response_id" not in shadow_call
        assert "instructions" not in shadow_call

    @pytest.mark.parametrize("payload_shape", ["typed", "dict"])
    @pytest.mark.parametrize("call_type", ["aresponses", "responses"])
    async def test_responses_arms_normalize_bare_string_input_and_instructions(self, call_type, payload_shape):
        from litellm.types.llms.openai import ResponsesAPIResponse

        hook_kwargs = _success_kwargs(call_type=call_type)
        hook_kwargs["messages"] = "what is 8+8"
        hook_kwargs["instructions"] = "you are terse"
        response = (
            ResponsesAPIResponse.model_validate(RESPONSES_API_RESPONSE)
            if payload_shape == "typed"
            else RESPONSES_API_RESPONSE
        )

        prisma, router = await self._drive(hook_kwargs, response)

        shadow_call = router.acompletion.call_args_list[0].kwargs
        shadow_messages = shadow_call["messages"]
        assert shadow_messages[0]["role"] == "system"
        assert shadow_messages[1]["role"] == "user"
        assert shadow_messages[1]["content"] == "what is 8+8"
        assert "tools" not in shadow_call
        prisma.db.litellm_shadowevalattempt.create.assert_called_once()

    @pytest.mark.parametrize(
        "response_mutation,kwargs_mutation",
        [
            ("chat-tool-calls", {}),
            ("responses-function-call", {"call_type": "aresponses"}),
        ],
        ids=["tool-final-chat-turn", "tool-final-responses-turn"],
    )
    async def test_unjudgeable_turns_are_skipped_without_consuming_budget(self, response_mutation, kwargs_mutation):
        from litellm.types.llms.openai import ResponsesAPIResponse

        hook_kwargs = _success_kwargs(**({"call_type": "acompletion"} | kwargs_mutation))
        response = RESPONSE
        if response_mutation == "chat-tool-calls":
            response = {
                "choices": [
                    {
                        "message": {
                            "content": "let me check",
                            "tool_calls": [
                                {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                            ],
                        }
                    }
                ]
            }
        elif response_mutation == "responses-function-call":
            hook_kwargs["messages"] = "do the thing"
            response = ResponsesAPIResponse.model_validate(
                RESPONSES_API_RESPONSE
                | {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "f",
                            "arguments": "{}",
                            "call_id": "c1",
                            "id": "fc1",
                            "status": "completed",
                        }
                    ]
                }
            )

        prisma, router = await self._drive(hook_kwargs, response)

        router.acompletion.assert_not_called()
        prisma.db.litellm_shadowevalattempt.create.assert_not_called()

    @pytest.mark.parametrize(
        "call_type,guardrail_mode,sampled",
        [
            ("anthropic_messages", ["logging_only", "pre_call"], False),
            ("aresponses", GuardrailEventHooks.pre_call, False),
            ("anthropic_messages", "post_call", True),
            ("acompletion", "pre_call", True),
        ],
        ids=["anthropic-pre-call-list", "responses-pre-call-enum", "anthropic-post-call-only", "chat-pre-call"],
    )
    async def test_guardrail_rewritten_requests_never_replay_the_wire_body(self, call_type, guardrail_mode, sampled):
        """The proxy snapshots the wire body before the guardrail pre-call hook, so the
        wire-sourced surfaces skip requests a request-mutating guardrail ran on rather
        than replay stripped tools or unmasked content; chat sources the dispatched
        call and keeps sampling, as do requests only response-mode guardrails touched."""
        hook_kwargs = _success_kwargs(
            call_type=call_type,
            request_metadata={
                "standard_logging_guardrail_information": [{"guardrail_name": "g", "guardrail_mode": guardrail_mode}]
            },
        )
        response = RESPONSE
        if call_type == "anthropic_messages":
            hook_kwargs["messages"] = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        elif call_type == "aresponses":
            hook_kwargs["messages"] = "hi"
            response = RESPONSES_API_RESPONSE

        prisma, router = await self._drive(hook_kwargs, response)

        if sampled:
            prisma.db.litellm_shadowevalattempt.create.assert_called_once()
        else:
            router.acompletion.assert_not_called()
            prisma.db.litellm_shadowevalattempt.create.assert_not_called()

    @pytest.mark.parametrize(
        "call_type,messages,response_obj",
        [
            ("anthropic_messages", "not-a-message-list", RESPONSE),
            ("acompletion", [{"role": "user", "content": "hi"}], {"unexpected": "shape"}),
            ("aresponses", "hi", RESPONSE),
        ],
        ids=["rejected-request-shape", "malformed-chat-response", "responses-response-without-output"],
    )
    async def test_unsampleable_shapes_fail_closed(self, call_type, messages, response_obj):
        """A request or response shape the normalizers reject is skipped without a
        provider call or an attempt row, never raised."""
        hook_kwargs = _success_kwargs(call_type=call_type)
        hook_kwargs["messages"] = messages

        prisma, router = await self._drive(hook_kwargs, response_obj)

        router.acompletion.assert_not_called()
        prisma.db.litellm_shadowevalattempt.create.assert_not_called()


class TestSampling:
    def test_boundaries_and_determinism(self):
        assert not any(_sample_hits(f"req-{i}", "job", 0.0) for i in range(100))
        assert all(_sample_hits(f"req-{i}", "job", 100.0) for i in range(100))
        assert len({_sample_hits("req-1", "job-1", 50.0) for _ in range(10)}) == 1

    def test_distribution_close_to_percentage(self):
        hits = sum(_sample_hits(f"req-{i}", "job-x", 10.0) for i in range(10_000))
        assert 800 < hits < 1200

    def test_different_jobs_sample_independently(self):
        agreements = sum(
            _sample_hits(f"req-{i}", "job-a", 50.0) == _sample_hits(f"req-{i}", "job-b", 50.0) for i in range(1000)
        )
        assert 300 < agreements < 700


@pytest.mark.parametrize(
    "raw,real_is_a,expected",
    [
        ("A", True, "real"),
        ("a", True, "real"),
        ("A", False, "shadow"),
        ("B", True, "shadow"),
        ("B", False, "real"),
        ("tie", True, "tie"),
        ("garbage", True, "tie"),
        ("", False, "tie"),
    ],
)
def test_unmask_preference(raw, real_is_a, expected):
    assert _unmask_preference(raw, real_is_a) == expected


def test_failure_detail_names_the_raising_frame():
    try:
        raise TypeError("'tuple' object does not support item assignment")
    except TypeError as e:
        detail = _failure_detail(e)
        lineno = e.__traceback__.tb_lineno
    assert (
        detail == f"TypeError at test_shadow_eval_logger.py:{lineno}: 'tuple' object does not support item assignment"
    )

    try:
        raise ValueError("p" * 5 * _MAX_ERROR_CHARS)
    except ValueError as long_e:
        truncated_row_error = _failure_detail(long_e)[:_MAX_ERROR_CHARS]
    assert "ValueError at test_shadow_eval_logger.py:" in truncated_row_error


def test_call_cost_prefers_the_billed_figure_over_the_public_price_map(monkeypatch):
    """The router client stamps _hidden_params.response_cost from the deployment's own
    pricing; the public map reads 0 for deployment-priced models, so budgets gated on it
    would never close. The map is only the fallback for responses with no stamp."""
    import litellm as litellm_module
    from litellm.integrations.shadow_eval_logger import _call_cost

    monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
    stamped = MagicMock()
    stamped._hidden_params = {"response_cost": 0.04}
    assert _call_cost(stamped) == 0.04

    from litellm.types.utils import HiddenParams

    object_stamped = MagicMock()
    object_stamped._hidden_params = HiddenParams(response_cost=0.03)
    assert _call_cost(object_stamped) == 0.03

    unstamped = MagicMock()
    unstamped._hidden_params = {"response_cost": None}
    assert _call_cost(unstamped) == 0.005
    assert _call_cost({"choices": []}) == 0.005


@pytest.mark.asyncio
async def test_a_cold_or_reset_counter_degrades_to_the_fill_floor_not_zero(monkeypatch: pytest.MonkeyPatch):
    """The design leans on one owner contract: for a spend:shadow_eval:* key (no DB
    reseed by design), get_current_spend returns the caller's fill-sum fallback whenever
    the counter reads lower. A reset counter therefore degrades to the <=10s-stale DB
    sum, never to zero, so a Redis expiry cannot re-open a spent budget by a full cap."""
    from litellm.proxy import proxy_server

    counter_key = "spend:shadow_eval:job-cold-test"
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    proxy_server.spend_counter_cache.in_memory_cache.set_cache(key=counter_key, value=0.05)
    try:
        assert (
            await proxy_server.get_current_spend(counter_key=counter_key, fallback_spend=0.42, max_budget=1.0) == 0.42
        )
        proxy_server.spend_counter_cache.in_memory_cache.delete_cache(key=counter_key)
        assert (
            await proxy_server.get_current_spend(counter_key=counter_key, fallback_spend=0.42, max_budget=1.0) == 0.42
        )
    finally:
        proxy_server.spend_counter_cache.in_memory_cache.delete_cache(key=counter_key)


@pytest.mark.asyncio
async def test_an_unverifiable_budget_skips_the_sample_instead_of_spending():
    """A raising spend read (fail-closed enforcement, or an owner bug) must skip the
    sample before any provider call, never admit it on a guess."""

    async def unverifiable(key, fallback_spend, max_budget):
        raise RuntimeError("budget unverifiable")

    prisma = _prisma()
    router = _router()
    logger = _logger(router=router, prisma=prisma, jobs=(_job(max_budget=1.0),))
    logger._read_job_spend = unverifiable

    await logger.async_log_success_event(_success_kwargs(), RESPONSE, None, None)
    await _drain(logger)

    router.acompletion.assert_not_called()
    prisma.db.litellm_shadowevalattempt.create.assert_not_called()
    assert logger._test_funnel == [("job-1", "withheld")]


def test_judge_prompt_is_bounded_however_large_the_inputs():
    prompt = _judge_user_prompt("c" * 200_000, "a" * 200_000, "b" * 200_000)
    assert len(prompt) < _MAX_JUDGE_PROMPT_CHARS + 100
    assert prompt.endswith("Which response is better?")
    small = _judge_user_prompt("conv", "alpha", "beta")
    assert "conv" in small and "alpha" in small and "beta" in small


@pytest.mark.asyncio
class TestSuccessHookSkipChain:
    async def test_happy_path_writes_exactly_one_attempt_row(self, monkeypatch: pytest.MonkeyPatch):
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma, jobs=(_job(),))

        await logger.async_log_success_event(_success_kwargs(), RESPONSE, None, None)
        await _drain(logger)

        shadow_call = router.acompletion.call_args_list[0].kwargs
        assert shadow_call["temperature"] == 0.5
        assert "stream" not in shadow_call
        create = prisma.db.litellm_shadowevalattempt.create
        create.assert_awaited_once()
        row = create.call_args.kwargs["data"]
        assert row["job_id"] == "job-1"
        assert row["request_id"] == "req-1"
        assert row["outcome"] in ("real", "shadow")
        assert row["tier"] == "SIMPLE"
        assert row["real_model"] == "claude-opus"
        assert row["shadow_model"] == "cheap-model"
        assert row["confidence"] == 0.9
        assert row["judge_cost"] == 0.005
        assert row["shadow_cost"] == 0.005
        assert row["error"] is None
        assert prisma.db.litellm_shadowevaljob.find_many.await_count == 0

    async def test_judge_call_carries_the_verdict_schema(self, monkeypatch: pytest.MonkeyPatch):
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
        router = _router()
        logger = _logger(router=router, prisma=_prisma(), jobs=(_job(),))

        await logger.async_log_success_event(_success_kwargs(), RESPONSE, None, None)
        await _drain(logger)

        judge_call = next(
            c.kwargs
            for c in router.acompletion.call_args_list
            if c.kwargs["metadata"].get(INTERNAL_CALL_ORIGIN_METADATA_KEY) == SHADOW_EVAL_JUDGE_CALL_ORIGIN
        )
        assert judge_call["response_format"] == PAIRWISE_JUDGE_RESPONSE_FORMAT
        schema = judge_call["response_format"]["json_schema"]["schema"]
        assert schema["required"] == ["preference", "confidence"]
        assert schema["properties"]["preference"]["enum"] == ["A", "B", "tie"]

    async def test_shadow_call_messages_survive_in_place_provider_rewrites(self, monkeypatch: pytest.MonkeyPatch):
        """Provider transforms (anthropic factory, cache-control hook) rewrite messages with
        `messages[i] = ...`; the logger's immutable snapshot must never reach them directly."""
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
        prisma = _prisma()
        router = _router()
        inner = router.acompletion.side_effect

        async def mutating_acompletion(**kwargs):
            kwargs["messages"][0] = dict(kwargs["messages"][0])
            return await inner(**kwargs)

        router.acompletion = MagicMock(side_effect=mutating_acompletion)
        logger = _logger(router=router, prisma=prisma, jobs=(_job(),))

        await logger.async_log_success_event(_success_kwargs(), RESPONSE, None, None)
        await _drain(logger)

        row = prisma.db.litellm_shadowevalattempt.create.call_args.kwargs["data"]
        assert row["error"] is None
        assert row["outcome"] in ("real", "shadow", "tie")

    async def test_pipeline_continues_judging_after_a_failed_attempt(self, monkeypatch: pytest.MonkeyPatch):
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
        prisma = _prisma()
        router = _router()
        inner = router.acompletion.side_effect
        shadow_calls = {"count": 0}

        async def flaky_acompletion(**kwargs):
            if kwargs["metadata"].get(INTERNAL_CALL_ORIGIN_METADATA_KEY) == SHADOW_EVAL_ROUTER_CALL_ORIGIN:
                shadow_calls["count"] += 1
                if shadow_calls["count"] == 1:
                    raise RuntimeError("provider exploded")
            return await inner(**kwargs)

        router.acompletion = MagicMock(side_effect=flaky_acompletion)
        logger = _logger(router=router, prisma=prisma, jobs=(_job(),))

        await logger.async_log_success_event(_success_kwargs(), RESPONSE, None, None)
        await _drain(logger)
        await logger.async_log_success_event(_success_kwargs(request_id="req-2"), RESPONSE, None, None)
        await _drain(logger)

        rows = [c.kwargs["data"] for c in prisma.db.litellm_shadowevalattempt.create.call_args_list]
        assert [rows[0]["outcome"], rows[1]["outcome"] in ("real", "shadow")] == ["error", True]
        assert "provider exploded" in rows[0]["error"]
        assert rows[1]["request_id"] == "req-2"
        assert rows[1]["error"] is None
        assert logger._inflight_shadow_tasks == 0

    @pytest.mark.parametrize(
        "kwargs_mutation,job_mutation",
        [
            ({"request_metadata": {INTERNAL_CALL_ORIGIN_METADATA_KEY: "shadow_eval_router"}}, {}),
            ({"api_key_hash": "other-key"}, {}),
            ({"call_type": "aembedding"}, {}),
            ({"call_type": None}, {}),
            ({"request_metadata": {"routing_decision": {"router_model_name": "my-router"}}}, {}),
            ({}, {"ends_at": datetime.now(timezone.utc) - timedelta(seconds=1)}),
            ({}, {"attempts": 200}),
            ({}, {"attempts": 199, "max_turns": 200, "_starts": 1}),
            ({}, {"max_budget": 0.10, "spend": 0.10}),
        ],
        ids=[
            "internal-origin",
            "no-job-for-key",
            "non-chat",
            "missing-call-type",
            "self-shadow",
            "past-end",
            "turn-budget-reached",
            "budget-consumed-by-started-tasks",
            "spend-budget-reached",
        ],
    )
    async def test_skip_paths_store_nothing(self, kwargs_mutation, job_mutation):
        starts = job_mutation.pop("_starts", 0)
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma, jobs=(_job(**job_mutation),))
        logger._job_starts = {"job-1": starts}

        await logger.async_log_success_event(_success_kwargs(**kwargs_mutation), RESPONSE, None, None)
        await _drain(logger)

        prisma.db.litellm_shadowevalattempt.create.assert_not_called()
        assert logger._job_starts.get("job-1", 0) == starts

    async def test_completed_pipelines_hold_turn_budget_within_a_cache_generation(self):
        """A finished pipeline frees its concurrency slot but not its slice of the turn
        budget; the budget only reopens when a cache refill absorbs the written rows."""
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma, jobs=(_job(attempts=199, max_turns=200),))

        await logger.async_log_success_event(_success_kwargs(request_id="req-1"), RESPONSE, None, None)
        await _drain(logger)
        await logger.async_log_success_event(_success_kwargs(request_id="req-2"), RESPONSE, None, None)
        await _drain(logger)

        assert prisma.db.litellm_shadowevalattempt.create.await_count == 1

    async def test_completed_pipelines_hold_spend_budget_within_a_cache_generation(self, monkeypatch):
        """An attempt's recorded cost lands in the spend counter immediately, so the
        second sample is skipped before any provider call even though the cached fill
        still reads spend 0."""
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma, jobs=(_job(max_budget=0.009, spend=0.0),))

        await logger.async_log_success_event(_success_kwargs(request_id="req-1"), RESPONSE, None, None)
        await _drain(logger)
        await logger.async_log_success_event(_success_kwargs(request_id="req-2"), RESPONSE, None, None)
        await _drain(logger)

        assert prisma.db.litellm_shadowevalattempt.create.await_count == 1
        assert logger._test_counter["spend:shadow_eval:job-1"] == 0.01

    async def test_a_sibling_pod_sees_spend_through_the_shared_counter(self, monkeypatch):
        """Two pods share the cross-pod counter: once pod A's attempts spend the budget,
        pod B skips before its shadow call even though pod B's cached fill reads 0."""
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
        shared = {}
        prisma_a = _prisma()
        pod_a = _logger(
            router=_router(), prisma=prisma_a, jobs=(_job(max_budget=0.009, spend=0.0),), counter_store=shared
        )
        router_b = _router()
        prisma_b = _prisma()
        pod_b = _logger(
            router=router_b, prisma=prisma_b, jobs=(_job(max_budget=0.009, spend=0.0),), counter_store=shared
        )

        await pod_a.async_log_success_event(_success_kwargs(request_id="req-1"), RESPONSE, None, None)
        await _drain(pod_a)
        await pod_b.async_log_success_event(_success_kwargs(request_id="req-2"), RESPONSE, None, None)
        await _drain(pod_b)

        assert prisma_a.db.litellm_shadowevalattempt.create.await_count == 1
        prisma_b.db.litellm_shadowevalattempt.create.assert_not_called()
        router_b.acompletion.assert_not_called()

    async def test_legacy_jobs_without_a_spend_budget_sample_on_turns_alone(self):
        """A pre-migration job carries max_budget None: recorded spend must never gate it,
        only its own max_turns can."""
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma, jobs=(_job(max_budget=None, spend=999.0, attempts=5),))

        await logger.async_log_success_event(_success_kwargs(request_id="req-1"), RESPONSE, None, None)
        await _drain(logger)

        assert prisma.db.litellm_shadowevalattempt.create.await_count == 1

    async def test_v1_messages_surface_forwards_identity_from_litellm_metadata(self):
        """/v1/messages stores identity in litellm_params.litellm_metadata, so the hook
        resolves the bucket through the shared helper; every surface forwards the same
        identity to the shadow and judge calls."""
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma, jobs=(_job(),))

        hook_kwargs = _success_kwargs()
        hook_kwargs["litellm_params"] = {
            "litellm_metadata": {"user_api_key_hash": "key-hash", "user_api_key_team_id": "team-1"}
        }
        await logger.async_log_success_event(hook_kwargs, RESPONSE, None, None)
        await _drain(logger)

        shadow_call = router.acompletion.call_args_list[0].kwargs
        assert shadow_call["metadata"]["user_api_key_hash"] == "key-hash"
        assert shadow_call["metadata"]["user_api_key_team_id"] == "team-1"

    async def test_redacted_requests_are_never_shadowed(self):
        """Redaction rewrites the logged messages before callbacks run, so this hook only
        ever sees placeholders for opted-out traffic; the skip uses the redactor's own
        predicate, so every redaction source counts."""
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma, jobs=(_job(),))

        hook_kwargs = _success_kwargs()
        hook_kwargs["standard_callback_dynamic_params"] = {"turn_off_message_logging": True}
        await logger.async_log_success_event(hook_kwargs, RESPONSE, None, None)
        await _drain(logger)

        router.acompletion.assert_not_called()
        prisma.db.litellm_shadowevalattempt.create.assert_not_called()

    async def test_inflight_cap_sheds_instead_of_queueing(self):
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma, jobs=(_job(),))
        logger._inflight_shadow_tasks = _MAX_CONCURRENT_SHADOW_TASKS

        await logger.async_log_success_event(_success_kwargs(), RESPONSE, None, None)

        assert logger._inflight_shadow_tasks == _MAX_CONCURRENT_SHADOW_TASKS
        prisma.db.litellm_shadowevalattempt.create.assert_not_called()


JWT_IDENTITY = {"user_api_key_hash": None, "user_api_key_team_id": "team-eng", "user_api_key_user_id": "dev-alice"}


@pytest.mark.asyncio
class TestTargetMatching:
    """A request qualifies for a job through ANY of its resolved identities: key hash,
    team id, or user id. Team and user jobs must therefore sample JWT-authenticated
    traffic, which carries no key hash at all."""

    @pytest.mark.parametrize(
        "target,sampled",
        [
            (("team", "team-eng"), True),
            (("user", "dev-alice"), True),
            (("key", "some-key"), False),
        ],
        ids=["team-job-samples-jwt-traffic", "user-job-samples-jwt-traffic", "key-jobs-never-match-keyless-traffic"],
    )
    async def test_jwt_shaped_traffic_matches_team_and_user_jobs_but_no_key_job(self, target, sampled):
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma, jobs_by_target={target: (_job(),)})
        hook_kwargs = _success_kwargs()
        hook_kwargs["standard_logging_object"]["metadata"] = dict(JWT_IDENTITY)

        await logger.async_log_success_event(hook_kwargs, RESPONSE, None, None)
        await _drain(logger)

        if sampled:
            prisma.db.litellm_shadowevalattempt.create.assert_awaited_once()
            assert prisma.db.litellm_shadowevalattempt.create.call_args.kwargs["data"]["job_id"] == "job-1"
        else:
            router.acompletion.assert_not_called()
            prisma.db.litellm_shadowevalattempt.create.assert_not_called()

    async def test_an_event_with_no_identity_early_returns_without_a_cache_read(self):
        prisma = _prisma()
        router = _router()
        cache = MagicMock(spec=InMemoryCache)
        cache.async_get_cache = AsyncMock()
        logger = ShadowEvalLogger(
            router_provider=lambda: router,
            prisma_provider=lambda: prisma,
            jobs_cache=cache,
        )
        hook_kwargs = _success_kwargs()
        hook_kwargs["standard_logging_object"]["metadata"] = {}

        await logger.async_log_success_event(hook_kwargs, RESPONSE, None, None)

        cache.async_get_cache.assert_not_awaited()
        router.acompletion.assert_not_called()
        prisma.db.litellm_shadowevalattempt.create.assert_not_called()

    async def test_an_event_matching_a_key_job_and_a_team_job_fires_both(self):
        """A request's key and its team can each hold a job; the two are separately
        budgeted experiments, so both fire and each counts its own start."""
        prisma = _prisma()
        logger = _logger(
            router=_router(),
            prisma=prisma,
            jobs_by_target={
                ("key", "key-hash"): (_job(id="key-job"),),
                ("team", "team-eng"): (_job(id="team-job"),),
            },
        )
        hook_kwargs = _success_kwargs()
        hook_kwargs["standard_logging_object"]["metadata"] = {
            "user_api_key_hash": "key-hash",
            "user_api_key_team_id": "team-eng",
        }

        await logger.async_log_success_event(hook_kwargs, RESPONSE, None, None)
        await _drain(logger)

        rows = [call.kwargs["data"] for call in prisma.db.litellm_shadowevalattempt.create.call_args_list]
        assert sorted(row["job_id"] for row in rows) == ["key-job", "team-job"]
        assert logger._job_starts == {"key-job": 1, "team-job": 1}


@pytest.mark.asyncio
class TestActiveJobsCache:
    async def test_cache_miss_reads_db_once_then_serves_from_cache(self):
        job = _job()
        prisma = _prisma(jobs=[_job_record(job)], attempt_counts=[("job-1", 7)])
        logger = ShadowEvalLogger(
            router_provider=lambda: None,
            prisma_provider=lambda: prisma,
            jobs_cache=InMemoryCache(max_size_in_memory=4, default_ttl=60),
        )

        first = await logger._active_jobs()
        second = await logger._active_jobs()

        assert [job.id for job in first[("key", "key-hash")]] == ["job-1"]
        assert second[("key", "key-hash")][0].attempts == 7
        assert prisma.db.litellm_shadowevaljob.find_many.await_count == 1
        where = prisma.db.litellm_shadowevaljob.find_many.call_args.kwargs["where"]
        assert where["stopped_at"] is None
        assert "gt" in where["ends_at"]
        count_where = prisma.db.litellm_shadowevalattempt.group_by.call_args.kwargs["where"]
        assert count_where == {"job_id": {"in": ["job-1"]}}

    async def test_no_active_jobs_is_cached_too(self):
        prisma = _prisma(jobs=[])
        logger = ShadowEvalLogger(
            router_provider=lambda: None,
            prisma_provider=lambda: prisma,
            jobs_cache=InMemoryCache(max_size_in_memory=4, default_ttl=60),
        )

        assert await logger._active_jobs() == {}
        assert await logger._active_jobs() == {}
        assert prisma.db.litellm_shadowevaljob.find_many.await_count == 1
        prisma.db.litellm_shadowevalattempt.group_by.assert_not_called()

    async def test_db_fault_returns_empty_without_caching_the_fault(self):
        prisma = _prisma()
        prisma.db.litellm_shadowevaljob.find_many = AsyncMock(side_effect=RuntimeError("db blip"))
        logger = ShadowEvalLogger(
            router_provider=lambda: None,
            prisma_provider=lambda: prisma,
            jobs_cache=InMemoryCache(max_size_in_memory=4, default_ttl=60),
        )

        assert await logger._active_jobs() == {}
        assert await logger._active_jobs() == {}
        assert prisma.db.litellm_shadowevaljob.find_many.await_count == 2

    async def test_cache_refill_resets_the_starts_counter(self):
        job = _job()
        prisma = _prisma(jobs=[_job_record(job)], attempt_counts=[("job-1", 7)], attempt_costs=[("job-1", 0.02, 0.03)])
        logger = ShadowEvalLogger(
            router_provider=lambda: None,
            prisma_provider=lambda: prisma,
            jobs_cache=InMemoryCache(max_size_in_memory=4, default_ttl=60),
        )
        logger._job_starts = {"job-1": 5}

        jobs = await logger._active_jobs()

        assert logger._job_starts == {}
        assert jobs[("key", "key-hash")][0].attempts == 7
        assert jobs[("key", "key-hash")][0].spend == 0.05


@pytest.mark.asyncio
class TestShadowPipeline:
    async def test_no_prisma_means_no_provider_spend(self):
        router = _router()
        logger = _logger(router=router, prisma=None)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            real_text="real answer",
            real_model="claude-opus",
            real_cost=0.0,
            real_classifier_cost=0.0,
            real_cache_hit=False,
            control_tier=None,
            shadow_params={},
            parent_metadata={},
        )

        router.acompletion.assert_not_called()
        assert logger._test_funnel == [("job-1", "withheld")]

    async def test_over_budget_key_skips_before_any_call(self, monkeypatch: pytest.MonkeyPatch):
        """The gate delegates to the auth path's own budget owner, so an over-budget
        verdict there (BudgetExceededError) skips the shadow before any provider call."""
        from litellm.exceptions import BudgetExceededError
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.auth import auth_checks

        monkeypatch.setattr(
            auth_checks,
            "_virtual_key_max_budget_check",
            AsyncMock(side_effect=BudgetExceededError(current_cost=11.0, max_budget=10.0)),
        )
        router = _router()
        prisma = _prisma()
        logger = _logger(router=router, prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            real_text="real answer",
            real_model="claude-opus",
            real_cost=0.0,
            real_classifier_cost=0.0,
            real_cache_hit=False,
            control_tier=None,
            shadow_params={},
            parent_metadata={"user_api_key_auth": UserAPIKeyAuth(api_key="sk-abc", max_budget=10.0)},
        )

        router.acompletion.assert_not_called()
        prisma.db.litellm_shadowevalattempt.create.assert_not_called()
        assert logger._test_funnel == [("job-1", "withheld")]

    @pytest.mark.parametrize(
        "router_factory,expected_error,expected_cost,expected_shadow_cost",
        [
            (lambda: _failing_router(), "provider exploded", 0.0, 0.0),
            (lambda: _router(judge_json="I prefer response A, definitely"), "unparseable judge verdict", 0.007, 0.007),
            (lambda: _router(judge_json='{"preference": "'), "unparseable judge verdict", 0.007, 0.007),
            (lambda: _router(judge_json="{}"), "unparseable judge verdict", 0.007, 0.007),
            (
                lambda: _router(judge_json='{"preference": "A", "confidence": "0.8'),
                "unparseable judge verdict",
                0.007,
                0.007,
            ),
        ],
        ids=[
            "shadow-call-fails",
            "judge-verdict-unparseable",
            "verdict-truncated-before-fields",
            "verdict-empty-object",
            "verdict-truncated-inside-confidence",
        ],
    )
    async def test_failures_become_error_rows_and_keep_billed_judge_cost(
        self, router_factory, expected_error, expected_cost, expected_shadow_cost, monkeypatch: pytest.MonkeyPatch
    ):
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.007)
        prisma = _prisma()
        logger = _logger(router=router_factory(), prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            real_text="real answer",
            real_model="claude-opus",
            real_cost=0.0,
            real_classifier_cost=0.0,
            real_cache_hit=False,
            control_tier=None,
            shadow_params={},
            parent_metadata={},
        )

        row = prisma.db.litellm_shadowevalattempt.create.call_args.kwargs["data"]
        assert row["outcome"] == "error"
        assert expected_error in row["error"]
        assert row["confidence"] is None
        assert row["judge_cost"] == expected_cost
        assert row["shadow_cost"] == expected_shadow_cost

    async def test_an_empty_shadow_reply_still_bills_its_cost(self, monkeypatch: pytest.MonkeyPatch):
        """A shadow call that returns no extractable text has still billed; pricing it at
        zero would keep the dollar gate open while shadow calls keep charging the key."""
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.007)
        prisma = _prisma()
        logger = _logger(router=_router(shadow_text=""), prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            real_text="real answer",
            real_model="claude-opus",
            real_cost=0.0,
            real_classifier_cost=0.0,
            real_cache_hit=False,
            control_tier=None,
            shadow_params={},
            parent_metadata={},
        )

        row = prisma.db.litellm_shadowevalattempt.create.call_args.kwargs["data"]
        assert row["outcome"] == "error"
        assert "empty response" in row["error"]
        assert row["shadow_cost"] == 0.007
        assert logger._test_counter["spend:shadow_eval:job-1"] == 0.007

    async def test_a_pipeline_error_after_the_shadow_call_keeps_its_billed_cost(self, monkeypatch: pytest.MonkeyPatch):
        """An unexpected error between the billed shadow call and the attempt write must
        still record the shadow cost, or the per-key dollar gate undercounts forever."""
        import litellm as litellm_module
        import litellm.integrations.shadow_eval_logger as shadow_eval_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.007)

        def explode(conversation, response_a, response_b):
            raise RuntimeError("judge prompt build failed")

        monkeypatch.setattr(shadow_eval_module, "_judge_user_prompt", explode)
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            real_text="real answer",
            real_model="claude-opus",
            real_cost=0.0,
            real_classifier_cost=0.0,
            real_cache_hit=False,
            control_tier=None,
            shadow_params={},
            parent_metadata={},
        )

        row = prisma.db.litellm_shadowevalattempt.create.call_args.kwargs["data"]
        assert row["outcome"] == "error"
        assert "pipeline error" in row["error"]
        assert row["shadow_cost"] == 0.007
        assert row["judge_cost"] == 0.0
        assert logger._test_counter["spend:shadow_eval:job-1"] == 0.007

    async def test_sub_calls_carry_identity_and_origin_but_never_parent_request_state(self):
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma)
        parent_metadata = {
            "user_api_key_hash": "key-hash",
            "user_api_key_team_id": "team-1",
            "user_api_key_budget_reservation": {"amount": 1.0},
            "routing_decision": {"router_model_name": "other-router"},
        }

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            real_text="real answer",
            real_model="claude-opus",
            real_cost=0.0,
            real_classifier_cost=0.0,
            real_cache_hit=False,
            control_tier=None,
            shadow_params={"temperature": 0.2},
            parent_metadata=parent_metadata,
        )

        shadow_call = router.acompletion.call_args_list[0].kwargs
        judge_call = router.acompletion.call_args_list[1].kwargs
        for call in (shadow_call, judge_call):
            assert call["num_retries"] == 0
            assert call["fallbacks"] == []
            assert call["metadata"]["user_api_key_hash"] == "key-hash"
            assert call["metadata"]["user_api_key_team_id"] == "team-1"
            assert "user_api_key_budget_reservation" not in call["metadata"]
        assert shadow_call["metadata"][INTERNAL_CALL_ORIGIN_METADATA_KEY] == SHADOW_EVAL_ROUTER_CALL_ORIGIN
        assert judge_call["metadata"][INTERNAL_CALL_ORIGIN_METADATA_KEY] == SHADOW_EVAL_JUDGE_CALL_ORIGIN
        assert "routing_decision" not in judge_call["metadata"]
        assert shadow_call["temperature"] == 0.2
        assert judge_call["max_tokens"] == JUDGE_MAX_OUTPUT_TOKENS


def _reverse_job(**overrides) -> ActiveShadowEvalJob:
    return _job(**{"direction": "reverse", "baseline_model": "baseline-model", **overrides})


class TestJobValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"direction": "reverse"},
            {"baseline_model": "baseline-model"},
            {"direction": "sideways", "baseline_model": "baseline-model"},
            {"direction": "reverse", "baseline_model": "baseline-model", "router_names": ("a", "b")},
        ],
        ids=["reverse-without-baseline", "forward-with-baseline", "unknown-direction", "reverse-with-router-set"],
    )
    def test_unsamplable_shapes_are_rejected(self, overrides):
        with pytest.raises(ValidationError):
            _job(**overrides)

    def test_arm_target_follows_direction(self):
        assert _job().arm_target("my-router") == "my-router"
        assert _reverse_job().arm_target("my-router") == "baseline-model"

    def test_rows_from_before_router_names_carry_their_set_in_router_name(self):
        assert _job().arm_router_names == ("my-router",)
        assert _job(router_names=("my-router", "alt-router")).arm_router_names == ("my-router", "alt-router")


@pytest.mark.asyncio
class TestDirection:
    @pytest.mark.parametrize(
        "job,routed_by,attempt_rows",
        [
            (_job(), None, 1),
            (_job(), "my-router", 0),
            (_job(), "other-router", 1),
            (_reverse_job(), "my-router", 1),
            (_reverse_job(), None, 0),
            (_reverse_job(), "other-router", 0),
            (_job(router_names=("my-router", "alt-router")), "alt-router", 0),
            (_job(router_names=("my-router", "alt-router")), "other-router", 2),
        ],
        ids=[
            "forward-samples-unrouted",
            "forward-skips-its-own-router",
            "forward-samples-another-router",
            "reverse-samples-its-own-router",
            "reverse-skips-unrouted",
            "reverse-skips-another-router",
            "forward-skips-any-candidates-own-traffic",
            "forward-multi-samples-once-per-arm",
        ],
    )
    async def test_direction_decides_which_traffic_is_sampled(self, job, routed_by, attempt_rows):
        """The two directions partition the key's traffic: whatever one samples, the other
        skips, so a key running both never judges the same turn twice for the same reason.
        A multi-router job extends the forward skip to every candidate: a request one
        candidate served must not be judged as the incumbent against another candidate."""
        prisma = _prisma()
        logger = _logger(router=_router(sibling_router_texts={"alt-router": "alt answer"}), prisma=prisma, jobs=(job,))

        await logger.async_log_success_event(
            _success_kwargs(request_metadata=_routed_by(routed_by) if routed_by else {}), RESPONSE, None, None
        )
        await _drain(logger)

        assert prisma.db.litellm_shadowevalattempt.create.await_count == attempt_rows

    async def test_reverse_duplicates_against_the_baseline_model(self):
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma, jobs=(_reverse_job(),))

        await logger.async_log_success_event(_success_kwargs(request_metadata=_routed_by()), RESPONSE, None, None)
        await _drain(logger)

        assert router.acompletion.call_args_list[0].kwargs["model"] == "baseline-model"

    async def test_reverse_row_orients_arms_and_reads_tier_off_the_served_request(self):
        """real is what the caller received, so in reverse it is the router's own pick and
        the tier that produced it; only the shadow arm moves to the baseline."""
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma, jobs=(_reverse_job(),))

        await logger.async_log_success_event(
            _success_kwargs(request_metadata=_routed_by(tier="COMPLEX"), model="router-pick"), RESPONSE, None, None
        )
        await _drain(logger)

        row = prisma.db.litellm_shadowevalattempt.create.call_args.kwargs["data"]
        assert row["real_model"] == "router-pick"
        assert row["shadow_model"] == "baseline-model"
        assert row["tier"] == "COMPLEX"

    async def test_forward_row_still_reads_tier_off_the_shadow_call(self):
        """A forward job's tier describes the arm being evaluated, which is the shadow one,
        so a routing decision on the incumbent request must not leak into it."""
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma, jobs=(_job(),))

        await logger.async_log_success_event(
            _success_kwargs(request_metadata=_routed_by("other-router", tier="CONTROL_TIER")), RESPONSE, None, None
        )
        await _drain(logger)

        row = prisma.db.litellm_shadowevalattempt.create.call_args.kwargs["data"]
        assert row["tier"] == "SIMPLE"
        assert row["shadow_model"] == "cheap-model"

    async def test_a_key_running_both_directions_dispatches_both(self):
        """One request can qualify for a forward job on a router that did not serve it and a
        reverse job on the router that did. The two are separately budgeted experiments, so
        both fire rather than one silently losing the turn."""
        prisma = _prisma()
        logger = _logger(
            router=_router(),
            prisma=prisma,
            jobs=(_job(id="forward-job", router_name="other-router"), _reverse_job(id="reverse-job")),
        )

        await logger.async_log_success_event(_success_kwargs(request_metadata=_routed_by()), RESPONSE, None, None)
        await _drain(logger)

        rows = [call.kwargs["data"] for call in prisma.db.litellm_shadowevalattempt.create.call_args_list]
        assert sorted(row["job_id"] for row in rows) == ["forward-job", "reverse-job"]
        assert logger._job_starts == {"forward-job": 1, "reverse-job": 1}


@pytest.mark.asyncio
class TestMultiRouterArms:
    async def test_every_arm_judges_the_same_request_and_stamps_its_own_row(self):
        """One sampled request, one row per candidate router, both judged against the same
        real response: the paired comparison that makes multi-router win rates comparable."""
        prisma = _prisma()
        router = _router(sibling_router_texts={"alt-router": "alt answer"})
        logger = _logger(router=router, prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(router_names=("my-router", "alt-router")),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            real_text="real answer",
            real_model="claude-opus",
            real_cost=0.001,
            real_classifier_cost=0.0,
            real_cache_hit=False,
            control_tier=None,
            shadow_params={},
            parent_metadata={},
        )

        rows = [call.kwargs["data"] for call in prisma.db.litellm_shadowevalattempt.create.await_args_list]
        assert [row["router_name"] for row in rows] == ["my-router", "alt-router"]
        assert {row["request_id"] for row in rows} == {"req-1"}
        assert [row["shadow_model"] for row in rows] == ["cheap-model", "alt-router-pick"]
        assert all(row["outcome"] in ("real", "shadow", "tie") for row in rows)
        assert all(row["real_cost"] == 0.001 for row in rows)

    async def test_a_single_router_job_stamps_its_router_on_the_row(self):
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            real_text="real answer",
            real_model="claude-opus",
            real_cost=0.0,
            real_classifier_cost=0.0,
            real_cache_hit=False,
            control_tier=None,
            shadow_params={},
            parent_metadata={},
        )

        row = prisma.db.litellm_shadowevalattempt.create.call_args.kwargs["data"]
        assert row["router_name"] == "my-router"

    async def test_one_arms_failure_never_silences_the_sibling(self):
        prisma = _prisma()
        router = _router(sibling_router_texts={"alt-router": "alt answer"})
        healthy = router.acompletion.side_effect

        async def first_arm_explodes(**kwargs):
            if kwargs["model"] == "my-router":
                raise RuntimeError("provider exploded")
            return await healthy(**kwargs)

        router.acompletion.side_effect = first_arm_explodes
        logger = _logger(router=router, prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(router_names=("my-router", "alt-router")),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            real_text="real answer",
            real_model="claude-opus",
            real_cost=0.0,
            real_classifier_cost=0.0,
            real_cache_hit=False,
            control_tier=None,
            shadow_params={},
            parent_metadata={},
        )

        rows = [call.kwargs["data"] for call in prisma.db.litellm_shadowevalattempt.create.await_args_list]
        assert [row["router_name"] for row in rows] == ["my-router", "alt-router"]
        assert rows[0]["outcome"] == "error"
        assert "provider exploded" in rows[0]["error"]
        assert rows[1]["outcome"] in ("real", "shadow", "tie")

    async def test_the_turn_valve_counts_every_arm_a_start_will_write(self):
        """max_turns is a row ceiling and one sampled request writes one row per arm, so
        admission pre-counts the arms: a two-arm job with two turns of budget admits one
        request, not two."""
        prisma = _prisma()
        router = _router(sibling_router_texts={"alt-router": "alt answer"})
        logger = _logger(
            router=router, prisma=prisma, jobs=(_job(router_names=("my-router", "alt-router"), max_turns=2),)
        )

        await logger.async_log_success_event(_success_kwargs(request_id="req-1"), RESPONSE, None, None)
        await logger.async_log_success_event(_success_kwargs(request_id="req-2"), RESPONSE, None, None)
        await _drain(logger)

        rows = [call.kwargs["data"] for call in prisma.db.litellm_shadowevalattempt.create.await_args_list]
        assert {row["request_id"] for row in rows} == {"req-1"}
        assert len(rows) == 2

    async def test_a_withheld_request_runs_no_arm_and_counts_once(self):
        """The budget gates run once per sampled request, before any arm: funnel counters
        stay per-request, so coverage math is arm-count independent."""
        prisma = _prisma()
        router = _router(sibling_router_texts={"alt-router": "alt answer"})
        logger = _logger(router=router, prisma=prisma)

        await logger._run_shadow_eval(
            job=_job(router_names=("my-router", "alt-router"), max_budget=1.0, spend=2.0),
            request_id="req-1",
            messages=({"role": "user", "content": "hi"},),
            real_text="real answer",
            real_model="claude-opus",
            real_cost=0.0,
            real_classifier_cost=0.0,
            real_cache_hit=False,
            control_tier=None,
            shadow_params={},
            parent_metadata={},
        )

        router.acompletion.assert_not_called()
        prisma.db.litellm_shadowevalattempt.create.assert_not_called()
        assert logger._test_funnel == [("job-1", "withheld")]


@pytest.mark.asyncio
class TestActiveJobsFailClosed:
    async def test_a_row_the_sampler_cannot_read_is_dropped_not_guessed(self):
        """A reverse row with no baseline model has no second arm to call, so it is skipped
        rather than silently dispatched at the router it is supposed to be judging."""
        broken = _job_record(_job(id="job-broken"))
        broken.direction = "reverse"
        broken.baseline_model = None
        prisma = _prisma(jobs=[broken, _job_record(_job(id="job-ok"))], attempt_counts=[("job-ok", 1)])
        logger = ShadowEvalLogger(
            router_provider=lambda: None,
            prisma_provider=lambda: prisma,
            jobs_cache=InMemoryCache(max_size_in_memory=4, default_ttl=60),
        )

        assert [job.id for job in (await logger._active_jobs())[("key", "key-hash")]] == ["job-ok"]

    async def test_every_targets_jobs_survive_the_lookup_keyed_by_type_and_id(self):
        records = [
            _job_record(_job(id="job-forward")),
            _job_record(_reverse_job(id="job-reverse")),
            _job_record(_job(id="job-other"), target_id="other-key"),
            _job_record(_job(id="job-team"), target_type="team", target_id="team-eng"),
        ]
        prisma = _prisma(jobs=records, attempt_counts=[("job-reverse", 3)])
        logger = ShadowEvalLogger(
            router_provider=lambda: None,
            prisma_provider=lambda: prisma,
            jobs_cache=InMemoryCache(max_size_in_memory=4, default_ttl=60),
        )

        jobs = await logger._active_jobs()

        assert sorted(job.id for job in jobs[("key", "key-hash")]) == ["job-forward", "job-reverse"]
        assert [job.id for job in jobs[("key", "other-key")]] == ["job-other"]
        assert [job.id for job in jobs[("team", "team-eng")]] == ["job-team"]
        assert ("team-eng",) not in jobs and "team-eng" not in jobs
        assert {job.id: job.attempts for job in jobs[("key", "key-hash")]}["job-reverse"] == 3


def _failing_router():
    router = MagicMock()
    router.model_group_alias = {}
    router.get_model_list = MagicMock(return_value=None)
    router.acompletion = AsyncMock(side_effect=RuntimeError("provider exploded"))
    return router


@pytest.mark.asyncio
async def test_judge_call_resolves_its_arm_under_the_shadowed_keys_team(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start-time validation resolves the judge under the key's team, so the dispatch has to
    as well or the two disagree about the same name.

    A team-public judge resolves to a real deployment for its own team and to nothing for
    anybody else. Choosing the arm without the team sends the literal name to the SDK, which
    has never heard of it, so every judge call fails on a job validation just accepted.
    """
    import litellm
    from litellm.litellm_core_utils.llm_judge import judge_acompletion

    router = litellm.Router(
        model_list=[
            {
                "model_name": "row_team_a",
                "litellm_params": {"model": "anthropic/claude-sonnet-5", "api_key": "fake"},
                "model_info": {"team_id": "team-a", "team_public_model_name": "house-judge"},
            }
        ]
    )
    router.acompletion = AsyncMock(  # pyright: ignore[reportAttributeAccessIssue]  # fake the call, not the resolution
        return_value={"choices": [{"message": {"content": "router answer"}}]}
    )
    sdk = AsyncMock(return_value={"choices": [{"message": {"content": "sdk answer"}}]})
    monkeypatch.setattr(litellm, "acompletion", sdk)

    await judge_acompletion(router, "house-judge", [{"role": "user", "content": "hi"}], team_id="team-a")

    router.acompletion.assert_awaited_once()
    sdk.assert_not_called()


@pytest.mark.asyncio
class TestCostComparison:
    """The attempt row prices BOTH arms with what each actually billed: the real arm's
    payload cost plus its own classifier when it routed, the shadow arm's completion plus
    its write-back classifier cost, and the exact-cache flag that voids the comparison."""

    async def test_success_row_records_both_arms_and_the_classifier(self, monkeypatch: pytest.MonkeyPatch):
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
        router = _router(classifier_cost=0.0007)
        prisma = _prisma()
        logger = _logger(router=router, prisma=prisma, jobs=(_job(),))

        await logger.async_log_success_event(_success_kwargs(response_cost=0.002), RESPONSE, None, None)
        await _drain(logger)

        row = prisma.db.litellm_shadowevalattempt.create.call_args.kwargs["data"]
        assert row["real_cost"] == 0.002
        assert row["real_classifier_cost"] == 0.0
        assert row["shadow_classifier_cost"] == 0.0007
        assert row["real_cache_hit"] is False
        assert logger._test_funnel == []

    async def test_reverse_job_prices_the_real_arms_classifier(self, monkeypatch: pytest.MonkeyPatch):
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
        router = _router()
        prisma = _prisma()
        job = _job(direction="reverse", baseline_model="gpt-4o-mini")
        logger = _logger(router=router, prisma=prisma, jobs=(job,))
        metadata = _routed_by()
        metadata["routing_decision"]["classifier_cost"] = 0.0004

        await logger.async_log_success_event(
            _success_kwargs(request_metadata=metadata, response_cost=0.003), RESPONSE, None, None
        )
        await _drain(logger)

        row = prisma.db.litellm_shadowevalattempt.create.call_args.kwargs["data"]
        assert row["real_cost"] == 0.003
        assert row["real_classifier_cost"] == 0.0004
        assert row["shadow_classifier_cost"] == 0.0

    async def test_shadow_classifier_cost_charges_the_eval_budget_counter(self, monkeypatch: pytest.MonkeyPatch):
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
        logger = _logger(router=_router(classifier_cost=0.0007), prisma=_prisma(), jobs=(_job(),))

        await logger.async_log_success_event(_success_kwargs(response_cost=0.002), RESPONSE, None, None)
        await _drain(logger)

        assert logger._test_counter["spend:shadow_eval:job-1"] == pytest.approx(0.005 + 0.005 + 0.0007)

    async def test_real_cost_never_charges_the_eval_budget_counter(self, monkeypatch: pytest.MonkeyPatch):
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
        logger = _logger(router=_router(), prisma=_prisma(), jobs=(_job(),))

        await logger.async_log_success_event(_success_kwargs(response_cost=99.0), RESPONSE, None, None)
        await _drain(logger)

        assert logger._test_counter["spend:shadow_eval:job-1"] == pytest.approx(0.005 + 0.005)

    async def test_cache_served_turn_is_flagged_on_the_row(self, monkeypatch: pytest.MonkeyPatch):
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma, jobs=(_job(),))

        await logger.async_log_success_event(_success_kwargs(response_cost=0.0, cache_hit=True), RESPONSE, None, None)
        await _drain(logger)

        row = prisma.db.litellm_shadowevalattempt.create.call_args.kwargs["data"]
        assert row["real_cache_hit"] is True
        assert row["real_cost"] == 0.0

    async def test_failed_shadow_call_still_records_its_classifier_cost(self, monkeypatch: pytest.MonkeyPatch):
        import litellm as litellm_module

        monkeypatch.setattr(litellm_module, "completion_cost", lambda completion_response: 0.005)
        router = _router(classifier_cost=0.0007)

        async def failing_acompletion(**kwargs):
            if kwargs["metadata"].get(INTERNAL_CALL_ORIGIN_METADATA_KEY) == SHADOW_EVAL_ROUTER_CALL_ORIGIN:
                kwargs["metadata"]["routing_decision"] = {"tier_label": "SIMPLE", "classifier_cost": 0.0007}
                raise RuntimeError("provider down")
            return {"choices": [{"message": {"content": "unused"}}]}

        router.acompletion = MagicMock(side_effect=failing_acompletion)
        prisma = _prisma()
        logger = _logger(router=router, prisma=prisma, jobs=(_job(),))

        await logger.async_log_success_event(_success_kwargs(response_cost=0.002), RESPONSE, None, None)
        await _drain(logger)

        row = prisma.db.litellm_shadowevalattempt.create.call_args.kwargs["data"]
        assert row["outcome"] == "error"
        assert row["shadow_classifier_cost"] == 0.0007
        assert row["real_cost"] == 0.002
        assert logger._test_counter["spend:shadow_eval:job-1"] == pytest.approx(0.0007)


@pytest.mark.asyncio
class TestSamplingFunnel:
    async def test_a_budget_reached_admission_counts_withheld_not_nothing(self):
        """The in-flight burst as a job crosses max_budget must stay in the coverage
        identity: admitted samples the budget gate holds land in withheld."""
        counter = {"spend:shadow_eval:job-1": 5.0}
        prisma = _prisma()
        router = _router()
        logger = _logger(router=router, prisma=prisma, jobs=(_job(max_budget=1.0, spend=0.0),), counter_store=counter)

        await logger.async_log_success_event(_success_kwargs(), RESPONSE, None, None)
        await _drain(logger)

        router.acompletion.assert_not_called()
        prisma.db.litellm_shadowevalattempt.create.assert_not_called()
        assert logger._test_funnel == [("job-1", "withheld")]

    """Skips an admitting job cannot derive from attempt rows are counted per leg, so the
    judged rows can be weighed against the eligible traffic they stand for."""

    async def test_a_lost_sampling_dice_roll_counts_not_sampled(self):
        from litellm.integrations.shadow_eval_logger import _sample_hits

        job = _job(shadow_percentage=1.0)
        missing_id = next(
            f"req-miss-{n}" for n in range(10_000) if not _sample_hits(f"req-miss-{n}", job.id, job.shadow_percentage)
        )
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma, jobs=(job,))

        await logger.async_log_success_event(_success_kwargs(request_id=missing_id), RESPONSE, None, None)
        await _drain(logger)

        assert logger._test_funnel == [("job-1", "not_sampled")]
        prisma.db.litellm_shadowevalattempt.create.assert_not_awaited()

    async def test_an_unjudgeable_sampled_request_counts_unjudgeable(self):
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma, jobs=(_job(),))
        tool_final = {"choices": [{"message": {"content": None, "tool_calls": [{"type": "function", "function": {}}]}}]}

        await logger.async_log_success_event(_success_kwargs(), tool_final, None, None)
        await _drain(logger)

        assert logger._test_funnel == [("job-1", "unjudgeable")]
        prisma.db.litellm_shadowevalattempt.create.assert_not_awaited()

    async def test_a_concurrency_shed_counts_shed_and_starts_nothing(self):
        prisma = _prisma()
        logger = _logger(router=_router(), prisma=prisma, jobs=(_job(),))
        logger._inflight_shadow_tasks = 16

        await logger.async_log_success_event(_success_kwargs(), RESPONSE, None, None)

        assert logger._test_funnel == [("job-1", "shed")]
        assert logger._job_starts == {}
        prisma.db.litellm_shadowevalattempt.create.assert_not_awaited()
        logger._inflight_shadow_tasks = 0

    async def test_direction_mismatch_and_saturated_jobs_count_nothing(self):
        prisma = _prisma()
        saturated = _job(id="job-full", max_turns=1, attempts=1)
        wrong_direction = _job(id="job-rev", direction="reverse", baseline_model="gpt-4o-mini")
        logger = _logger(router=_router(), prisma=prisma, jobs=(saturated, wrong_direction))

        await logger.async_log_success_event(_success_kwargs(), RESPONSE, None, None)
        await _drain(logger)

        assert logger._test_funnel == []
        prisma.db.litellm_shadowevalattempt.create.assert_not_awaited()
