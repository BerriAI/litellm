"""
A cascade router's step once the attempt answered, on the Rust router core (``_maybe_cascade_escalate``,
``_cascade_strong_call`` and ``_settle_cascade_cheap_leg`` of the Python router): the cheap deployment
router-core picked has answered; the cascade judges the answer and, when it falls short, calls the strong
group through the core as the request's own continuation. router-core counts the attempt's usage from the
token count the attempt hands back, so the attempt hands back the cheap answer's; the strong leg's own drive
counts the strong deployment's.
"""

import contextlib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final  # noqa: TID251  # request kwargs forwarded to acompletion are untyped

from litellm._internal_context import internal_sub_call
from litellm._logging import verbose_router_logger
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY, ROUTER_BUDGET_ATTRIBUTED_SPEND_METADATA_KEY
from litellm.litellm_core_utils.core_helpers import get_parent_otel_span_from_kwargs
from litellm.router_strategy.cascade_router import cascade_router as _cascade_mod
from litellm.router_strategy.cascade_router.cascade_router import CascadeRouter, StrongOutcome
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    attributed_spend_marker,
    find_deployment_metadata,
)
from litellm.types.utils import CASCADE_VERIFIER_CALL_ORIGIN, ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.router_rust.core import RustCore
    from litellm.router_rust.invoke.endpoints import AttemptCall
    from litellm.router_rust.surface.registry import HostRegistry


@dataclass(frozen=True)
class CascadeAttempt:
    """A chat attempt a cascade judges: its strategy, the request's logging object, and whether this attempt
    parked that object's success log (the cascade fires it once the final answer is known)."""

    strategy: CascadeRouter
    logging_obj: "Logging | None"
    owns_deferral: bool


@dataclass(frozen=True)
class CascadeOutcome:
    """What the attempt serves once the cascade judged: the final answer, and the answer whose tokens
    router-core counts against the deployment it picked (the cheap one)."""

    response: object
    counted: object


def cascade_for_request(hosted: "HostRegistry", kwargs: Mapping[str, object]) -> CascadeRouter | None:
    """The cascade strategy that routed this request, read back from the stamped routing decision."""
    registry: Final = hosted._core_cascade_routers  # pyright: ignore[reportPrivateUsage]  # the registry lives on the host registry
    if not registry:
        return None
    litellm_metadata: Final = _cascade_mod.mapping_or_none(kwargs.get("litellm_metadata")) or {}
    metadata: Final = _cascade_mod.mapping_or_none(kwargs.get("metadata")) or {}
    decision: Final = _cascade_mod.mapping_or_none(
        litellm_metadata.get("routing_decision")
    ) or _cascade_mod.mapping_or_none(metadata.get("routing_decision"))
    if decision is None or decision.get("router_type") != "cascade" or decision.get("tier") != "cheap":
        return None
    tagged: Final = registry.get(str(decision.get("router_model_name")))
    strategy: Final[object] = tagged[0].strategy if tagged else None
    return strategy if isinstance(strategy, CascadeRouter) else None


def judging_cascade(hosted: "HostRegistry", kwargs: Mapping[str, object]) -> CascadeRouter | None:
    """The cascade judging this attempt, if any. A cascade reached from inside another cascade's leg (its strong
    call runs under ``cascade_leg``, its verifier call carries the verifier origin) is refused here: the two would
    share the request's logging object and bookkeeping, and no registration-time rule sees every path a name can
    take (an alias added at runtime, a pattern route). Other callers that route through the core as an internal
    sub-call (the RAG pipeline) are not inside a cascade and are served."""
    if not hosted._core_cascade_routers:  # pyright: ignore[reportPrivateUsage]  # the registry lives on the host registry
        return None
    if not _cascade_mod.inside_cascade_leg():
        # Every request that is not a cascade's own leg starts with clean per-request markers: an SDK caller may
        # hand the previous request's metadata dict (and even its call id) back in. The leg keeps the request's.
        _cascade_mod.reset_request_bookkeeping(kwargs)
    strategy: Final = cascade_for_request(hosted, kwargs)
    if strategy is None:
        return None
    metadata: Final = _cascade_mod.mapping_or_none(kwargs.get("metadata")) or {}
    if (
        _cascade_mod.inside_cascade_leg()
        or metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY) == CASCADE_VERIFIER_CALL_ORIGIN
    ):
        raise ValueError(f"cascade {strategy.model_name!r} cannot serve another cascade's leg; cascades do not nest")
    return strategy


def prepare_cascade_attempt(core: "RustCore", call: "AttemptCall") -> CascadeAttempt | None:
    """Before a chat attempt's provider call: the cascade judging it (none for most attempts), the request's
    logging object (built here on the SDK path) and that object's success log parked, so the cascade can re-point
    it at the final answer (summed cost) before it fires."""
    strategy: Final = judging_cascade(core._hosted, call.kwargs)  # pyright: ignore[reportPrivateUsage]  # the step serves its own core
    if strategy is None:
        return None
    logging_obj: Final = _request_logging_obj(call)
    return CascadeAttempt(strategy, logging_obj, _cascade_mod.park_success_log(logging_obj))


def _request_logging_obj(call: "AttemptCall") -> "Logging | None":
    """SDK path: give a cascade request its logging object before the cheap call (the proxy hands one in).
    ``litellm.acompletion`` would otherwise create it inside the cheap call, where the cascade cannot park its
    success log, share it with the strong leg or bill the final answer on it; the caller's callbacks would then
    record the rejected cheap answer as the request's. Same setup the wrapper does, as the prompt-management
    helper also does ahead of its call."""
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.utils import Rules, function_setup

    existing: Final = call.kwargs.get("litellm_logging_obj")
    if existing is not None:
        return existing if isinstance(existing, Logging) else None
    request: Final[dict[str, Any]] = {  # mutable-ok: handed to function_setup
        key: value for key, value in call.served.items() if key != "model"
    }
    call_id: Final = request.get("litellm_call_id")
    request["litellm_call_id"] = call_id if isinstance(call_id, str) else str(uuid.uuid4())
    logging_obj, _ = function_setup(
        original_function="acompletion",
        rules_obj=Rules(),
        start_time=datetime.now(),  # noqa: DTZ005  # the wrapper times calls on this naive clock
        model=call.group,
        **request,
    )
    # `callbacks` is consumed by function_setup, as the wrapper would have.
    stamps: Final[dict[str, object]] = {  # mutable-ok: merged below
        "litellm_logging_obj": logging_obj,
        "litellm_call_id": request["litellm_call_id"],
    }
    call.kwargs = {  # rebind-ok: the attempt's call, given its logging object as the Python router gave its kwargs
        **{key: value for key, value in call.kwargs.items() if key != "callbacks"},
        **stamps,
    }
    if isinstance(call.served, dict):
        served: Final[dict[str, object]] = call.served  # pyright: ignore[reportUnknownVariableType]  # mutable-ok: the request kwargs dict, edited in place as the Python router did
        _ = served.pop("callbacks", None)
        served.update(stamps)
    return logging_obj


def provider_call_scope(kwargs: Mapping[str, object]) -> contextlib.AbstractContextManager[None]:
    """A cascade's strong leg logs no row of its own (the request's one row bills both legs), so its own provider
    call, the one carrying the request's shared logging object, runs as an internal sub-call. What the leg's
    routing starts (a classifier, the deployment's silent experiment) logs as its own."""
    return internal_sub_call() if _cascade_mod.is_cascade_legs_own_call(kwargs) else contextlib.nullcontext()


async def judge_cascade_answer(
    core: "RustCore", pending: CascadeAttempt, call: "AttemptCall", response: object, *, blocked: bool
) -> CascadeOutcome:
    """Judge a cheap answer served for a cascade group and call the strong group when it falls short.

    The returned response is what the caller receives: the cheap answer when it passes, the strong answer when
    it does not. Either way it carries the usage and cost of the answers taken (cheap, plus strong on an
    escalation), a ``cascade`` record in its hidden params, and the request's ``routing_decision`` is rewritten
    to say what happened. A stream, a non-chat shape or a blocked answer is not judged: the log parked for the
    cascade is released and the attempt serves what it got (router-core raises over a blocked answer).
    """
    logging_obj: Final = pending.logging_obj
    if not isinstance(response, ModelResponse) or call.kwargs.get("stream") or blocked:
        if pending.owns_deferral:
            _cascade_mod.release_deferred_logging(logging_obj)
        return CascadeOutcome(response, response)
    kwargs: Final[dict[str, Any]] = call.kwargs if isinstance(call.kwargs, dict) else dict(call.kwargs)  # pyright: ignore[reportUnknownVariableType]  # mutable-ok: the attempt's kwargs; an escalation re-stamps them in place
    messages: Final = kwargs.get("messages")
    strategy: Final = pending.strategy
    cfg: Final = strategy.config
    caller_capped: Final = any(kwargs.get(k) is not None for k in ("max_tokens", "max_completion_tokens"))
    verdict: Final = await strategy.async_verify(
        messages if isinstance(messages, list) else [],  # pyright: ignore[reportUnknownArgumentType]  # the request's messages, as given
        response,
        caller_capped=caller_capped,
        request_kwargs=call.served,
    )
    calls: Final[list[Mapping[str, object]]] = [  # mutable-ok: the strong leg is appended below
        _cascade_mod.inner_call_record("cheap", cfg.cheap_model, response),
        *verdict.calls,
    ]
    # The strong leg is the request continued on another deployment, like a fallback hop: it shares the
    # request's logging object and metadata, runs as an internal sub-call (no log row of its own), and the
    # core stamps the deployment that answered. Each leg's cost breakdown is read off the shared object.
    cheap_breakdown: Final = _cascade_mod.breakdown_of(logging_obj)
    outcome: Final = (
        await _strong_call(core, pending, call, kwargs, messages, response)
        if verdict.escalate
        else StrongOutcome(response=None, error=None)
    )
    strong: Final = outcome.response
    strong_breakdown: Final = _cascade_mod.breakdown_of(logging_obj)
    escalated: Final = strong is not None
    if strong is not None:
        calls.append(_cascade_mod.inner_call_record("strong", cfg.strong_model, strong))
        _cascade_mod.record_summed_breakdown(logging_obj, response, cheap_breakdown, strong, strong_breakdown)
    # Billing follows upstream's two-round rule and the auto-router convention: the request's one row is
    # charged for the answers it took (cheap, plus strong on an escalation), each leg as it was priced; the
    # verifier is the decision's `classifier_cost`, billed on its own internal-call row like the complexity
    # router's classifier. The caller gets a copy: the cheap call's own response (its log may be queued on
    # the SDK path) is left as the call made it.
    final: Final = (
        _cascade_mod.escalated_response(response, cheap_breakdown, strong, strong_breakdown)
        if strong is not None
        else _cascade_mod.detached_copy(response)
    )
    billing: Final = _cascade_mod.billing_of(final, calls)
    served_by: Final = cfg.strong_model if escalated else cfg.cheap_model
    path: Final = "strong" if escalated else "cheap"
    reason: Final = "strong_error" if outcome.error is not None else verdict.reason
    decision: Final = strategy.final_decision(
        verdict, served_by=served_by, path=path, escalated=escalated, strong_error=outcome.error, calls=calls
    )
    _record_decision(kwargs, decision)
    hidden: Final = _cascade_mod.hidden_params_of(final)
    hidden["response_cost"] = billing.billed
    # visible to the caller without a spend-log database (the proxy forwards additional_headers)
    hidden["additional_headers"] = {
        **(_cascade_mod.mapping_or_none(hidden.get("additional_headers")) or {}),
        "x-litellm-cascade-path": path,
        "x-litellm-cascade-reason": reason or "pass",
        "x-litellm-cascade-served-by": served_by,
    }
    hidden["cascade"] = {
        "path": path,
        "reason": reason,
        "legs_settled": verdict.escalate,
        "verifier_error": verdict.verifier_error,
        "strong_error": outcome.error,
        "served_by": served_by,
        "calls": calls,
        "billed_cost": billing.billed,
        "total_cost": billing.total,
        "routing_decision": dict(decision),
    }
    _cascade_mod.sync_request_logging(
        logging_obj, final=final, billed_cost=billing.billed, fire_deferred=pending.owns_deferral
    )
    verbose_router_logger.info(
        "cascade %s: path=%s reason=%s served_by=%s billed=%.6f total=%.6f calls=%s",
        strategy.model_name,
        path,
        reason,
        served_by,
        billing.billed,
        billing.total,
        _cascade_mod.calls_summary(calls),
    )
    return CascadeOutcome(final, response)


def _record_decision(
    kwargs: dict[str, Any],  # mutable-ok: the attempt's kwargs
    decision: Mapping[str, object],
) -> None:
    """Through the recorder, like every other strategy: redaction, OTel span attributes, baseline stamp."""
    from litellm.router_rust.steps.prerouting import (
        _record_routing_decision,  # pyright: ignore[reportPrivateUsage]  # the recorder every strategy's decision goes through
    )

    _record_routing_decision(kwargs, decision)  # pyright: ignore[reportArgumentType]  # the strategy's StandardLoggingRoutingDecision


async def _strong_call(
    core: "RustCore",
    pending: CascadeAttempt,
    call: "AttemptCall",
    kwargs: dict[str, Any],  # mutable-ok: the attempt's kwargs; the strong leg shares their metadata buckets
    messages: object,
    cheap: ModelResponse,
) -> StrongOutcome:
    """Call the strong group for an escalated request, as the request's own continuation.

    The strong leg shares the request's logging object and metadata buckets, as a fallback hop does, and its
    provider call runs as an internal sub-call (``provider_call_scope``, under ``cascade_leg``): it logs no row of
    its own, while the calls its routing starts (a classifier, a silent experiment) log as their own; the core
    stamps the deployment that answered into the request's metadata (the request's own group and fallback stamps
    are put back), and the request's one row bills both legs. Returns the strong answer, or the error being served
    around when ``on_strong_error`` is ``cheap``, in which case the request is put back to the cheap call's state
    (its deployment, its priced cost). Otherwise the failure propagates, marked so the cooldown logic does not blame
    the cheap deployment (the strong call was routed, retried and cooled down on its own); the cheap leg's usage is
    counted against its deployment (router-core counts nothing for a failed attempt), the parked cheap success log
    is discarded, the request's failure is logged billing the paid cheap leg, and that leg is carried as a failed
    attempt's usage for whatever fallback answers next.
    """
    from litellm.responses.mcp.round_billing import first_round_billed_on_failure
    from litellm.router_utils.cooldown_handlers import mark_cascade_escalation_failure
    from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import async_io_token_reconcile_success

    strategy: Final = pending.strategy
    logging_obj: Final = pending.logging_obj
    cfg: Final = strategy.config
    strong_kwargs: Final[dict[str, Any]] = {  # mutable-ok: unpacked into acompletion
        **_cascade_mod.forwarded_request_params(call.served),
        **_cascade_mod.shared_request_kwargs(call.served, logging_obj),
        "model": cfg.strong_model,
        "messages": messages,
    }
    parent_otel_span: Final = get_parent_otel_span_from_kwargs(kwargs)
    cache: Final = core._cache  # pyright: ignore[reportPrivateUsage]  # the step serves its own core
    # The cheap deployment's itpm/otpm reservation is settled against the cheap usage now, before the strong
    # leg's routing would refund it as a stale attempt's and before the snapshot (so the snapshot holds no
    # reservation to put back); the strong leg's is settled against its own usage below, so the request's one
    # success log (summed usage) has nothing left to reconcile.
    await async_io_token_reconcile_success(cache, kwargs, cheap, parent_otel_span=parent_otel_span)
    await _attribute_cheap_spend(core, cheap, kwargs)
    state: Final = _cascade_mod.snapshot_request_state(kwargs, logging_obj)
    _cascade_mod.forget_cache_state(logging_obj)  # the strong leg is a fresh call, priced as one
    try:
        with first_round_billed_on_failure(logging_obj, cheap), _cascade_mod.cascade_leg(logging_obj):
            strong: Final = await core.acompletion(stream=False, **strong_kwargs)
    except Exception as exc:
        if cfg.on_strong_error != "cheap":
            _cascade_mod.restore_request_identity(kwargs, state)  # the failure row: this request, the strong deployment
            _cascade_mod.clear_attributed_spend(kwargs)  # the cheap leg is settled; a fallback's row pays its own way
            mark_cascade_escalation_failure(exc)
            await _count_cheap_leg(core, cfg.cheap_model, cheap)
            await _cascade_mod.fail_request_logging(logging_obj, exc, owns_deferral=pending.owns_deferral)
            raise
        _cascade_mod.restore_request_state(kwargs, logging_obj, state)  # the cheap call is what gets served
        strong_error: Final = f"{type(exc).__name__}: {str(exc)[:200]}"
        verbose_router_logger.warning(
            "cascade %s: strong %s failed, serving the cheap answer: %s",
            strategy.model_name,
            cfg.strong_model,
            strong_error,
        )
        return StrongOutcome(response=None, error=strong_error)
    finally:
        # also when the leg is cancelled (client gone): the request's row must still name the request
        _cascade_mod.restore_request_identity(kwargs, state)
    if not isinstance(strong, ModelResponse):
        raise TypeError(
            f"cascade {strategy.model_name}: strong group {cfg.strong_model!r} returned {type(strong).__name__}"
        )
    await async_io_token_reconcile_success(cache, kwargs, strong, parent_otel_span=parent_otel_span)
    return StrongOutcome(response=strong, error=None)


async def _attribute_cheap_spend(
    core: "RustCore",
    cheap: ModelResponse,
    kwargs: dict[str, Any],  # mutable-ok: the attempt's kwargs; the attributed-spend marker is stamped in place
) -> None:
    """Before the strong leg: the completed cheap leg is charged to its own deployment's and provider's budgets,
    whatever the strong leg then does. The spend is stamped as attributed so the request's one success row charges
    the deployment that answered only the rest."""
    from litellm.litellm_core_utils.get_llm_provider_logic import inferred_provider

    cheap_id, cheap_name = _cascade_mod.leg_deployment(cheap)
    hosted: Final = core._hosted  # pyright: ignore[reportPrivateUsage]  # the step serves its own core
    limiter: Final = hosted._core_get_router_deployment_budget_limiter()  # pyright: ignore[reportPrivateUsage]  # the host registry's budget callback
    deployment_metadata: Final = find_deployment_metadata(kwargs)
    if cheap_id is None or limiter is None or deployment_metadata is None:
        return
    cheap_cost: Final = _cascade_mod.inner_call_cost(cheap) or 0.0
    deployment: Final = hosted._core_get_deployment(model_id=cheap_id)  # pyright: ignore[reportPrivateUsage]  # the host registry's id lookup
    declared: Final = deployment.litellm_params.custom_llm_provider if deployment is not None else None
    await limiter.attribute_spend(
        model_id=cheap_id,
        custom_llm_provider=declared if isinstance(declared, str) else inferred_provider(cheap_name),
        response_cost=cheap_cost,
    )
    # Bound to this request's call id: a caller that reuses its metadata dict for the next request, or an
    # ordinary request after this one, must not have this request's cheap leg subtracted from its charge.
    deployment_metadata[ROUTER_BUDGET_ATTRIBUTED_SPEND_METADATA_KEY] = attributed_spend_marker(
        kwargs.get("litellm_call_id"), cheap_cost
    )


async def _count_cheap_leg(core: "RustCore", cheap_group: str, cheap: ModelResponse) -> None:
    """The cheap leg's usage (tokens and the request) against its deployment when the attempt is about to fail
    over the strong leg: router-core counts a served attempt from what it hands back, and nothing for a failed one."""
    cheap_id, cheap_name = _cascade_mod.leg_deployment(cheap)
    if cheap_id is None:
        return
    hosted: Final = core._hosted  # pyright: ignore[reportPrivateUsage]  # the step serves its own core
    group: Final = hosted._core_get_model_from_alias(cheap_group) or cheap_group  # pyright: ignore[reportPrivateUsage]  # the host registry's alias lookup
    report: Final = core._handle.report_success(  # pyright: ignore[reportPrivateUsage]  # see above
        cheap_id,
        float(_cascade_mod.total_tokens_of(cheap)),
        1.0,
        cheap_name,
        core.deployment_tracks_usage(cheap_id, group),
    )
    _ = await core._drive_cache_steps(report)  # pyright: ignore[reportPrivateUsage]  # see above


async def fail_judged_attempt(pending: CascadeAttempt, call: "AttemptCall", error: Exception) -> None:
    """The judgment raised somewhere other than the strong leg's own failure path: the request's failure is logged
    once (the strong leg's path already did, in which case the logging object says so) and the parked cheap success
    log is discarded, so a fallback can claim the request's one success log."""
    from litellm.router_rust.surface.health import failure_already_logged

    if failure_already_logged(call.kwargs, "async_failure"):
        return
    await _cascade_mod.fail_request_logging(pending.logging_obj, error, owns_deferral=pending.owns_deferral)


__all__ = [
    "CascadeAttempt",
    "CascadeOutcome",
    "cascade_for_request",
    "fail_judged_attempt",
    "judge_cascade_answer",
    "judging_cascade",
    "prepare_cascade_attempt",
    "provider_call_scope",
]
