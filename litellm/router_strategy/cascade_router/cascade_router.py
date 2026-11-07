"""Cascade router: cheap deployment first, verify the answer, escalate to the strong deployment.

The pre-call routers (complexity, quality, adaptive, semantic) pick a deployment before the
request is sent, and fallbacks react to errors. Neither looks at the answer. The cascade does:
the Router sends the request to `cheap_model`, this strategy judges the response, and the Router
calls `strong_model` when it falls short. The caller sees one request on the cascade's model
group, is charged once for the answers it took (cheap, plus strong on an escalation, on the request's
own log row, the way a fallback hop or a multi-round request is billed; the verifier is the decision's
`classifier_cost`, like the auto-router classifier), and gets a `routing_decision` that says which
path was taken and why.

Checks run cheapest first: an empty answer, a refusal, an answer under `min_chars`, or one that
stopped on the token limit escalates without paying for a verifier. Otherwise the optional
verifier is asked one question about the answer, through `Router.adecisions` for a decision
model or a one-word YES/NO prompt for a chat model. Every inner call is routed by the Router.

Non-streaming only: a streamed cheap answer has already reached the caller by the time it could
be judged, so streaming requests are served by the cheap deployment without verification.
"""

import copy
import datetime
import re
import time
import traceback
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final  # noqa: TID251  # request kwargs forwarded to acompletion are untyped

from pydantic import BaseModel, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_router_logger
from litellm.constants import (
    INTERNAL_CALL_ORIGIN_METADATA_KEY,
    KUBERNETES_POD_ROUTING_KEY,
    ROUTER_BUDGET_ATTRIBUTED_SPEND_CALL_DETAIL_KEY,
)
from litellm.litellm_core_utils.internal_call_metadata import (
    IO_TOKEN_RESERVATION_KEYS,
    effective_turn_off_message_logging,
    forwarded_internal_call_metadata,
    parent_session_kwargs,
)
from litellm.router_strategy.cascade_router.config import CascadeRouterConfig
from litellm.types.router import PreRoutingHookResponse
from litellm.types.utils import (
    CASCADE_VERIFIER_CALL_ORIGIN,
    Choices,
    CostBreakdown,
    ModelResponse,
    RoutingDecisionCause,
    StandardLoggingRoutingDecision,
    Usage,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.router_api import RoutingContext
    from litellm.router_strategy.savings_baseline import Baseline

# Request kwargs the strong call must not inherit: the cascade sets these itself, or they steer the
# Router rather than the model. Everything LiteLLM owns (metadata, logging object, routing stamps,
# credentials) is excluded by `filter_out_litellm_params`; the caller's own parameters pass.
_NOT_FORWARDED_TO_STRONG: Final = frozenset(
    {"model", "messages", "stream", "stream_options", "specific_deployment", "deployment", "priority"}
)
_SAVINGS_BASELINE_TTL_SECONDS: Final = 60.0

REFUSAL_RE: Final = re.compile(
    r"\b(i'?m sorry|i cannot|i can'?t|as an ai|i am unable|i'?m unable|cannot help with)\b", re.IGNORECASE
)
VERIFIER_CHAT_SYSTEM: Final = (
    "You judge whether an answer serves the user's request. {question} Reply with exactly one word: YES or NO."
)
# Request kwargs the strong leg shares with the request (the same objects): what makes it the request's own
# continuation rather than a new request. Its logging object is added when the request has one.
_SHARED_WITH_STRONG: Final = (
    "litellm_call_id",
    "litellm_session_id",
    "litellm_trace_id",
    "metadata",
    "litellm_metadata",
)
_METADATA_BUCKETS: Final = ("metadata", "litellm_metadata")
# LiteLLM-owned request kwargs that are the request's policies, not its credentials or routing: they hold on every
# call the cascade makes for the request (the strong leg, the verifier) as they held on the cheap call.
REQUEST_POLICY_PARAMS: Final = ("guardrails", "allowed_model_region", "turn_off_message_logging", "no-log", "cache")
# Stamps that describe the request rather than the leg that answered it; the strong leg's own routing rewrites
# them (its group, a fresh fallback count) and they are put back whatever the leg's outcome.
_REQUEST_IDENTITY_STAMPS: Final = ("model_group", "model_group_alias", "original_model_group", "attempted_fallbacks")
# Everything the strong leg's own routing writes into the request's metadata: the identity stamps, its deployment
# stamps, its retry bookkeeping and its itpm/otpm reservation. Restoring the cheap call's state puts these back and
# removes the ones the cheap call never had; anything else written meanwhile (a guardrail's record) is kept.
_LEG_STAMPS: Final = (
    *_REQUEST_IDENTITY_STAMPS,
    "deployment",
    "model_info",
    "api_base",
    "deployment_model_name",
    KUBERNETES_POD_ROUTING_KEY,
    "attempted_retries",
    "max_retries",
    "model_group_size",
    "previous_models",
    "request_retry_count",
    *IO_TOKEN_RESERVATION_KEYS,
)
VERIFIER_QUESTION_KEY: Final = "answered"
# Set while a cascade's strong leg runs, to the request's logging object the leg shares (or a sentinel when the
# request has none). Two readers: a cascade reached inside the leg is a cascade inside a cascade, which the Router
# refuses; and the leg's own provider call, the one carrying that logging object, is the one call that logs no
# row of its own. Calls the leg's routing starts (a classifier, a silent experiment) carry their own logging and
# log as usual. The generic internal-call flag is neither signal: other callers (the RAG pipeline) route through
# the Router under it with no cascade around them, and it would silence the leg's independent sub-calls.
_LEG_WITHOUT_LOG: Final = object()
_INSIDE_CASCADE_LEG: Final[ContextVar[object | None]] = ContextVar("litellm_inside_cascade_leg", default=None)


@contextmanager
def cascade_leg(logging_obj: "Logging | None") -> Generator[None]:
    """Marks the strong leg of a cascade, for `inside_cascade_leg` and `is_cascade_legs_own_call`."""
    token: Final = _INSIDE_CASCADE_LEG.set(logging_obj if logging_obj is not None else _LEG_WITHOUT_LOG)
    try:
        yield
    finally:
        _INSIDE_CASCADE_LEG.reset(token)


def inside_cascade_leg() -> bool:
    return _INSIDE_CASCADE_LEG.get() is not None


def is_cascade_legs_own_call(request_kwargs: Mapping[str, object]) -> bool:
    """True for the strong leg's own provider call: the Router call made inside the leg that carries the request's
    shared logging object. A classifier or a silent experiment started by the leg's routing carries its own."""
    leg: Final = _INSIDE_CASCADE_LEG.get()
    return leg is not None and leg is not _LEG_WITHOUT_LOG and request_kwargs.get("litellm_logging_obj") is leg


def as_float(value: object) -> float | None:
    """A numeric record value as a float; anything else (absent, None, text) is no value."""
    return float(value) if isinstance(value, (int, float)) else None


def as_int(value: object) -> int:
    """A numeric record value as an int; anything else counts as zero."""
    return int(value) if isinstance(value, (int, float)) else 0


_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_CONTENT_PARTS_ADAPTER: Final = TypeAdapter(list[object])


def mapping_or_none(value: object) -> Mapping[str, object] | None:
    """`value` as a string-keyed mapping (metadata, a routing decision, a header dict), else None."""
    try:
        mapping: Final = _MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None
    return mapping


def parent_metadata_of(request_kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    """The request's metadata for an inner call, from both buckets when both exist.

    `litellm_metadata` is layered over `metadata`, except the caller's identity keys (`user_api_key*`), which
    stay the proxy's (`metadata`) values, and `litellm_received_at`, carried over when `litellm_metadata` lacks
    it: the precedence `get_litellm_metadata_from_kwargs` applies to the request's own spend row, so the
    sub-call rows are attributed to the same key and team. A superset of that merge: every other `metadata` key is forwarded too,
    as it is when the request carries `metadata` alone."""
    metadata: Final = mapping_or_none(request_kwargs.get("metadata"))
    litellm_metadata: Final = mapping_or_none(request_kwargs.get("litellm_metadata"))
    if metadata is None or litellm_metadata is None:
        return litellm_metadata or metadata
    identity: Final = {
        key: value
        for key, value in metadata.items()
        if "user_api_key" in key or (key == "litellm_received_at" and key not in litellm_metadata)
    }
    return {**metadata, **litellm_metadata, **identity}


def needs_escalation(
    text: str,
    finish_reason: str | None,
    min_chars: int,
    escalate_on_refusal: bool = True,
    escalate_on_truncation: bool = True,
    caller_capped: bool = False,
    has_tool_calls: bool = False,
) -> str | None:
    """The cheapest reasons to skip a cheap answer, in the order they are cheapest to check.

    `caller_capped` is True when the request itself set `max_tokens` / `max_completion_tokens`: a
    truncation under the caller's own cap is not a model failure, and the strong model would be
    cut at the same point, so it never escalates."""
    stripped: Final = (text or "").strip()
    if has_tool_calls:
        # a tool-call answer has no text by design; only the length-cap check applies
        return "truncated" if escalate_on_truncation and finish_reason == "length" and not caller_capped else None
    if not stripped:
        return "empty"
    if escalate_on_refusal and REFUSAL_RE.search(stripped[:200]):
        return "refusal"
    if len(stripped) < min_chars:
        return "short"
    if escalate_on_truncation and finish_reason == "length" and not caller_capped:
        return "truncated"
    return None


def inner_call_cost(response: object, call_type: str | None = None) -> float | None:
    """Cost of one inner call. The Router attaches `response_cost` to hidden params only once the
    async success logger has run, which is after `acompletion` returns, so price the response
    ourselves when it is missing, with the deployment's own pricing (`router_model_id`). A
    decisions response is priced as one (`call_type="adecisions"`) so decisions pricing applies."""
    hidden: Final[Mapping[str, object]] = getattr(response, "_hidden_params", None) or {}
    cost: Final = as_float(hidden.get("response_cost"))
    if cost is not None:
        return cost
    model_id: Final = hidden.get("model_id")
    router_model_id: Final = model_id if isinstance(model_id, str) else None
    try:
        priced: Final = litellm.completion_cost(
            completion_response=response,
            custom_pricing=True if router_model_id else None,
            router_model_id=router_model_id,
            **({"call_type": call_type} if call_type is not None else {}),
        )
    except Exception:  # noqa: BLE001 - unpriced deployments are legal; the record then shows no cost
        return None
    return float(priced)


def request_policy_kwargs(
    request_kwargs: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: unpacked into a call
    """The request's policies, for a call the cascade makes on its behalf; `turn_off_message_logging` as the
    request's callbacks resolve it (set at the top level or in its metadata)."""
    policies: Final[dict[str, object]] = {  # mutable-ok: built below
        key: value for key in REQUEST_POLICY_PARAMS if (value := request_kwargs.get(key)) is not None
    }
    privacy: Final = effective_turn_off_message_logging(request_kwargs)
    if privacy is not None:
        policies["turn_off_message_logging"] = privacy
    return policies


def forwarded_request_params(
    request_kwargs: Mapping[str, object],
) -> dict[str, Any]:  # mutable-ok: unpacked into acompletion
    """The caller's own parameters, for the strong call: everything the cheap call was sent except
    what LiteLLM owns and what the cascade sets itself, so `n`, `response_format`, `reasoning_effort`,
    `extra_body`, provider-specific params and the rest shape the strong answer the same way."""
    from litellm.utils import filter_out_litellm_params

    return {
        key: value
        for key, value in filter_out_litellm_params(request_kwargs, excluding=_NOT_FORWARDED_TO_STRONG).items()
        if value is not None and not key.startswith("_")
    }


def hidden_params_of(
    response: ModelResponse,
) -> dict[str, object]:  # mutable-ok: the response's own dict, written in place
    """The response's `_hidden_params`, where LiteLLM keeps per-response metadata (cost, model id, headers)."""
    hidden: Final[dict[str, object] | None] = getattr(response, "_hidden_params", None)  # mutable-ok: its own dict
    if isinstance(hidden, dict):
        return hidden
    fresh: Final[dict[str, object]] = {}  # mutable-ok: becomes the response's hidden-params dict
    response._hidden_params = fresh  # pyright: ignore[reportPrivateUsage]  # rebind-ok: LiteLLM's own hidden-params slot
    return fresh


def detached_copy(response: ModelResponse) -> ModelResponse:
    """A shallow copy whose hidden params and usage are the cascade's to rewrite.

    The inner call's own response object is what its success log describes (on the SDK path that log is
    already queued when the cascade runs), so the sums and the `cascade` record go on a copy."""
    final: Final = copy.copy(response)
    final._hidden_params = dict(hidden_params_of(response))  # pyright: ignore[reportPrivateUsage]  # same slot, own dict
    usage: Final = getattr(response, "usage", None)
    if isinstance(usage, Usage):
        final.usage = copy.copy(usage)  # pyright: ignore[reportAttributeAccessIssue]  # ModelResponse sets `usage` dynamically
    return final


def parse_verdict(raw: str | None) -> bool:
    """Strict: only an answer that starts with YES passes. Anything else escalates."""
    return re.match(r"^\s*y(es)?\b", raw or "", re.IGNORECASE) is not None


def _first_choice(response: ModelResponse) -> Choices | None:
    """The first choice of a chat response, or None when it has none."""
    choices: Final[Sequence[object] | None] = getattr(response, "choices", None)
    first: Final = choices[0] if choices else None
    return first if isinstance(first, Choices) else None


def _text_of(response: ModelResponse) -> str:
    choice: Final = _first_choice(response)
    content: Final = choice.message.content if choice is not None else None
    return content or ""


def _has_tool_calls(response: ModelResponse) -> bool:
    choice: Final = _first_choice(response)
    return bool(choice is not None and choice.message.tool_calls)


def _finish_reason_of(response: ModelResponse) -> str | None:
    choice: Final = _first_choice(response)
    return choice.finish_reason if choice is not None else None


class VerifierUnusable(Exception):
    """The verifier answered, but not in a form the cascade can read (no answer, no choice field)."""


def _token_count(usage: object, *names: str) -> int:
    """The first of `names` set on the usage object, as an int (zero when none is)."""
    return next((as_int(value) for value in (getattr(usage, name, None) for name in names) if value is not None), 0)


def _usage_of(response: object) -> tuple[int, int]:
    """(prompt, completion) tokens of a chat response, or (input, output) of a decisions response."""
    usage: Final = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return _token_count(usage, "prompt_tokens", "input_tokens"), _token_count(
        usage, "completion_tokens", "output_tokens"
    )


def inner_call_record(role: str, model: str, response: object, call_type: str | None = None) -> Mapping[str, object]:
    """One inner call as the `cascade` record lists it: who answered, what it cost, how many tokens."""
    usage: Final = _usage_of(response)
    return {
        "role": role,
        "model": model,
        "cost": inner_call_cost(response, call_type=call_type),
        "prompt_tokens": usage[0],
        "completion_tokens": usage[1],
    }


def calls_summary(calls: Sequence[Mapping[str, object]]) -> Sequence[tuple[object, object]]:
    """(role, cost) per inner call, for the log line."""
    return [(call.get("role"), call.get("cost")) for call in calls]


def shared_request_kwargs(
    request_kwargs: Mapping[str, object], logging_obj: "Logging | None"
) -> dict[str, object]:  # mutable-ok: unpacked into acompletion
    """The strong leg continues the request: same logging object, call and session ids, and the same metadata
    dicts, so the Router's deployment stamps and the cascade's final decision land where the request's one row
    reads them, as they do on a fallback hop."""
    shared: Final[dict[str, object]] = {  # mutable-ok: the logging object is added below
        key: value for key in _SHARED_WITH_STRONG if (value := request_kwargs.get(key)) is not None
    }
    if logging_obj is not None:
        shared["litellm_logging_obj"] = logging_obj
    return shared


@dataclass(frozen=True)
class RequestState:
    """What the strong leg overwrites on the request it continues: its metadata buckets (deployment stamps, group,
    fallback bookkeeping) and the logging object's view of the call (model, params, priced breakdown), by slot."""

    buckets: Mapping[str, Mapping[str, object]]
    call_slots: Mapping[str, object]
    model_call_details: Mapping[str, object] | None


def snapshot_request_state(request_kwargs: Mapping[str, object], logging_obj: "Logging | None") -> RequestState:
    """Taken before the strong leg (and after the cheap leg's reservation is settled, so the snapshot holds no
    reservation to resurrect), so the request can be put back to the cheap call's state when the strong leg
    fails and the cheap answer is what gets served."""
    buckets: Final = {
        name: dict(bucket)
        for name in _METADATA_BUCKETS
        if (bucket := mapping_or_none(request_kwargs.get(name))) is not None
    }
    if logging_obj is None:
        return RequestState(buckets=buckets, call_slots={}, model_call_details=None)
    call_slots: Final[dict[str, object]] = {  # mutable-ok: a fresh snapshot
        "model": logging_obj.model,
        "litellm_params": logging_obj.litellm_params,  # pyright: ignore[reportUnknownMemberType]  # untyped dict slot upstream
        "optional_params": logging_obj.optional_params,  # pyright: ignore[reportUnknownMemberType]  # untyped dict slot upstream
        "custom_llm_provider": logging_obj.custom_llm_provider,
        "custom_pricing": logging_obj.custom_pricing,
        "completion_start_time": logging_obj.completion_start_time,
        "caching_details": logging_obj.caching_details,
        "cost_breakdown": logging_obj.cost_breakdown,
    }
    return RequestState(buckets=buckets, call_slots=call_slots, model_call_details=dict(logging_obj.model_call_details))


def restore_request_identity(request_kwargs: Mapping[str, object], state: RequestState) -> None:
    """One request on the cascade's alias, whichever deployment answered (`deployment_model_name` says which):
    the group and fallback stamps go back, the strong leg's deployment stamps stay."""
    for name, saved in state.buckets.items():
        bucket = request_kwargs.get(name)
        if isinstance(bucket, dict):
            bucket.update({key: saved[key] for key in _REQUEST_IDENTITY_STAMPS if key in saved})  # pyright: ignore[reportUnknownMemberType]  # the request's own metadata dict
            for key in _REQUEST_IDENTITY_STAMPS:
                if key not in saved:
                    bucket.pop(key, None)  # pyright: ignore[reportUnknownMemberType]  # as above


def restore_request_state(
    request_kwargs: Mapping[str, object], logging_obj: "Logging | None", state: RequestState
) -> None:
    """The strong leg failed and the cheap answer is served: the request is the cheap call again, so its
    deployment stamps, its priced cost and the logging object's view of the call go back to the cheap call's.
    Only what was snapshotted is put back (and the leg stamps the cheap call never had are removed), so a record
    something else wrote in the meantime, a guardrail's for one, is kept."""
    for name, saved in state.buckets.items():
        bucket = request_kwargs.get(name)
        if isinstance(bucket, dict):
            bucket.update(saved)  # pyright: ignore[reportUnknownMemberType]  # the request's own metadata dict
            for key in _LEG_STAMPS:
                if key not in saved:
                    bucket.pop(key, None)  # pyright: ignore[reportUnknownMemberType]  # as above
    if logging_obj is None or state.model_call_details is None:
        return
    for name, value in state.call_slots.items():
        setattr(logging_obj, name, value)  # back to the cheap call's view of the call
    logging_obj.model_call_details.update(state.model_call_details)


def forget_cache_state(logging_obj: "Logging | None") -> None:
    """The cheap leg may have been served from the response cache; the strong leg is a fresh call and must be
    priced as one, so the shared logging object forgets the cheap leg's cache result before the strong leg runs
    (the snapshot, taken first, keeps it for the case where the cheap answer is what gets served)."""
    if logging_obj is None:
        return
    logging_obj.model_call_details.pop("cache_hit", None)
    logging_obj.reset_caching_details()


def clear_attributed_spend(logging_obj: "Logging | None") -> None:
    """The cascade attempt failed: its cheap leg was charged to its own deployment directly and its failure row never
    reaches the budget limiter, so the attributed-spend marker must not ride into a fallback attempt's row (which
    would be charged its cost minus the cheap leg's)."""
    if logging_obj is not None:
        _ = logging_obj.model_call_details.pop(ROUTER_BUDGET_ATTRIBUTED_SPEND_CALL_DETAIL_KEY, None)


def leg_deployment(response: ModelResponse) -> tuple[str | None, str | None]:
    """(deployment id, provider model name) of the deployment that produced a leg, from its hidden params."""
    hidden: Final = hidden_params_of(response)
    model_id: Final = hidden.get("model_id")
    name: Final = hidden.get("litellm_model_name")
    return (model_id if isinstance(model_id, str) else None, name if isinstance(name, str) else None)


def total_tokens_of(response: object) -> int:
    prompt_tokens, completion_tokens = _usage_of(response)
    return prompt_tokens + completion_tokens


@dataclass(frozen=True)
class Billing:
    """What the request is charged: the answers it took, and the total including the verifier."""

    billed: float
    total: float


def _cost_of(calls: Sequence[Mapping[str, object]], *roles: str) -> float:
    return float(
        sum(
            cost
            for cost in (as_float(call.get("cost")) for call in calls if call.get("role") in roles)
            if cost is not None
        )
    )


def billing_of(final: ModelResponse, calls: Sequence[Mapping[str, object]]) -> Billing:
    """What the request is charged: the answers it took (cheap, plus strong on an escalation), as priced on the
    response the caller gets; `total` adds the verifier, which is billed on its own row and attributed through the
    decision's `classifier_cost`, like the auto-router classifier."""
    priced: Final = as_float(hidden_params_of(final).get("response_cost"))
    billed: Final = priced if priced is not None else _cost_of(calls, "cheap", "strong")
    return Billing(billed=billed, total=billed + _cost_of(calls, "verifier"))


def breakdown_of(logging_obj: "Logging | None") -> CostBreakdown | None:
    """The cost breakdown of the leg the request's logging object last priced."""
    return None if logging_obj is None else logging_obj.cost_breakdown


def record_summed_breakdown(
    logging_obj: "Logging | None",
    cheap: ModelResponse,
    cheap_breakdown: CostBreakdown | None,
    strong: ModelResponse,
    strong_breakdown: CostBreakdown | None,
) -> None:
    """The request's breakdown is its legs' breakdowns summed, each as priced, the fixed per-request margin once;
    None when a leg was priced without one, so the breakdown never disagrees with the billed cost (the rule
    upstream's two-round billing follows)."""
    if logging_obj is None:
        return
    from litellm.responses.mcp.round_billing import summed_cost_breakdown

    logging_obj.cost_breakdown = summed_cost_breakdown(  # rebind-ok: the request's breakdown now covers both legs
        first=cheap, first_breakdown=cheap_breakdown, final=strong, final_breakdown=strong_breakdown
    )


def escalated_response(
    cheap: ModelResponse,
    cheap_breakdown: CostBreakdown | None,
    strong: ModelResponse,
    strong_breakdown: CostBreakdown | None,
) -> ModelResponse:
    """What the caller gets after an escalation: the strong answer carrying the usage and cost of both legs, each
    leg as it was priced (not the summed usage at one model's rates), the fixed margin once, under the request's
    own id. A copy: the Router's own response objects stay as their calls left them."""
    from litellm.responses.mcp.round_billing import billed_for_every_round

    final: Final = billed_for_every_round(
        first=cheap, first_breakdown=cheap_breakdown, final=strong, final_breakdown=strong_breakdown
    )
    final.id = cheap.id  # the id the request's logging started with
    return final


async def fail_request_logging(logging_obj: "Logging | None", exc: Exception, owns_deferral: bool) -> None:
    """The strong leg failed and the request fails with it (`on_strong_error: raise`).

    The parked cheap success log is discarded, the failure is logged the way `litellm.acompletion`'s wrapper
    logs a failed attempt (the failing call was an internal sub-call, so the wrapper did not), billing the cheap
    leg that `first_round_billed_on_failure` recorded, and that leg is then carried as a failed attempt's usage:
    a fallback attempt on the same request logs its own answer and cost, while the request's failure log, if the
    fallback fails too, still bills every attempt. The same hand-off upstream makes between attempts."""
    if logging_obj is None:
        return
    discard_parked_success_log(logging_obj, owns_deferral)
    trace: Final = traceback.format_exc()
    logging_obj.failure_handler(exc, trace, for_async_request=True)
    await logging_obj.async_failure_handler(exc, trace)
    logging_obj.carry_failed_attempt_usage()


def park_success_log(logging_obj: "Logging | None") -> bool:
    """Defer the request's success log so the cascade can re-point it at the final answer before it fires.

    False when there is no logging object or it is already deferred (the proxy parks it itself)."""
    if logging_obj is None or getattr(logging_obj, "defer_async_logging", False):
        return False
    logging_obj.defer_async_logging = True  # rebind-ok: the cascade owns the deferral now
    return True


def sync_request_logging(
    logging_obj: "Logging | None",
    final: object,
    billed_cost: float,
    fire_deferred: bool,
) -> None:
    """Make the request's logging object describe the cascade's final answer.

    The Router binds the request's `litellm_logging_obj` to the first (cheap) deployment call, so by the
    time the cascade escalates, that object already parked a success log for the cheap result
    (`deferred_success_result` plus the `enqueue_deferred_logging` closure that captured it) and
    recorded the cheap cost. The strong leg then ran on the same object (its model and deployment are
    already those that answered). Re-point the parked log at the final response and the billed cost,
    drop any cached cost preparation, and fire the parked log when nobody else will (SDK path; the
    proxy fires it itself after its post-call hooks)."""
    if logging_obj is None:
        return
    _repoint_deferred_success_log(logging_obj, final)
    details: Final = getattr(logging_obj, "model_call_details", None)
    if isinstance(details, dict):
        details["response_cost"] = billed_cost
    for attr in ("prepared_cost", "finalized_cost"):
        if hasattr(logging_obj, attr):
            setattr(logging_obj, attr, None)
    if fire_deferred:
        release_deferred_logging(logging_obj)


def _repoint_deferred_success_log(logging_obj: "Logging", final: object) -> None:
    """Re-point the parked success log (result, start, end) and its enqueue closure at the final answer."""
    deferred: Final = getattr(logging_obj, "deferred_success_result", None)
    if deferred is None:
        return
    start: Final = deferred[1]
    end: Final = datetime.datetime.now()  # noqa: DTZ005  # the parked start time is on the same naive clock
    logging_obj.deferred_success_result = (final, start, end)  # rebind-ok: parked log, final answer
    if getattr(logging_obj, "enqueue_deferred_logging", None) is not None:
        enqueue: Final = _deferred_success_log_for(logging_obj, final, start, end)
        logging_obj.enqueue_deferred_logging = enqueue  # rebind-ok: closure re-made for the final answer


def release_deferred_logging(logging_obj: "Logging | None") -> None:
    """Fire a success log the cascade parked but has no final answer to swap in (streaming, no judge)."""
    if logging_obj is None:
        return
    enqueue: Final = getattr(logging_obj, "enqueue_deferred_logging", None)
    logging_obj.enqueue_deferred_logging = None  # rebind-ok: parked log fires now
    if hasattr(logging_obj, "defer_async_logging"):
        logging_obj.defer_async_logging = False  # rebind-ok: deferral ends here
    if enqueue is not None:
        enqueue()


def discard_parked_success_log(logging_obj: "Logging", owns_deferral: bool) -> None:
    """The request is failing after the cheap call succeeded: the parked cheap success log must never fire, and the
    one async success log is given back so a fallback attempt that answers next can claim it. The proxy keeps its
    deferral (it fires the fallback's parked log after its post-call hooks); the cascade's own ends here."""
    logging_obj.deferred_success_result = None  # rebind-ok: parked result dropped
    logging_obj.enqueue_deferred_logging = None  # rebind-ok: parked log dropped
    logging_obj.release_async_success_log()
    if owns_deferral:
        logging_obj.defer_async_logging = False  # rebind-ok: the cascade's deferral ends here


def _deferred_success_log_for(
    logging_obj: "Logging", result: object, start: datetime.datetime, end: datetime.datetime
) -> Callable[[], None]:
    """The same closure `litellm.utils._schedule_async_success_logging` parks, built for `result`."""

    def _enqueue() -> None:
        import asyncio

        from litellm.responses.streaming_iterator import spawn_logging_task
        from litellm.utils import (
            _client_async_logging_helper,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # the parked closure uses it
        )

        if getattr(logging_obj, "defer_async_logging", False):
            logging_obj.handle_sync_success_callbacks_for_async_calls(result=result, start_time=start, end_time=end)
        logging_obj.mark_async_success_log_enqueued()
        spawn_logging_task(
            asyncio.get_running_loop(),
            _client_async_logging_helper(
                logging_obj=logging_obj,
                result=result,
                start_time=start,
                end_time=end,
                is_completion_with_fallbacks=False,
            ),
            task_name="cascade success callback enqueue",
        )

    return _enqueue


def _content_text(content: object) -> str:
    """A message's text: the string itself, or the text parts of a content list joined by spaces."""
    if isinstance(content, str):
        return content
    try:
        parts: Final = _CONTENT_PARTS_ADAPTER.validate_python(content)
    except ValidationError:
        return ""
    texts: Final = (
        str(part["text"])
        for part in (mapping_or_none(item) for item in parts)
        if part is not None and part.get("type") == "text" and isinstance(part.get("text"), str)
    )
    return " ".join(texts)


def _turn_mapping(message: object) -> Mapping[str, object] | None:
    """A message-history entry as a mapping: a dict as is, a pydantic message (LiteLLM's `Message`, an SDK
    object an agent loop appended) through `model_dump`, anything else None."""
    if isinstance(message, BaseModel):
        return mapping_or_none(message.model_dump())
    return mapping_or_none(message)


def last_user_text(messages: Sequence[object] | None, max_chars: int = 4000) -> str:
    """The latest user turn's text, for the verifier's question.

    The cascade reads the caller's list as `_acompletion` holds it, before LiteLLM's own message conversion,
    so entries may be dicts or pydantic message objects; anything else is skipped."""
    for message in reversed(messages or []):
        turn = _turn_mapping(message)
        if turn is not None and turn.get("role") == "user":
            return _content_text(turn.get("content"))[:max_chars]
    return ""


@dataclass(frozen=True)
class CascadeVerdict:
    """What the cascade decided about one cheap answer."""

    reason: str | None  # None = keep the cheap answer
    cause: RoutingDecisionCause
    calls: tuple[Mapping[str, object], ...] = ()  # role/model/cost/tokens per inner call
    verifier_error: str | None = None  # why the verifier could not be used, when that is the story

    @property
    def escalate(self) -> bool:
        return self.reason is not None


@dataclass(frozen=True)
class StrongOutcome:
    """What the strong call produced: the answer, or the error being served around (`on_strong_error: cheap`)."""

    response: ModelResponse | None
    error: str | None


class CascadeRouter:
    """`PreRoutingStrategy` that routes to the cheap group and judges the answer afterwards."""

    def __init__(
        self,
        model_name: str,
        ctx: "RoutingContext | None",
        cascade_router_config: Mapping[str, object] | None,
    ) -> None:
        self.model_name = model_name
        self.ctx = ctx
        self.config: Final = CascadeRouterConfig.model_validate(dict(cascade_router_config or {}))
        if self.model_name in (self.config.cheap_model, self.config.strong_model):
            raise ValueError(
                f"cascade_router_config: {self.model_name!r} cannot be its own cheap_model or strong_model"
            )
        self._savings_baseline: Baseline | None = None
        self._savings_baseline_resolved_at: float | None = None

    def _router(self) -> "RoutingContext":
        if self.ctx is None:
            raise VerifierUnusable(f"cascade {self.model_name}: no Router to route the verifier call through")
        return self.ctx

    def savings_baseline(self) -> "Baseline | None":
        """The counterfactual: the strong group's most expensive deployment, what every request would
        have cost without the cascade. Cached briefly, like the complexity router's, so a deployment
        added or removed at runtime still moves it without a pool walk per request."""
        if self.ctx is None:
            return None
        from litellm.router_strategy.savings_baseline import resolve_baseline

        now: Final = time.monotonic()
        if (
            self._savings_baseline_resolved_at is None
            or now - self._savings_baseline_resolved_at > _SAVINGS_BASELINE_TTL_SECONDS
        ):
            self._savings_baseline = resolve_baseline(self.ctx, (self.config.strong_model,))
            self._savings_baseline_resolved_at = now
        return self._savings_baseline

    def base_decision(
        self, routed_model: str, tier: str, cause: RoutingDecisionCause, escalated: bool
    ) -> StandardLoggingRoutingDecision:
        decision: Final[StandardLoggingRoutingDecision] = {
            "router_model_name": self.model_name,
            "router_type": "cascade",
            "routed_model": routed_model,
            "tier": tier,
            "cause": cause,
            "escalated": escalated,
        }
        # The baseline is stamped as the savings calculator prices it: the resolved provider model (a group
        # name like `strong` cannot be priced) with its deployment, as the complexity router stamps its own.
        baseline: Final = self.savings_baseline()
        if baseline is not None:
            decision["savings_baseline_model"] = baseline.model
            if baseline.deployment_id is not None:
                decision["savings_baseline_deployment_id"] = baseline.deployment_id
        return decision

    def final_decision(
        self,
        verdict: CascadeVerdict,
        served_by: str,
        path: str,
        escalated: bool,
        strong_error: str | None,
        calls: Sequence[Mapping[str, object]],
    ) -> StandardLoggingRoutingDecision:
        """The request's routing decision once the cascade is done: path, cause, the check that escalated, and
        the verifier as the decision's classifier (its cost is `classifier_cost`, like the auto-router's)."""
        decision: Final = self.base_decision(
            routed_model=served_by,
            tier=path,
            cause="cascade_strong_error" if strong_error is not None else verdict.cause,
            escalated=escalated,
        )
        if verdict.reason is not None:
            decision["escalation_reason"] = verdict.reason  # the check that asked for the strong model
        verifier_call: Final = next((call for call in calls if call.get("role") == "verifier"), None)
        if verifier_call is not None:
            decision["classifier_model"] = str(verifier_call["model"])
            verifier_cost: Final = as_float(verifier_call.get("cost"))
            if verifier_cost is not None:
                decision["classifier_cost"] = verifier_cost
        if verdict.verifier_error is None:
            return decision
        return {
            **decision,
            "classifier_failure_reason": "classifier_error",
            "classifier_error_type": verdict.verifier_error,
        }

    # ---- pre-routing: always the cheap group ------------------------------------------------

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: dict[str, Any],  # mutable-ok: PreRoutingStrategy protocol signature
        messages: list[dict[str, Any]] | None = None,  # mutable-ok: PreRoutingStrategy protocol signature
        input: str | list[Any] | None = None,  # mutable-ok: PreRoutingStrategy protocol signature
        specific_deployment: bool | None = False,
        call_type: str | None = None,
    ) -> PreRoutingHookResponse | None:
        decision: Final = self.base_decision(
            routed_model=self.config.cheap_model, tier="cheap", cause="cascade_cheap_first", escalated=False
        )
        return PreRoutingHookResponse(model=self.config.cheap_model, messages=messages, routing_decision=decision)

    # ---- post-response: judge the cheap answer ----------------------------------------------

    async def async_verify(
        self,
        messages: Sequence[Mapping[str, object]] | None,
        response: ModelResponse,
        caller_capped: bool = False,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> CascadeVerdict:
        cfg: Final = self.config
        reason: Final = needs_escalation(
            _text_of(response),
            _finish_reason_of(response),
            cfg.min_chars,
            escalate_on_refusal=cfg.escalate_on_refusal,
            escalate_on_truncation=cfg.escalate_on_truncation,
            caller_capped=caller_capped,
            has_tool_calls=_has_tool_calls(response),
        )
        if reason is not None:
            return CascadeVerdict(reason=reason, cause="cascade_heuristic_escalation")
        if cfg.verifier_model is None or _has_tool_calls(response):
            # nothing to judge in a tool call; the caller's loop decides what to do with it
            return CascadeVerdict(reason=None, cause="cascade_pass")
        try:
            passed, call = await self._ask_verifier(last_user_text(messages), _text_of(response), request_kwargs)
        except Exception as exc:  # noqa: BLE001 - the verifier is advisory; the config says what its failure means
            verbose_router_logger.warning(
                "cascade %s: verifier %s unusable: %s", self.model_name, cfg.verifier_model, exc
            )
            if cfg.on_verifier_error == "accept":
                return CascadeVerdict(reason=None, cause="cascade_pass", verifier_error=str(exc)[:200])
            return CascadeVerdict(
                reason="verifier_error", cause="cascade_verifier_error", verifier_error=str(exc)[:200]
            )
        if passed:
            return CascadeVerdict(reason=None, cause="cascade_pass", calls=(call,))
        return CascadeVerdict(reason="verifier_no", cause="cascade_verifier_escalation", calls=(call,))

    async def _ask_verifier(
        self,
        question: str,
        answer: str,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> tuple[bool, Mapping[str, object]]:
        cfg: Final = self.config
        if cfg.verifier_model is None:
            raise VerifierUnusable(f"cascade {self.model_name}: no verifier_model configured")
        router: Final = self._router()
        parent: Final = request_kwargs or {}
        # The cheap deployment's itpm/otpm reservation stays with the request, whose legs settle it against the
        # cheap usage; a copy on the verifier's call would be refunded at its deployment pick as a stale attempt's,
        # and the settlement of the original would then take that refund a second time.
        forwarded: Final = forwarded_internal_call_metadata(parent_metadata_of(parent), CASCADE_VERIFIER_CALL_ORIGIN)
        metadata: Final = {
            **{key: value for key, value in forwarded.items() if key not in IO_TOKEN_RESERVATION_KEYS},
            INTERNAL_CALL_ORIGIN_METADATA_KEY: CASCADE_VERIFIER_CALL_ORIGIN,  # stamped also when the request has none
            "cascade_router": self.model_name,
        }
        call_kwargs: Final[dict[str, Any]] = {  # mutable-ok: unpacked into the verifier call
            "metadata": metadata,
            # the verifier's own call and row carry the request and the cheap answer: the request's region
            # restriction, logging opt-outs, cache policy and guardrails hold for it
            **request_policy_kwargs(parent),
            **parent_session_kwargs(parent),
        }
        prompt: Final = f"Request:\n{question}\n\nAnswer:\n{answer}"
        if cfg.verifier_kind == "decision":
            return await self._ask_decision_verifier(router, cfg.verifier_model, prompt, call_kwargs)
        return await self._ask_chat_verifier(router, cfg.verifier_model, prompt, call_kwargs)

    async def _ask_decision_verifier(
        self, router: "RoutingContext", verifier_model: str, prompt: str, call_kwargs: Mapping[str, Any]
    ) -> tuple[bool, Mapping[str, object]]:
        cfg: Final = self.config
        decision_response: Final = await _route_decisions(
            router,
            {
                "model": verifier_model,
                "input": prompt,
                "questions": {
                    VERIFIER_QUESTION_KEY: {
                        "type": "choice",
                        "instructions": cfg.verifier_question,
                        "criteria": {"yes": "the answer serves the request", "no": "it does not"},
                    }
                },
                "num_retries": 0,
                "disable_fallbacks": True,
                **call_kwargs,
            },
        )
        answers: Final = mapping_or_none(getattr(decision_response, "answers", None)) or {}
        verdict: Final = answers.get(VERIFIER_QUESTION_KEY)
        choice: Final = getattr(verdict, "choice", None)
        if not isinstance(choice, str) or not choice:
            raise VerifierUnusable(f"decision verifier returned no choice for {VERIFIER_QUESTION_KEY!r}")
        confidence: Final = as_float(getattr(verdict, "confidence", 1.0)) or 0.0
        passed: Final = choice.lower() == "yes" and confidence >= cfg.verifier_min_confidence
        record: Final = {
            **inner_call_record("verifier", verifier_model, decision_response, call_type="adecisions"),
            "choice": choice,
            "confidence": confidence,
        }
        return passed, record

    async def _ask_chat_verifier(
        self, router: "RoutingContext", verifier_model: str, prompt: str, call_kwargs: Mapping[str, Any]
    ) -> tuple[bool, Mapping[str, object]]:
        chat_response: Final = await router.acompletion(  # pyright: ignore[reportUnknownMemberType]  # untyped **kwargs upstream
            model=verifier_model,
            messages=[
                {"role": "system", "content": VERIFIER_CHAT_SYSTEM.format(question=self.config.verifier_question)},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            max_tokens=2,
            temperature=0,
            drop_params=True,  # reasoning models reject temperature; a dropped param beats a failed verifier
            **call_kwargs,
        )
        if not isinstance(chat_response, ModelResponse):
            raise VerifierUnusable("chat verifier returned a stream")
        text: Final = _text_of(chat_response)
        if not text.strip():
            raise VerifierUnusable("chat verifier returned no text")
        return parse_verdict(text), inner_call_record("verifier", verifier_model, chat_response)


async def _route_decisions(router: "RoutingContext", call_kwargs: Mapping[str, Any]) -> object:
    """`RoutingContext.adecisions` as an opaque object: its return type is untyped upstream, the fields are read by name."""
    return await router.adecisions(**call_kwargs)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # untyped upstream
