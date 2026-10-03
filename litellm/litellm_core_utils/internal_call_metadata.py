"""Metadata a request forwards to the internal LLM sub-calls it triggers.

Internal calls retain the caller's routing identity. Ordinary classifier and embedding
calls also retain its billing identity; shadow evaluations project billing receipts onto
the evaluation creator without changing routing metadata. Two fields need special handling:

* ``user_api_key_budget_reservation`` (and the reservation nested inside
  ``user_api_key_auth``) belongs to the parent completion. If a sub-call's cost callback
  sees it, that callback finalizes the reservation and the parent's own callback then
  skips incrementing the key/team budget counters, losing the parent's spend.
  ``user_api_key_auth`` itself is kept, sanitized, because model access-group filtering
  needs it.
* The sub-call is stamped with ``INTERNAL_CALL_ORIGIN_METADATA_KEY`` so its spend log row
  records that it is not traffic the caller sent.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY, NON_INFERENCE_CALL_TYPES
from litellm.litellm_core_utils.initialize_dynamic_callback_params import initialize_standard_callback_dynamic_params
from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN, InternalCallOrigin

BUDGET_RESERVATION_METADATA_KEYS: Final = frozenset({"user_api_key_budget_reservation"})


@dataclass(frozen=True, slots=True)
class EvaluationBillingOwner:
    user_id: str
    user_model_max_budget: Mapping[str, object] | None = None
    max_budget: float | None = None
    spend: float = 0.0


EVALUATION_BILLING_OWNER_KEY: Final = "_evaluation_billing_owner"
EVALUATION_BUDGET_RESERVATION_KEY: Final = "_evaluation_budget_reservation"
_EVALUATION_BILLING_OWNER: Final[ContextVar[EvaluationBillingOwner | None]] = ContextVar(
    "evaluation_billing_owner", default=None
)
_BILLING_MAPPING: Final = TypeAdapter(Mapping[str, object])
_EMPTY_BILLING_FIELDS: Final[Mapping[str, object]] = MappingProxyType({})
_BILLING_IDENTITY_FIELDS: Final = frozenset(
    {"user_api_key", "user_api_end_user_max_budget", "team_id", "team_alias", "agent_id", "billing_agent_id"}
)


def get_evaluation_billing_owner() -> EvaluationBillingOwner | None:
    return _EVALUATION_BILLING_OWNER.get()


@contextmanager
def evaluation_billing_context(owner: EvaluationBillingOwner | None) -> Generator[None]:
    token: Final = _EVALUATION_BILLING_OWNER.set(owner)
    try:
        yield
    finally:
        _EVALUATION_BILLING_OWNER.reset(token)


def get_evaluation_billing_owner_from_kwargs(kwargs: Mapping[str, object]) -> EvaluationBillingOwner | None:
    owner: Final = kwargs.get(EVALUATION_BILLING_OWNER_KEY)
    return owner if isinstance(owner, EvaluationBillingOwner) else None


def _billing_mapping(value: object) -> Mapping[str, object] | None:
    return _BILLING_MAPPING.validate_python(value) if isinstance(value, Mapping) else None


def _evaluation_budget_reservation(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    handle: Final = kwargs.get(EVALUATION_BUDGET_RESERVATION_KEY)
    if handle is None:
        return None
    from litellm.proxy.spend_tracking.evaluation_budget import EvaluationBudgetReservation

    return handle.total if isinstance(handle, EvaluationBudgetReservation) else None


def _billing_metadata(
    metadata: Mapping[str, object] | None, owner: EvaluationBillingOwner, reservation: Mapping[str, object] | None
) -> Mapping[str, object]:
    return {
        **{
            key: None if key.startswith("user_api_key_") or key in _BILLING_IDENTITY_FIELDS else item
            for key, item in (metadata or _EMPTY_BILLING_FIELDS).items()
        },
        "user_api_key_user_id": owner.user_id,
        "user_api_key_user_model_max_budget": owner.user_model_max_budget,
        "user_api_key_budget_reservation": reservation,
        "tags": [],
    }


def _billing_request(value: object) -> object:
    request: Final = _billing_mapping(value)
    if request is None:
        return value
    body: Final = _billing_mapping(request.get("body"))
    if body is None:
        return dict(request)
    return {**request, "body": {**body, "user": None}}


def _billing_fields(
    fields: Mapping[str, object], owner: EvaluationBillingOwner, reservation: Mapping[str, object] | None
) -> Mapping[str, object]:
    alternate: Final = _billing_mapping(fields.get("litellm_metadata"))
    return {
        **fields,
        "agent_id": None,
        "billing_agent_id": None,
        **({"user": owner.user_id} if "user" in fields else {}),
        "metadata": _billing_metadata(_billing_mapping(fields.get("metadata")), owner, reservation),
        **({"litellm_metadata": _billing_metadata(alternate, owner, reservation)} if alternate else {}),
        **(
            {"proxy_server_request": _billing_request(fields["proxy_server_request"])}
            if "proxy_server_request" in fields
            else {}
        ),
    }


def project_evaluation_billing_kwargs(
    kwargs: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: existing SDK receipt consumers require dictionaries
    """Copy an evaluation's receipt onto its creator without changing request state."""
    owner: Final = get_evaluation_billing_owner_from_kwargs(kwargs)
    if owner is None:
        return kwargs if isinstance(kwargs, dict) else dict(kwargs)
    reservation: Final = _evaluation_budget_reservation(kwargs)
    params: Final = _billing_mapping(kwargs.get("litellm_params")) or _EMPTY_BILLING_FIELDS
    payload: Final = _billing_mapping(kwargs.get("standard_logging_object"))
    receipt_fields: Final[Mapping[str, object]] = {
        "user": owner.user_id,
        "end_user": None,
        "request_tags": [],
        "request_model_access_groups": (),
    }
    return {
        **_billing_fields(kwargs, owner, reservation),
        **receipt_fields,
        "user_api_key_end_user_id": None,
        "litellm_params": {
            **_billing_fields(params, owner, reservation),
            "user_api_key_end_user_id": None,
        },
        **(
            {
                "standard_logging_object": {
                    **_billing_fields(payload, owner, reservation),
                    **receipt_fields,
                }
            }
            if payload is not None
            else {}
        ),
    }


MODEL_ACCESS_GROUP_METADATA_KEY: Final = "user_api_key_matched_model_access_groups"
"""Where auth records the model access groups that authorized the request, for the spend writer.

The ``user_api_key`` prefix is load-bearing, not cosmetic: when a request carries both
``metadata`` and ``litellm_metadata``, ``get_litellm_metadata_from_kwargs`` returns the latter and
copies a key across only when ``user_api_key`` appears in its name."""

_USER_API_KEY_AUTH_KEY: Final = "user_api_key_auth"

FORWARDABLE_IDENTITY_METADATA_KEYS: Final = frozenset(
    {
        "user_api_key",
        "user_api_key_hash",
        "user_api_key_alias",
        "user_api_key_team_id",
        "user_api_key_org_id",
        "user_api_key_user_id",
        "user_api_key_end_user_id",
        _USER_API_KEY_AUTH_KEY,
    }
)
"""The caller-identity subset a detached sub-call needs to be attributed and
budget-checked like the request that spawned it. Everything else on the parent's metadata
(routing decision, guardrail state, logging payload) describes the parent call and would
be a lie on a sub-call that runs after it returned."""


def is_background_response(response: object) -> bool:
    """Whether a retrieved object is a response created with ``background=true``.

    Such a create returns ``status="queued"`` and no usage at all, so nothing has billed the
    job by the time anyone reads it back. Accepts the response as a mapping or a model,
    because the callers hold it in both shapes.
    """
    if isinstance(response, Mapping):
        return response.get("background") is True
    return getattr(response, "background", None) is True


def is_unbilled_non_inference_call(
    call_type: str | None,
    metadata: Mapping[str, object] | None,
    response: object,
) -> bool:
    """A read/management route priced at zero, because the usage it reports belongs to the
    call that created the object it just read.

    Retrieving a background response is the exception, and the enterprise cost poller's read
    is the same exception seen from the other side: that job's create billed nothing, so its
    retrieval is the only place the spend is ever visible. Pricing those at zero would lose
    the spend rather than deduplicate it.
    """
    if call_type not in NON_INFERENCE_CALL_TYPES:
        return False
    if is_background_response(response):
        return False
    if metadata is None:
        return True
    return metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY) != BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN


def is_unbilled_non_inference_call_from_params(
    call_type: str | None,
    litellm_params: Mapping[str, object] | None,
    response: object,
) -> bool:
    """:func:`is_unbilled_non_inference_call` for callers holding raw ``litellm_params``.

    The call-type membership test runs first so that inference traffic, which is every
    request in a normal workload, never pays for the metadata merge behind it.
    """
    if call_type not in NON_INFERENCE_CALL_TYPES:
        return False
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    metadata: Final = (
        StandardLoggingPayloadSetup.merge_litellm_metadata(litellm_params) if litellm_params is not None else None
    )
    return is_unbilled_non_inference_call(call_type, metadata, response)


def sanitize_user_api_key_auth(auth: object) -> object:
    """Copy of the auth object with its budget reservation removed; the cost callback
    falls back to reading the reservation from inside the auth object."""
    if isinstance(auth, dict):
        return {k: v for k, v in auth.items() if k != "budget_reservation"}
    reservation: Final[object] = getattr(auth, "budget_reservation", None)
    model_copy: Final[object] = getattr(auth, "model_copy", None)
    if reservation is not None and callable(model_copy):
        return model_copy(update={"budget_reservation": None})
    return auth


def _sanitized(parent_metadata: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: SDK metadata kwarg
    return {
        k: sanitize_user_api_key_auth(v) if k == _USER_API_KEY_AUTH_KEY else v
        for k, v in parent_metadata.items()
        if k not in BUDGET_RESERVATION_METADATA_KEYS
    }


def forwarded_internal_call_metadata(
    parent_metadata: Mapping[str, object] | None,
    call_origin: InternalCallOrigin,
) -> dict[str, object]:  # mutable-ok: SDK metadata kwarg
    """Parent metadata, minus its budget reservation, stamped with the sub-call's origin.

    For sub-calls made inside the parent request (classifier, embeddings), where the
    parent's full context still describes the call being made.
    """
    if not parent_metadata:
        return {}
    return _sanitized(parent_metadata) | {INTERNAL_CALL_ORIGIN_METADATA_KEY: call_origin}


def parent_session_kwargs(request_kwargs: Mapping[str, object] | None) -> Mapping[str, str]:
    kwargs: Final = request_kwargs or MappingProxyType({})
    return MappingProxyType(
        {k: v for k in ("litellm_session_id", "litellm_trace_id") if isinstance(v := kwargs.get(k), str)}
    )


def effective_turn_off_message_logging(request_kwargs: Mapping[str, object] | None) -> bool | None:
    return initialize_standard_callback_dynamic_params(dict(request_kwargs) if request_kwargs else None).get(
        "turn_off_message_logging"
    )


def sanitized_forwardable_call_metadata(
    parent_metadata: Mapping[str, object],
    call_origin: InternalCallOrigin,
) -> dict[str, object]:  # mutable-ok: SDK metadata kwarg
    """Just the caller's identity, stamped with the sub-call's origin.

    For sub-calls detached from the parent request (shadow eval), which outlive it and
    must not inherit per-request state such as its routing decision or logging payload.
    """
    identity: Final = {k: v for k, v in parent_metadata.items() if k in FORWARDABLE_IDENTITY_METADATA_KEYS}
    return _sanitized(identity) | {INTERNAL_CALL_ORIGIN_METADATA_KEY: call_origin}
