"""Metadata a request forwards to the internal LLM sub-calls it triggers.

Internal features (the auto-router's classifier and embeddings, shadow eval's shadow and
judge calls) bill real provider spend that nobody typed a prompt for. That spend must land
on the same key/team/org/user as the request that caused it, so the sub-call carries the
caller's identity metadata, minus two things that must never be forwarded as-is:

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

from collections.abc import Mapping
from typing import Final

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY, NON_INFERENCE_CALL_TYPES
from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN, InternalCallOrigin

BUDGET_RESERVATION_METADATA_KEYS: Final = frozenset({"user_api_key_budget_reservation"})

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
        return {k: v for k, v in auth.items() if k != "budget_reservation"}  # mutable-ok: SDK metadata value
    reservation: Final[object] = getattr(auth, "budget_reservation", None)
    model_copy: Final[object] = getattr(auth, "model_copy", None)
    if reservation is not None and callable(model_copy):
        return model_copy(update={"budget_reservation": None})  # mutable-ok: pydantic update payload
    return auth


def _sanitized(parent_metadata: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: SDK metadata kwarg
    return {  # mutable-ok: SDK metadata kwarg
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
        return {}  # mutable-ok: SDK metadata kwarg
    return _sanitized(parent_metadata) | {  # mutable-ok: SDK metadata kwarg
        INTERNAL_CALL_ORIGIN_METADATA_KEY: call_origin
    }


def sanitized_forwardable_call_metadata(
    parent_metadata: Mapping[str, object],
    call_origin: InternalCallOrigin,
) -> dict[str, object]:  # mutable-ok: SDK metadata kwarg
    """Just the caller's identity, stamped with the sub-call's origin.

    For sub-calls detached from the parent request (shadow eval), which outlive it and
    must not inherit per-request state such as its routing decision or logging payload.
    """
    identity: Final = {k: v for k, v in parent_metadata.items() if k in FORWARDABLE_IDENTITY_METADATA_KEYS}
    return _sanitized(identity) | {INTERNAL_CALL_ORIGIN_METADATA_KEY: call_origin}  # mutable-ok: SDK metadata kwarg
