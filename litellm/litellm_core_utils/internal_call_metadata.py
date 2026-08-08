"""Metadata a request forwards to the internal LLM sub-calls it triggers.

Internal features (the auto-router's classifier and its keyword embeddings, shadow
eval's shadow + judge calls) bill real provider spend that nobody typed a prompt for.
That spend must land on the same key/team/org/user as the request that caused it, so
the sub-call has to carry the caller's identity metadata: the proxy's cost callback
reads ``user_api_key`` / ``user_api_key_team_id`` / ... straight off the call's
metadata, and drops the spend log entirely when they are missing.

Two things must never be forwarded as-is:

* ``user_api_key_budget_reservation`` (and the same reservation nested inside
  ``user_api_key_auth``) belongs to the parent completion, not to the sub-call. If a
  sub-call's cost callback sees it, that callback finalizes the reservation and the
  parent's own callback then skips incrementing the key/team budget counters, losing
  the parent's spend. ``user_api_key_auth`` itself is kept, sanitized, because model
  access-group filtering needs it.
* The origin of the sub-call. ``INTERNAL_CALL_ORIGIN_METADATA_KEY`` stamps which
  feature made the call so a spend-log row can say it is not traffic the caller sent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.types.utils import InternalCallOrigin

BUDGET_RESERVATION_METADATA_KEYS: Final = frozenset({"user_api_key_budget_reservation"})

_USER_API_KEY_AUTH_KEY: Final = "user_api_key_auth"

# The caller-identity subset a detached sub-call needs to be attributed and
# budget-checked exactly like the request that spawned it. Everything else on the
# parent's metadata (routing decisions, guardrail state, the standard logging object,
# the proxy request body) describes the parent call and would be a lie on a sub-call
# that runs after it has already returned.
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


def sanitize_user_api_key_auth(auth: object) -> object:
    """Copy of the auth object with its budget reservation removed.

    The proxy's cost callback falls back to reading the reservation from inside the
    auth object when the top-level key is absent, so forwarding it unsanitized
    re-creates the double-finalization that stripping the top-level key prevents.
    """
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

    For sub-calls made *inside* the parent request (the auto-router classifier and its
    embeddings), where the parent's full context still describes the call being made.
    A parent with no metadata at all had nothing to attribute in the first place, so
    the sub-call is left unstamped rather than carrying a lone origin marker.
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
    The origin stamp is unconditional here: these calls are always internal, whether or
    not there was an identity to forward.
    """
    identity: Final = {k: v for k, v in parent_metadata.items() if k in FORWARDABLE_IDENTITY_METADATA_KEYS}
    return _sanitized(identity) | {INTERNAL_CALL_ORIGIN_METADATA_KEY: call_origin}  # mutable-ok: SDK metadata kwarg
