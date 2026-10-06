"""Metadata a request forwards to the internal LLM sub-calls it triggers.

Internal features (the auto-router's classifier and embeddings, shadow eval's shadow and
judge calls) bill real provider spend that nobody typed a prompt for. Sub-calls retain the
caller's runtime identity; an evaluation's billing owner is projected only at financial
consumers. Two things must never be forwarded as-is:

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
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY, NON_INFERENCE_CALL_TYPES
from litellm.litellm_core_utils.initialize_dynamic_callback_params import initialize_standard_callback_dynamic_params
from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN, InternalCallOrigin

BUDGET_RESERVATION_METADATA_KEYS: Final = frozenset({"user_api_key_budget_reservation"})
BILLING_USER_ID_METADATA_KEY: Final = "user_api_key_billing_user_id"
BILLING_MODEL_MAX_BUDGET_METADATA_KEY: Final = "user_api_key_billing_model_max_budget"
_METADATA_BUCKETS: Final = ("metadata", "litellm_metadata")
_OBJECT_MAPPING: Final = TypeAdapter(Mapping[str, object])
_EMPTY_METADATA: Final[Mapping[str, object]] = MappingProxyType({})
_BILLING_IDENTITY_FIELDS: Final = frozenset(
    {
        "user_api_key",
        "user",
        "end_user",
        "agent_id",
        "billing_agent_id",
        "tags",
        "request_tags",
        "team_id",
        "team_alias",
        "user_api_end_user_max_budget",
        "request_model_access_groups",
    }
)

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
        BILLING_USER_ID_METADATA_KEY,
        BILLING_MODEL_MAX_BUDGET_METADATA_KEY,
        _USER_API_KEY_AUTH_KEY,
    }
)
"""The caller-identity subset a detached sub-call needs to be attributed and
budget-checked like the request that spawned it. Everything else on the parent's metadata
(routing decision, guardrail state, logging payload) describes the parent call and would
be a lie on a sub-call that runs after it returned."""


def _mapping(value: object) -> Mapping[str, object]:
    return _OBJECT_MAPPING.validate_python(value) if isinstance(value, Mapping) else _EMPTY_METADATA


def _without_billing_identity(value: Mapping[str, object]) -> Mapping[str, object]:
    return {
        key: item
        for key, item in value.items()
        if not key.startswith("user_api_key_") and key not in _BILLING_IDENTITY_FIELDS
    }


def _billing_metadata(value: object, user_id: str, model_max_budget: object) -> Mapping[str, object]:
    return {
        **_without_billing_identity(_mapping(value)),
        "user_api_key_user_id": user_id,
        "user_api_key_user_model_max_budget": model_max_budget,
        BILLING_USER_ID_METADATA_KEY: user_id,
        BILLING_MODEL_MAX_BUDGET_METADATA_KEY: model_max_budget,
    }


def _billing_request(value: object) -> object:
    request: Final = _mapping(value)
    body: Final = _mapping(request.get("body"))
    if not isinstance(request.get("body"), Mapping):
        return value
    return {**request, "body": {key: item for key, item in body.items() if key != "user"}}


def billing_kwargs(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """A financial receipt for a server-stamped payer, leaving the shared runtime context intact."""
    params: Final = _mapping(kwargs.get("litellm_params"))
    owner_metadata: Final = next(
        (
            metadata
            for key in reversed(_METADATA_BUCKETS)
            if isinstance((metadata := _mapping(params.get(key))).get(BILLING_USER_ID_METADATA_KEY), str)
            and metadata.get(BILLING_USER_ID_METADATA_KEY)
        ),
        _EMPTY_METADATA,
    )
    user_id: Final = owner_metadata.get(BILLING_USER_ID_METADATA_KEY)
    if not isinstance(user_id, str) or not user_id:
        return kwargs
    model_max_budget: Final = owner_metadata.get(BILLING_MODEL_MAX_BUDGET_METADATA_KEY)
    metadata_updates: Final = {
        key: _billing_metadata(params[key], user_id, model_max_budget) for key in _METADATA_BUCKETS if params.get(key)
    }
    payload: Final = _mapping(kwargs.get("standard_logging_object"))
    projected_payload: Final[Mapping[str, object]] = (
        {
            "standard_logging_object": {
                **_without_billing_identity(payload),
                "metadata": _billing_metadata(payload.get("metadata"), user_id, model_max_budget),
                "end_user": None,
                "request_tags": [],
                "request_model_access_groups": [],
            }
        }
        if isinstance(kwargs.get("standard_logging_object"), Mapping)
        else {}
    )
    return {
        **_without_billing_identity(kwargs),
        "litellm_params": {
            **{key: item for key, item in params.items() if key != "user_api_key_end_user_id"},
            **metadata_updates,
            **(
                {"proxy_server_request": _billing_request(params["proxy_server_request"])}
                if "proxy_server_request" in params
                else {}
            ),
        },
        **projected_payload,
    }


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
