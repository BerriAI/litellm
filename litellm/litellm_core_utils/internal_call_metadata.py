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


EVALUATION_BILLING_OWNER_KEY: Final = "_evaluation_billing_owner"
_EVALUATION_BILLING_OWNER: Final[ContextVar[EvaluationBillingOwner | None]] = ContextVar(
    "evaluation_billing_owner", default=None
)
_BILLING_MAPPING: Final = TypeAdapter(Mapping[str, object])
_BILLING_DICT: Final = TypeAdapter(dict[str, object])
_BILLING_TAGS: Final = TypeAdapter(list[str])
_EMPTY_BILLING_FIELDS: Final[Mapping[str, object]] = MappingProxyType({})
_BILLING_IDENTITY_FIELDS: Final = frozenset(
    {"user_api_key", "user_api_end_user_max_budget", "team_id", "team_alias", "agent_id"}
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


def _billing_metadata(value: object, owner: EvaluationBillingOwner) -> Mapping[str, object]:
    metadata: Final = _BILLING_MAPPING.validate_python(value) if isinstance(value, Mapping) else _EMPTY_BILLING_FIELDS
    return _BILLING_DICT.validate_python(
        MappingProxyType(
            {
                **MappingProxyType(
                    {
                        key: None if key.startswith("user_api_key_") or key in _BILLING_IDENTITY_FIELDS else item
                        for key, item in metadata.items()
                    }
                ),
                "user_api_key_user_id": owner.user_id,
                "user_api_key_user_model_max_budget": owner.user_model_max_budget,
                "tags": _BILLING_TAGS.validate_python(()),
            }
        )
    )


def _billing_request(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    request: Final = _BILLING_MAPPING.validate_python(value)
    body_value: Final = request.get("body")
    if not isinstance(body_value, Mapping):
        return _BILLING_DICT.validate_python(request)
    body: Final = _BILLING_MAPPING.validate_python(body_value)
    return _BILLING_DICT.validate_python(
        MappingProxyType({**request, "body": _BILLING_DICT.validate_python(MappingProxyType({**body, "user": None}))})
    )


def project_evaluation_billing_kwargs(
    kwargs: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: existing SDK receipt consumers require dictionaries
    """Copy an evaluation's receipt onto its creator without changing request state."""
    owner: Final = get_evaluation_billing_owner_from_kwargs(kwargs)
    if owner is None:
        return kwargs if isinstance(kwargs, dict) else _BILLING_DICT.validate_python(kwargs)
    params_value: Final = kwargs.get("litellm_params")
    params: Final = (
        _BILLING_MAPPING.validate_python(params_value) if isinstance(params_value, Mapping) else _EMPTY_BILLING_FIELDS
    )
    alternate_value: Final = params.get("litellm_metadata")
    alternate_metadata: Final = (
        _BILLING_MAPPING.validate_python(alternate_value) if isinstance(alternate_value, Mapping) else None
    )
    payload_value: Final = kwargs.get("standard_logging_object")
    payload: Final = _BILLING_MAPPING.validate_python(payload_value) if isinstance(payload_value, Mapping) else None
    projected_params: Final = _BILLING_DICT.validate_python(
        MappingProxyType(
            {
                **params,
                **(MappingProxyType({"user": owner.user_id}) if "user" in params else _EMPTY_BILLING_FIELDS),
                "user_api_key_end_user_id": None,
                "metadata": _billing_metadata(params.get("metadata"), owner),
                # A truthy alternate bucket wins metadata resolution; keep empty
                # buckets empty so origin and model group stay on the selected one.
                **(
                    MappingProxyType({"litellm_metadata": _billing_metadata(alternate_metadata, owner)})
                    if alternate_metadata
                    else _EMPTY_BILLING_FIELDS
                ),
                **(
                    MappingProxyType({"proxy_server_request": _billing_request(params["proxy_server_request"])})
                    if "proxy_server_request" in params
                    else _EMPTY_BILLING_FIELDS
                ),
            }
        )
    )
    projected_payload: Final = (
        _BILLING_DICT.validate_python(
            MappingProxyType(
                {
                    **payload,
                    **(MappingProxyType({"user": owner.user_id}) if "user" in payload else _EMPTY_BILLING_FIELDS),
                    **(MappingProxyType({"agent_id": None}) if "agent_id" in payload else _EMPTY_BILLING_FIELDS),
                    "metadata": _billing_metadata(payload.get("metadata"), owner),
                    "end_user": None,
                    "request_tags": _BILLING_TAGS.validate_python(()),
                    "request_model_access_groups": (),
                }
            )
        )
        if payload is not None
        else None
    )
    return _BILLING_DICT.validate_python(
        MappingProxyType(
            {
                **kwargs,
                "user": owner.user_id,
                "agent_id": None,
                "end_user": None,
                "user_api_key_end_user_id": None,
                "request_tags": _BILLING_TAGS.validate_python(()),
                "request_model_access_groups": (),
                **MappingProxyType(
                    {
                        key: _billing_metadata(kwargs[key], owner)
                        for key in ("metadata", "litellm_metadata")
                        if isinstance(kwargs.get(key), Mapping) and kwargs[key]
                    }
                ),
                **(
                    MappingProxyType({"proxy_server_request": _billing_request(kwargs["proxy_server_request"])})
                    if "proxy_server_request" in kwargs
                    else _EMPTY_BILLING_FIELDS
                ),
                "litellm_params": projected_params,
                **(
                    MappingProxyType({"standard_logging_object": projected_payload})
                    if projected_payload is not None
                    else _EMPTY_BILLING_FIELDS
                ),
            }
        )
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
    return _sanitized(identity) | {INTERNAL_CALL_ORIGIN_METADATA_KEY: call_origin}  # mutable-ok: SDK metadata kwarg
