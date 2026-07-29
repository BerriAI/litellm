"""The single translation layer between a request's metadata and the spans.

Every field is parsed **once**, here, out of the ``StandardLoggingPayload`` (or a
``UserAPIKeyAuth`` at the auth boundary), so span data, baggage, and the mappers
read typed fields instead of the raw ``metadata`` / ``hidden_params`` dicts.

:class:`RequestIdentity` holds caller identity (team/key/end-user), seeded into
Baggage at the auth boundary before routing has picked a deployment; its
``provider_model`` is absent from that early seed and filled only from the payload
at close. :class:`RequestContext` is the fully-resolved view at close, wrapping it.

The request-vs-provider model split: a caller asks for a *model group* (``gpt-4o``)
that routes to a concrete deployment (``azure/my-deployment``). ``gen_ai.request.model``
records the group and ``litellm.provider.model`` the dispatched model; they coincide
on the SDK path, which has no group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, cast


from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL
from litellm.integrations.otel.model.destination import OtelDestination
from litellm.integrations.otel.model.semconv import resolve_operation
from litellm.integrations.otel.model.utils import as_str, to_seconds
from litellm.integrations.otel.plumbing.context import request_destinations

if TYPE_CHECKING:
    from litellm.types.utils import StandardLoggingPayload


@dataclass(frozen=True)
class RequestIdentity:
    call_id: str | None = None
    team_id: str | None = None
    team_alias: str | None = None
    # The team's free-form metadata, carried raw; filtered to an operator allowlist at Baggage-promotion time.
    team_metadata: Mapping[str, Any] | None = None
    key_hash: str | None = None
    end_user: str | None = None
    # The model litellm dispatched to the provider; known only at close, so absent from the auth-time seed.
    provider_model: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: "StandardLoggingPayload") -> "RequestIdentity":
        """Parse caller identity (incl. resolved ``provider_model``) from a closed request's payload."""
        raw_meta = cast(Mapping[str, object], payload.get("metadata") or {})
        metadata = {key: str(value) for key, value in raw_meta.items() if isinstance(value, (str, bool, int, float))}
        return cls(
            call_id=as_str(payload.get("litellm_call_id")) or as_str(payload.get("id")),
            # Prefer the canonical ``user_api_key_team_id``; the bare ``team_id`` is a legacy alias.
            team_id=as_str(raw_meta.get("user_api_key_team_id")) or as_str(raw_meta.get("team_id")),
            team_alias=as_str(raw_meta.get("user_api_key_team_alias")) or as_str(raw_meta.get("team_alias")),
            team_metadata=_team_metadata_dict(raw_meta.get("user_api_key_team_metadata")),
            key_hash=as_str(raw_meta.get("user_api_key_hash")),
            end_user=as_str(payload.get("end_user")) or as_str(raw_meta.get("user_api_key_end_user_id")),
            provider_model=resolve_provider_model(payload),
            metadata=metadata,
        )

    @classmethod
    def from_user_api_key_auth(cls, auth: object) -> "RequestIdentity":
        """Identity from a ``UserAPIKeyAuth`` (duck-typed to avoid a proxy import).

        Seeds Baggage at the pre-call hook so every span inherits identity. Metadata
        sub-keys use the ``user_api_key_*`` names Baggage promotion expects.
        """
        get = lambda name: getattr(auth, name, None)  # noqa: E731
        metadata = {
            meta_key: str(value)
            for meta_key, attr in (
                ("user_api_key_user_id", "user_id"),
                ("user_api_key_org_id", "org_id"),
                ("user_api_key_alias", "key_alias"),
                ("user_api_key_end_user_id", "end_user_id"),
            )
            if (value := get(attr))
        }
        return cls(
            team_id=as_str(get("team_id")),
            team_alias=as_str(get("team_alias")),
            team_metadata=_team_metadata_dict(get("team_metadata")),
            key_hash=as_str(get("api_key")),
            end_user=as_str(get("end_user_id")),
            # ``provider_model`` is unknown at the auth boundary (routing hasn't picked a deployment yet).
            metadata=metadata,
        )


@dataclass(frozen=True)
class RequestContext:
    """The fully-resolved view of a closed request, parsed once from the payload."""

    request_model: str
    response_model: str | None
    model_group: str | None
    model_id: str | None
    api_base: str | None
    identity: RequestIdentity

    @property
    def provider_model(self) -> str | None:
        """The dispatched-model name, carried on the identity for Baggage."""
        return self.identity.provider_model

    @classmethod
    def from_standard_logging_payload(cls, payload: "StandardLoggingPayload") -> "RequestContext":
        raw_meta = cast(Mapping[str, object], payload.get("metadata") or {})
        hidden = cast(Mapping[str, object], payload.get("hidden_params") or {})
        raw_response = payload.get("response")
        response = cast(Mapping[str, object], raw_response if isinstance(raw_response, dict) else {})
        model_group = as_str(payload.get("model_group")) or as_str(raw_meta.get("model_group"))
        return cls(
            # The user asked for the group; fall back to the call model on the SDK
            # path, which has no group. Empty string (never None) so the span name
            # builder and the mapper see a plain string.
            request_model=model_group or as_str(payload.get("model")) or "",
            response_model=as_str(response.get("model")),
            model_group=model_group,
            model_id=as_str(payload.get("model_id")) or _model_info_id(raw_meta.get("model_info")),
            api_base=as_str(payload.get("api_base")) or as_str(hidden.get("api_base")),
            identity=RequestIdentity.from_payload(payload),
        )


# --- live-callback kwargs parsing ------------------------------------------- #
# Parse the live callback ``kwargs`` god object and raw pre/post-call ``data`` dicts —
# the untyped request state reaching a ``CustomLogger`` before a ``StandardLoggingPayload``.


@dataclass(frozen=True)
class LLMCallEvent:
    """The typed view of the live callback ``kwargs`` (``model_call_details``), parsed once."""

    # The ``litellm_call_id`` correlating ``pre_call`` with the close callback; the stable
    # key for the open-call carrier.
    call_id: str | None
    # The success/failure payload; ``None`` at ``pre_call`` or if the call closed with no payload.
    payload: "StandardLoggingPayload | None"
    otel_destinations: tuple[OtelDestination, ...]
    # True for synthetic proxy-gate logs (auth/rate-limit rejections): no upstream call, so no span.
    is_no_upstream_call: bool
    # Best-effort ``"{operation} {model}"`` name at ``pre_call``; only matters for a leaked span (renamed at close).
    provisional_span_name: str
    time_to_first_chunk_seconds: float | None

    @classmethod
    def from_dict(cls, kwargs: Mapping[str, Any]) -> "LLMCallEvent":
        raw_payload = kwargs.get("standard_logging_object")
        payload = cast("StandardLoggingPayload", raw_payload) if raw_payload else None
        operation = resolve_operation(as_str(kwargs.get("call_type")))
        model = as_str(kwargs.get("model")) or ""
        return cls(
            call_id=_call_id(payload, kwargs),
            payload=payload,
            otel_destinations=request_destinations(),
            is_no_upstream_call=bool(kwargs.get(LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL)),
            provisional_span_name=f"{operation.value} {model}".strip(),
            time_to_first_chunk_seconds=time_to_first_chunk_seconds(kwargs),
        )


def time_to_first_chunk_seconds(kwargs: Mapping[str, Any]) -> float | None:
    """Seconds from upstream request (``api_call_start_time``) to first streamed chunk
    (``completion_start_time``); ``None`` for non-streaming calls."""
    optional_params = cast(Mapping[str, Any], kwargs.get("optional_params") or {})
    if not optional_params.get("stream"):
        return None
    api_call_start = to_seconds(kwargs.get("api_call_start_time"))
    completion_start = to_seconds(kwargs.get("completion_start_time"))
    if api_call_start is None or completion_start is None:
        return None
    return completion_start - api_call_start


def _call_id(payload: "StandardLoggingPayload | None", kwargs: Mapping[str, Any]) -> str | None:
    """The call id from the payload (when closed) or the bare kwargs (at pre_call)."""
    if payload is not None:
        call_id = as_str(payload.get("litellm_call_id")) or as_str(payload.get("id"))
        if call_id:
            return call_id
    return as_str(kwargs.get("litellm_call_id"))


def model_from_request_data(data: object) -> str | None:
    """The user-facing ``model`` from a pre-call ``data`` dict (``None`` if absent)."""
    if isinstance(data, Mapping):
        return as_str(data.get("model"))
    return None


def resolve_provider_model(payload: "StandardLoggingPayload") -> str | None:
    """The model litellm dispatched to the provider, from the payload.

    Prefers ``metadata.deployment``, then ``hidden_params.litellm_model_name``, then the
    top-level ``model`` (already resolved to the provider-prefixed deployment name).
    """
    raw_meta = cast(Mapping[str, object], payload.get("metadata") or {})
    hidden = cast(Mapping[str, object], payload.get("hidden_params") or {})
    return (
        # ``deployment`` (most precise) survives only on paths that don't strip it from metadata.
        as_str(raw_meta.get("deployment")) or as_str(hidden.get("litellm_model_name")) or as_str(payload.get("model"))
    )


def _model_info_id(model_info: object) -> str | None:
    """The deployment id from a ``metadata.model_info`` sub-dict, if present."""
    if isinstance(model_info, Mapping):
        return as_str(model_info.get("id"))
    return None


def _team_metadata_dict(value: object) -> Mapping[str, Any] | None:
    """The team's free-form metadata as a raw mapping, or ``None`` when missing or empty."""
    if isinstance(value, Mapping) and value:
        return dict(value)
    return None
