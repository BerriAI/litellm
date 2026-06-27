"""The single translation layer between a request's metadata and the spans.

Every relevant field litellm exposes about a request — the user-facing model,
the model actually dispatched to the provider, the deployment, and the caller's
identity (team, key, end-user) — is parsed **once**, here, out of the
``StandardLoggingPayload`` (or a ``UserAPIKeyAuth`` at the auth boundary). Span
data, baggage promotion, and the mappers then read these typed fields instead of
each digging into the raw ``metadata`` / ``hidden_params`` dicts.

Two models live here because a request's identity is known *before* its model
resolution is:

* :class:`RequestIdentity` — team / key / end-user, seeded into Baggage at the
  auth boundary (``from_user_api_key_auth``), before routing has picked a
  deployment. ``provider_model`` is therefore absent from that early seed and is
  only filled in from the payload once the call closes.
* :class:`RequestContext` — the full picture available at close: the resolved
  request vs. provider model split, plus the response model, model group, model
  id, and api base, wrapping the :class:`RequestIdentity`.

The request-vs-provider model split is the subtle part. On the proxy a caller
asks for a *model group* (e.g. ``gpt-4o``) that routes to a concrete deployment
(e.g. ``azure/my-deployment``); the two are distinct and both worth recording.
``StandardLoggingPayload`` exposes them as:

* ``model_group`` — the user-facing name the caller requested.
* ``model`` — already reconstructed (see ``reconstruct_model_name``) to the name
  litellm dispatched to the provider (the deployment, provider-prefixed).
* ``hidden_params.litellm_model_name`` — a secondary source for the dispatched
  model (populated only on some call paths, e.g. files).

So ``gen_ai.request.model`` is the *group* (falling back to the call model on the
SDK path, which has no group), and ``litellm.provider.model`` is the *dispatched*
model. They coincide on the SDK path, which is correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, cast

from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL
from litellm.integrations.otel.model.semconv import resolve_operation
from litellm.integrations.otel.model.utils import as_str

if TYPE_CHECKING:
    from litellm.types.utils import StandardLoggingPayload


@dataclass(frozen=True)
class RequestIdentity:
    call_id: str | None = None
    team_id: str | None = None
    team_alias: str | None = None
    # The team's free-form metadata, carried raw (empty/missing -> None) and
    # filtered to an operator allowlist only at Baggage-promotion time, so an
    # unconfigured deployment never promotes any of it.
    team_metadata: Mapping[str, Any] | None = None
    key_hash: str | None = None
    end_user: str | None = None
    # The model litellm dispatched to the provider. Only known once the call
    # completes (routing has picked a deployment), so it's absent from the
    # auth-time seed and filled only from the payload.
    provider_model: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: "StandardLoggingPayload") -> "RequestIdentity":
        """Parse caller identity out of a closed request's payload metadata.

        ``provider_model`` is resolved here too (see :func:`resolve_provider_model`)
        so the identity carried into Baggage labels every span with the dispatched
        model, not just the user-facing one.
        """
        raw_meta = cast(Mapping[str, object], payload.get("metadata") or {})
        metadata = {key: str(value) for key, value in raw_meta.items() if isinstance(value, (str, bool, int, float))}
        return cls(
            call_id=as_str(payload.get("litellm_call_id")) or as_str(payload.get("id")),
            # StandardLoggingMetadata's canonical key is ``user_api_key_team_id``;
            # the bare ``team_id`` is a legacy alias and is often empty, so prefer
            # the canonical key and fall back to the alias.
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
        """Identity from a ``UserAPIKeyAuth`` (duck-typed to keep this module
        free of a proxy import).

        Used in the pre-call hook to seed Baggage early — before any LLM,
        guardrail, or service span is created — so the whole request's spans
        inherit identity, not just the LLM-call span. Metadata sub-keys use the
        ``user_api_key_*`` names that ``baggage.DEFAULT_BAGGAGE_METADATA_KEYS``
        promotes.
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
            # ``provider_model`` is unknown at the auth boundary — routing hasn't
            # picked a deployment yet — so it's only populated from the payload.
            metadata=metadata,
        )


@dataclass(frozen=True)
class RequestContext:
    """The fully-resolved view of a closed request, parsed once from the payload.

    ``request_model`` is the user-facing requested model and ``provider_model``
    (on :attr:`identity`) is the model litellm dispatched to the provider; the two
    differ on the proxy (group vs. deployment) and coincide on the SDK path.
    """

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
#
# The model and helpers below parse the *live* callback ``kwargs`` god object (and
# the raw pre/post-call ``data`` dicts) — the untyped request state that reaches a
# ``CustomLogger`` before, or instead of, a ``StandardLoggingPayload``. They live
# here, with the payload/auth parsers, so every read out of a request's raw dicts
# is in one place rather than scattered across the ``CustomLogger``.


@dataclass(frozen=True)
class LLMCallEvent:
    """The typed view of the live callback ``kwargs`` (``model_call_details``).

    litellm hands every callback an untyped ``kwargs`` god object. The fields the
    OTel logger needs out of it are parsed **once**, here, so the ``CustomLogger``
    reads typed attributes instead of digging into the dict at each boundary.
    """

    # The ``litellm_call_id`` correlating ``pre_call`` with the close callback.
    # Present in ``model_call_details`` at ``pre_call`` and in both the kwargs and
    # the ``standard_logging_object`` at success/failure, so it's a stable key for
    # the open-call carrier — no back-reference to the logging object required (the
    # object isn't reachable from the callback kwargs at ``pre_call`` time).
    call_id: str | None
    # The ``StandardLoggingPayload`` carried on a success/failure callback; ``None``
    # at ``pre_call``, or when the call closed before any payload materialized (so
    # there is nothing to stamp on the span).
    payload: "StandardLoggingPayload | None"
    # The ``standard_callback_dynamic_params`` routing the call to a per-tenant
    # tracer (its own exporter/endpoint), or ``None`` when the call isn't scoped.
    dynamic_params: Any
    # True for synthetic proxy-gate logs (auth / rate-limit rejections): they fire
    # the ``pre_call`` hook but never made an upstream call, so they get no span.
    is_no_upstream_call: bool
    # A best-effort ``"{operation} {model}"`` name known at ``pre_call`` time. The
    # span is renamed from the typed payload at close (``finish_span``); this only
    # needs to be reasonable for a span that never gets closed (a leak).
    provisional_span_name: str

    @classmethod
    def from_dict(cls, kwargs: Mapping[str, Any]) -> "LLMCallEvent":
        raw_payload = kwargs.get("standard_logging_object")
        payload = cast("StandardLoggingPayload", raw_payload) if raw_payload else None
        operation = resolve_operation(as_str(kwargs.get("call_type")))
        model = as_str(kwargs.get("model")) or ""
        return cls(
            call_id=_call_id(payload, kwargs),
            payload=payload,
            dynamic_params=kwargs.get("standard_callback_dynamic_params"),
            is_no_upstream_call=bool(kwargs.get(LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL)),
            provisional_span_name=f"{operation.value} {model}".strip(),
        )


def _call_id(payload: "StandardLoggingPayload | None", kwargs: Mapping[str, Any]) -> str | None:
    """The call id from the payload (when closed) or the bare kwargs (at pre_call)."""
    if payload is not None:
        call_id = as_str(payload.get("litellm_call_id")) or as_str(payload.get("id"))
        if call_id:
            return call_id
    return as_str(kwargs.get("litellm_call_id"))


def model_from_request_data(data: object) -> str | None:
    """The user-facing ``model`` from a pre-call ``data`` dict (``None`` if absent).

    Read at the auth boundary to label early Baggage before routing has resolved
    a deployment; ``data`` is duck-typed since it arrives untyped from the proxy.
    """
    if isinstance(data, Mapping):
        return as_str(data.get("model"))
    return None


def resolve_provider_model(payload: "StandardLoggingPayload") -> str | None:
    """The model litellm dispatched to the provider, from the payload.

    Prefers the explicit ``hidden_params.litellm_model_name`` (set on call paths
    that know it, e.g. files), then the top-level ``model`` — which
    ``reconstruct_model_name`` has already resolved to the deployment's
    provider-prefixed name. Returns ``None`` only when neither is present.
    """
    raw_meta = cast(Mapping[str, object], payload.get("metadata") or {})
    hidden = cast(Mapping[str, object], payload.get("hidden_params") or {})
    return (
        # ``deployment`` survives only on paths that don't strip it from metadata;
        # harmless (and most precise) to prefer it when present.
        as_str(raw_meta.get("deployment")) or as_str(hidden.get("litellm_model_name")) or as_str(payload.get("model"))
    )


def _model_info_id(model_info: object) -> str | None:
    """The deployment id from a ``metadata.model_info`` sub-dict, if present."""
    if isinstance(model_info, Mapping):
        return as_str(model_info.get("id"))
    return None


def _team_metadata_dict(value: object) -> Mapping[str, Any] | None:
    """The team's free-form metadata as a raw mapping, or ``None`` when missing
    or empty.

    Carried raw on the identity and filtered to an operator allowlist only at
    Baggage-promotion time (see ``baggage.promoted_baggage``), so an empty case
    is dropped rather than carrying a useless ``{}``.
    """
    if isinstance(value, Mapping) and value:
        return dict(value)
    return None
