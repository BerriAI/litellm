import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    TypeAlias,
    cast,  # noqa: TID251  # SSE byte frames ride the ModelResponseStream-typed pipe (bedrock precedent)
)

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
from litellm.caching import DualCache
from litellm.exceptions import GuardrailRaisedException
from litellm.exceptions import Timeout as LitellmTimeout
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import (
    LiteLLM_ManagementEndpoint_MetadataFields,
    LiteLLM_ManagementEndpoint_MetadataFields_Premium,
    UserAPIKeyAuth,
)
from litellm.proxy.guardrails.anthropic_sse import (
    anthropic_sse_chunks_from_body,
    anthropic_sse_error_frames,
    assemble_anthropic_sse_body,
    sse_stream_text,
)
from litellm.proxy.guardrails.stream_surface import (
    StreamSurface,
    classify_stream,
    final_responses_api_response,
    is_terminal_error_stream,
    responses_deltas_absent_from_body,
)
from litellm.proxy.litellm_pre_call_utils import (
    _UNTRUSTED_METADATA_CONTROL_FIELDS,  # pyright: ignore[reportPrivateUsage]  # shared list
    _UNTRUSTED_ROOT_CONTROL_FIELDS,  # pyright: ignore[reportPrivateUsage]  # shared list
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.proxy.guardrails.guardrail_hooks.thirdlaw import (
    ThirdlawGuardrailRequest,
    ThirdlawGuardrailRequestMetadata,
    ThirdlawGuardrailResponse,
)
from litellm.types.utils import (
    CallTypesLiteral,
    GuardrailStatus,
    LLMResponseTypes,
    ModelResponse,
    ModelResponseStream,
)

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME: Final = "thirdlaw"

_ENDPOINT_PATH: Final = "/guardrails/litellm/v2"

_UNREACHABLE_STATUS_CODES: Final = frozenset({502, 503, 504})

# Credential-bearing LiteLLM parameters that are not declared on ``CredentialLiteLLMParams``
# because callers supply them through **kwargs. ``extra_headers`` / ``default_headers`` are
# here because either can carry a provider Authorization header.
_KWARGS_ONLY_CREDENTIAL_KEYS: Final = frozenset(
    {
        "azure_ad_token_provider",
        "default_headers",
        "extra_headers",
        "oci_fingerprint",
        "oci_key",
        "oci_key_file",
        "oci_signer",
        "oci_tenancy",
        "oci_user",
    }
)


def _credential_keys() -> frozenset[str]:
    """Every LiteLLM parameter that can carry a provider credential.

    Derived from ``CredentialLiteLLMParams`` rather than enumerated here, so a provider
    credential field added upstream is withheld from ThirdLaw without a matching edit in
    this module. The model also carries non-secret routing fields (``api_base``,
    ``vertex_location``, ``aws_region_name``); those are not provider body content either,
    so stripping the whole model costs nothing.
    """
    from litellm.types.router import CredentialLiteLLMParams

    return frozenset(CredentialLiteLLMParams.model_fields) | _KWARGS_ONLY_CREDENTIAL_KEYS


# Not part of the provider request body. ``secret_fields`` holds plaintext Authorization
# values, and every credential-bearing litellm parameter is withheld alongside it: a caller
# authenticating with aws_secret_access_key, azure_ad_token, client_secret or
# vertex_credentials must not have that value serialized out to the guardrail service.
_BODY_STRIP_KEYS: Final = _credential_keys() | frozenset(
    {
        "guardrail_config",
        "guardrails",
        "headers",
        "litellm_call_id",
        "litellm_logging_obj",
        "litellm_metadata",
        "litellm_session_id",
        "litellm_trace_id",
        "metadata",
        "provider_specific_header",
        "proxy_server_request",
        "response",
        "responses",
        "secret_fields",
        "user_api_key_dict",
    }
)

# Deployment selectors: either one redirects the call to different credentials or a different
# provider than the key/team was authorized against.
_ROUTING_SELECTOR_KEYS: Final = frozenset({"custom_llm_provider", "litellm_credential_name"})

# Never writable by modify_request: ``guardrails`` gates which guardrails run, ``model`` was
# authorized against the key/team before this hook, ``stream`` changes the wire protocol mid-request.
# ``_UNTRUSTED_ROOT_CONTROL_FIELDS`` is litellm's own list of root controls stripped from caller
# input by add_litellm_data_to_request; a guardrail response arrives after that strip, so without
# this it could put back agentic-loop and sandbox-interception state the caller was denied.
_WRITE_BACK_DENY_KEYS: Final = (
    _BODY_STRIP_KEYS
    | _ROUTING_SELECTOR_KEYS
    | frozenset(_UNTRUSTED_ROOT_CONTROL_FIELDS)
    | frozenset({"model", "policies", "stream", "user"})
)

_USER_METADATA_FIELDS: Final = (
    "user_api_key_hash",
    "user_api_key_alias",
    "user_api_key_user_id",
    "user_api_key_user_email",
    "user_api_key_team_id",
    "user_api_key_team_alias",
    "user_api_key_end_user_id",
    "user_api_key_org_id",
    "user_api_key_project_id",
    "user_api_key_project_alias",
    "user_api_key_org_alias",
    # Unprefixed unlike its siblings above: litellm writes this key bare, not as
    # "user_api_key_agent_id" (see add_user_api_key_auth_to_request_metadata).
    "agent_id",
)

# Admin-facing control keys litellm reserves inside a key/team/project/org's own custom
# `metadata` dict -- never admin-typed custom data. Combines the management-endpoint
# reserved fields (rate/budget overrides, guardrails, tags, etc.) with litellm's own
# client-forgery denylist (guardrail-bypass and internal routing/logging signals): the
# same reasoning that makes these untrustworthy coming from a caller makes them
# unsuitable to forward to a third-party guardrail. ``model_config`` is added
# separately -- it can carry ``litellm_credentials`` selector names for provider
# routing, which is proxy operational config, not admin-authored metadata.
_RESERVED_METADATA_KEYS: Final = frozenset(
    (
        *LiteLLM_ManagementEndpoint_MetadataFields,
        *LiteLLM_ManagementEndpoint_MetadataFields_Premium,
        *_UNTRUSTED_METADATA_CONTROL_FIELDS,
        "model_config",
    )
)

_WireEvent: TypeAlias = Literal["pre_call", "during_call", "post_call"]

# What a buffered stream assembles into, one shape per surface that has an assembler
_AssembledStream: TypeAlias = ModelResponse | ResponsesAPIResponse | Mapping[str, object]

# Never writable by modify_response on a /v1/responses stream: litellm encrypts the response id and
# the client chains the next turn off it with previous_response_id.
_RESPONSE_WRITE_BACK_DENY_KEYS: Final = frozenset({"id"})

_JSON_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])

# tags metadata rides in as an untyped list; validating through a TypeAdapter (its
# `validate_python` parameter is `Any`, so it accepts an untyped value without complaint)
# resolves it to a concrete element type before use.
_JSON_LIST_ADAPTER: Final = TypeAdapter(list[object])

_EMPTY_MAP: Final[Mapping[str, object]] = MappingProxyType({})

_EMPTY_STR_MAP: Final[Mapping[str, str]] = MappingProxyType({})


def _dict_of(value: object) -> Mapping[str, object] | None:
    return _JSON_DICT_ADAPTER.validate_python(value) if isinstance(value, dict) else None


def _custom_metadata(value: object) -> Mapping[str, object] | None:
    """A key+team (pre-merged by litellm core), project, or org's custom metadata, minus reserved keys."""
    as_dict: Final = _dict_of(value)
    if as_dict is None:
        return None
    filtered: Final = {k: v for k, v in as_dict.items() if k not in _RESERVED_METADATA_KEYS}
    return MappingProxyType(filtered) if filtered else None


def _embedded_user_api_key_auth(merged: Mapping[str, object]) -> UserAPIKeyAuth | None:
    auth: Final = merged.get("user_api_key_auth")
    return auth if isinstance(auth, UserAPIKeyAuth) else None


# UserAPIKeyAuth declares these two as a bare `dict | None`, so reading them is itself a
# partially-unknown member access; each accessor isolates that one read behind an explicit
# `object` return type so the rest of this module never touches the untyped field directly.
def _project_metadata_of(auth: UserAPIKeyAuth) -> object:
    return auth.project_metadata  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # untyped


def _organization_metadata_of(auth: UserAPIKeyAuth) -> object:
    return auth.organization_metadata  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # untyped


def _jsonable_dict(value: Mapping[str, object]) -> Mapping[str, object]:
    """Round-trip through JSON so the payload cannot carry live objects."""
    plain: Final = dict(value)  # mutable-ok: json.dumps requires a real dict; consumed immediately
    return _JSON_DICT_ADAPTER.validate_python(json.loads(json.dumps(plain, default=str)))


def _merged_request_metadata(request_data: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {
            **(_dict_of(request_data.get("metadata")) or _EMPTY_MAP),
            **(_dict_of(request_data.get("litellm_metadata")) or _EMPTY_MAP),
        }
    )


def _request_metadata(request_data: Mapping[str, object]) -> ThirdlawGuardrailRequestMetadata:
    merged: Final = _merged_request_metadata(request_data)
    user_fields: Final = MappingProxyType(
        {field: value for field in _USER_METADATA_FIELDS if isinstance(value := merged.get(field), str)}
    )
    token: Final = merged.get("user_api_key_token")
    hash_fallback: Final[Mapping[str, str]] = (
        MappingProxyType({"user_api_key_hash": token})
        if "user_api_key_hash" not in user_fields and isinstance(token, str)
        else _EMPTY_STR_MAP
    )
    call_id: Final = request_data.get("litellm_call_id")
    trace_id: Final = request_data.get("litellm_trace_id")
    # litellm_session_id (call-level, header/Anthropic-metadata-derived) and metadata.session_id
    # (settable directly by a caller per litellm's own "missing_session_id" documentation) are
    # two independent, equally valid sources; litellm core itself treats either as sufficient.
    session_id: Final = request_data.get("litellm_session_id") or merged.get("session_id")
    model: Final = request_data.get("model")
    tags_value: Final = merged.get("tags")
    tags_list: Final = _JSON_LIST_ADAPTER.validate_python(tags_value) if isinstance(tags_value, list) else None
    tags: Final = tuple(tag for tag in tags_list if isinstance(tag, str)) if tags_list is not None else None
    auth_obj: Final = _embedded_user_api_key_auth(merged)
    project_metadata: Final = _dict_of(_project_metadata_of(auth_obj)) if auth_obj is not None else None
    organization_metadata: Final = _dict_of(_organization_metadata_of(auth_obj)) if auth_obj is not None else None
    return ThirdlawGuardrailRequestMetadata(
        litellm_version=litellm_version,
        litellm_call_id=call_id if isinstance(call_id, str) else None,
        litellm_trace_id=trace_id if isinstance(trace_id, str) else None,
        litellm_session_id=session_id if isinstance(session_id, str) else None,
        model=model if isinstance(model, str) else None,
        tags=tags or None,
        # Already includes the team's own custom metadata (litellm core layers it on top when
        # building this dict), so team_metadata is intentionally not forwarded separately here.
        user_api_key_auth_metadata=_custom_metadata(merged.get("user_api_key_auth_metadata")),
        project_metadata=_custom_metadata(project_metadata),
        organization_metadata=_custom_metadata(organization_metadata),
        **user_fields,
        **hash_fallback,
    )


def _proxy_server_request(request_data: Mapping[str, object]) -> Mapping[str, object]:
    return _dict_of(request_data.get("proxy_server_request")) or _EMPTY_MAP


def _request_url(request_data: Mapping[str, object]) -> str | None:
    url: Final = _proxy_server_request(request_data).get("url")
    return url if isinstance(url, str) else None


def _redacted_inbound_headers(request_data: Mapping[str, object]) -> Mapping[str, object] | None:
    proxy_headers: Final = _dict_of(_proxy_server_request(request_data).get("headers"))
    if proxy_headers:
        return proxy_headers
    return _dict_of(_merged_request_metadata(request_data).get("headers")) or None


def _raw_inbound_headers(request_data: Mapping[str, object]) -> Mapping[str, object]:
    secret_fields: Final = _dict_of(request_data.get("secret_fields")) or _EMPTY_MAP
    return _dict_of(secret_fields.get("raw_headers")) or _EMPTY_MAP


def _outbound_request_headers(
    request_data: Mapping[str, object], raw_value_header_names: frozenset[str]
) -> Mapping[str, str] | None:
    """Every inbound header, using LiteLLM's credential-redacted values; names in
    ``raw_value_header_names`` carry the raw value the client sent.

    The proxy strips the header used for LiteLLM auth (e.g. ``authorization``) from its
    sanitized copy entirely, so opted-in names are also re-added from the raw headers
    rather than only substituted in place.
    """
    redacted: Final = _redacted_inbound_headers(request_data)
    if redacted is None:
        return None
    raw_by_lower: Final = MappingProxyType(
        {name.lower(): str(value) for name, value in _raw_inbound_headers(request_data).items()}
    )
    from_redacted: Final = MappingProxyType(
        {
            name: (
                raw_by_lower[name.lower()]
                if name.lower() in raw_value_header_names and name.lower() in raw_by_lower
                else str(value)
            )
            for name, value in redacted.items()
        }
    )
    present_lower: Final = frozenset(name.lower() for name in from_redacted)
    reintroduced: Final = MappingProxyType(
        {
            name: raw_by_lower[name]
            for name in raw_value_header_names
            if name in raw_by_lower and name not in present_lower
        }
    )
    return MappingProxyType({**from_redacted, **reintroduced})


def _request_body(request_data: Mapping[str, object], prefer_snapshot: bool) -> Mapping[str, object] | None:
    """Best-effort provider request body; never raises because a guardrail that throws
    here would fail live traffic.

    Pre-call must read live ``request_data``: the ``proxy_server_request.body`` snapshot
    is taken before pre-call hooks run, so it would resend content an earlier guardrail
    (e.g. Presidio masking) already rewrote.
    """
    snapshot: Final = _dict_of(_proxy_server_request(request_data).get("body")) if prefer_snapshot else None
    source: Final = snapshot if snapshot is not None else request_data
    try:
        return _jsonable_dict(
            MappingProxyType({key: value for key, value in source.items() if key not in _BODY_STRIP_KEYS})
        )
    except Exception:  # noqa: BLE001  # best-effort capture; a raise here would fail live traffic
        verbose_proxy_logger.warning("ThirdLaw guardrail: could not serialize request body", exc_info=True)
        return None


def _response_payload(response: object) -> Mapping[str, object] | None:
    dump: Final = getattr(response, "model_dump", None)
    try:
        raw: Final[object] = dump(mode="json") if callable(dump) else response
    except Exception:  # noqa: BLE001  # best-effort capture; a raise here would fail live traffic
        verbose_proxy_logger.warning("ThirdLaw guardrail: could not serialize response body", exc_info=True)
        return None
    as_dict: Final = _dict_of(raw)
    if as_dict is None:
        return None
    try:
        return _jsonable_dict(as_dict)
    except Exception:  # noqa: BLE001  # best-effort capture; a raise here would fail live traffic
        verbose_proxy_logger.warning("ThirdLaw guardrail: could not serialize response body", exc_info=True)
        return None


def _stream_chunk_payload(item: object) -> Mapping[str, object] | None:
    """One buffered stream event as JSON, keeping a sequence number the bridge stamps off-model.

    LiteLLM's chat-to-Responses bridge writes ``sequence_number`` straight onto ``__dict__``, which
    ``model_dump`` drops, so it is read back with ``getattr`` and restored on the payload.
    """
    dump: Final = getattr(item, "model_dump", None)
    if not callable(dump):
        as_dict: Final = _dict_of(item)
        return _jsonable_dict(as_dict) if as_dict is not None else None
    try:
        dumped: Final[object] = dump(mode="json")
    except Exception:  # noqa: BLE001  # one unserializable event must not drop the whole stream from the payload
        return None
    dumped_dict: Final = _dict_of(dumped)
    if dumped_dict is None:
        return None
    sequence: Final = getattr(item, "sequence_number", None)
    if isinstance(sequence, int) and "sequence_number" not in dumped_dict:
        return _jsonable_dict(MappingProxyType({**dumped_dict, "sequence_number": sequence}))
    return _jsonable_dict(dumped_dict)


def _is_unreachable_error(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in _UNREACHABLE_STATUS_CODES
    return isinstance(error, (httpx.RequestError, LitellmTimeout))


def _service_error_message(error: Exception) -> str:
    """Prefer ThirdLaw's own explanation over httpx's generic status text.

    ThirdLaw returns ``{"detail": {"error", "reason", "message"}}`` on a fail-closed
    response. Anything else (an ingress 502, a non-JSON body, an older service version)
    falls back to the status line.
    """
    if not isinstance(error, httpx.HTTPStatusError):
        return str(error)
    try:
        detail: Final[object] = error.response.json().get("detail")
    except Exception:  # noqa: BLE001  # non-JSON or non-object error body: keep the status line
        return str(error)
    if isinstance(detail, str) and detail:
        return detail
    detail_fields: Final = _dict_of(detail)
    if detail_fields is None:
        return str(error)
    message: Final = detail_fields.get("message")
    if not isinstance(message, str) or not message:
        return str(error)
    reason: Final = detail_fields.get("reason")
    return f"{message} (reason={reason})" if isinstance(reason, str) and reason else message


def _decision_trace(decision: ThirdlawGuardrailResponse) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "action": decision.action,
            "message": decision.message,
            "response_status": decision.response_status,
            "modified_request": decision.request_body is not None,
            "modified_response": decision.response_body is not None,
        }
    )


class ThirdlawGuardrailMissingConfig(ValueError):
    pass


class ThirdlawGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        additional_headers: str | None = None,
        guardrail_timeout: int | None = 60,
        streaming_buffer_until_moderated: bool = True,
        streaming_end_of_stream_only: bool = True,
        streaming_sampling_rate: int = 5,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        unscannable_stream_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        send_stream_chunks: bool = False,
        additional_provider_specific_params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        extra_headers: Sequence[str] | None = None,
        async_handler: AsyncHTTPHandler | None = None,
        **kwargs,  # noqa: ANN003  # kwargs-ok: forwarded verbatim to CustomGuardrail, which owns their types
    ) -> None:
        resolved_base: Final = api_base or get_secret_str("THIRDLAW_API_BASE")
        if not resolved_base:
            raise ThirdlawGuardrailMissingConfig(
                "ThirdLaw api_base is required. Set api_base in the guardrail "
                "config or the THIRDLAW_API_BASE environment variable."
            )
        if streaming_sampling_rate < 1:
            raise ValueError(f"streaming_sampling_rate must be >= 1 (got {streaming_sampling_rate})")

        trimmed_base: Final = resolved_base.rstrip("/")
        self.api_base = trimmed_base if trimmed_base.endswith(_ENDPOINT_PATH) else f"{trimmed_base}{_ENDPOINT_PATH}"

        resolved_key: Final = api_key or get_secret_str("THIRDLAW_API_KEY")
        auth_header: Final[Mapping[str, str]] = (
            MappingProxyType({"Authorization": f"Bearer {resolved_key}"}) if resolved_key else _EMPTY_STR_MAP
        )
        self.http_headers: Mapping[str, str] = MappingProxyType(
            {"Content-Type": "application/json", **(headers or _EMPTY_STR_MAP), **auth_header}
        )

        configured_names: Final = tuple(additional_headers.split(",")) if additional_headers else ()
        self.raw_value_header_names: frozenset[str] = frozenset(
            stripped.lower() for name in (*configured_names, *(extra_headers or ())) if (stripped := name.strip())
        )

        self.guardrail_timeout = httpx.Timeout(timeout=guardrail_timeout or 60, connect=5.0)
        self.streaming_buffer_until_moderated = streaming_buffer_until_moderated
        self.streaming_end_of_stream_only = streaming_end_of_stream_only
        self.streaming_sampling_rate = streaming_sampling_rate
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = unreachable_fallback
        self.unscannable_stream_fallback: Literal["fail_closed", "fail_open"] = unscannable_stream_fallback
        self.send_stream_chunks: bool = send_stream_chunks
        self.additional_provider_specific_params: Mapping[str, object] = (
            additional_provider_specific_params or _EMPTY_MAP
        )

        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))  # mutable-ok: base API
        super().__init__(**kwargs)

        self.async_handler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
            params={"timeout": self.guardrail_timeout},  # mutable-ok: one-shot client-factory argument
        )

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: CustomGuardrail base-class contract
        return [  # mutable-ok: CustomGuardrail base-class contract returns a list
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.during_call,
        ]

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.thirdlaw import (
            ThirdlawGuardrailConfigModel,
        )

        return ThirdlawGuardrailConfigModel

    def _build_wire_request(
        self,
        *,
        wire_event: _WireEvent,
        request_data: dict[str, object],  # mutable-ok: proxy-shared request dict, forwarded to base-class helpers
        response_body: Mapping[str, object] | None,
        streamed_deltas: Sequence[str] | None = None,
        response_chunks: Sequence[Mapping[str, object]] | None = None,
        response_sse: str | None = None,
    ) -> ThirdlawGuardrailRequest:
        dynamic_params: Final = _JSON_DICT_ADAPTER.validate_python(
            self.get_guardrail_dynamic_request_body_params(request_data)
        )
        combined_params: Final = MappingProxyType({**self.additional_provider_specific_params, **dynamic_params})
        return ThirdlawGuardrailRequest(
            event_type=wire_event,
            metadata=_request_metadata(request_data),
            request_url=_request_url(request_data),
            request_headers=_outbound_request_headers(request_data, self.raw_value_header_names),
            request_body=_request_body(request_data, prefer_snapshot=wire_event != "pre_call"),
            response_body=response_body,
            response_chunks=tuple(response_chunks) if response_chunks else None,
            response_sse=response_sse or None,
            streamed_deltas_not_in_body=tuple(streamed_deltas) if streamed_deltas else None,
            additional_provider_specific_params=combined_params or None,
        )

    def _record_trace(
        self,
        *,
        request_data: dict[str, object],  # mutable-ok: the trace helper appends to this dict's metadata bucket
        event_type: GuardrailEventHooks,
        status: GuardrailStatus,
        started_at: datetime,
        trace: Mapping[str, object],
    ) -> None:
        now: Final = datetime.now(timezone.utc)
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=dict(trace),  # mutable-ok: the logging helper requires a plain dict
            request_data=request_data,
            guardrail_status=status,
            start_time=started_at.timestamp(),
            end_time=now.timestamp(),
            duration=(now - started_at).total_seconds(),
            event_type=event_type,
        )

    def _handle_call_failure(self, *, error: Exception, wire_event: _WireEvent) -> None:
        """Fail-open logs and returns; everything else raises."""
        if _is_unreachable_error(error) and self.unreachable_fallback == "fail_open":
            verbose_proxy_logger.critical(
                "ThirdLaw guardrail unreachable (fail-open). Proceeding without guardrail. "
                "guardrail_name=%s api_base=%s event=%s",
                self.guardrail_name,
                self.api_base,
                wire_event,
                exc_info=error,
            )
            return
        raise GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message=f"ThirdLaw guardrail request failed: {_service_error_message(error)}",
        ) from error

    async def _run_thirdlaw(
        self,
        *,
        event_type: GuardrailEventHooks,
        wire_event: _WireEvent,
        request_data: dict[str, object],  # mutable-ok: proxy-shared request dict; trace records land in it
        response_body: Mapping[str, object] | None = None,
        streamed_deltas: Sequence[str] | None = None,
        response_chunks: Sequence[Mapping[str, object]] | None = None,
        response_sse: str | None = None,
    ) -> ThirdlawGuardrailResponse | None:
        """POST the full payload to ThirdLaw and return its decision.

        Returns None when the service is unreachable and the guardrail is configured
        fail-open; raises for every other failure and records the guardrail trace on
        all paths (the decorator on the lifecycle hooks skips its auto-record when an
        entry was already written here).
        """
        started_at: Final = datetime.now(timezone.utc)
        payload: Final = self._build_wire_request(
            wire_event=wire_event,
            request_data=request_data,
            response_body=response_body,
            streamed_deltas=streamed_deltas,
            response_chunks=response_chunks,
            response_sse=response_sse,
        )
        try:
            http_response: Final = await self.async_handler.post(
                url=self.api_base,
                headers=dict(self.http_headers),  # mutable-ok: the HTTP client requires a plain dict
                json=payload.model_dump(mode="json", exclude_none=True),
            )
            http_response.raise_for_status()
            decision: Final = ThirdlawGuardrailResponse.model_validate(http_response.json())
        except Exception as error:  # noqa: BLE001  # every transport/parse failure funnels into the fallback policy
            self._record_trace(
                request_data=request_data,
                event_type=event_type,
                status="guardrail_failed_to_respond",
                started_at=started_at,
                trace={"error": _service_error_message(error)},  # mutable-ok: one-shot trace payload
            )
            self._handle_call_failure(error=error, wire_event=wire_event)
            return None
        status: Final[GuardrailStatus] = "guardrail_intervened" if decision.action == "block" else "success"
        self._record_trace(
            request_data=request_data,
            event_type=event_type,
            status=status,
            started_at=started_at,
            trace=_decision_trace(decision),
        )
        return decision

    def _block_status(self, response_status: int | None) -> int:
        """A block is a refusal, so it can only travel as a 4xx/5xx.

        Honoring a success status the service sent would hand the caller a 200 that
        reads as an allow.
        """
        if response_status is None:
            return 400
        if 400 <= response_status <= 599:
            return response_status
        verbose_proxy_logger.warning(
            "ThirdLaw guardrail: ignoring non-error response_status %s on a block decision; using 400",
            response_status,
        )
        return 400

    def _block_exception(self, decision: ThirdlawGuardrailResponse) -> GuardrailRaisedException:
        return GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message=decision.message or "Content violates ThirdLaw policy",
            should_wrap_with_default_message=False,
            status_code=self._block_status(decision.response_status),
            blocked_content=True,
        )

    def _applied_request_modifications(
        self, *, data: Mapping[str, object], decision: ThirdlawGuardrailResponse
    ) -> dict[str, object]:  # mutable-ok: the pre-call hook contract returns the (new) request dict
        replacement: Final = decision.request_body
        if not replacement:
            verbose_proxy_logger.warning(
                "ThirdLaw guardrail: modify_request decision carried no request_body; request unchanged"
            )
            return dict(data)  # mutable-ok: the pre-call hook contract returns a plain request dict
        accepted: Final = {  # mutable-ok: one-shot filter merged into the returned request dict
            key: value for key, value in replacement.items() if key not in _WRITE_BACK_DENY_KEYS
        }
        denied: Final = replacement.keys() - accepted.keys()
        if denied:
            verbose_proxy_logger.warning(
                "ThirdLaw guardrail: dropping protected keys from modify_request write-back: %s",
                sorted(denied),
            )
        return {**data, **accepted}  # mutable-ok: the proxy owns the replaced request dict

    @staticmethod
    def _carry_hidden_params(*, source: object, target: object) -> None:
        """model_validate() builds a fresh instance with no memory of the source's private
        attributes -- ``_hidden_params`` (which carries ``additional_headers`` for response-
        header forwarding) is one of those, so copy it across explicitly.
        """
        hidden_params = getattr(source, "_hidden_params", None)
        if hidden_params is not None:
            setattr(target, "_hidden_params", hidden_params)  # target's concrete type varies by call site

    def _without_response_id(self, replacement: Mapping[str, object]) -> Mapping[str, object]:
        """Drop ``id`` from a modify_response write-back.

        litellm hands out an encrypted response id that the client sends back as
        previous_response_id and that carries the deployment the turn was routed to, so a
        rewrite that renames it breaks both continuation and routing.
        """
        if not _RESPONSE_WRITE_BACK_DENY_KEYS & replacement.keys():
            return replacement
        verbose_proxy_logger.warning(
            "ThirdLaw guardrail: ignoring %s in a modify_response; the response id is encrypted "
            "and the client chains the next turn off it",
            sorted(_RESPONSE_WRITE_BACK_DENY_KEYS & replacement.keys()),
        )
        return MappingProxyType(
            {key: value for key, value in replacement.items() if key not in _RESPONSE_WRITE_BACK_DENY_KEYS}
        )

    def _modified_response(self, *, response: object, replacement: Mapping[str, object]) -> object:
        replacement = self._without_response_id(replacement)  # rebind-ok: one guard for every shape below
        if isinstance(response, ModelResponse):
            # ModelResponse validation coerces a non-list ``choices`` into a single
            # empty choice instead of raising, which would silently blank the response.
            if "choices" in replacement and not isinstance(replacement.get("choices"), list):
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message="ThirdLaw guardrail returned a malformed modified response: choices must be a list",
                )
            merged: Final = {  # mutable-ok: one-shot overlay consumed immediately by model_validate
                **_JSON_DICT_ADAPTER.validate_python(response.model_dump()),
                **replacement,
            }
            try:
                validated: Final = ModelResponse.model_validate(merged)
            except ValidationError as error:
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"ThirdLaw guardrail returned a malformed modified response: {error}",
                ) from error
            self._carry_hidden_params(source=response, target=validated)
            return validated
        response_dict: Final = _dict_of(response)
        if response_dict is not None:
            return {**response_dict, **replacement}  # mutable-ok: the proxy owns the replaced response dict
        if isinstance(response, BaseModel):
            merged_body: Final = {  # mutable-ok: one-shot overlay consumed immediately by model_validate
                **_JSON_DICT_ADAPTER.validate_python(response.model_dump(mode="json")),
                **replacement,
            }
            try:
                revalidated: Final = type(response).model_validate(merged_body)
            except ValidationError as error:
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"ThirdLaw guardrail returned a malformed modified response: {error}",
                ) from error
            self._carry_hidden_params(source=response, target=revalidated)
            return revalidated
        verbose_proxy_logger.warning(
            "ThirdLaw guardrail: modify_response is not supported for %s responses; returning original",
            type(response).__name__,
        )
        return response

    def _mark_applied(self, request_data: dict[str, object]) -> None:  # mutable-ok: header helper appends to metadata
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        add_guardrail_to_applied_guardrails_header(request_data=request_data, guardrail_name=self.guardrail_name)

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],  # mutable-ok: proxy-shared request dict, per the CustomLogger hook contract
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:  # mutable-ok: the proxy replaces its request dict with this return value
        decision: Final = await self._run_thirdlaw(
            event_type=GuardrailEventHooks.pre_call, wire_event="pre_call", request_data=data
        )
        if decision is None:
            return data
        if decision.action == "block":
            raise self._block_exception(decision)
        self._mark_applied(data)
        if decision.action == "modify_request":
            return self._applied_request_modifications(data=data, decision=decision)
        if decision.action == "modify_response":
            verbose_proxy_logger.warning("ThirdLaw guardrail: ignoring modify_response decision on pre_call")
        return data

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict[str, object],  # mutable-ok: proxy-shared request dict, per the CustomLogger hook contract
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:  # mutable-ok: the hook contract returns the request dict
        decision: Final = await self._run_thirdlaw(
            event_type=GuardrailEventHooks.during_call, wire_event="during_call", request_data=data
        )
        if decision is None:
            return data
        if decision.action == "block":
            raise self._block_exception(decision)
        if decision.action != "allow":
            verbose_proxy_logger.warning(
                "ThirdLaw guardrail: %s decision cannot be applied on during_call (runs parallel to the LLM call)",
                decision.action,
            )
        self._mark_applied(data)
        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict[str, object],  # mutable-ok: proxy-shared request dict, per the CustomLogger hook contract
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        response_body: Final = _response_payload(response)
        if response_body is None:
            verbose_proxy_logger.warning(
                "ThirdLaw guardrail: skipping post_call scan for unsupported response type %s",
                str(type(response).__name__),
            )
            return response
        decision: Final = await self._run_thirdlaw(
            event_type=GuardrailEventHooks.post_call,
            wire_event="post_call",
            request_data=data,
            response_body=response_body,
        )
        if decision is None:
            return response
        if decision.action == "block":
            raise self._block_exception(decision)
        self._mark_applied(data)
        if decision.action != "modify_response" or not decision.response_body:
            return response
        if self.run_in_parallel:
            verbose_proxy_logger.warning(
                "ThirdLaw guardrail: modify_response is discarded when run_in_parallel=True (block-only mode)"
            )
        modified: Final = self._modified_response(response=response, replacement=decision.response_body)
        return cast(LLMResponseTypes, modified)  # cast-ok: dict passthrough mirrors the proxy /v1/messages contract

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncIterator[object],
        request_data: dict[str, object],  # mutable-ok: proxy-shared request dict; trace records land in it
    ) -> AsyncGenerator[ModelResponseStream, None]:
        buffer: Final = self._buffer_until_moderated()
        end_of_stream_only: Final = buffer or self.streaming_end_of_stream_only
        iterator: Final = (
            self._end_of_stream_moderated_stream(response=response, request_data=request_data, buffer=buffer)
            if end_of_stream_only
            else self._sampled_stream(response=response, request_data=request_data)
        )
        async for item in iterator:
            # Raw-SSE frames (bytes) ride through the same pipe; the proxy's data
            # generator forwards str/bytes chunks verbatim, matching bedrock's hook.
            yield cast(ModelResponseStream, item)  # cast-ok: raw-SSE byte frames share the typed stream (bedrock)

    @staticmethod
    def _assembled_stream_response(collected: Sequence[object], surface: StreamSurface) -> _AssembledStream | None:
        """Assemble the buffered stream into the scannable body its surface produces."""
        match surface:
            case StreamSurface.ANTHROPIC_MESSAGES:
                return assemble_anthropic_sse_body(collected)
            case StreamSurface.RESPONSES:
                return ThirdlawGuardrail._assembled_responses_stream_response(collected)
            case StreamSurface.OPAQUE_SSE:
                return None
            case StreamSurface.CHAT_COMPLETIONS:
                return ThirdlawGuardrail._assembled_chat_stream_response(collected)

    @staticmethod
    def _assembled_responses_stream_response(collected: Sequence[object]) -> ResponsesAPIResponse | None:
        """The finished Responses body, which is the shape the non-streaming route already posts."""
        return final_responses_api_response(collected)

    @staticmethod
    def _streamed_deltas_not_in_body(
        collected: Sequence[object], assembled: _AssembledStream, surface: StreamSurface
    ) -> tuple[str, ...]:
        """Delta text the client already received that the scanned body does not carry.

        Reasoning summaries and tool-call arguments reach a /v1/responses client through delta
        events some providers never repeat in the terminal body. Posting them beside the body
        lets the scan cover the whole turn without widening the body contract.
        """
        if surface is not StreamSurface.RESPONSES or not isinstance(assembled, ResponsesAPIResponse):
            return ()
        return responses_deltas_absent_from_body(collected, assembled)

    def _stream_payload(
        self, collected: Sequence[object], surface: StreamSurface
    ) -> tuple[tuple[Mapping[str, object], ...] | None, str | None]:
        """The buffered stream as the service receives it beside the assembled body.

        The body is what LiteLLM folded; the stream is what the client received. Posting both lets
        the service fold on its own side and fall back to whichever side has no gap, with no
        LiteLLM release in between. Returns ``(chunks, sse_text)``; at most one is set.
        """
        if not self.send_stream_chunks:
            return None, None
        match surface:
            case StreamSurface.ANTHROPIC_MESSAGES:
                return None, sse_stream_text(collected)
            case StreamSurface.RESPONSES | StreamSurface.CHAT_COMPLETIONS:
                chunks: Final = tuple(
                    payload for item in collected if (payload := _stream_chunk_payload(item)) is not None
                )
                return (chunks or None), None
            case StreamSurface.OPAQUE_SSE:
                return None, None

    @staticmethod
    def _assembled_chat_stream_response(collected: Sequence[object]) -> ModelResponse | None:
        from litellm.main import stream_chunk_builder

        try:
            assembled: Final = stream_chunk_builder(chunks=list(collected))  # mutable-ok: builder requires a list
        except Exception:  # noqa: BLE001  # unassembleable streams route to the explicit fail-closed/pass-through path
            verbose_proxy_logger.warning("ThirdLaw guardrail: could not assemble streamed response", exc_info=True)
            return None
        return assembled if isinstance(assembled, ModelResponse) else None

    def _buffer_until_moderated(self) -> bool:
        """Whether to withhold the stream until the scan decides.

        Buffering replays the original chunks on release, so a guardrail asked to mask content would
        hand back the very text it was told to redact. Masking therefore wins over buffering.
        """
        requested: Final = self.streaming_buffer_until_moderated
        if requested and self.mask_response_content:
            verbose_proxy_logger.warning(
                "ThirdLaw guardrail: streaming_buffer_until_moderated is disabled for %s because "
                "mask_response_content=True -- buffered replay would release unredacted original chunks",
                self.guardrail_name,
            )
            return False
        return requested

    def _streaming_block_error(self, message: str) -> Exception:
        from litellm.proxy.proxy_server import StreamingCallbackError

        return StreamingCallbackError(message)

    @staticmethod
    def _stream_error_items(
        message: str, surface: StreamSurface, collected: Sequence[object]
    ) -> Sequence[object] | None:
        """Terminal stream items framing a refusal in this surface's wire format.

        ``None`` means the surface cannot frame its own error, so the caller raises instead and
        lets the proxy's data generator serialize it.
        """
        match surface:
            case StreamSurface.ANTHROPIC_MESSAGES:
                return anthropic_sse_error_frames(message)
            case StreamSurface.RESPONSES:
                return ThirdlawGuardrail._responses_error_events(message, collected)
            case StreamSurface.CHAT_COMPLETIONS | StreamSurface.OPAQUE_SSE:
                return None

    @staticmethod
    def _responses_error_events(message: str, collected: Sequence[object]) -> Sequence[object]:
        """A ``/v1/responses`` refusal as an in-stream error event.

        A raised exception reaches the client as the proxy's ``{"error": ...}`` blob, which carries
        no top-level ``type`` and so is not a Responses event at all. ``collected`` is forwarded so
        the error event continues the stream's sequence numbering rather than restarting at zero.
        """
        from fastapi import HTTPException

        from litellm.llms.openai.responses.guardrail_translation.handler import (
            OpenAIResponsesHandler,
        )

        return (
            OpenAIResponsesHandler().build_stream_error_items(
                HTTPException(status_code=400, detail=message),
                responses_so_far=collected,
            )
            or ()
        )

    async def _end_of_stream_moderated_stream(
        self,
        *,
        response: AsyncIterator[object],
        request_data: dict[str, object],  # mutable-ok: proxy-shared request dict; trace records land in it
        buffer: bool,
    ) -> AsyncGenerator[object, None]:
        started: Final = time.monotonic()
        collected: Final[list[object]] = []  # mutable-ok: streaming chunk buffer
        async for item in response:
            collected.append(item)
            if not buffer:
                yield item

        surface: Final = classify_stream(collected)
        assembled: Final = self._assembled_stream_response(collected, surface)
        if assembled is None:
            async for item in self._handle_unassembleable(collected=collected, surface=surface, buffer=buffer):
                yield item
            return

        stream_chunks, stream_sse = self._stream_payload(collected, surface)
        try:
            decision: Final = await self._run_thirdlaw(
                event_type=GuardrailEventHooks.post_call,
                wire_event="post_call",
                request_data=request_data,
                response_body=_response_payload(assembled),
                streamed_deltas=self._streamed_deltas_not_in_body(collected, assembled, surface),
                response_chunks=stream_chunks,
                response_sse=stream_sse,
            )
        except Exception as error:  # noqa: BLE001  # after keepalive flush a raise cannot reach the client; send a frame
            flushed_frames: Final = (
                self._stream_error_items(
                    f"ThirdLaw guardrail request failed: {_service_error_message(error)}", surface, collected
                )
                if self._sse_headers_flushed(started)
                else None
            )
            if flushed_frames is None:
                raise
            for frame in flushed_frames:
                yield frame
            return
        async for item in self._emit_end_of_stream_outcome(
            decision=decision, assembled=assembled, collected=collected, surface=surface, buffer=buffer
        ):
            yield item

    async def _emit_end_of_stream_outcome(
        self,
        *,
        decision: ThirdlawGuardrailResponse | None,
        assembled: _AssembledStream,
        collected: Sequence[object],
        surface: StreamSurface,
        buffer: bool,
    ) -> AsyncGenerator[object, None]:
        if decision is None or decision.action in ("allow", "modify_request"):
            if decision is not None and decision.action == "modify_request":
                verbose_proxy_logger.warning("ThirdLaw guardrail: ignoring modify_request decision on a stream")
            if buffer:
                for item in collected:
                    yield item
            return

        if decision.action == "block":
            message: Final = decision.message or "Content violates ThirdLaw policy"
            block_items: Final = self._stream_error_items(message, surface, collected)
            if block_items is None:
                raise self._streaming_block_error(message)
            for item in block_items:
                yield item
            return

        if not buffer:
            verbose_proxy_logger.warning(
                "ThirdLaw guardrail: modify_response arrived after chunks were already delivered "
                "(streaming_buffer_until_moderated=False); response not modified"
            )
            return
        if not decision.response_body:
            for item in collected:
                yield item
            return
        async for item in self._emit_modified_stream(
            assembled=assembled, replacement=decision.response_body, surface=surface
        ):
            yield item

    async def _handle_unassembleable(
        self,
        *,
        collected: Sequence[object],
        surface: StreamSurface,
        buffer: bool,
    ) -> AsyncGenerator[object, None]:
        """Refuse a stream that could not be assembled for scanning.

        A stream whose surface has no assembler, or that died before carrying a body, is refused
        unless ``unscannable_stream_fallback`` is ``fail_open``, because forwarding it unscanned
        lets a caller pick an endpoint to dodge the guardrail.
        """
        if is_terminal_error_stream(collected):
            # Only the refusal an earlier guardrail in the chain already emitted; replacing it
            # would hide the rejection the client is owed.
            for item in collected:
                yield item
            return
        refusal: Final = f"{self.guardrail_name}: streamed response could not be assembled for scanning, blocking it"
        if any(isinstance(item, ModelResponseStream) for item in collected):
            # Real chat-completions chunks that will not assemble are a failure to scan a supported
            # shape, not an unsupported one, so the fail_open opt-in does not cover them.
            raise self._streaming_block_error(refusal)
        # Only the explicit opt-in opens: Literal is not enforced at runtime, so a typo
        # in the config must fail closed rather than silently disable the scan.
        if self.unscannable_stream_fallback == "fail_open":
            verbose_proxy_logger.warning(
                "ThirdLaw guardrail: unsupported stream shape passed through unscanned "
                "(unscannable_stream_fallback=fail_open)"
            )
            if buffer:
                for item in collected:
                    yield item
            return
        refusal_items: Final = self._stream_error_items(refusal, surface, collected)
        if refusal_items is None:
            raise self._streaming_block_error(refusal)
        for item in refusal_items:
            yield item

    async def _emit_modified_stream(
        self,
        *,
        assembled: _AssembledStream,
        replacement: Mapping[str, object],
        surface: StreamSurface,
    ) -> AsyncGenerator[object, None]:
        if surface is StreamSurface.RESPONSES:
            async for event in self._emit_modified_responses_stream(assembled=assembled, replacement=replacement):
                yield event
            return
        if surface is StreamSurface.ANTHROPIC_MESSAGES:
            for frame in self._modified_anthropic_frames(assembled=assembled, replacement=replacement):
                yield frame
            return
        modified: Final = self._modified_response(response=assembled, replacement=replacement)
        if not isinstance(modified, ModelResponse):
            raise self._streaming_block_error(f"{self.guardrail_name}: modified streamed response failed validation")
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator

        async for chunk in MockResponseIterator(model_response=modified):
            yield chunk

    def _modified_anthropic_frames(
        self, *, assembled: _AssembledStream, replacement: Mapping[str, object]
    ) -> Sequence[bytes]:
        """Re-emit a rewritten Messages body as the SSE frames its client expects.

        A chat-shaped rewrite fails closed rather than being merged: the overlay is a shallow
        merge, so ``choices`` would land beside the untouched ``content`` and the stream would
        re-emit the very text the rewrite asked to redact.
        """
        if "choices" in replacement:
            raise self._streaming_block_error(
                f"{self.guardrail_name}: modify_response answered a /v1/messages stream with "
                '"choices"; the Messages body carries its text in "content"'
            )
        modified: Final = self._modified_response(response=assembled, replacement=replacement)
        body: Final = _dict_of(modified)
        if body is None:
            raise self._streaming_block_error(f"{self.guardrail_name}: modified streamed response failed validation")
        return anthropic_sse_chunks_from_body(body)

    async def _emit_modified_responses_stream(
        self,
        *,
        assembled: _AssembledStream,
        replacement: Mapping[str, object],
    ) -> AsyncGenerator[object, None]:
        """Re-emit a rewritten Responses body as the events its client expects."""
        from litellm.responses.streaming_iterator import (
            MockResponsesAPIStreamingIterator,
            build_synthetic_response_events,
        )

        modified: Final = self._modified_response(response=assembled, replacement=replacement)
        if not isinstance(modified, ResponsesAPIResponse):
            raise self._streaming_block_error(f"{self.guardrail_name}: modified streamed response failed validation")
        for event in build_synthetic_response_events(
            transformed=modified,
            logging_obj=None,
            chunk_size=MockResponsesAPIStreamingIterator.CHUNK_SIZE,
        ):
            yield event

    async def _sampled_stream(
        self,
        *,
        response: AsyncIterator[object],
        request_data: dict[str, object],  # mutable-ok: proxy-shared request dict; trace records land in it
    ) -> AsyncGenerator[object, None]:
        sampling_rate: Final = self.streaming_sampling_rate
        collected: Final[list[object]] = []  # mutable-ok: streaming chunk buffer
        async for item in response:
            collected.append(item)
            yield item
            if len(collected) % sampling_rate != 0 or classify_stream(collected) is not StreamSurface.CHAT_COMPLETIONS:
                continue
            interim = self._assembled_stream_response(collected, StreamSurface.CHAT_COMPLETIONS)
            if interim is None:
                continue
            interim_decision = await self._run_thirdlaw(
                event_type=GuardrailEventHooks.post_call,
                wire_event="post_call",
                request_data=request_data,
                response_body=_response_payload(interim),
            )
            if interim_decision is not None and interim_decision.action == "block":
                raise self._streaming_block_error(interim_decision.message or "Content violates ThirdLaw policy")

        surface: Final = classify_stream(collected)
        if surface is StreamSurface.RESPONSES:
            # Interim scans need an assembled body and a Responses stream carries none until its
            # terminal event, so nothing was scanned until every chunk had already been delivered.
            verbose_proxy_logger.warning(
                "ThirdLaw guardrail: a /v1/responses stream was scanned only once its turn "
                "finished, because interim scans have no body to read; set "
                "streaming_buffer_until_moderated=True to withhold it until the scan decides"
            )
        assembled: Final = self._assembled_stream_response(collected, surface)
        if assembled is None:
            async for item in self._handle_unassembleable(collected=collected, surface=surface, buffer=False):
                yield item
            return
        stream_chunks, stream_sse = self._stream_payload(collected, surface)
        final_decision: Final = await self._run_thirdlaw(
            event_type=GuardrailEventHooks.post_call,
            wire_event="post_call",
            request_data=request_data,
            response_body=_response_payload(assembled),
            streamed_deltas=self._streamed_deltas_not_in_body(collected, assembled, surface),
            response_chunks=stream_chunks,
            response_sse=stream_sse,
        )
        if final_decision is None:
            return
        if final_decision.action == "block":
            message: Final = final_decision.message or "Content violates ThirdLaw policy"
            block_items: Final = self._stream_error_items(message, surface, collected)
            if block_items is None:
                raise self._streaming_block_error(message)
            for item in block_items:
                yield item
            return
        if final_decision.action == "modify_response":
            verbose_proxy_logger.warning(
                "ThirdLaw guardrail: modify_response arrived after chunks were already delivered "
                "(streaming_end_of_stream_only=False); response not modified"
            )

    @staticmethod
    def _sse_headers_flushed(started_monotonic: float) -> bool:
        from litellm.proxy.common_utils.sse_keepalive import keepalive_ping_has_fired

        return keepalive_ping_has_fired(
            time.monotonic() - started_monotonic, litellm.anthropic_sse_ping_interval_seconds
        )
