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
from pydantic import TypeAdapter, ValidationError

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
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.anthropic_sse import (
    anthropic_sse_chunks_from_response,
    anthropic_sse_error_frames,
    assemble_anthropic_sse_stream,
    is_raw_sse_stream,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks
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

# LiteLLM bookkeeping and transport-credential keys that ride along in ``request_data``
# but are not part of the provider request body. ``secret_fields`` holds plaintext
# Authorization values and ``api_key`` can carry a client-forwarded provider key, so
# neither may leave the proxy; ``proxy_server_request`` is self-referential.
_BODY_STRIP_KEYS: Final = frozenset(
    {
        "api_key",
        "guardrail_config",
        "guardrails",
        "headers",
        "litellm_call_id",
        "litellm_logging_obj",
        "litellm_metadata",
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

# Keys a modify_request decision may never write back. Beyond the stripped keys:
# ``guardrails`` gates which guardrails run, ``model`` was authorized against the
# key/team before this hook, ``stream`` changes the wire protocol mid-request.
_WRITE_BACK_DENY_KEYS: Final = _BODY_STRIP_KEYS | frozenset({"model", "policies", "stream", "user"})

_USER_METADATA_FIELDS: Final = (
    "user_api_key_hash",
    "user_api_key_alias",
    "user_api_key_user_id",
    "user_api_key_user_email",
    "user_api_key_team_id",
    "user_api_key_team_alias",
    "user_api_key_end_user_id",
    "user_api_key_org_id",
)

_WireEvent: TypeAlias = Literal["pre_call", "during_call", "post_call"]

_JSON_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])

_EMPTY_MAP: Final[Mapping[str, object]] = MappingProxyType({})

_EMPTY_STR_MAP: Final[Mapping[str, str]] = MappingProxyType({})


def _dict_of(value: object) -> Mapping[str, object] | None:
    return _JSON_DICT_ADAPTER.validate_python(value) if isinstance(value, dict) else None


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
    model: Final = request_data.get("model")
    return ThirdlawGuardrailRequestMetadata(
        litellm_version=litellm_version,
        litellm_call_id=call_id if isinstance(call_id, str) else None,
        litellm_trace_id=trace_id if isinstance(trace_id, str) else None,
        model=model if isinstance(model, str) else None,
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


def _is_unreachable_error(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in _UNREACHABLE_STATUS_CODES
    return isinstance(error, (httpx.RequestError, LitellmTimeout))


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
            message=f"ThirdLaw guardrail request failed: {error}",
        ) from error

    async def _run_thirdlaw(
        self,
        *,
        event_type: GuardrailEventHooks,
        wire_event: _WireEvent,
        request_data: dict[str, object],  # mutable-ok: proxy-shared request dict; trace records land in it
        response_body: Mapping[str, object] | None = None,
    ) -> ThirdlawGuardrailResponse | None:
        """POST the full payload to ThirdLaw and return its decision.

        Returns None when the service is unreachable and the guardrail is configured
        fail-open; raises for every other failure and records the guardrail trace on
        all paths (the decorator on the lifecycle hooks skips its auto-record when an
        entry was already written here).
        """
        started_at: Final = datetime.now(timezone.utc)
        payload: Final = self._build_wire_request(
            wire_event=wire_event, request_data=request_data, response_body=response_body
        )
        try:
            http_response: Final = await self.async_handler.post(
                url=self.api_base,
                headers=dict(self.http_headers),  # mutable-ok: the HTTP client requires a plain dict
                json=payload.model_dump(mode="json", exclude_none=True),
            )
            if http_response is None:
                raise ValueError("ThirdLaw guardrail HTTP client returned no response")
            http_response.raise_for_status()
            decision: Final = ThirdlawGuardrailResponse.model_validate(http_response.json())
        except Exception as error:  # noqa: BLE001  # every transport/parse failure funnels into the fallback policy
            self._record_trace(
                request_data=request_data,
                event_type=event_type,
                status="guardrail_failed_to_respond",
                started_at=started_at,
                trace={"error": str(error)},  # mutable-ok: one-shot trace payload
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

    def _block_exception(self, decision: ThirdlawGuardrailResponse) -> GuardrailRaisedException:
        return GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message=decision.message or "Content violates ThirdLaw policy",
            should_wrap_with_default_message=False,
            status_code=decision.response_status or 400,
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

    def _modified_response(self, *, response: object, replacement: Mapping[str, object]) -> object:
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
                return ModelResponse.model_validate(merged)
            except ValidationError as error:
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"ThirdLaw guardrail returned a malformed modified response: {error}",
                ) from error
        response_dict: Final = _dict_of(response)
        if response_dict is not None:
            return {**response_dict, **replacement}  # mutable-ok: the proxy owns the replaced response dict
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
        end_of_stream_only: Final = self.streaming_buffer_until_moderated or self.streaming_end_of_stream_only
        iterator: Final = (
            self._end_of_stream_moderated_stream(
                response=response, request_data=request_data, buffer=self.streaming_buffer_until_moderated
            )
            if end_of_stream_only
            else self._sampled_stream(response=response, request_data=request_data)
        )
        async for item in iterator:
            # Raw-SSE frames (bytes) ride through the same pipe; the proxy's data
            # generator forwards str/bytes chunks verbatim, matching bedrock's hook.
            yield cast(ModelResponseStream, item)  # cast-ok: raw-SSE byte frames share the typed stream (bedrock)

    @staticmethod
    def _assembled_stream_response(collected: Sequence[object], raw_sse: bool) -> ModelResponse | None:
        if raw_sse:
            return assemble_anthropic_sse_stream(collected, restore_identity=True)
        from litellm.main import stream_chunk_builder

        try:
            assembled: Final = stream_chunk_builder(chunks=list(collected))  # mutable-ok: builder requires a list
        except Exception:  # noqa: BLE001  # unassembleable streams route to the explicit fail-closed/pass-through path
            verbose_proxy_logger.warning("ThirdLaw guardrail: could not assemble streamed response", exc_info=True)
            return None
        return assembled if isinstance(assembled, ModelResponse) else None

    def _streaming_block_error(self, message: str) -> Exception:
        from litellm.proxy.proxy_server import StreamingCallbackError

        return StreamingCallbackError(message)

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

        raw_sse: Final = is_raw_sse_stream(collected)
        assembled: Final = self._assembled_stream_response(collected, raw_sse)
        if assembled is None:
            async for item in self._handle_unassembleable(collected=collected, raw_sse=raw_sse, buffer=buffer):
                yield item
            return

        try:
            decision: Final = await self._run_thirdlaw(
                event_type=GuardrailEventHooks.post_call,
                wire_event="post_call",
                request_data=request_data,
                response_body=_response_payload(assembled),
            )
        except Exception as error:  # noqa: BLE001  # after keepalive flush a raise cannot reach the client; send a frame
            if raw_sse and self._sse_headers_flushed(started):
                for frame in anthropic_sse_error_frames(f"ThirdLaw guardrail request failed: {error}"):
                    yield frame
                return
            raise
        async for item in self._emit_end_of_stream_outcome(
            decision=decision, assembled=assembled, collected=collected, raw_sse=raw_sse, buffer=buffer
        ):
            yield item

    async def _emit_end_of_stream_outcome(
        self,
        *,
        decision: ThirdlawGuardrailResponse | None,
        assembled: ModelResponse,
        collected: Sequence[object],
        raw_sse: bool,
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
            # A raw-SSE block always travels as an Anthropic error frame: a raised
            # exception is serialized as an OpenAI-shape error blob the Anthropic SDK
            # cannot parse.
            if raw_sse:
                for frame in anthropic_sse_error_frames(decision.message or "Content violates ThirdLaw policy"):
                    yield frame
                return
            raise self._streaming_block_error(decision.message or "Content violates ThirdLaw policy")

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
            assembled=assembled, replacement=decision.response_body, raw_sse=raw_sse
        ):
            yield item

    async def _handle_unassembleable(
        self,
        *,
        collected: Sequence[object],
        raw_sse: bool,
        buffer: bool,
    ) -> AsyncGenerator[object, None]:
        """Fail closed for scannable stream shapes; pass through the rest.

        A blanket fail-closed would break /v1/responses and text-completion streams
        (which assemble to non-ModelResponse shapes) for default_on deployments.
        """
        if raw_sse:
            for frame in anthropic_sse_error_frames(
                f"{self.guardrail_name}: streamed response could not be assembled for scanning, blocking it"
            ):
                yield frame
            return
        if any(isinstance(item, ModelResponseStream) for item in collected):
            raise self._streaming_block_error(
                f"{self.guardrail_name}: streamed response could not be assembled for scanning, blocking it"
            )
        verbose_proxy_logger.warning("ThirdLaw guardrail: unsupported stream shape; passing through without scanning")
        if buffer:
            for item in collected:
                yield item

    async def _emit_modified_stream(
        self,
        *,
        assembled: ModelResponse,
        replacement: Mapping[str, object],
        raw_sse: bool,
    ) -> AsyncGenerator[object, None]:
        modified: Final = self._modified_response(response=assembled, replacement=replacement)
        if not isinstance(modified, ModelResponse):
            raise self._streaming_block_error(f"{self.guardrail_name}: modified streamed response failed validation")
        if raw_sse:
            for frame in anthropic_sse_chunks_from_response(modified):
                yield frame
            return
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator

        async for chunk in MockResponseIterator(model_response=modified):
            yield chunk

    async def _sampled_stream(
        self,
        *,
        response: AsyncIterator[object],
        request_data: dict[str, object],  # mutable-ok: proxy-shared request dict; trace records land in it
    ) -> AsyncGenerator[object, None]:
        collected: Final[list[object]] = []  # mutable-ok: streaming chunk buffer
        async for item in response:
            collected.append(item)
            yield item
            if len(collected) % self.streaming_sampling_rate != 0 or is_raw_sse_stream(collected):
                continue
            interim = self._assembled_stream_response(collected, raw_sse=False)
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

        raw_sse: Final = is_raw_sse_stream(collected)
        assembled: Final = self._assembled_stream_response(collected, raw_sse)
        if assembled is None:
            async for item in self._handle_unassembleable(collected=collected, raw_sse=raw_sse, buffer=False):
                yield item
            return
        final_decision: Final = await self._run_thirdlaw(
            event_type=GuardrailEventHooks.post_call,
            wire_event="post_call",
            request_data=request_data,
            response_body=_response_payload(assembled),
        )
        if final_decision is None:
            return
        if final_decision.action == "block":
            if raw_sse:
                for frame in anthropic_sse_error_frames(final_decision.message or "Content violates ThirdLaw policy"):
                    yield frame
                return
            raise self._streaming_block_error(final_decision.message or "Content violates ThirdLaw policy")
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
