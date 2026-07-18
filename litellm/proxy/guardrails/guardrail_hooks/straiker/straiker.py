from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NoReturn
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
from litellm.exceptions import (
    BadRequestError,
    GuardrailRaisedException,
    ModifyResponseException,
    Timeout,
)
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    get_session_id_from_request_data,
    log_guardrail_information,
)
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.straiker import (
    STRAIKER_WEBHOOK_SCHEMA_VERSION,
    StraikerGuardrailConfigModel,
    StraikerWebhookApplication,
    StraikerWebhookContent,
    StraikerWebhookContext,
    StraikerWebhookEvent,
    StraikerWebhookIdentity,
    StraikerWebhookRequest,
    StraikerWebhookResponse,
    StraikerWebhookStream,
    StraikerWebhookUsage,
)
from litellm.types.utils import GenericGuardrailAPIInputs, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME = "straiker"
DEFAULT_BLOCK_MESSAGE = "Content violates policy"
DEFAULT_API_BASE = "https://api.prod.straiker.ai"
DEFAULT_MAX_PAYLOAD_BYTES = 524288
WEBHOOK_PATH = "/api/v1/detect/webhook"
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
UNREACHABLE_STATUS = frozenset({502, 503, 504})
_APPLICATION_METADATA_KEYS = frozenset({"agent_id", "app_name"})
_OPAQUE_METADATA_SCALAR_TYPES = (str, int, float, bool)


@dataclass(frozen=True, slots=True)
class _WebhookFailure:
    message: str
    is_unreachable: bool


def _as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _merged_metadata(request_data: dict) -> dict:
    return {
        **_as_dict(request_data.get("metadata")),
        **_as_dict(request_data.get("litellm_metadata")),
    }


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _build_webhook_metadata(request_data: dict, default_metadata: dict[str, str]) -> dict[str, object] | None:
    out: dict[str, object] = {}
    for key, value in _as_dict(request_data.get("metadata")).items():
        if key in _APPLICATION_METADATA_KEYS or key.startswith("user_api_key_"):
            continue
        if key == "session_id":
            continue
        if isinstance(value, _OPAQUE_METADATA_SCALAR_TYPES):
            out[key] = value
    out.update(default_metadata)
    return out or None


def _extract_identity(request_data: dict) -> StraikerWebhookIdentity:
    meta = _merged_metadata(request_data)
    return StraikerWebhookIdentity(
        litellm_key=_as_optional_str(meta.get("user_api_key_alias"))
        or _as_optional_str(meta.get("user_api_key_hash"))
        or _as_optional_str(meta.get("user_api_key_token")),
        litellm_team=_as_optional_str(meta.get("user_api_key_team_alias"))
        or _as_optional_str(meta.get("user_api_key_team_id")),
        litellm_user_id=_as_optional_str(meta.get("user_api_key_user_id")),
        litellm_user_email=_as_optional_str(meta.get("user_api_key_user_email")),
        litellm_org_id=_as_optional_str(meta.get("user_api_key_org_id")),
        end_user_id=_as_optional_str(meta.get("user_api_key_end_user_id")),
    )


def _resolve_provider(request_data: dict, model: str | None) -> str | None:
    litellm_params = _as_dict(request_data.get("litellm_params"))
    custom_llm_provider = request_data.get("custom_llm_provider") or litellm_params.get("custom_llm_provider")
    if custom_llm_provider:
        return custom_llm_provider
    if not model:
        return None
    try:
        _, provider, _, _ = get_llm_provider(
            model=model,
            api_base=request_data.get("api_base") or litellm_params.get("api_base"),
            api_key=request_data.get("api_key") or litellm_params.get("api_key"),
        )
    except BadRequestError:
        return None
    return provider or None


def _resolve_destination(request_data: dict) -> str | None:
    litellm_params = _as_dict(request_data.get("litellm_params"))
    api_base = request_data.get("api_base") or litellm_params.get("api_base")
    if not isinstance(api_base, str):
        return None
    try:
        return urlsplit(api_base).hostname
    except ValueError:
        return None


def _resolve_call_surface(logging_obj: LiteLLMLoggingObj | None, request_data: dict) -> str:
    call_type = (
        (getattr(logging_obj, "call_type", None) if logging_obj is not None else None)
        or request_data.get("call_type")
        or request_data.get("litellm_call_type")
    )
    return call_type if isinstance(call_type, str) and call_type else "unknown"


def _response_finish_reason(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list):
        return None
    for choice in choices:
        reason = getattr(choice, "finish_reason", None)
        if isinstance(reason, str) and reason:
            return reason
    return None


def _build_usage(response: object) -> StraikerWebhookUsage | None:
    usage = getattr(response, "usage", None)
    if not isinstance(usage, Usage):
        return None
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    if input_tokens is None and output_tokens is None:
        return None
    return StraikerWebhookUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _is_streamed_request(request_data: dict) -> bool:
    if request_data.get("stream") is True:
        return True
    body = _as_dict(_as_dict(request_data.get("proxy_server_request")).get("body"))
    return body.get("stream") is True


class StraikerGuardrail(CustomGuardrail):
    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel]:
        return StraikerGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        api_key: str,
        api_base: str = DEFAULT_API_BASE,
        source: str = "LiteLLM Gateway",
        timeout: float = 5.0,
        max_retries: int = 2,
        initial_backoff: float = 0.1,
        max_backoff: float = 2.0,
        unreachable_fallback: Literal["fail_open", "fail_closed"] = "fail_closed",
        fail_on_error: bool = True,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        custom_headers: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        verbose: bool = False,
        async_handler: httpx.AsyncClient | None = None,
        **kwargs: object,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        if unreachable_fallback not in ("fail_open", "fail_closed"):
            raise ValueError(f"unreachable_fallback must be 'fail_open' or 'fail_closed'; got {unreachable_fallback!r}")

        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.source = source
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))
        self.initial_backoff = max(0.0, float(initial_backoff))
        self.max_backoff = max(self.initial_backoff, float(max_backoff))
        self.unreachable_fallback = unreachable_fallback
        self.fail_on_error = fail_on_error
        self.max_payload_bytes = int(max_payload_bytes)
        self.custom_headers = dict(custom_headers) if custom_headers else {}
        self.default_metadata = dict(metadata) if metadata else {}
        self.verbose = bool(verbose)

        self.streaming_end_of_stream_only = True
        self.streaming_buffer_until_moderated = True

        self.async_handler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        super().__init__(**kwargs)

    def _webhook_url(self) -> str:
        return f"{self.api_base}{WEBHOOK_PATH}"

    def _headers(self) -> dict[str, str]:
        reserved = {"authorization", "content-type", "x-straiker-webhook-format"}
        extra = {k: v for k, v in self.custom_headers.items() if k.lower() not in reserved}
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Straiker-Webhook-Format": "litellm",
            **extra,
        }

    def _build_application(self, request_data: dict) -> StraikerWebhookApplication:
        meta = _merged_metadata(request_data)
        agent_id = _as_optional_str(meta.get("agent_id"))
        return StraikerWebhookApplication(
            source=agent_id or self.source,
            name=_as_optional_str(meta.get("app_name")),
        )

    def _build_context(
        self,
        request_data: dict,
        model: str | None,
        logging_obj: LiteLLMLoggingObj | None,
    ) -> StraikerWebhookContext:
        return StraikerWebhookContext(
            call_surface=_resolve_call_surface(logging_obj, request_data),
            model=model,
            model_provider=_resolve_provider(request_data, model),
            destination=_resolve_destination(request_data),
            session_id=get_session_id_from_request_data(request_data),
            litellm_call_id=getattr(logging_obj, "litellm_call_id", None) if logging_obj else None,
            litellm_trace_id=getattr(logging_obj, "litellm_trace_id", None) if logging_obj else None,
            litellm_version=litellm_version,
        )

    def _build_envelope(
        self,
        *,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None,
    ) -> StraikerWebhookRequest:
        model = inputs.get("model") or request_data.get("model")
        call_id = getattr(logging_obj, "litellm_call_id", None) if logging_obj else None
        event_id = f"{call_id or 'litellm'}:{input_type}"

        content = StraikerWebhookContent(
            texts=list(inputs.get("texts") or []),
            images=list(inputs.get("images") or []),
            structured_messages=inputs.get("structured_messages"),
            tools=inputs.get("tools"),
            tool_calls=inputs.get("tool_calls"),
        )

        if input_type == "request":
            event = StraikerWebhookEvent(type="pre_call", id=event_id)
            return StraikerWebhookRequest(
                event=event,
                request=content,
                context=self._build_context(request_data, model, logging_obj),
                identity=_extract_identity(request_data),
                application=self._build_application(request_data),
                metadata=_build_webhook_metadata(request_data, self.default_metadata),
            )

        response_obj = request_data.get("response")
        content.finish_reason = _response_finish_reason(response_obj)
        original_messages = request_data.get("messages")
        request_content = StraikerWebhookContent(
            structured_messages=original_messages if isinstance(original_messages, list) else None,
        )
        phase: Literal["none", "assembled"] = "assembled" if _is_streamed_request(request_data) else "none"
        event = StraikerWebhookEvent(type="post_call", id=event_id, stream=StraikerWebhookStream(phase=phase))
        return StraikerWebhookRequest(
            event=event,
            request=request_content,
            response=content,
            context=self._build_context(request_data, model, logging_obj),
            identity=_extract_identity(request_data),
            application=self._build_application(request_data),
            usage=_build_usage(response_obj),
            metadata=_build_webhook_metadata(request_data, self.default_metadata),
        )

    async def _post_webhook(self, payload: dict) -> tuple[StraikerWebhookResponse | None, _WebhookFailure | None]:
        try:
            body = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as error:
            return None, _WebhookFailure(f"request serialization failed: {error}", is_unreachable=False)
        body_bytes = len(body)
        if body_bytes > self.max_payload_bytes:
            return None, _WebhookFailure(
                f"payload {body_bytes}B exceeds max_payload_bytes {self.max_payload_bytes}",
                is_unreachable=False,
            )

        url = self._webhook_url()
        headers = self._headers()
        attempts = self.max_retries + 1
        last_failure: _WebhookFailure | None = None

        if self.verbose:
            verbose_proxy_logger.info(
                json.dumps(
                    {
                        "event": "straiker.webhook_request",
                        "url": url,
                        "bytes": body_bytes,
                        "payload": payload,
                    },
                    default=str,
                )
            )

        for attempt in range(attempts):
            try:
                resp = await self.async_handler.post(url, content=body, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    try:
                        body = resp.json()
                        parsed = StraikerWebhookResponse.model_validate(body)
                    except (ValidationError, json.JSONDecodeError) as ve:
                        return None, _WebhookFailure(f"invalid response schema: {ve}", is_unreachable=False)
                    if self.verbose:
                        verbose_proxy_logger.info(
                            json.dumps(
                                {
                                    "event": "straiker.webhook_response",
                                    "status_code": resp.status_code,
                                    "body": body,
                                },
                                default=str,
                            )
                        )
                    return parsed, None
                last_failure = _WebhookFailure(
                    f"HTTP {resp.status_code}: {resp.text[:200]}",
                    is_unreachable=resp.status_code in UNREACHABLE_STATUS,
                )
                if resp.status_code not in RETRY_STATUS:
                    return None, last_failure
            except (httpx.RequestError, asyncio.TimeoutError, Timeout) as e:
                last_failure = _WebhookFailure(f"{type(e).__name__}: {e}", is_unreachable=True)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                return None, _WebhookFailure(f"{type(e).__name__}: {e}", is_unreachable=False)

            if attempt < attempts - 1:
                backoff = min(self.initial_backoff * (2**attempt), self.max_backoff)
                await asyncio.sleep(random.uniform(0, backoff))

        return None, last_failure or _WebhookFailure("unknown error", is_unreachable=True)

    def _record(
        self,
        *,
        request_data: dict,
        logging_obj: LiteLLMLoggingObj | None,
        parsed: StraikerWebhookResponse,
    ) -> None:
        if not self.verbose:
            return
        response_obj = request_data.get("response")
        hidden = getattr(response_obj, "_hidden_params", None)
        if isinstance(hidden, dict):
            straiker_hidden = hidden.setdefault("straiker", {})
            if isinstance(straiker_hidden, dict):
                straiker_hidden.update({"action": parsed.action, "turn_id": parsed.turn_id})

    def _fail(
        self,
        *,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        error: str,
        is_unreachable: bool,
    ) -> GenericGuardrailAPIInputs:
        fail_open = (is_unreachable and self.unreachable_fallback == "fail_open") or not self.fail_on_error
        verbose_proxy_logger.error(
            json.dumps(
                {
                    "event": "straiker.error",
                    "input_type": input_type,
                    "error": error,
                    "fail_open": fail_open,
                },
                default=str,
            )
        )
        if fail_open:
            return inputs
        self._block(
            request_data=request_data,
            input_type=input_type,
            message=f"Straiker detection unavailable: {error}",
        )

    def _block(
        self,
        *,
        request_data: dict,
        input_type: Literal["request", "response"],
        message: str,
    ) -> NoReturn:
        if input_type == "request":
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name or GUARDRAIL_NAME,
                message=message,
                should_wrap_with_default_message=False,
            )
        raise ModifyResponseException(
            message=message,
            model=request_data.get("model", "unknown") or "unknown",
            request_data=request_data,
            guardrail_name=self.guardrail_name or GUARDRAIL_NAME,
            original_response=request_data.get("response"),
        )

    @staticmethod
    def _intervened_inputs(
        inputs: GenericGuardrailAPIInputs,
        parsed: StraikerWebhookResponse,
    ) -> GenericGuardrailAPIInputs:
        return_inputs: GenericGuardrailAPIInputs = {}
        return_inputs.update(inputs)
        if parsed.texts is not None:
            return_inputs["texts"] = parsed.texts
        return return_inputs

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        try:
            envelope = self._build_envelope(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                logging_obj=logging_obj,
            )
            payload = envelope.model_dump(mode="json", exclude_none=True)
        except (ValidationError, TypeError, ValueError) as error:
            return self._fail(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                error=str(error),
                is_unreachable=False,
            )

        parsed, failure = await self._post_webhook(payload)
        if failure is not None:
            return self._fail(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                error=failure.message,
                is_unreachable=failure.is_unreachable,
            )

        if parsed is None:
            return self._fail(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                error="empty response from Straiker",
                is_unreachable=False,
            )
        self._record(request_data=request_data, logging_obj=logging_obj, parsed=parsed)

        if parsed.schema_version is not None and parsed.schema_version != STRAIKER_WEBHOOK_SCHEMA_VERSION:
            verbose_proxy_logger.warning(
                json.dumps(
                    {
                        "event": "straiker.schema_drift",
                        "expected": STRAIKER_WEBHOOK_SCHEMA_VERSION,
                        "received": parsed.schema_version,
                    }
                )
            )

        if parsed.action == "BLOCKED":
            self._block(
                request_data=request_data,
                input_type=input_type,
                message=parsed.blocked_reason or DEFAULT_BLOCK_MESSAGE,
            )
        if parsed.action == "GUARDRAIL_INTERVENED":
            is_streamed_response = input_type == "response" and _is_streamed_request(request_data)
            if parsed.texts is None or is_streamed_response:
                self._block(
                    request_data=request_data,
                    input_type=input_type,
                    message=parsed.blocked_reason or DEFAULT_BLOCK_MESSAGE,
                )
            return self._intervened_inputs(inputs, parsed)
        return inputs
