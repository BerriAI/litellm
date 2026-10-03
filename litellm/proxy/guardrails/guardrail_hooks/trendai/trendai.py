# Derived from tm-v1-ai-guard-litellm-plugin revision 6cafc143f62962a98d4eb5abe9f608c61ff194d4.
# This file has been modified for integration into LiteLLM.
# Licensed under the Apache License, Version 2.0. See LICENSE.txt in this directory.

import asyncio
import os
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NoReturn, Protocol
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx
from pydantic import ValidationError
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
from litellm.exceptions import GuardrailRaisedException, Timeout
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,  # pyright: ignore[reportUnknownVariableType]  # legacy decorator has an untyped signature
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # legacy client factory has an untyped params map
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import GenericGuardrailAPIInputs, GuardrailStatus

from ._models import (
    TrendAIAllow,
    TrendAIBlock,
    TrendAIChatCompletionPayload,
    TrendAIProviderFailure,
    TrendAIRedactedPrompt,
    TrendAIRedactedResponse,
    TrendAIResponse,
    TrendAIScanResult,
    TrendAIWindowScan,
)
from ._text import apply_window_redaction, locate_request_prompt, redact_request_texts, utf8_windows

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GUARDRAIL_NAME: Final = "trendai"
OPENAI_CHAT_COMPLETION_RESPONSE_V1: Final = "OpenAIChatCompletionResponseV1"
RESPONSE_CONTENT_CHUNK_SIZE_BYTES: Final = 49_500
TMV1_CLIENT_NAME: Final = "litellm"
PLUGIN_VERSION: Final = "0.1.2"
PROVIDER_UNAVAILABLE_STATUS: Final = 503
_APPLY_GUARDRAILS_PATH: Final = "/applyGuardrails"
_TREND_AI_SECURITY_PATH: Final = "/v3.0/aiSecurity"
_RESPONSE_MODEL: Final = "guardrailed-response"
_REDACTION_FAILED_MESSAGE: Final = "Trend AI Guard could not apply the requested redaction"
_NO_ENTITIES: Final[Mapping[str, int]] = MappingProxyType({})


class _AsyncHTTPClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        json: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response: ...


class TrendAIGuardrail(CustomGuardrail):
    records_own_guardrail_information: ClassVar[bool] = True

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        app_name: str | None = None,
        fallback_on_error: Literal["block", "allow"] = "block",
        timeout: float = 5.0,
        stream_overlap_size: int = 256,
        response_content_chunk_size_bytes: int = RESPONSE_CONTENT_CHUNK_SIZE_BYTES,
        async_handler: _AsyncHTTPClient | None = None,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None = None,
        default_on: bool = False,
    ) -> None:
        resolved_api_key: Final = api_key or os.environ.get("TMV1_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Trend AI Guard requires an API key. Pass api_key or set the TMV1_API_KEY environment variable."
            )

        resolved_api_base: Final = api_base or os.environ.get("TRENDAI_AI_GUARD_BASE_URL")
        if not resolved_api_base:
            raise ValueError(
                "Trend AI Guard requires an API base URL. Pass api_base or set the "
                "TRENDAI_AI_GUARD_BASE_URL environment variable."
            )
        if fallback_on_error not in ("block", "allow"):
            raise ValueError("fallback_on_error must be 'block' or 'allow'")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if stream_overlap_size < 0:
            raise ValueError("stream_overlap_size must be non-negative")
        if response_content_chunk_size_bytes < 1:
            raise ValueError("response_content_chunk_size_bytes must be greater than zero")

        self.api_key: str = resolved_api_key
        self.api_url: str = _build_apply_guardrails_url(resolved_api_base)
        self.app_name: str = app_name or os.environ.get("TMV1_APPLICATION_NAME", "litellm")
        self.fallback_on_error: Literal["block", "allow"] = fallback_on_error
        self.timeout: float = timeout
        self.stream_overlap_size: int = stream_overlap_size
        self.response_content_chunk_size_bytes: int = response_content_chunk_size_bytes
        self.async_handler: _AsyncHTTPClient = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # base constructor retains untyped extension kwargs
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            supported_event_hooks=self.get_supported_event_hooks(),
        )

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: CustomGuardrail contract
        return [  # mutable-ok: CustomGuardrail contract
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
        ]

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: CustomGuardrail hook contract is a plain dict
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = tuple(inputs.get("texts") or ())
        match input_type:
            case "request":
                return await self._apply_request_guardrail(inputs, texts, request_data)
            case "response":
                return await self._apply_response_guardrail(inputs, texts, request_data)
            case _:
                assert_never(input_type)

    async def _apply_request_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        texts: tuple[str, ...],
        request_data: dict[str, object],  # mutable-ok: CustomGuardrail hook contract is a plain dict
    ) -> GenericGuardrailAPIInputs:
        prompt: Final = locate_request_prompt(texts, inputs.get("structured_messages"))
        if prompt is None:
            verbose_proxy_logger.debug("Trend AI Guard: no user prompt to scan in request inputs")
            return inputs
        started_at: Final = time.time()
        result: Final = await self._scan_payload(MappingProxyType({"prompt": prompt.prompt}))
        self._record_scan(result, request_data, started_at, event_type=None)
        redacted: Final = self._enforce(result)
        if redacted is None:
            return inputs
        redacted_inputs: Final[GenericGuardrailAPIInputs] = {
            **inputs,
            "texts": list(redact_request_texts(texts, prompt, redacted)),  # mutable-ok: GenericGuardrailAPIInputs field
        }
        return redacted_inputs

    async def _apply_response_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        texts: tuple[str, ...],
        request_data: dict[str, object],  # mutable-ok: CustomGuardrail hook contract is a plain dict
    ) -> GenericGuardrailAPIInputs:
        if not texts:
            verbose_proxy_logger.debug("Trend AI Guard: no response text to scan")
            return inputs
        model: Final = inputs.get("model") or _RESPONSE_MODEL
        started_at: Final = time.time()
        scans: Final = tuple([await self._scan_response_windows(text, model) for text in texts])
        all_scans: Final = tuple(scan for text_scans in scans for scan in text_scans)
        verdict: Final = _window_verdict(all_scans)
        self._record_scan(
            verdict,
            request_data,
            started_at,
            event_type=None,
            redacted=any(_is_redacting(scan.result) for scan in all_scans),
        )
        self._enforce(verdict)
        redacted_texts: Final = tuple(
            _merge_window_redactions(text, text_scans) for text, text_scans in zip(texts, scans, strict=True)
        )
        merged_texts: Final = tuple(text for text in redacted_texts if text is not None)
        if len(merged_texts) != len(texts):
            self._raise_redaction_failure(request_data, started_at, event_type=None)
        redacted_inputs: Final[GenericGuardrailAPIInputs] = {
            **inputs,
            "texts": list(merged_texts),  # mutable-ok: GenericGuardrailAPIInputs field
        }
        return redacted_inputs

    async def _scan_response_windows(self, content: str, model: str) -> tuple[TrendAIWindowScan, ...]:
        return tuple([scan async for scan in self._iter_response_window_scans(content, model)])

    async def _iter_response_window_scans(self, content: str, model: str) -> AsyncIterator[TrendAIWindowScan]:
        """Scan ``content`` window by window, stopping at the first block or provider failure."""
        for window in utf8_windows(
            content,
            chunk_size_bytes=self.response_content_chunk_size_bytes,
            overlap_chars=self.stream_overlap_size,
        ):
            result = await self._scan_payload(
                _chat_completion_payload(window.text, model),
                request_type=OPENAI_CHAT_COMPLETION_RESPONSE_V1,
            )
            yield TrendAIWindowScan(window=window, result=result)
            if not isinstance(result, TrendAIAllow):
                return

    def _raise_redaction_failure(
        self,
        request_data: dict[str, object],  # mutable-ok: CustomGuardrail hook contract is a plain dict
        started_at: float,
        *,
        event_type: GuardrailEventHooks | None,
    ) -> NoReturn:
        self._record_failure(_REDACTION_FAILED_MESSAGE, request_data, started_at, event_type=event_type)
        raise GuardrailRaisedException(
            guardrail_name=self.guardrail_name,
            message=_REDACTION_FAILED_MESSAGE,
            should_wrap_with_default_message=False,
            status_code=PROVIDER_UNAVAILABLE_STATUS,
        )

    def _enforce(self, result: TrendAIScanResult) -> str | None:
        """Raise for a block or a fail-closed provider failure; otherwise return any redacted content."""
        match result:
            case TrendAIAllow(redacted_content=redacted_content):
                return redacted_content
            case TrendAIBlock(reason=reason, status_code=status_code):
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"Blocked by Trend AI Guard. Security violation: {reason}",
                    should_wrap_with_default_message=False,
                    status_code=status_code,
                    blocked_content=True,
                )
            case TrendAIProviderFailure(reason=reason, status_code=status_code):
                if self.fallback_on_error == "allow":
                    verbose_proxy_logger.warning(
                        "Trend AI Guard: %s (status=%s); allowing traffic (fallback_on_error=allow)",
                        reason,
                        status_code,
                    )
                    return None
                verbose_proxy_logger.error(
                    "Trend AI Guard: %s (status=%s); blocking traffic (fallback_on_error=block)", reason, status_code
                )
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"Security Guard Error: {reason}",
                    should_wrap_with_default_message=False,
                    status_code=PROVIDER_UNAVAILABLE_STATUS,
                )
            case _:
                assert_never(result)

    def _record_scan(
        self,
        result: TrendAIScanResult,
        request_data: dict[str, object],  # mutable-ok: CustomGuardrail hook contract is a plain dict
        started_at: float,
        *,
        event_type: GuardrailEventHooks | None,
        redacted: bool | None = None,
    ) -> None:
        match result:
            case TrendAIAllow(redacted_content=redacted_content, masked_entity_count=masked_entity_count):
                self._record(
                    _allow_record(redacted_content is not None if redacted is None else redacted),
                    "success",
                    request_data,
                    started_at,
                    event_type=event_type,
                    masked_entity_count=_merge_entity_counts(_NO_ENTITIES, masked_entity_count),
                )
            case TrendAIBlock(reason=reason):
                self._record(
                    MappingProxyType({"action": "block", "reason": reason}),
                    "guardrail_intervened",
                    request_data,
                    started_at,
                    event_type=event_type,
                )
            case TrendAIProviderFailure(reason=reason, status_code=status_code):
                self._record(
                    MappingProxyType(
                        {"description": reason, "status_code": status_code, "fallback_on_error": self.fallback_on_error}
                    ),
                    "guardrail_failed_to_respond",
                    request_data,
                    started_at,
                    event_type=event_type,
                )
            case _:
                assert_never(result)

    def _record_failure(
        self,
        description: str,
        request_data: dict[str, object],  # mutable-ok: CustomGuardrail hook contract is a plain dict
        started_at: float,
        *,
        event_type: GuardrailEventHooks | None,
    ) -> None:
        self._record(
            MappingProxyType({"description": description}),
            "guardrail_failed_to_respond",
            request_data,
            started_at,
            event_type=event_type,
        )

    def _record(
        self,
        guardrail_json_response: Mapping[str, object],
        guardrail_status: GuardrailStatus,
        request_data: dict[str, object],  # mutable-ok: CustomGuardrail hook contract is a plain dict
        started_at: float,
        *,
        event_type: GuardrailEventHooks | None,
        masked_entity_count: Mapping[str, int] = _NO_ENTITIES,
    ) -> None:
        ended_at: Final = time.time()
        self.add_standard_logging_guardrail_information_to_request_data(  # pyright: ignore[reportUnknownMemberType]  # base signature takes an untyped dict
            guardrail_json_response=dict(guardrail_json_response),  # mutable-ok: base signature takes a plain dict
            request_data=request_data,
            guardrail_status=guardrail_status,
            start_time=started_at,
            end_time=ended_at,
            duration=ended_at - started_at,
            event_type=event_type,
            masked_entity_count=dict(masked_entity_count) or None,  # mutable-ok: base signature takes a plain dict
        )

    def _build_request_headers(self, request_type: str | None = None) -> Mapping[str, str]:
        headers: Final = (
            ("TMV1-Application-Name", self.app_name),
            ("Authorization", f"Bearer {self.api_key}"),
            ("Content-Type", "application/json"),
            ("TMV1-Client-Name", TMV1_CLIENT_NAME),
            ("TMV1-Client-Version", litellm_version),
            ("TMV1-Plugin-Version", PLUGIN_VERSION),
            ("prefer", "redact-pii,return=representation"),
            *_optional_header("TMV1-Request-Type", request_type),
        )
        return MappingProxyType(dict(headers))

    async def _scan_payload(
        self,
        payload: Mapping[str, object],
        request_type: str | None = None,
    ) -> TrendAIScanResult:
        try:
            response: Final = await self.async_handler.post(
                self.api_url,
                json=dict(payload),  # mutable-ok: httpx takes a plain dict
                headers=dict(self._build_request_headers(request_type)),  # mutable-ok: httpx takes a plain dict
                timeout=self.timeout,
            )
            response.raise_for_status()
            parsed: Final = TrendAIResponse.model_validate_json(response.content)
        except httpx.HTTPStatusError as error:
            return TrendAIProviderFailure(
                reason="Trend AI Guard returned an HTTP error",
                status_code=error.response.status_code,
            )
        except (httpx.RequestError, Timeout, asyncio.TimeoutError) as error:
            return TrendAIProviderFailure(reason=f"Trend AI Guard request failed: {type(error).__name__}")
        except ValidationError:
            return TrendAIProviderFailure(reason="Trend AI Guard returned an invalid response")

        normalized_action: Final = parsed.action.strip().lower()
        if normalized_action == "block":
            reason: Final = ", ".join(parsed.reasons) or parsed.reason or "Content policy violation"
            return TrendAIBlock(reason=reason)
        if normalized_action != "allow":
            return TrendAIProviderFailure(reason=f"Trend AI Guard returned an unsupported action: {parsed.action!r}")

        redacted_content: Final = _extract_redacted_content(request_type, parsed)
        if parsed.redacted_request is not None and redacted_content is None:
            return TrendAIProviderFailure(reason="Trend AI Guard returned an invalid redacted payload")

        entity_counts: Final = (
            tuple(
                sorted(
                    (
                        rule_id,
                        sum(1 for candidate in parsed.sensitive_information.rules if candidate.id.strip() == rule_id),
                    )
                    for rule_id in frozenset(
                        rule.id.strip() for rule in parsed.sensitive_information.rules if rule.id.strip()
                    )
                )
            )
            if parsed.sensitive_information is not None and redacted_content is not None
            else ()
        )
        return TrendAIAllow(redacted_content=redacted_content, masked_entity_count=entity_counts)


def _chat_completion_payload(content: str, model: str) -> Mapping[str, object]:
    return MappingProxyType(TrendAIChatCompletionPayload.for_content(content, model).model_dump())


def _optional_header(name: str, value: str | None) -> tuple[tuple[str, str], ...]:
    return () if value is None else ((name, value),)


def _allow_record(redacted: bool) -> Mapping[str, object]:
    return MappingProxyType({"action": "allow", "redacted": redacted})


def _is_redacting(result: TrendAIScanResult) -> bool:
    return isinstance(result, TrendAIAllow) and result.redacted_content is not None


def _window_verdict(scans: Sequence[TrendAIWindowScan]) -> TrendAIScanResult:
    """Collapse window scans into one verdict: the first non-allow result, else an allow carrying every entity count."""
    terminal: Final = next((scan.result for scan in scans if not isinstance(scan.result, TrendAIAllow)), None)
    if terminal is not None:
        return terminal
    counts: Final = tuple(
        pair for scan in scans if isinstance(scan.result, TrendAIAllow) for pair in scan.result.masked_entity_count
    )
    return TrendAIAllow(masked_entity_count=tuple(sorted(_merge_entity_counts(_NO_ENTITIES, counts).items())))


def _merge_entity_counts(counts: Mapping[str, int], additions: Sequence[tuple[str, int]]) -> Mapping[str, int]:
    """Per-entity maxima: overlapping scans may report the same entity more than once."""
    entities: Final = frozenset(counts) | frozenset(entity for entity, _ in additions)
    return MappingProxyType(
        {
            entity: max((counts.get(entity, 0), *(count for candidate, count in additions if candidate == entity)))
            for entity in entities
        }
    )


def _merge_window_redactions(content: str, scans: Sequence[TrendAIWindowScan]) -> str | None:
    """Fold every window's redaction into ``content``; None if any window cannot be merged positionally."""
    merged = content  # rebind-ok: folded across the windows
    for scan in scans:
        if not isinstance(scan.result, TrendAIAllow) or scan.result.redacted_content is None:
            continue
        merged = apply_window_redaction(merged, scan.window, scan.result.redacted_content)
        if merged is None:
            return None
    return merged


def _build_apply_guardrails_url(api_base: str) -> str:
    parsed: Final = urlsplit(api_base.strip())
    normalized_path: Final = parsed.path.rstrip("/")
    path_with_product: Final = (
        f"{normalized_path}{_TREND_AI_SECURITY_PATH}"
        if parsed.hostname is not None
        and parsed.hostname.lower().endswith(".trendmicro.com")
        and "/aiSecurity" not in normalized_path
        else normalized_path
    )
    final_path: Final = (
        path_with_product
        if path_with_product.endswith(_APPLY_GUARDRAILS_PATH)
        else f"{path_with_product}{_APPLY_GUARDRAILS_PATH}"
    )
    return urlunsplit(SplitResult(parsed.scheme, parsed.netloc, final_path, parsed.query, parsed.fragment))


def _extract_redacted_content(request_type: str | None, response: TrendAIResponse) -> str | None:
    payload: Final = response.redacted_request
    if payload is None:
        return None
    try:
        if request_type == OPENAI_CHAT_COMPLETION_RESPONSE_V1:
            parsed_response: Final = TrendAIRedactedResponse.model_validate(payload)
            content: Final = " ".join(
                choice.message.content
                for choice in parsed_response.choices
                if choice.message is not None and choice.message.content
            )
            return content or None
        return TrendAIRedactedPrompt.model_validate(payload).prompt or None
    except ValidationError:
        return None
