"""NeuralTrust TrustGuard native LiteLLM guardrail.

Calls TrustGuard POST /v1/evaluate on pre_call (input) and post_call (output).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import httpx
from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import Timeout
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    get_session_id_from_request_data,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.neuraltrust import DEFAULT_API_BASE
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

EVALUATE_PATH: Final = "/v1/evaluate"
DEFAULT_TIMEOUT: Final = 5.0
STATUS_BLOCK: Final = "block"
STATUS_TRANSFORM: Final = "transform"
STATUS_REPORT: Final = "report"
STATUS_ALLOW: Final = "allow"
KNOWN_STATUSES: Final = frozenset({STATUS_ALLOW, STATUS_BLOCK, STATUS_TRANSFORM, STATUS_REPORT})
UNREACHABLE_HTTP_STATUSES: Final = frozenset({502, 504})


class _TrustGuardUnreachable(Exception):
    """Transport or availability failure; eligible for unreachable_fallback."""


def _message_text(message: Mapping[str, object]) -> str | None:
    content: Final = message.get("content")
    return content if isinstance(content, str) and content else None


def _copy_messages(messages: list[object]) -> list[dict[str, object]] | None:
    copied: list[dict[str, object]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        copied.append(dict(message))
    return copied


def _texts_from_messages(messages: list[dict[str, object]]) -> list[str]:
    return [text for message in messages if (text := _message_text(message)) is not None]


def _rewrite_last_user_message(
    messages: list[dict[str, object]],
    redacted: str,
) -> list[dict[str, object]]:
    rewritten: Final = [dict(message) for message in messages]
    last_user: int | None = None
    for index, message in enumerate(rewritten):
        if message.get("role") == "user":
            last_user = index
    target: Final = last_user if last_user is not None else len(rewritten) - 1
    if target < 0:
        return [{"role": "user", "content": redacted}]
    rewritten[target] = {**rewritten[target], "content": redacted}
    return rewritten


def _model_name(
    inputs: GenericGuardrailAPIInputs,
    logging_obj: LiteLLMLoggingObj | None,
) -> str:
    if logging_obj is not None and logging_obj.model:
        return str(logging_obj.model)
    return str(inputs.get("model") or "")


class NeuralTrustGuardrail(CustomGuardrail):
    """LiteLLM hook that evaluates prompts and completions with TrustGuard."""

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel]:
        from litellm.types.proxy.guardrails.guardrail_hooks.neuraltrust import (
            NeuralTrustGuardrailConfigModel,
        )

        return NeuralTrustGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        collector_key: str | None = None,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        timeout: float | None = None,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None = None,
        default_on: bool = False,
    ) -> None:
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )
        self.api_base = (api_base or os.environ.get("TRUSTGUARD_API_BASE") or DEFAULT_API_BASE).rstrip("/")
        self.api_key = api_key or os.environ.get("TRUSTGUARD_API_KEY") or ""
        if not self.api_key:
            raise ValueError(
                "TrustGuard API key is required. Set TRUSTGUARD_API_KEY or pass api_key in litellm_params."
            )
        self.collector_key = collector_key or os.environ.get("TRUSTGUARD_COLLECTOR_KEY") or ""
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = unreachable_fallback
        resolved_timeout: Final = DEFAULT_TIMEOUT if timeout is None else float(timeout)
        self.timeout = resolved_timeout
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=list(self.get_supported_event_hooks()),
            event_hook=event_hook,
            default_on=default_on,
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,  # mutable-ok: CustomGuardrail.apply_guardrail contract
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        body: Final = self._evaluate_body(inputs, request_data, input_type, logging_obj)
        try:
            result: Final = await self._call_evaluate(body)
        except HTTPException:
            raise
        except _TrustGuardUnreachable as exc:
            return self._handle_unreachable(inputs, exc)

        status: Final = result["status"]
        if status == STATUS_BLOCK:
            raise HTTPException(
                status_code=400,
                detail={  # mutable-ok: FastAPI HTTPException.detail is a JSON object
                    "error": "Violated guardrail policy",
                    "neuraltrust_guardrail_response": "Blocked by NeuralTrust TrustGuard.",
                    "trace_id": result.get("trace_id"),
                    "request_id": result.get("request_id"),
                },
            )
        if status == STATUS_TRANSFORM:
            return self._apply_transform(inputs, result.get("transformed_payload"))
        if status == STATUS_REPORT:
            verbose_proxy_logger.info("TrustGuard report-only findings trace_id=%s", result.get("trace_id"))
        return inputs

    def _evaluate_body(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None,
    ) -> dict[str, object]:
        body: dict[str, object] = {  # mutable-ok: outbound JSON
            "payload": self._payload(inputs, input_type),
            "direction": "input" if input_type == "request" else "output",
            "protocol": "llm",
            "attributes": {
                "content_type": "application/json",
                "model": {"name": _model_name(inputs, logging_obj)},
            },
        }
        if self.collector_key:
            body["collector_key"] = self.collector_key
        session_id: Final = get_session_id_from_request_data(request_data)
        if session_id:
            body["session_id"] = session_id
        return body

    @staticmethod
    def _payload(
        inputs: GenericGuardrailAPIInputs,
        input_type: Literal["request", "response"],
    ) -> dict[str, object]:
        if input_type == "request":
            structured: Final = inputs.get("structured_messages")
            payload: dict[str, object] = {  # mutable-ok: outbound JSON
                "messages": structured
                if structured
                else [{"role": "user", "content": text} for text in (inputs.get("texts") or ())],
            }
            tools: Final = inputs.get("tools")
            if tools:
                payload["tools"] = tools
            return payload

        texts: Final = list(inputs.get("texts") or ())
        tool_calls: Final = inputs.get("tool_calls")
        messages: list[dict[str, object]] = [{"role": "assistant", "content": text} for text in texts]
        if tool_calls:
            if messages:
                messages[-1] = {**messages[-1], "tool_calls": tool_calls}
            else:
                messages = [{"role": "assistant", "content": None, "tool_calls": tool_calls}]
        if not messages:
            messages = [{"role": "assistant", "content": ""}]
        return {"messages": messages}

    async def _call_evaluate(self, body: dict[str, object]) -> dict[str, object]:
        url: Final = f"{self.api_base}{EVALUATE_PATH}"
        headers: Final = MappingProxyType(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        try:
            response: Final = await self.async_handler.post(
                url,
                json=body,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Timeout as exc:
            raise _TrustGuardUnreachable(exc) from exc
        except httpx.HTTPStatusError as exc:
            status_code: Final = exc.response.status_code
            if status_code == 503:
                raise HTTPException(
                    status_code=503,
                    detail="TrustGuard entitlements unavailable",
                ) from exc
            if status_code in (401, 403):
                raise HTTPException(
                    status_code=status_code,
                    detail="TrustGuard authentication failed",
                ) from exc
            if status_code in UNREACHABLE_HTTP_STATUSES:
                raise _TrustGuardUnreachable(exc) from exc
            raise HTTPException(
                status_code=503,
                detail="TrustGuard request failed",
            ) from exc
        except httpx.RequestError as exc:
            raise _TrustGuardUnreachable(exc) from exc

        try:
            parsed: Final[object] = response.json()
        except ValueError as exc:
            raise _TrustGuardUnreachable("TrustGuard returned non-JSON body") from exc
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=503, detail="TrustGuard returned an invalid response")
        status: Final = parsed.get("status")
        if not isinstance(status, str) or status.lower() not in KNOWN_STATUSES:
            raise HTTPException(status_code=503, detail="TrustGuard returned an unknown verdict")
        parsed["status"] = status.lower()
        return parsed

    def _handle_unreachable(
        self,
        inputs: GenericGuardrailAPIInputs,
        error: Exception,
    ) -> GenericGuardrailAPIInputs:
        if self.unreachable_fallback == "fail_open":
            verbose_proxy_logger.critical(
                "TrustGuard unreachable (fail-open): %s",
                error,
                exc_info=error,
            )
            return inputs
        verbose_proxy_logger.error("TrustGuard unreachable (fail-closed): %s", error)
        raise HTTPException(
            status_code=503,
            detail="TrustGuard guardrail service unreachable",
        ) from error

    @staticmethod
    def _apply_transform(
        inputs: GenericGuardrailAPIInputs,
        transformed: object,
    ) -> GenericGuardrailAPIInputs:
        if not isinstance(transformed, Mapping):
            raise HTTPException(status_code=400, detail="TrustGuard transform missing payload")

        raw_messages: Final = transformed.get("messages")
        if isinstance(raw_messages, list) and raw_messages:
            rewritten_messages: Final = _copy_messages(raw_messages)
            if rewritten_messages is None:
                raise HTTPException(status_code=400, detail="TrustGuard transform missing payload")
            texts_from_messages: Final = _texts_from_messages(rewritten_messages)
            return {  # mutable-ok: GenericGuardrailAPIInputs is a TypedDict
                **inputs,
                "structured_messages": rewritten_messages,
                "texts": texts_from_messages or inputs.get("texts"),
            }

        raw_input: Final = transformed.get("input")
        if not isinstance(raw_input, str) or not raw_input:
            raise HTTPException(status_code=400, detail="TrustGuard transform missing payload")

        original_messages: Final = inputs.get("structured_messages")
        if isinstance(original_messages, list) and original_messages:
            copied: Final = _copy_messages(original_messages)
            if copied is None:
                raise HTTPException(status_code=400, detail="TrustGuard transform missing payload")
            rewritten: Final = _rewrite_last_user_message(copied, raw_input)
            return {  # mutable-ok: GenericGuardrailAPIInputs is a TypedDict
                **inputs,
                "structured_messages": rewritten,
                "texts": _texts_from_messages(rewritten) or inputs.get("texts"),
            }

        original_texts: Final = list(inputs.get("texts") or ())
        if not original_texts:
            raise HTTPException(status_code=400, detail="TrustGuard transform missing payload")
        rewritten_texts: Final = list(original_texts)
        rewritten_texts[-1] = raw_input
        return {**inputs, "texts": rewritten_texts}  # mutable-ok: GenericGuardrailAPIInputs is a TypedDict
