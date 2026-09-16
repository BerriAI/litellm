"""NeuralTrust TrustGuard native LiteLLM guardrail.

Calls TrustGuard POST /v1/evaluate on pre_call (input) and post_call (output).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import httpx
from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

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
from litellm.types.proxy.guardrails.guardrail_hooks.neuraltrust import DEFAULT_API_BASE, DEFAULT_TIMEOUT
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

EVALUATE_PATH: Final = "/v1/evaluate"
CONSUMER_ID_KEYS: Final = (
    ("user_api_key_alias", "user_api_key_key_alias"),
    ("user_api_key_user_email",),
    ("user_api_key_user_id",),
    ("user_api_key_team_alias",),
)
METADATA_ADAPTER: Final = TypeAdapter(Mapping[str, object])
EMPTY_METADATA: Final[Mapping[str, object]] = MappingProxyType({})
STATUS_BLOCK: Final = "block"
STATUS_ASK: Final = "ask"
STATUS_TRANSFORM: Final = "transform"
STATUS_REPORT: Final = "report"
STATUS_ALLOW: Final = "allow"
BLOCKING_STATUSES: Final = frozenset({STATUS_BLOCK, STATUS_ASK})
KNOWN_STATUSES: Final = frozenset({STATUS_ALLOW, STATUS_TRANSFORM, STATUS_REPORT, *BLOCKING_STATUSES})
UNREACHABLE_HTTP_STATUSES: Final = frozenset({502, 504})
TRANSFORM_MISSING: Final = "TrustGuard transform missing payload"


class _TrustGuardUnreachable(Exception):
    """Transport or availability failure; eligible for unreachable_fallback."""


def _metadata(block: object) -> Mapping[str, object]:
    try:
        return METADATA_ADAPTER.validate_python(block)
    except ValidationError:
        return EMPTY_METADATA


def _consumer_id(request_data: Mapping[str, object]) -> str | None:
    blocks: Final = tuple(_metadata(request_data.get(source)) for source in ("litellm_metadata", "metadata"))
    candidates: Final = (block.get(name) for names in CONSUMER_ID_KEYS for name in names for block in blocks)
    return next((value for value in candidates if isinstance(value, str) and value), None)


def _message_text(message: Mapping[str, object]) -> str:
    content: Final = message.get("content")
    return content if isinstance(content, str) else ""


def _copy_message(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}  # mutable-ok: shallow copy for write-back


def _copy_messages(messages: Sequence[object]) -> tuple[Mapping[str, object], ...] | None:
    copied: Final = tuple(copy for message in messages if (copy := _copy_message(message)) is not None)
    return copied if len(copied) == len(messages) else None


def _texts_from_messages(messages: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(_message_text(message) for message in messages)


def _tool_calls_in_message(message: Mapping[str, object]) -> tuple[object, ...] | None:
    raw: Final = message.get("tool_calls")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail=TRANSFORM_MISSING)
    return tuple(raw)


def _tool_calls_from_messages(messages: Sequence[Mapping[str, object]]) -> tuple[object, ...] | None:
    groups: Final = tuple(_tool_calls_in_message(message) for message in messages)
    if all(group is None for group in groups):
        return None
    return tuple(tool_call for group in groups if group is not None for tool_call in group)


def _rewrite_last_user_message(
    messages: Sequence[Mapping[str, object]],
    redacted: str,
) -> tuple[Mapping[str, object], ...]:
    user_indices: Final = tuple(index for index, message in enumerate(messages) if message.get("role") == "user")
    target: Final = user_indices[-1] if user_indices else len(messages) - 1
    return tuple(
        {**message, "content": redacted} if index == target else dict(message)  # mutable-ok: write-back message
        for index, message in enumerate(messages)
    )


def _model_name(
    inputs: GenericGuardrailAPIInputs,
    logging_obj: LiteLLMLoggingObj | None,
) -> str:
    if logging_obj is not None and logging_obj.model:
        return str(logging_obj.model)
    return str(inputs.get("model") or "")


def _assistant_message(text: str | None, tool_calls: object) -> Mapping[str, object]:
    if tool_calls:
        return {"role": "assistant", "content": text, "tool_calls": tool_calls}  # mutable-ok: outbound JSON
    return {"role": "assistant", "content": text}  # mutable-ok: outbound JSON


def _assistant_messages(texts: Sequence[str], tool_calls: object) -> tuple[Mapping[str, object], ...]:
    if not texts:
        return (_assistant_message(None if tool_calls else "", tool_calls),)
    last: Final = len(texts) - 1
    return tuple(_assistant_message(text, tool_calls if index == last else None) for index, text in enumerate(texts))


def _sent_messages(
    inputs: GenericGuardrailAPIInputs,
    input_type: Literal["request", "response"],
) -> Sequence[Mapping[str, object]]:
    if input_type == "response":
        return _assistant_messages(tuple(inputs.get("texts") or ()), inputs.get("tool_calls"))
    structured: Final = inputs.get("structured_messages")
    if structured:
        return structured
    return tuple({"role": "user", "content": text} for text in (inputs.get("texts") or ()))  # mutable-ok: outbound JSON


def _inputs_with_messages(
    inputs: GenericGuardrailAPIInputs,
    messages: Sequence[Mapping[str, object]],
    *,
    replace_tool_calls: bool,
) -> GenericGuardrailAPIInputs:
    extracted: Final = _tool_calls_from_messages(messages) if replace_tool_calls else None
    original_tool_calls: Final = inputs.get("tool_calls")
    if extracted is not None and original_tool_calls is not None and len(extracted) != len(original_tool_calls):
        raise HTTPException(status_code=400, detail=TRANSFORM_MISSING)
    merged: Final[GenericGuardrailAPIInputs] = {  # mutable-ok: GenericGuardrailAPIInputs is a TypedDict
        **inputs,
        "structured_messages": list(messages),  # mutable-ok: GenericGuardrailAPIInputs.structured_messages is a list
    }
    rebuilt: Final[GenericGuardrailAPIInputs] = (
        {**merged, "texts": list(_texts_from_messages(messages))}  # mutable-ok: TypedDict field is a list
        if inputs.get("texts")
        else merged
    )
    if extracted is None:
        return rebuilt
    return {**rebuilt, "tool_calls": list(extracted)}  # mutable-ok: GenericGuardrailAPIInputs.tool_calls is a list


class NeuralTrustGuardrail(CustomGuardrail):
    """LiteLLM hook that evaluates prompts and completions with TrustGuard."""

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel]:
        from litellm.types.proxy.guardrails.guardrail_hooks.neuraltrust import (
            NeuralTrustGuardrailConfigModel,
        )

        return NeuralTrustGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: CustomGuardrail contract
        return [  # mutable-ok: CustomGuardrail.supported_event_hooks is a list
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
        event_hook: GuardrailEventHooks | Mode | str | Sequence[str] | None = None,
        default_on: bool | None = None,
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
        if resolved_timeout <= 0:
            raise ValueError("TrustGuard timeout must be a positive number of seconds.")
        self.timeout = resolved_timeout
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=self.get_supported_event_hooks(),
            # LitellmParams.mode is str | list[str] | Mode, which CustomGuardrail narrows to the enum
            event_hook=event_hook,  # pyright: ignore[reportArgumentType]  # config supplies the raw mode string
            default_on=bool(default_on),
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
        if status in BLOCKING_STATUSES:
            raise HTTPException(
                status_code=400,
                detail={  # mutable-ok: FastAPI HTTPException.detail is a JSON object
                    "error": "Violated guardrail policy",
                    "neuraltrust_guardrail_response": "Blocked by NeuralTrust TrustGuard.",
                    "verdict": status,
                    "trace_id": result.get("trace_id"),
                    "request_id": result.get("request_id"),
                },
            )
        if status == STATUS_TRANSFORM:
            return self._apply_transform(
                inputs,
                result.get("transformed_payload"),
                sent_count=len(_sent_messages(inputs, input_type)),
            )
        if status == STATUS_REPORT:
            verbose_proxy_logger.info("TrustGuard report-only findings trace_id=%s", result.get("trace_id"))
        return inputs

    def _evaluate_body(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,  # mutable-ok: CustomGuardrail.apply_guardrail contract
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None,
    ) -> dict[str, object]:  # mutable-ok: outbound JSON
        session_id: Final = get_session_id_from_request_data(request_data)
        consumer_id: Final = _consumer_id(request_data)
        return {  # mutable-ok: outbound JSON
            "payload": self._payload(inputs, input_type),
            "direction": "input" if input_type == "request" else "output",
            "protocol": "llm",
            "attributes": {  # mutable-ok: outbound JSON
                "content_type": "application/json",
                "model": {"name": _model_name(inputs, logging_obj)},  # mutable-ok: outbound JSON
            },
            **({"collector_key": self.collector_key} if self.collector_key else {}),  # mutable-ok: outbound JSON
            **({"session_id": session_id} if session_id else {}),  # mutable-ok: outbound JSON
            **({"consumer_id": consumer_id} if consumer_id is not None else {}),  # mutable-ok: outbound JSON
        }

    @staticmethod
    def _payload(
        inputs: GenericGuardrailAPIInputs,
        input_type: Literal["request", "response"],
    ) -> Mapping[str, object]:
        messages: Final = _sent_messages(inputs, input_type)
        tools: Final = inputs.get("tools") if input_type == "request" else None
        if tools:
            return {"messages": messages, "tools": tools}  # mutable-ok: outbound JSON
        return {"messages": messages}  # mutable-ok: outbound JSON

    async def _call_evaluate(self, body: dict[str, object]) -> dict[str, object]:  # mutable-ok: TrustGuard JSON
        url: Final = f"{self.api_base}{EVALUATE_PATH}"
        headers: Final = {  # mutable-ok: AsyncHTTPHandler.post declares headers as dict
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
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
        return {**parsed, "status": status.lower()}  # mutable-ok: TrustGuard JSON object

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
        *,
        sent_count: int,
    ) -> GenericGuardrailAPIInputs:
        if not isinstance(transformed, Mapping):
            raise HTTPException(status_code=400, detail=TRANSFORM_MISSING)

        raw_messages: Final = transformed.get("messages")
        if isinstance(raw_messages, list) and raw_messages:
            rewritten_messages: Final = _copy_messages(raw_messages)
            if rewritten_messages is None or len(rewritten_messages) != sent_count:
                raise HTTPException(status_code=400, detail=TRANSFORM_MISSING)
            return _inputs_with_messages(inputs, rewritten_messages, replace_tool_calls=True)

        raw_input: Final = transformed.get("input")
        if not isinstance(raw_input, str) or not raw_input:
            raise HTTPException(status_code=400, detail=TRANSFORM_MISSING)

        original_messages: Final = inputs.get("structured_messages")
        if isinstance(original_messages, list) and original_messages:
            copied: Final = _copy_messages(original_messages)
            if copied is None:
                raise HTTPException(status_code=400, detail=TRANSFORM_MISSING)
            return _inputs_with_messages(
                inputs,
                _rewrite_last_user_message(copied, raw_input),
                replace_tool_calls=False,
            )

        original_texts: Final = tuple(inputs.get("texts") or ())
        if not original_texts:
            raise HTTPException(status_code=400, detail=TRANSFORM_MISSING)
        rewritten_texts: Final = (*original_texts[:-1], raw_input)
        return {**inputs, "texts": list(rewritten_texts)}  # mutable-ok: GenericGuardrailAPIInputs.texts is a list
