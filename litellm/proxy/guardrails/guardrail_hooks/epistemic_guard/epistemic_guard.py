"""
Epistemic Guard post-call guardrail.

Deterministically verifies assistant action-completion claims against actual tool-call
receipts in the conversation history (no LLM-as-a-judge required). Requires the optional
`epistemic-guard` package (`pip install epistemic-guard>=0.6.5`).
"""

import json
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Literal, Optional, cast

from fastapi import HTTPException
from typing_extensions import TypedDict, Unpack

from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.proxy.guardrails.guardrail_hooks.epistemic_guard import (
    EpistemicGuardDetection,
)
from litellm.types.utils import (
    GenericGuardrailAPIInputs,
    GuardrailStatus,
    GuardrailTracingDetail,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class _CustomGuardrailOptions(TypedDict, total=False, extra_items=object):
    """Base-class constructor options forwarded untouched to CustomGuardrail."""


def _tool_payload_indicates_failure(content: object) -> bool:
    """Return True if a tool message payload explicitly signals failure via JSON or text."""
    obj: Mapping[str, Any] | None = None
    if isinstance(content, Mapping):
        obj = content
    elif isinstance(content, str):
        stripped = content.strip()
        if stripped.lower().startswith(("error:", "traceback")):
            return True
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, Mapping):
                    obj = parsed
            except (TypeError, ValueError):
                return False
    if obj is None:
        return False
    if obj.get("ok") is False or obj.get("success") is False:
        return True
    if obj.get("isError") is True or obj.get("is_error") is True:
        return True
    err_val = obj.get("error")
    if err_val not in (None, False, "", [], {}):
        return True
    status_val = str(obj.get("status", "")).strip().lower()
    return status_val in {"error", "failed", "failure", "denied", "rejected"}


def _normalize_tool_error_messages(
    messages: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Mark explicit structured JSON tool failures with `is_error=True` before receipt extraction."""
    normalized: list[Mapping[str, Any]] = []
    for msg in messages:
        if (
            str(msg.get("role", "")).lower() == "tool"
            and not msg.get("is_error")
            and _tool_payload_indicates_failure(msg.get("content"))
        ):
            mut = dict(msg)
            mut["is_error"] = True
            normalized.append(mut)
            continue
        normalized.append(msg)
    return normalized


def _import_epistemic_guard() -> tuple[Any, Any]:
    """Lazily import `epistemic_guard` helpers with an actionable install message."""
    try:
        from epistemic_guard import (
            ReceiptSupervisor,
            verify_chat_turn,
        )
    except ImportError as exc:
        raise ImportError(
            "The epistemic_guard guardrail requires the optional `epistemic-guard` "
            "package. Install it with: pip install 'epistemic-guard>=0.6.5'"
        ) from exc
    return ReceiptSupervisor, verify_chat_turn


class EpistemicGuardrail(CustomGuardrail):
    """
    Post-call guardrail that cross-checks assistant completion claims against tool
    execution receipts present in `request_data["messages"]`.

    Catches ungrounded claims (e.g. claiming a file was deleted, email sent, or command
    succeeded when no matching successful tool receipt exists or the tool failed).
    """

    def __init__(
        self,
        guardrail_name: str | None = None,
        on_fail: Literal["block", "rewrite", "log_only"] = "block",
        verify_output_values: bool = False,
        allow_creative_context: bool = False,
        device: str | None = None,
        event_hook: Literal["post_call"] | list[str] | None = None,
        default_on: bool = False,
        **kwargs: Unpack[_CustomGuardrailOptions],
    ) -> None:
        if on_fail not in ("block", "rewrite", "log_only"):
            raise ValueError("on_fail must be 'block', 'rewrite', or 'log_only'")

        _event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | None = None
        if event_hook is not None:
            if isinstance(event_hook, list):
                _event_hook = [
                    GuardrailEventHooks(h) if isinstance(h, str) else h
                    for h in event_hook
                ]
            else:
                _event_hook = GuardrailEventHooks(event_hook)

        super().__init__(
            guardrail_name=guardrail_name or "epistemic_guard",
            supported_event_hooks=list(self.get_supported_event_hooks()),
            event_hook=_event_hook or [GuardrailEventHooks.post_call],
            default_on=default_on,
            **kwargs,
        )
        self.on_fail: Literal["block", "rewrite", "log_only"] = on_fail
        self.verify_output_values = verify_output_values
        self.allow_creative_context = allow_creative_context
        self.device = device
        self._supervisor: Any = None

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.epistemic_guard import (
            EpistemicGuardConfigModel,
        )

        return EpistemicGuardConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [GuardrailEventHooks.post_call]

    def _get_supervisor(self) -> tuple[Any, Any]:
        receipt_supervisor_cls, verify_chat_turn = _import_epistemic_guard()
        if self._supervisor is None:
            self._supervisor = receipt_supervisor_cls(
                device=self.device,
                verify_output_values=self.verify_output_values,
                allow_creative_context=self.allow_creative_context,
                warn_on_missing=False,
            )
        return self._supervisor, verify_chat_turn

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type != "response":
            return inputs

        start_time: Final = datetime.now()
        detections: Final[list[EpistemicGuardDetection]] = []
        status: GuardrailStatus = "success"
        exception_str = ""

        try:
            texts: Final = inputs.get("texts", [])
            if not texts:
                return inputs

            supervisor, verify_chat_turn = self._get_supervisor()
            raw_msgs = request_data.get("messages")
            messages = _normalize_tool_error_messages(
                [m for m in raw_msgs if isinstance(m, Mapping)]
                if isinstance(raw_msgs, list)
                else []
            )

            processed: Final[list[str]] = []
            for text in texts:
                verdict = verify_chat_turn(
                    messages,
                    response=text,
                    supervisor=supervisor,
                    verify_output_values=self.verify_output_values,
                )
                if verdict.is_valid:
                    processed.append(text)
                    continue

                unsupported = list(verdict.unsupported_claims)
                detections.append(
                    cast(
                        EpistemicGuardDetection,
                        {
                            "type": "ungrounded_action_claim",
                            "unsupported_claims": unsupported,
                            "clauses_checked": int(verdict.clauses_checked),
                            "device": str(verdict.device),
                            "latency_ms": round(float(verdict.latency_ms), 3),
                            "action_taken": self.on_fail,
                        },
                    )
                )

                client_message = (
                    "Response blocked: assistant made an ungrounded action claim "
                    "not supported by verified tool execution receipts."
                )
                is_streaming = bool(request_data.get("stream", False))
                if self.on_fail == "block" or (
                    self.on_fail == "rewrite" and is_streaming
                ):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": client_message,
                            "guardrail": self.guardrail_name,
                            "unsupported_claim_count": len(unsupported),
                        },
                    )
                if self.on_fail == "rewrite":
                    processed.append(f"[UNGROUNDED CLAIM BLOCKED] {client_message}")
                else:
                    processed.append(text)

            inputs["texts"] = processed
            return inputs
        except HTTPException:
            status = "guardrail_intervened"
            raise
        except ModifyResponseException:
            status = "guardrail_intervened"
            raise
        except Exception as exc:
            status = "guardrail_failed_to_respond"
            exception_str = str(exc)
            raise
        finally:
            detection_dicts: Final[list[dict[str, object]]] = [
                dict(d) for d in detections
            ]
            guardrail_response: Final[list[dict[str, object]] | str] = (
                exception_str
                if status != "success" and not detections
                else detection_dicts
            )
            tracing_kw: Final[GuardrailTracingDetail] = {
                "guardrail_id": self.guardrail_name,
                "detection_method": "epistemic_guard_receipt_verification",
                "match_details": guardrail_response,
            }
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="epistemic_guard",
                guardrail_json_response=guardrail_response,
                request_data=request_data,
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
                event_type=GuardrailEventHooks.post_call,
                tracing_detail=GuardrailTracingDetail(**tracing_kw),
            )
