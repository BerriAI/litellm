"""
Spanda Guardrail Integration for LiteLLM (BRHMN Labs).

Sub-millisecond epistemic uncertainty quantification ($R_{sc}$) and 2-tier
cascaded guardrail detecting hallucinations and 120B mode collapse.
Runs on CPU without requiring external LLM-as-a-judge calls.
"""

from __future__ import annotations

import asyncio
import difflib
import importlib.util
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal

from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs


class SpandaReceipt(TypedDict):
    rsc: ReadOnly[float]
    grounding_residual: ReadOnly[float]
    is_safe: ReadOnly[bool]
    decision: ReadOnly[str]
    tier_used: ReadOnly[int]
    samples_analyzed: ReadOnly[int]


if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.spanda import (
        SpandaGuardrailConfigModel,
    )


def _is_spanda_available() -> bool:
    return importlib.util.find_spec("spanda") is not None


_HAS_SPANDA_PKG: Final = _is_spanda_available()

_MAX_SAMPLES_TO_COMPARE: Final = 10
_MAX_SAMPLE_CHAR_LENGTH: Final = 2000


def _compute_fallback_rsc(samples: Sequence[str]) -> float:
    clamped: Final = tuple(s[:_MAX_SAMPLE_CHAR_LENGTH] for s in samples[:_MAX_SAMPLES_TO_COMPARE])
    k: Final = len(clamped)
    if k < 2:
        return 0.0
    dists: Final = tuple(
        1.0 - difflib.SequenceMatcher(None, clamped[i], clamped[j]).ratio() for i in range(k) for j in range(i + 1, k)
    )
    return sum(dists) / len(dists) if dists else 0.0


def _compute_fallback_grounding(response: str, context: str) -> float:
    if not context or not response:
        return 0.0
    resp_tokens: Final = frozenset(re.findall(r"\b\w{4,}\b", response.lower()))
    if not resp_tokens:
        return 0.0
    ctx_tokens: Final = frozenset(re.findall(r"\b\w{4,}\b", context.lower()))
    ungrounded: Final = resp_tokens - ctx_tokens
    return len(ungrounded) / len(resp_tokens)


def _init_guardrail_instance(
    uncertainty_threshold: float,
    grounding_threshold: float,
) -> object | None:
    if not _HAS_SPANDA_PKG:
        return None
    try:
        from spanda.guardrails import CascadedGuardrail

        return CascadedGuardrail(
            uncertainty_threshold=uncertainty_threshold,
            grounding_threshold=grounding_threshold,
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


def _extract_msg_context(m: object) -> str | None:
    if isinstance(m, Mapping):
        role_val: Final[object] = m.get("role")
        content_val: Final[object] = m.get("content")
        role: Final = role_val if isinstance(role_val, str) else ""
        content: Final = content_val if isinstance(content_val, str) else ""
        if role == "tool" and content:
            return content
        if content and (content.lower().startswith("context:") or content.lower().startswith("reference:")):
            return content
        return None

    role_attr: Final[object] = getattr(m, "role", "")
    content_attr: Final[object] = getattr(m, "content", "")
    role_s: Final = role_attr if isinstance(role_attr, str) else ""
    content_s: Final = content_attr if isinstance(content_attr, str) else ""
    if role_s == "tool" and content_s:
        return content_s
    if content_s and (content_s.lower().startswith("context:") or content_s.lower().startswith("reference:")):
        return content_s
    return None


def _extract_choice_text(choice: object) -> str:
    if isinstance(choice, Mapping):
        msg_obj: Final[object] = choice.get("message")
        if isinstance(msg_obj, Mapping):
            content_val: Final[object] = msg_obj.get("content")
            if isinstance(content_val, str):
                return content_val
        text_val: Final[object] = choice.get("text")
        return text_val if isinstance(text_val, str) else ""
    msg: Final[object] = getattr(choice, "message", None)
    if msg is not None:
        content_attr: Final[object] = getattr(msg, "content", "")
        if isinstance(content_attr, str):
            return content_attr
    text_attr: Final[object] = getattr(choice, "text", "")
    return text_attr if isinstance(text_attr, str) else ""


class SpandaGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_base: str | None = None,
        uncertainty_threshold: float | None = 0.35,
        grounding_threshold: float | None = 0.15,
        block_mode: bool = False,
        guardrail_name: str = "spanda",
        event_hook: GuardrailEventHooks | Sequence[GuardrailEventHooks] | str | None = None,
        default_on: bool = False,
        guardrail_instance: object | None = None,
        **kwargs: object,  # kwargs-ok: forwarded to CustomGuardrail.__init__
    ) -> None:
        self.api_base: Final = api_base
        self.uncertainty_threshold: Final = float(0.35 if uncertainty_threshold is None else uncertainty_threshold)
        self.grounding_threshold: Final = float(0.15 if grounding_threshold is None else grounding_threshold)
        self.block_mode: Final = bool(block_mode)
        self._guardrail: Final = (
            guardrail_instance
            if guardrail_instance is not None
            else _init_guardrail_instance(self.uncertainty_threshold, self.grounding_threshold)
        )
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=self.get_supported_event_hooks(),
            event_hook=event_hook or GuardrailEventHooks.post_call,
            default_on=default_on,
            **kwargs,  # pyright: ignore[reportArgumentType]  # kwargs-ok: forwarded to CustomGuardrail.__init__
        )

    @classmethod
    def get_supported_event_hooks(
        cls,
    ) -> list[GuardrailEventHooks]:  # mutable-ok: overrides CustomGuardrail signature
        return [  # mutable-ok: framework expects list
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
        ]

    @staticmethod
    def get_config_model() -> type[SpandaGuardrailConfigModel]:
        from litellm.types.proxy.guardrails.guardrail_hooks.spanda import (
            SpandaGuardrailConfigModel,
        )

        return SpandaGuardrailConfigModel

    def evaluate_texts(self, texts: Sequence[str], context: str | None = None) -> Mapping[str, object]:
        if self._guardrail is not None and len(texts) >= 2:
            evaluate_fn: Final[object] = getattr(self._guardrail, "evaluate", None)
            if callable(evaluate_fn):
                receipt_obj: Final[object] = evaluate_fn(
                    sampled_responses=list(texts),  # mutable-ok: external library expects list
                    context=context,
                )
                to_dict_fn: Final[object] = getattr(receipt_obj, "to_dict", None)
                if callable(to_dict_fn):
                    receipt_from_lib: Final[object] = to_dict_fn()
                    if isinstance(receipt_from_lib, Mapping):
                        return receipt_from_lib

        rsc: Final = _compute_fallback_rsc(texts) if len(texts) >= 2 else 0.0
        gr: Final = _compute_fallback_grounding(texts[0], context) if (texts and context) else 0.0

        outcome: Final = (
            (False, "FLAG_HIGH_UNCERTAINTY")
            if len(texts) >= 2 and rsc > self.uncertainty_threshold
            else (False, "FLAG_UNGROUNDED")
            if gr > self.grounding_threshold
            else (True, "PASS_SINGLE_SAMPLE_UNCHECKED")
            if len(texts) < 2 and not context
            else (True, "PASS")
        )
        is_safe: Final[bool] = outcome[0]
        decision: Final[str] = outcome[1]

        receipt_dict: Final[SpandaReceipt] = {
            "rsc": round(rsc, 4),
            "grounding_residual": round(gr, 4),
            "is_safe": is_safe,
            "decision": decision,
            "tier_used": 2 if gr > 0 else 1,
            "samples_analyzed": len(texts),
        }
        return receipt_dict

    def _extract_context(self, data: Mapping[str, object]) -> str | None:
        explicit_ctx: Final[object] = data.get("context") or data.get("spanda_context") or data.get("grounding_context")
        if explicit_ctx and isinstance(explicit_ctx, str):
            return explicit_ctx

        metadata: Final[object] = data.get("metadata")
        if isinstance(metadata, Mapping):
            meta_ctx: Final[object] = (
                metadata.get("context") or metadata.get("spanda_context") or metadata.get("grounding_context")
            )
            if meta_ctx and isinstance(meta_ctx, str):
                return meta_ctx

        raw_messages: Final[object] = data.get("messages")
        if isinstance(raw_messages, Sequence) and not isinstance(raw_messages, (str, bytes)):
            for m in raw_messages:
                if (extracted := _extract_msg_context(m)) is not None:
                    return extracted
        return None

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: framework passes mutable request_data dict
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type != "response":
            return inputs

        texts: Final = inputs.get("texts") or ()
        if not texts:
            return inputs

        context: Final = self._extract_context(request_data)

        receipt: Final = await asyncio.to_thread(self.evaluate_texts, texts, context)
        request_data["_spanda_receipt"] = receipt  # rebind-ok: attaching receipt to request data dictionary

        verbose_proxy_logger.debug(
            "Spanda guardrail evaluated: guardrail_name=%s decision=%s rsc=%s",
            self.guardrail_name,
            receipt.get("decision"),
            receipt.get("rsc"),
        )

        if self.block_mode and not receipt.get("is_safe", True):
            decision: Final = receipt.get("decision", "HIGH_UNCERTAINTY")
            rsc_val: Final = receipt.get("rsc", 0.0)
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Spanda guardrail intervention: {decision} (R_sc={rsc_val})",
                should_wrap_with_default_message=False,
                blocked_content=True,
            )

        return inputs

    async def async_post_call_success_hook(
        self,
        data: dict[str, object],  # mutable-ok: framework passes mutable data dict
        user_api_key_dict: object,
        response: object,
    ) -> None:
        try:
            choices_raw: Final[object] = (
                response.get("choices") if isinstance(response, Mapping) else getattr(response, "choices", None)
            )
            if not choices_raw or not isinstance(choices_raw, Sequence) or isinstance(choices_raw, (str, bytes)):
                return

            samples: Final[tuple[str, ...]] = tuple(text for c in choices_raw if (text := _extract_choice_text(c)))
            if not samples:
                return

            context: Final = self._extract_context(data)

            receipt: Final = await asyncio.to_thread(self.evaluate_texts, samples, context)

            if isinstance(response, dict):
                response["_spanda_receipt"] = receipt  # rebind-ok: attaching receipt to response dict
            elif hasattr(response, "__dict__"):
                try:
                    response._spanda_receipt = receipt  # pyright: ignore[reportAttributeAccessIssue]  # rebind-ok: attaching receipt to response
                except (AttributeError, TypeError):
                    pass

            model_extra: Final[object] = getattr(response, "model_extra", None)
            if isinstance(model_extra, dict):
                model_extra["_spanda_receipt"] = receipt  # rebind-ok: attaching receipt to Pydantic extra dict

            if self.block_mode and not receipt.get("is_safe", True):
                decision: Final = receipt.get("decision", "HIGH_UNCERTAINTY")
                rsc_val: Final = receipt.get("rsc", 0.0)
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"Spanda guardrail intervention: {decision} (R_sc={rsc_val})",
                    should_wrap_with_default_message=False,
                    blocked_content=True,
                )
        except GuardrailRaisedException:
            raise
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            verbose_proxy_logger.debug("Spanda async_post_call_success_hook error: %s", exc)
