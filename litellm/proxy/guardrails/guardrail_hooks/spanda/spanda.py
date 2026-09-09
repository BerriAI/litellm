"""
Spanda Guardrail Integration for LiteLLM (BRHMN Labs).

Sub-millisecond epistemic uncertainty quantification ($R_{sc}$) and 2-tier
cascaded guardrail detecting hallucinations and 120B mode collapse.
Runs on CPU without requiring external LLM-as-a-judge calls.
"""

from __future__ import annotations

import asyncio
import difflib
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.spanda import (
        SpandaGuardrailConfigModel,
    )

try:
    from spanda.guardrails import CascadedGuardrail

    _HAS_SPANDA_PKG: Final = True
except ImportError:
    _HAS_SPANDA_PKG: Final = False

_MAX_SAMPLES_TO_COMPARE: Final = 10
_MAX_SAMPLE_CHAR_LENGTH: Final = 2000


def _compute_fallback_rsc(samples: Sequence[str]) -> float:
    clamped: Final = tuple(s[:_MAX_SAMPLE_CHAR_LENGTH] for s in samples[:_MAX_SAMPLES_TO_COMPARE])
    k: Final = len(clamped)
    if k < 2:
        return 0.0
    total_dist = 0.0  # rebind-ok: accumulating distance sum
    pairs = 0  # rebind-ok: counting evaluated pairs
    for i in range(k):
        for j in range(i + 1, k):
            matcher: Final = difflib.SequenceMatcher(None, clamped[i], clamped[j])
            total_dist += 1.0 - matcher.ratio()  # rebind-ok: accumulating distance sum
            pairs += 1  # rebind-ok: counting evaluated pairs
    return total_dist / pairs if pairs > 0 else 0.0


def _compute_fallback_grounding(response: str, context: str) -> float:
    if not context or not response:
        return 0.0
    resp_tokens: Final = frozenset(re.findall(r"\b\w{4,}\b", response.lower()))
    if not resp_tokens:
        return 0.0
    ctx_tokens: Final = frozenset(re.findall(r"\b\w{4,}\b", context.lower()))
    ungrounded: Final = resp_tokens - ctx_tokens
    return len(ungrounded) / len(resp_tokens)


class SpandaGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_base: str | None = None,
        uncertainty_threshold: float = 0.35,
        grounding_threshold: float = 0.15,
        block_mode: bool = False,
        guardrail_name: str = "spanda",
        event_hook: GuardrailEventHooks | Sequence[GuardrailEventHooks] | str | None = None,
        default_on: bool = False,
        **kwargs: object,  # kwargs-ok: forwarded to CustomGuardrail.__init__
    ) -> None:
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=list(self.get_supported_event_hooks()),  # mutable-ok: framework expects list
            event_hook=event_hook or GuardrailEventHooks.post_call,
            default_on=default_on,
            **kwargs,  # kwargs-ok: forwarded to CustomGuardrail.__init__
        )
        self.api_base: Final = api_base
        self.uncertainty_threshold: Final = float(uncertainty_threshold or 0.35)
        self.grounding_threshold: Final = float(grounding_threshold or 0.15)
        self.block_mode: Final = bool(block_mode)

        if _HAS_SPANDA_PKG:
            self._guardrail: Final = CascadedGuardrail(
                uncertainty_threshold=self.uncertainty_threshold,
                grounding_threshold=self.grounding_threshold,
            )
        else:
            self._guardrail = None

    @classmethod
    def get_supported_event_hooks(cls) -> Sequence[GuardrailEventHooks]:
        return (
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
        )

    @staticmethod
    def get_config_model() -> type[SpandaGuardrailConfigModel]:
        from litellm.types.proxy.guardrails.guardrail_hooks.spanda import (
            SpandaGuardrailConfigModel,
        )

        return SpandaGuardrailConfigModel

    def evaluate_texts(self, texts: Sequence[str], context: str | None = None) -> Mapping[str, object]:
        if self._guardrail is not None and len(texts) >= 2:
            receipt: Final = self._guardrail.evaluate(
                sampled_responses=list(texts),  # mutable-ok: library boundary requires list
                context=context,
            )
            return receipt.to_dict()

        rsc: Final = _compute_fallback_rsc(texts) if len(texts) >= 2 else 0.0
        gr: Final = _compute_fallback_grounding(texts[0], context) if (texts and context) else 0.0

        is_safe: Final[bool]
        decision: Final[str]

        if len(texts) >= 2 and rsc > self.uncertainty_threshold:
            is_safe = False
            decision = "FLAG_HIGH_UNCERTAINTY"
        elif gr > self.grounding_threshold:
            is_safe = False
            decision = "FLAG_UNGROUNDED"
        elif len(texts) < 2 and not context:
            is_safe = True
            decision = "PASS_SINGLE_SAMPLE_UNCHECKED"
        else:
            is_safe = True
            decision = "PASS"

        receipt_dict: Final[dict[str, object]] = {  # mutable-ok: building receipt payload
            "rsc": round(rsc, 4),
            "grounding_residual": round(gr, 4),
            "is_safe": is_safe,
            "decision": decision,
            "tier_used": 2 if gr > 0 else 1,
            "samples_analyzed": len(texts),
        }
        return receipt_dict

    def _extract_context(self, data: Mapping[str, object]) -> str | None:
        explicit_ctx: Final = data.get("context") or data.get("spanda_context") or data.get("grounding_context")
        if explicit_ctx and isinstance(explicit_ctx, str):
            return explicit_ctx

        metadata: Final = data.get("metadata")
        if isinstance(metadata, Mapping):
            meta_ctx: Final = (
                metadata.get("context") or metadata.get("spanda_context") or metadata.get("grounding_context")
            )
            if meta_ctx and isinstance(meta_ctx, str):
                return meta_ctx

        raw_messages: Final = data.get("messages")
        if isinstance(raw_messages, (list, tuple)):
            for m in raw_messages:
                if isinstance(m, Mapping):
                    role: Final = m.get("role")
                    content: Final = m.get("content")
                    if role == "tool" and isinstance(content, str) and content:
                        return content
                    if isinstance(content, str) and (
                        content.lower().startswith("context:") or content.lower().startswith("reference:")
                    ):
                        return content
                elif hasattr(m, "role") and hasattr(m, "content"):
                    role_attr: Final = getattr(m, "role", "")
                    content_attr: Final = getattr(m, "content", "")
                    if role_attr == "tool" and isinstance(content_attr, str) and content_attr:
                        return content_attr
                    if isinstance(content_attr, str) and (
                        content_attr.lower().startswith("context:") or content_attr.lower().startswith("reference:")
                    ):
                        return content_attr
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
            choices_raw: Final = (
                response.get("choices") if isinstance(response, Mapping) else getattr(response, "choices", None)
            )
            if not choices_raw or not isinstance(choices_raw, (list, tuple)):
                return

            samples_acc = []  # mutable-ok: accumulating response samples
            for c in choices_raw:
                if isinstance(c, dict):
                    msg_obj: Final = c.get("message")
                    text = (msg_obj.get("content") if isinstance(msg_obj, dict) else None) or c.get("text", "")
                else:
                    msg = getattr(c, "message", None)
                    text = getattr(msg, "content", "") if msg else getattr(c, "text", "")
                if text and isinstance(text, str):
                    samples_acc.append(text)

            samples: Final = tuple(samples_acc)
            if not samples:
                return

            context: Final = self._extract_context(data)

            receipt: Final = await asyncio.to_thread(self.evaluate_texts, samples, context)

            if isinstance(response, dict):
                response["_spanda_receipt"] = receipt  # rebind-ok: attaching receipt to response dict
            elif hasattr(response, "__dict__"):
                try:
                    response._spanda_receipt = receipt  # pyright: ignore[reportAttributeAccessIssue]  # dynamic attribute attached to response
                except (AttributeError, TypeError):
                    pass

            if hasattr(response, "model_extra") and isinstance(response.model_extra, dict):  # pyright: ignore[reportAttributeAccessIssue]  # Pydantic v2 dynamic extra dict
                response.model_extra["_spanda_receipt"] = receipt  # rebind-ok: attaching receipt to Pydantic extra dict

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
