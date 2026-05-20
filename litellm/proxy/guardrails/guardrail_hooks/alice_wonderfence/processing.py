"""Pure transforms for Alice WonderFence: context build, text extract, action dispatch."""

from typing import Any, Literal, Optional, Tuple

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_last_user_message,
    set_last_user_message,
)
from litellm.types.utils import GenericGuardrailAPIInputs

from .credentials import get_metadata
from .exceptions import WonderFenceBlockedError


logger = verbose_proxy_logger.getChild("alice_wonderfence")


def build_analysis_context(
    request_data: dict,
    platform: Optional[str],
    context_class: Any,
) -> Any:
    """Build WonderFence AnalysisContext from request data."""
    metadata = get_metadata(request_data)
    model_str = request_data.get("model", "")

    provider = None
    model_name = model_str
    if model_str:
        try:
            model_name, provider, _, _ = litellm.get_llm_provider(model=model_str)
        except Exception:
            if "/" in model_str:
                provider, model_name = model_str.split("/", 1)

    user_id = (
        metadata.get("user_api_key_end_user_id")
        or metadata.get("end_user_id")
        or metadata.get("user_id")
    )

    session_id = (
        request_data.get("litellm_session_id")
        or metadata.get("litellm_session_id")
        or metadata.get("session_id")
    )

    return context_class(
        session_id=session_id,
        user_id=user_id,
        model_name=model_name,
        provider=provider,
        platform=platform,
    )


def extract_relevant_text(
    inputs: GenericGuardrailAPIInputs,
    input_type: Literal["request", "response"],
) -> Tuple[Optional[str], Optional[Literal["structured_messages", "texts"]]]:
    """Extract latest user message (request) or latest assistant message (response).

    Returns (text, source) — ``source`` identifies which slot the text came
    from so MASK can write the redacted version back to the same place.
    """
    if input_type == "request":
        structured_messages = inputs.get("structured_messages", [])
        if structured_messages:
            return (
                get_last_user_message(structured_messages),
                "structured_messages",
            )
        texts = inputs.get("texts", [])
        return (texts[-1] if texts else None), ("texts" if texts else None)
    texts = inputs.get("texts", [])
    return (texts[-1] if texts else None), ("texts" if texts else None)


def handle_action(
    result: Any,
    inputs: GenericGuardrailAPIInputs,
    text_source: Optional[Literal["structured_messages", "texts"]],
    guardrail_name: str,
    block_message: str,
) -> None:
    """Dispatch BLOCK/MASK/DETECT/NO_ACTION. Raises ``WonderFenceBlockedError`` on BLOCK.

    ``text_source`` identifies which inputs slot supplied the analyzed text;
    MASK writes the redacted value back to the same slot.
    """
    action = result.action.value if hasattr(result.action, "value") else result.action
    correlation_id = getattr(result, "correlation_id", None)

    if action == "BLOCK":
        detail: dict = {
            "error": block_message,
            "type": "alice_wonderfence_content_policy_violation",
            "guardrail_name": guardrail_name,
            "action": "BLOCK",
            "wonderfence_correlation_id": correlation_id,
        }
        if hasattr(result, "detections") and result.detections:
            detail["detections"] = [
                d.model_dump() if hasattr(d, "model_dump") else str(d)
                for d in result.detections
            ]
        raise WonderFenceBlockedError(detail)
    if action == "MASK":
        masked_text = result.action_text or "[MASKED]"
        wrote = False
        if text_source == "structured_messages":
            inputs["structured_messages"] = set_last_user_message(
                inputs.get("structured_messages", []), masked_text
            )
            wrote = True
        # Always also overwrite texts[-1] when texts is populated. The OpenAI
        # chat translation layer reads back only ``texts`` after
        # apply_guardrail returns and maps it onto messages — masking only
        # ``structured_messages`` lets the unmasked ``texts`` slot win and the
        # original prompt reaches the LLM.
        texts = inputs.get("texts")
        if texts:
            texts[-1] = masked_text
            inputs["texts"] = texts
            wrote = True
        if not wrote:  # pragma: no cover
            raise RuntimeError(
                "Alice WonderFence MASK requested but no text source — refusing "
                "to silently no-op."
            )
        logger.info(
            "Alice WonderFence (apply_guardrail): MASK applied guardrail=%s correlation_id=%s",
            guardrail_name,
            correlation_id,
        )
    elif action == "DETECT":
        logger.warning(
            "Alice WonderFence (apply_guardrail): DETECT guardrail=%s correlation_id=%s",
            guardrail_name,
            correlation_id,
        )
