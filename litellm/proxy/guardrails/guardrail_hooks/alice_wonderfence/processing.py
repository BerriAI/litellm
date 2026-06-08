"""Pure transforms for Alice WonderFence: context build, user-text mapping, verdict apply."""

from typing import Any, List, Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.types.utils import GenericGuardrailAPIInputs

from .chunked_evaluation import SegmentVerdict
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


def request_user_text_indices(
    structured_messages: Optional[List[Any]],
    texts: List[str],
) -> List[int]:
    """Return indices into ``texts`` that came from user-role messages.

    Replays the same flatten the translation layer uses to build ``texts``
    (string content -> one entry; list content -> one entry per item with a
    ``text`` field) over ``structured_messages`` and tags each entry's role. If
    ``structured_messages`` is absent or the replayed count diverges from
    ``len(texts)``, every index is returned: over-scanning is safe, mis-mapping
    a mask onto a non-user slot is not.
    """
    n = len(texts)
    if not structured_messages:
        return list(range(n))

    roles: List[str] = []
    for message in structured_messages:
        role = str(message.get("role") or "").lower()
        content = message.get("content", None)
        if content is None:
            continue
        if isinstance(content, str):
            roles.append(role)
        elif isinstance(content, list):
            for item in content:
                if item.get("text", None) is not None:
                    roles.append(role)

    if len(roles) != n:
        return list(range(n))
    return [i for i, role in enumerate(roles) if role == "user"]


def apply_verdicts(
    inputs: GenericGuardrailAPIInputs,
    indices: List[int],
    verdicts: List[SegmentVerdict],
    guardrail_name: str,
    block_message: str,
) -> GenericGuardrailAPIInputs:
    """Apply per-segment verdicts back onto ``inputs["texts"]``.

    Any BLOCK raises ``WonderFenceBlockedError`` with detections/correlation ids
    aggregated across all blocked segments. Otherwise each MASK verdict rewrites
    its mapped ``texts`` index and DETECT is logged.
    """
    blocked = [v for v in verdicts if v.action == "BLOCK"]
    if blocked:
        detections: list = []
        correlation_ids: List[str] = []
        for v in blocked:
            detections.extend(v.detections)
            correlation_ids.extend(v.correlation_ids)
        detail: dict = {
            "error": block_message,
            "type": "alice_wonderfence_content_policy_violation",
            "guardrail_name": guardrail_name,
            "action": "BLOCK",
            "wonderfence_correlation_id": (
                correlation_ids[0] if correlation_ids else None
            ),
            "wonderfence_correlation_ids": correlation_ids,
        }
        if detections:
            detail["detections"] = [
                d.model_dump() if hasattr(d, "model_dump") else d for d in detections
            ]
        raise WonderFenceBlockedError(detail)

    texts = inputs.get("texts") or []
    for idx, verdict in zip(indices, verdicts):
        if verdict.action == "MASK":
            texts[idx] = (
                verdict.masked_text if verdict.masked_text is not None else "[MASKED]"
            )
            logger.info(
                "Alice WonderFence (apply_guardrail): MASK applied guardrail=%s correlation_id=%s",
                guardrail_name,
                verdict.correlation_ids[0] if verdict.correlation_ids else None,
            )
        elif verdict.action == "DETECT":
            logger.warning(
                "Alice WonderFence (apply_guardrail): DETECT guardrail=%s correlation_id=%s",
                guardrail_name,
                verdict.correlation_ids[0] if verdict.correlation_ids else None,
            )
    inputs["texts"] = texts
    return inputs
