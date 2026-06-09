"""Pure transforms for Alice WonderFence: context build, user-text mapping, verdict apply."""

from typing import Any, List, Optional, Tuple

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


def tool_call_arg_segments(
    inputs: GenericGuardrailAPIInputs,
) -> Tuple[List[int], List[str]]:
    """Return (indices, argument strings) for tool calls carrying string args.

    ``inputs["tool_calls"]`` entries are dicts shaped
    ``{"function": {"arguments": "<json string>"}}``; the argument string is the
    caller- or model-controlled payload that reaches the model/client, so it is
    scanned like any other segment.
    """
    tool_calls = inputs.get("tool_calls") or []
    indices: List[int] = []
    segments: List[str] = []
    for i, tool_call in enumerate(tool_calls):
        fn = tool_call.get("function") if isinstance(tool_call, dict) else None
        args = fn.get("arguments") if isinstance(fn, dict) else None
        if isinstance(args, str) and args.strip():
            indices.append(i)
            segments.append(args)
    return indices, segments


def apply_verdicts(
    inputs: GenericGuardrailAPIInputs,
    indices: List[int],
    verdicts: List[SegmentVerdict],
    guardrail_name: str,
    block_message: str,
    tool_indices: Optional[List[int]] = None,
    tool_verdicts: Optional[List[SegmentVerdict]] = None,
) -> GenericGuardrailAPIInputs:
    """Apply per-segment verdicts back onto ``inputs["texts"]`` and tool-call args.

    Any BLOCK across text or tool-call segments raises ``WonderFenceBlockedError``
    with detections/correlation ids aggregated across all blocked segments.
    Otherwise each MASK verdict rewrites its mapped ``texts`` index or
    ``tool_calls[i]["function"]["arguments"]`` and DETECT is logged.
    """
    tool_indices = tool_indices or []
    tool_verdicts = tool_verdicts or []
    blocked = [v for v in (*verdicts, *tool_verdicts) if v.action == "BLOCK"]
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

    tool_calls = inputs.get("tool_calls") or []
    for idx, verdict in zip(tool_indices, tool_verdicts):
        if verdict.action == "MASK":
            tool_calls[idx]["function"]["arguments"] = (
                verdict.masked_text if verdict.masked_text is not None else "[MASKED]"
            )
            logger.info(
                "Alice WonderFence (apply_guardrail): MASK applied to tool_call args guardrail=%s correlation_id=%s",
                guardrail_name,
                verdict.correlation_ids[0] if verdict.correlation_ids else None,
            )
        elif verdict.action == "DETECT":
            logger.warning(
                "Alice WonderFence (apply_guardrail): DETECT tool_call args guardrail=%s correlation_id=%s",
                guardrail_name,
                verdict.correlation_ids[0] if verdict.correlation_ids else None,
            )
    return inputs
