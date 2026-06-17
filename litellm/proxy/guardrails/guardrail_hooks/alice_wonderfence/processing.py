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


def _description_strings(obj: Any, prefix: List[Any]) -> List[Tuple[List[Any], str]]:
    """Collect ``(path, text)`` for every non-blank ``description`` string under
    ``obj`` (a tool's ``function`` dict). Recurses into nested JSON-schema
    parameters so parameter descriptions are included, not just the top one."""
    out: List[Tuple[List[Any], str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "description" and isinstance(value, str) and value.strip():
                out.append((prefix + [key], value))
            elif isinstance(value, (dict, list)):
                out.extend(_description_strings(value, prefix + [key]))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                out.extend(_description_strings(item, prefix + [idx]))
    return out


def tool_definition_segments(
    inputs: GenericGuardrailAPIInputs,
) -> Tuple[List[List[Any]], List[str]]:
    """Return (paths, texts) for free-text in tool definitions.

    The chat translation layer passes caller-supplied ``inputs["tools"]`` to the
    model verbatim, so a tool's ``function.description`` and its nested parameter
    descriptions are scanned like any other request segment. Each path locates
    the string within ``inputs["tools"]`` so a MASK verdict can be written back.
    """
    tools = inputs.get("tools") or []
    paths: List[List[Any]] = []
    segments: List[str] = []
    for i, tool in enumerate(tools):
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict):
            continue
        for sub_path, text in _description_strings(fn, ["function"]):
            paths.append([i, *sub_path])
            segments.append(text)
    return paths, segments


def _set_by_path(root: Any, path: List[Any], value: Any) -> None:
    obj = root
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


def _block_detail(
    blocked: List[SegmentVerdict], guardrail_name: str, block_message: str
) -> dict:
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
        "wonderfence_correlation_id": correlation_ids[0] if correlation_ids else None,
        "wonderfence_correlation_ids": correlation_ids,
    }
    if detections:
        detail["detections"] = [
            d.model_dump() if hasattr(d, "model_dump") else d for d in detections
        ]
    return detail


def _masked_value(
    verdict: SegmentVerdict, guardrail_name: str, label: str
) -> Optional[str]:
    """Return the replacement string for a MASK verdict (logging as a side
    effect), or None for DETECT/NO_ACTION. The caller writes it to the slot the
    segment came from."""
    correlation_id = verdict.correlation_ids[0] if verdict.correlation_ids else None
    if verdict.action == "MASK":
        logger.info(
            "Alice WonderFence (apply_guardrail): MASK applied to %s guardrail=%s correlation_id=%s",
            label,
            guardrail_name,
            correlation_id,
        )
        return verdict.masked_text if verdict.masked_text is not None else "[MASKED]"
    if verdict.action == "DETECT":
        logger.warning(
            "Alice WonderFence (apply_guardrail): DETECT %s guardrail=%s correlation_id=%s",
            label,
            guardrail_name,
            correlation_id,
        )
    return None


def apply_verdicts(
    inputs: GenericGuardrailAPIInputs,
    indices: List[int],
    verdicts: List[SegmentVerdict],
    guardrail_name: str,
    block_message: str,
    tool_indices: Optional[List[int]] = None,
    tool_verdicts: Optional[List[SegmentVerdict]] = None,
    tool_def_paths: Optional[List[List[Any]]] = None,
    tool_def_verdicts: Optional[List[SegmentVerdict]] = None,
) -> GenericGuardrailAPIInputs:
    """Apply per-segment verdicts back onto request text, tool-call args, and
    tool-definition descriptions.

    Any BLOCK across any group raises ``WonderFenceBlockedError`` with
    detections/correlation ids aggregated across all blocked segments. Otherwise
    each MASK verdict rewrites the slot its segment came from and DETECT is
    logged.
    """
    tool_indices = tool_indices or []
    tool_verdicts = tool_verdicts or []
    tool_def_paths = tool_def_paths or []
    tool_def_verdicts = tool_def_verdicts or []

    blocked = [
        v
        for v in (*verdicts, *tool_verdicts, *tool_def_verdicts)
        if v.action == "BLOCK"
    ]
    if blocked:
        raise WonderFenceBlockedError(
            _block_detail(blocked, guardrail_name, block_message)
        )

    texts = inputs.get("texts") or []
    for idx, verdict in zip(indices, verdicts):
        masked = _masked_value(verdict, guardrail_name, "request text")
        if masked is not None:
            texts[idx] = masked
    inputs["texts"] = texts

    tool_calls = inputs.get("tool_calls") or []
    for idx, verdict in zip(tool_indices, tool_verdicts):
        masked = _masked_value(verdict, guardrail_name, "tool_call args")
        if masked is not None:
            tool_calls[idx]["function"]["arguments"] = masked

    tools = inputs.get("tools") or []
    for path, verdict in zip(tool_def_paths, tool_def_verdicts):
        masked = _masked_value(verdict, guardrail_name, "tool definition")
        if masked is not None:
            _set_by_path(tools, path, masked)

    return inputs
