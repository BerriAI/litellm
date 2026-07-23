"""Pure transforms for Alice WonderFence: context build, scan-piece gathering,
joined-document masked reconstruction, response-side verdict apply, total-work cap."""

from collections.abc import Sequence
from difflib import SequenceMatcher
from itertools import accumulate
from typing import Callable

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.types.utils import GenericGuardrailAPIInputs

from .chunked_evaluation import MAX_PROMPT_CHARS, SegmentVerdict
from .credentials import get_metadata
from .exceptions import WonderFenceBlockedError, WonderFenceScanBudgetExceeded

logger = verbose_proxy_logger.getChild("alice_wonderfence")

JOINER = "\n"

# Upper bound on the document ``reconstruct`` will align. ``SequenceMatcher`` is
# O(n*m) worst case and runs synchronously on the event loop, so a large
# repetitive MASK-triggering prompt could otherwise wedge it. MASK on a document
# larger than this fails closed (block) rather than run the quadratic alignment;
# non-MASK requests of any size are unaffected. Two chunks' worth keeps the
# worst case sub-second while still covering ordinary multi-message chats.
RECONSTRUCT_MAX_CHARS = 2 * MAX_PROMPT_CHARS


def build_analysis_context(
    request_data: dict,
    platform: str | None,
    context_class: Callable[..., object],
) -> object:
    """Build WonderFence AnalysisContext from request data."""
    metadata = get_metadata(request_data)
    model_str = request_data.get("model", "")

    provider = None
    model_name = model_str
    if model_str:
        try:
            model_name, provider, _, _ = litellm.get_llm_provider(model=model_str)
        except Exception:  # noqa: BLE001  # best-effort provider parse for telemetry; any failure falls back to manual split and must never break the guardrail
            if "/" in model_str:
                provider, model_name = model_str.split("/", 1)

    user_id = metadata.get("user_api_key_end_user_id") or metadata.get("end_user_id") or metadata.get("user_id")

    session_id = (
        request_data.get("litellm_session_id") or metadata.get("litellm_session_id") or metadata.get("session_id")
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
) -> tuple[list[int], list[str]]:
    """Return (indices, argument strings) for tool calls carrying string args.

    ``inputs["tool_calls"]`` entries are dicts shaped
    ``{"function": {"arguments": "<json string>"}}``; the argument string is the
    caller- or model-controlled payload that reaches the model/client, so it is
    scanned. ``indices`` is only used on the response side, where a MASK verdict
    is written back in place; on the request side the argument strings are
    appended to the joined document as detection-only pieces (see
    ``apply_guardrail``).
    """
    tool_calls = inputs.get("tool_calls") or []
    indices: list[int] = []
    segments: list[str] = []
    for i, tool_call in enumerate(tool_calls):
        fn = tool_call.get("function") if isinstance(tool_call, dict) else None
        args = fn.get("arguments") if isinstance(fn, dict) else None
        if isinstance(args, str) and args.strip():
            indices.append(i)
            segments.append(args)
    return indices, segments


def _description_texts(fn: object) -> list[str]:
    """Collect every non-blank ``description`` string under a tool's ``function``
    dict, walking nested JSON-schema parameters so parameter descriptions are
    included, not just the top one.

    Iterative (explicit stack) rather than recursive: caller-supplied tool
    schemas can nest arbitrarily, and unbounded recursion on request input is a
    DoS / stack-overflow risk. Only the strings are returned (no write-back
    paths): tool/function descriptions are scanned detection-only, so there is
    nothing to mask back into the schema.
    """
    out: list[str] = []
    stack: list[object] = [fn]
    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "description" and isinstance(value, str) and value.strip():
                    out.append(value)
                elif isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(obj, list):
            stack.extend(item for item in obj if isinstance(item, (dict, list)))
    return out


def tool_definition_segments(inputs: GenericGuardrailAPIInputs) -> list[str]:
    """Return description texts from ``inputs["tools"]`` (detection-only).

    The chat translation layer passes caller-supplied ``inputs["tools"]`` to the
    model verbatim, so a tool's ``function.description`` and its nested parameter
    descriptions are scanned. Detection-only: they are rendered into the joined
    document as extra pieces and can BLOCK/DETECT but are never masked back
    (there is no faithful place to splice a redaction into a schema).
    """
    tools = inputs.get("tools") or []
    return [
        text
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        for text in _description_texts(tool["function"])
    ]


def function_definition_segments(request_data: dict) -> list[str]:
    """Return description texts from the deprecated top-level ``functions[]``.

    Each entry is shaped like a tool's ``function`` object, so the same
    description walker applies. Detection-only, same rationale as
    ``tool_definition_segments``.
    """
    functions = request_data.get("functions") or []
    return [text for fn in functions if isinstance(fn, dict) for text in _description_texts(fn)]


def check_scan_budget(
    segments: list[str],
    max_scan_chars: int | None,
    max_scan_segments: int | None,
) -> None:
    """Fail-closed total-work cap; raises ``WonderFenceScanBudgetExceeded`` when
    the combined scan characters or segment count exceed the configured limits.

    WonderFence has no batch API (one HTTP POST per string), so without a cap a
    single crafted request with thousands of tiny parts or huge content could
    amplify into an unbounded number of upstream calls. This check runs before
    any WonderFence call and is never subject to ``fail_open`` — a caller must
    not be able to bypass scanning by overflowing the cap.
    """
    n = len(segments)
    if max_scan_segments is not None and n > max_scan_segments:
        raise WonderFenceScanBudgetExceeded(
            {
                "error": f"Alice WonderFence scan budget exceeded: {n} segments > max_scan_segments={max_scan_segments}",
                "type": "alice_wonderfence_scan_budget_exceeded",
                "limit": "max_scan_segments",
                "max_scan_segments": max_scan_segments,
                "segments": n,
            }
        )
    total_chars = sum(len(s) for s in segments)
    if max_scan_chars is not None and total_chars > max_scan_chars:
        raise WonderFenceScanBudgetExceeded(
            {
                "error": f"Alice WonderFence scan budget exceeded: {total_chars} chars > max_scan_chars={max_scan_chars}",
                "type": "alice_wonderfence_scan_budget_exceeded",
                "limit": "max_scan_chars",
                "max_scan_chars": max_scan_chars,
                "chars": total_chars,
            }
        )


def _map_index(x: int, ops: Sequence[tuple[str, int, int, int, int]], masked_len: int) -> int | None:
    """Map an index in the original joined document to its index in ``masked``.

    Uses the ``SequenceMatcher`` opcodes: an index inside (or at the end of) an
    ``equal`` block maps positionally; a boundary that lands at the very start
    of a changed block still maps (the range simply begins there); a boundary
    that lands *inside* a changed block is ambiguous and returns ``None`` so the
    caller fails closed rather than misassigning.
    """
    for tag, i1, i2, j1, _j2 in ops:
        if i1 <= x < i2 or (x == i2 and tag == "equal"):
            if tag == "equal":
                return j1 + (x - i1)
            return j1 if x == i1 else None
    return masked_len


def reconstruct(parts: list[str], masked: str) -> list[str] | None:
    """Recover per-part masked text from the masked joined document.

    ``parts`` were joined with ``JOINER`` (a plain ``"\\n"``) into the document
    that was scanned; ``masked`` is the service's masked version of that same
    document. We align original-vs-masked with ``difflib.SequenceMatcher`` (no
    sentinel injected) and map each part's char range through the alignment.

    Fails closed (returns ``None``) when the structure is not recoverable: the
    document exceeds ``RECONSTRUCT_MAX_CHARS`` (bounds the quadratic alignment
    cost); any ``JOINER`` between parts does not survive the mask as an
    unmodified ``\\n`` (a mask spanning a joiner would merge parts); or a part
    boundary lands inside a changed block. Returns one masked string per input
    part, in order; ``[]`` for no parts. Assumes masking is span substitution
    that preserves the non-masked characters; if the service reflows whitespace
    the joiner-survival check trips and we fail closed rather than misassign.
    """
    if not parts:
        return []

    original = JOINER.join(parts)
    if len(original) > RECONSTRUCT_MAX_CHARS or len(masked) > RECONSTRUCT_MAX_CHARS:
        return None
    starts = [0, *accumulate(len(p) + len(JOINER) for p in parts)][: len(parts)]
    ranges = [(s, s + len(p)) for s, p in zip(starts, parts)]
    joiners = [end for (_s, end) in ranges[:-1]]

    ops = SequenceMatcher(None, original, masked, autojunk=False).get_opcodes()

    joiner_survives = all(
        any(tag == "equal" and i1 <= j < i2 and masked[j1 + (j - i1)] == JOINER for tag, i1, i2, j1, _j2 in ops)
        for j in joiners
    )
    if not joiner_survives:
        return None

    mapped = [(_map_index(s, ops, len(masked)), _map_index(e, ops, len(masked))) for s, e in ranges]
    if any(ms is None or me is None or ms > me for ms, me in mapped):
        return None
    return [masked[ms:me] for ms, me in mapped]


def block_detail(blocked: list[SegmentVerdict], guardrail_name: str, block_message: str) -> dict:
    detections: list = []
    correlation_ids: list[str] = []
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
        detail["detections"] = [d.model_dump() if hasattr(d, "model_dump") else d for d in detections]
    return detail


def raise_if_blocked(verdicts: list[SegmentVerdict], guardrail_name: str, block_message: str) -> None:
    """Raise ``WonderFenceBlockedError`` if any verdict is BLOCK, aggregating
    detections / correlation ids across all blocked verdicts."""
    blocked = [v for v in verdicts if v.action == "BLOCK"]
    if blocked:
        raise WonderFenceBlockedError(block_detail(blocked, guardrail_name, block_message))


def _masked_value(verdict: SegmentVerdict, guardrail_name: str, label: str) -> str | None:
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


def apply_response_verdicts(
    inputs: GenericGuardrailAPIInputs,
    text_verdicts: list[SegmentVerdict],
    tool_indices: list[int],
    tool_verdicts: list[SegmentVerdict],
    guardrail_name: str,
    block_message: str,
) -> GenericGuardrailAPIInputs:
    """Response-side write-back: index-aligned MASK into ``texts`` (per choice)
    and ``tool_calls[i].function.arguments`` (model-generated).

    BLOCK across any segment raises first. Response text is written per-index
    (never joined) because the handler's response write-back is purely
    positional over the returned ``texts`` list and has no ``structured_messages``
    path, so collapsing choices into one string would dump every choice's text
    into choice 0.
    """
    raise_if_blocked([*text_verdicts, *tool_verdicts], guardrail_name, block_message)

    texts = inputs.get("texts") or []
    for idx, verdict in enumerate(text_verdicts):
        masked = _masked_value(verdict, guardrail_name, "response text")
        if masked is not None:
            texts[idx] = masked
    inputs["texts"] = texts

    tool_calls = inputs.get("tool_calls") or []
    for idx, verdict in zip(tool_indices, tool_verdicts):
        masked = _masked_value(verdict, guardrail_name, "tool_call args")
        if masked is not None:
            tool_calls[idx]["function"]["arguments"] = masked

    return inputs
