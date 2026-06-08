"""Chunk, evaluate in parallel, and aggregate per segment.

Guardrail-agnostic: the only coupling to WonderFence is the injected
``evaluate`` callable and the result shape it returns (``action``,
``action_text``, ``detections``, ``correlation_id``). Replace ``evaluate`` to
target a different backend.
"""

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

MAX_PROMPT_CHARS = 10000  # WonderFence server-side prompt limit
DEFAULT_MAX_CONCURRENCY = 10  # used when the client connection_pool_limit is unset


@dataclass
class SegmentVerdict:
    action: str  # "BLOCK" | "MASK" | "DETECT" | ""
    masked_text: Optional[str]
    detections: list
    correlation_ids: List[str]


def _split_text(text: str, max_chars: int) -> List[str]:
    """Split ``text`` into <= ``max_chars`` chunks with ``"".join(chunks) == text``.

    Splits at whitespace boundaries; whitespace runs are preserved as their own
    tokens so the rejoin is byte-identical. A single token longer than
    ``max_chars`` is force-split. Always returns at least one chunk.
    """
    if len(text) <= max_chars:
        return [text]

    tokens = re.findall(r"\S+|\s+", text)
    chunks: List[str] = []
    current = ""
    for token in tokens:
        if len(current) + len(token) <= max_chars:
            current += token
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(token) > max_chars:
            chunks.append(token[:max_chars])
            token = token[max_chars:]
        current = token
    if current:
        chunks.append(current)
    return chunks


def _action_str(result: Any) -> str:
    action = getattr(result, "action", "")
    return action.value if hasattr(action, "value") else (action or "")


def _aggregate(chunks: List[str], results: List[Any]) -> SegmentVerdict:
    actions = [_action_str(r) for r in results]
    detections: list = []
    correlation_ids: List[str] = []
    for r in results:
        detections.extend(getattr(r, "detections", None) or [])
        cid = getattr(r, "correlation_id", None)
        if cid:
            correlation_ids.append(cid)

    if "BLOCK" in actions:
        return SegmentVerdict("BLOCK", None, detections, correlation_ids)
    if "MASK" in actions:
        masked = "".join(
            (r.action_text or "[MASKED]") if _action_str(r) == "MASK" else chunk
            for chunk, r in zip(chunks, results)
        )
        return SegmentVerdict("MASK", masked, detections, correlation_ids)
    if "DETECT" in actions:
        return SegmentVerdict("DETECT", None, detections, correlation_ids)
    return SegmentVerdict("", None, detections, correlation_ids)


async def evaluate_segments(
    segments: List[str],
    evaluate: Callable[[str], Awaitable[Any]],
    max_chars: int = MAX_PROMPT_CHARS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> List[SegmentVerdict]:
    """Evaluate every segment (chunked) in parallel; return one verdict per segment.

    Each segment is split into <= ``max_chars`` chunks; every chunk across every
    segment is evaluated through a single ``asyncio.gather`` behind one shared
    ``Semaphore(max_concurrency)``. Results are grouped back per segment with
    action precedence BLOCK > MASK > DETECT > NO_ACTION.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(chunk: str) -> Any:
        async with semaphore:
            return await evaluate(chunk)

    seg_chunks = [_split_text(s, max_chars) for s in segments]
    flat_index = [
        (si, ci) for si, chunks in enumerate(seg_chunks) for ci in range(len(chunks))
    ]
    tasks = [run(seg_chunks[si][ci]) for si, ci in flat_index]
    results = await asyncio.gather(*tasks)

    per_segment: List[List[Any]] = [[None] * len(chunks) for chunks in seg_chunks]
    for (si, ci), res in zip(flat_index, results):
        per_segment[si][ci] = res

    return [_aggregate(seg_chunks[si], per_segment[si]) for si in range(len(segments))]
