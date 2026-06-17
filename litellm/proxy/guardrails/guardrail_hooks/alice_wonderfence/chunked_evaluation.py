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
# Detection-only overlap: when a segment is split into multiple chunks, content
# straddling a chunk boundary would be seen whole by neither chunk. We also
# evaluate a window spanning each boundary (last N chars of one chunk + first N
# of the next) so a blocked phrase up to ~2N chars long can't slip through the
# split. These windows feed BLOCK/DETECT only; masking still uses the disjoint
# chunks so the lossless rejoin invariant holds. Confirm sizing with the
# WonderFence team alongside MAX_PROMPT_CHARS.
CHUNK_OVERLAP_CHARS = 512


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


def _boundary_windows(chunks: List[str], overlap: int) -> List[str]:
    """Windows spanning each adjacent chunk boundary, for detection only.

    Each window is the last ``overlap`` chars of one chunk joined to the first
    ``overlap`` chars of the next, so a phrase split across the boundary is seen
    whole by the window (up to ~2*overlap long). Empty when there is one chunk.
    """
    if overlap <= 0:
        return []
    return [
        chunks[i][-overlap:] + chunks[i + 1][:overlap] for i in range(len(chunks) - 1)
    ]


def _aggregate(
    chunks: List[str],
    chunk_results: List[Any],
    boundary_results: List[Any],
) -> SegmentVerdict:
    chunk_actions = [_action_str(r) for r in chunk_results]
    boundary_actions = [_action_str(r) for r in boundary_results]
    detections: list = []
    correlation_ids: List[str] = []
    for r in (*chunk_results, *boundary_results):
        detections.extend(getattr(r, "detections", None) or [])
        cid = getattr(r, "correlation_id", None)
        if cid:
            correlation_ids.append(cid)

    if "BLOCK" in chunk_actions or "BLOCK" in boundary_actions:
        return SegmentVerdict("BLOCK", None, detections, correlation_ids)
    if "MASK" in chunk_actions:
        masked = "".join(
            (r.action_text or "[MASKED]") if _action_str(r) == "MASK" else chunk
            for chunk, r in zip(chunks, chunk_results)
        )
        return SegmentVerdict("MASK", masked, detections, correlation_ids)
    # A boundary window can only flag content that straddles a chunk split; we
    # cannot redact it across disjoint chunks, so surface it as DETECT rather
    # than dropping it. Per-chunk DETECT is folded in here too.
    if "DETECT" in chunk_actions or {"MASK", "DETECT"} & set(boundary_actions):
        return SegmentVerdict("DETECT", None, detections, correlation_ids)
    return SegmentVerdict("", None, detections, correlation_ids)


async def evaluate_segments(
    segments: List[str],
    evaluate: Callable[[str], Awaitable[Any]],
    max_chars: int = MAX_PROMPT_CHARS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> List[SegmentVerdict]:
    """Evaluate every segment (chunked) in parallel; return one verdict per segment.

    Each segment is split into <= ``max_chars`` disjoint chunks; multi-chunk
    segments also get a detection-only window spanning each chunk boundary (see
    ``_boundary_windows``). Every chunk and window across every segment is
    evaluated through a single ``asyncio.gather`` behind one shared
    ``Semaphore(max_concurrency)``. Results are grouped back per segment with
    action precedence BLOCK > MASK > DETECT > NO_ACTION; masking uses the
    disjoint chunks only so the lossless rejoin holds.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(text: str) -> Any:
        async with semaphore:
            return await evaluate(text)

    # Keep boundary windows within the prompt limit (<= 2*ov <= max_chars).
    ov = min(overlap, max_chars // 2)
    seg_chunks = [_split_text(s, max_chars) for s in segments]
    seg_boundaries = [_boundary_windows(chunks, ov) for chunks in seg_chunks]

    index: List[tuple] = []
    tasks = []
    for si in range(len(segments)):
        for ci, chunk in enumerate(seg_chunks[si]):
            index.append((si, False, ci))
            tasks.append(run(chunk))
        for bi, window in enumerate(seg_boundaries[si]):
            index.append((si, True, bi))
            tasks.append(run(window))
    results = await asyncio.gather(*tasks)

    chunk_res: List[List[Any]] = [[None] * len(c) for c in seg_chunks]
    bound_res: List[List[Any]] = [[None] * len(b) for b in seg_boundaries]
    for (si, is_boundary, idx), res in zip(index, results):
        (bound_res if is_boundary else chunk_res)[si][idx] = res

    return [
        _aggregate(seg_chunks[si], chunk_res[si], bound_res[si])
        for si in range(len(segments))
    ]
