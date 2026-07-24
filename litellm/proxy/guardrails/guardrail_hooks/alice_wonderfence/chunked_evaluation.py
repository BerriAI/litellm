"""Chunk, evaluate in parallel, and aggregate per segment.

Guardrail-agnostic: the only coupling to WonderFence is the injected
``evaluate`` callable and the result shape it returns (``action``,
``action_text``, ``detections``, ``correlation_id``). Replace ``evaluate`` to
target a different backend.
"""

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

MAX_PROMPT_CHARS = 10000  # WonderFence server-side prompt limit
DEFAULT_MAX_CONCURRENCY = 10  # used when the client connection_pool_limit is unset
# Overlap: a segment longer than the prompt limit is split into disjoint "owned"
# regions, but each chunk is scanned with the last N chars of the previous owned
# region prepended as a read-only prefix. A phrase straddling an owned-region
# seam (up to N chars into the left region) is therefore seen whole by one scan,
# so it can BLOCK/DETECT and, when the service masks it, the prefix bytes change
# and we fail closed (see ``_aggregate``) rather than stitch a half-masked seam.
# This replaces the old separate boundary-window calls. Confirm sizing with the
# WonderFence team alongside MAX_PROMPT_CHARS.
CHUNK_OVERLAP_CHARS = 512


@dataclass
class SegmentVerdict:
    action: str  # "BLOCK" | "MASK" | "DETECT" | ""
    masked_text: str | None
    detections: list
    correlation_ids: list[str]
    # Per owned-region ``(original, masked)`` pairs, set only on MASK. Lets the
    # caller align masking back to sub-structure (e.g. joined message parts) one
    # chunk at a time instead of over the whole document, so the alignment cost
    # is bounded by the chunk size rather than quadratic in the segment length.
    masked_chunks: list[tuple[str, str]] | None = None


@dataclass(frozen=True)
class WindowConfig:
    """Tuning for the chunk-seam overlap.

    ``overlap`` sizes the read-only prefix each chunk carries from the previous
    owned region (see ``_overlap_chunks``). There are no cross-segment windows:
    on the request side message parts are concatenated into one joined document
    before scanning (so their junctions are interior chunk seams); on the
    response side each segment is an independent choice or tool-call arg that the
    model never concatenates.
    """

    overlap: int = CHUNK_OVERLAP_CHARS


# Shared default so the ``evaluate_segments`` signature has a plain-name default
# (no call in the argument default); safe to share since ``WindowConfig`` is frozen.
_DEFAULT_WINDOW_CONFIG = WindowConfig()


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into <= ``max_chars`` chunks with ``"".join(chunks) == text``.

    Splits at whitespace boundaries; whitespace runs are preserved as their own
    tokens so the rejoin is byte-identical. A single token longer than
    ``max_chars`` is force-split. Always returns at least one chunk.
    """
    if len(text) <= max_chars:
        return [text]

    tokens = re.findall(r"\S+|\s+", text)
    chunks: list[str] = []
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


def _action_str(result: object) -> str:
    action = getattr(result, "action", "")
    return action.value if hasattr(action, "value") else (action or "")


def _overlap_chunks(text: str, max_chars: int, overlap: int) -> list[tuple[str, str]]:
    """Split ``text`` into overlapping scan chunks as ``(prefix, owned)`` pairs.

    ``owned`` regions are disjoint and concatenate back to ``text`` (lossless);
    ``prefix`` is the last ``overlap`` chars of the previous owned region (empty
    for the first). The scan input for a chunk is ``prefix + owned``, giving
    ``overlap`` chars of left-context so a phrase straddling the owned-region
    seam is seen whole. Reassembly strips the verbatim prefix back off, so the
    owned regions still rejoin losslessly.
    """
    owned = _split_text(text, max(1, max_chars - overlap))
    return [(owned[i - 1][-overlap:] if i and overlap > 0 else "", region) for i, region in enumerate(owned)]


def _aggregate(chunks: list[tuple[str, str]], results: list[Any]) -> SegmentVerdict:
    """Fold per-chunk results (scans of ``prefix + owned``) into one verdict.

    Precedence BLOCK > MASK > DETECT > NO_ACTION. A MASK whose masked text no
    longer starts with its verbatim ``prefix`` means the redaction reached into
    the prefix bytes -- i.e. content straddling (or sitting within ``overlap`` of)
    the owned-region seam. That cannot be stitched back without double-counting
    the overlap, so it fails closed (BLOCK) rather than leak the un-redacted half.
    Otherwise the prefix is stripped by its known length (no alignment needed)
    and the owned regions rejoin into the masked segment; the per-chunk
    ``(original, masked)`` pairs are carried on the verdict for bounded caller-side
    alignment.
    """
    actions = [_action_str(r) for r in results]
    detections: list = []
    correlation_ids: list[str] = []
    for r in results:
        detections.extend(getattr(r, "detections", None) or [])
        cid = getattr(r, "correlation_id", None)
        if cid:
            correlation_ids.append(cid)

    if "BLOCK" in actions:
        return SegmentVerdict("BLOCK", None, detections, correlation_ids)

    if "MASK" in actions:
        masked_chunks: list[tuple[str, str]] = []
        for (prefix, owned), r in zip(chunks, results):
            if _action_str(r) != "MASK":
                masked_chunks.append((owned, owned))
                continue
            masked = r.action_text if getattr(r, "action_text", None) is not None else prefix + "[MASKED]"
            if not masked.startswith(prefix):
                return SegmentVerdict("BLOCK", None, detections, correlation_ids)
            masked_chunks.append((owned, masked[len(prefix) :]))
        masked_text = "".join(m for _, m in masked_chunks)
        return SegmentVerdict("MASK", masked_text, detections, correlation_ids, masked_chunks)

    if "DETECT" in actions:
        return SegmentVerdict("DETECT", None, detections, correlation_ids)
    return SegmentVerdict("", None, detections, correlation_ids)


async def evaluate_segments(
    segments: list[str],
    evaluate: Callable[[str], Awaitable[Any]],
    max_chars: int = MAX_PROMPT_CHARS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    windows: WindowConfig = _DEFAULT_WINDOW_CONFIG,
) -> list[SegmentVerdict]:
    """Evaluate every segment (chunked) in parallel; return one verdict per segment.

    Each segment is split into <= ``max_chars`` overlapping chunks (disjoint
    ``owned`` regions each carrying an ``overlap``-char read-only prefix from the
    previous region, see ``_overlap_chunks``) so a phrase straddling an
    owned-region seam is seen whole by one scan without a separate boundary call.
    Every chunk across every segment is evaluated through a single
    ``asyncio.gather`` behind one shared ``Semaphore(max_concurrency)``. Results
    are folded per segment (see ``_aggregate``) with precedence
    BLOCK > MASK > DETECT > NO_ACTION.

    The request side passes a single joined document here (one segment) so the
    common case is one call; the response side passes one segment per choice /
    tool-call arg. There is no cross-segment window (see ``WindowConfig``).
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(text: str) -> object:
        async with semaphore:
            return await evaluate(text)

    # Keep each chunk's scan input (prefix + owned) within the prompt limit.
    ov = min(windows.overlap, max_chars // 2)
    seg_chunks = [_overlap_chunks(s, max_chars, ov) for s in segments]

    index: list[tuple[int, int]] = []
    tasks = []
    for si, chunks in enumerate(seg_chunks):
        for ci, (prefix, owned) in enumerate(chunks):
            index.append((si, ci))
            tasks.append(run(prefix + owned))
    results = await asyncio.gather(*tasks)

    chunk_res: list[list[Any]] = [[None] * len(c) for c in seg_chunks]
    for (si, ci), res in zip(index, results):
        chunk_res[si][ci] = res

    return [_aggregate(seg_chunks[si], chunk_res[si]) for si in range(len(segments))]
