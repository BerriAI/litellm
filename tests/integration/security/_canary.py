"""Canary values and the canary search used by every credential sweep.

A canary is a unique fake credential planted in one slot (one place the proxy can hold a
credential). Its value is ``<field prefix>lkc-<slot id>-<32 lowercase hex core>``; the slot id
names the source when a sweep finds it, and the random core is what every sweep searches for.

API:

- ``SLOTS``: slot id -> ``Slot(identity, description, prefix)``. Stacked suites add their slots
  here. ``MARKER`` is not a credential; it is the sensitivity marker sent in message content to
  prove that a sweep can see the surface it walks.
- ``canary(slot_id) -> Canary``: a fresh value per call. Call it inside the test (or the fixture
  that owns the config holding it), never at import time, so leftovers from earlier runs cannot
  match.
- ``find_canary(blob, canaries, *, budget_bytes=DECODE_BUDGET_BYTES) -> tuple[Match, ...]``:
  every canary whose core occurs in ``blob`` either raw, inside any base64-looking run after
  decoding it (standard and URL-safe alphabets, padded or not, at every 4-character alignment),
  or inside a gzip member wherever it starts in the blob. Decoding is applied recursively, so a
  gzip body carrying a ``Basic`` header value is still searched. JSON and URL encoding leave a
  hex core unchanged, so the raw search covers them. A properly masked value such as
  ``sk-...e71b`` is not a match. The search is bounded (three nested layers and ``budget_bytes``
  of decoded output per blob) and raises ``DecodeBudgetExceeded`` rather than returning a
  partial result.
"""

from __future__ import annotations

import binascii
import re
import uuid
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

_BASE64_RUN: Final = re.compile(rb"[A-Za-z0-9+/_-]{24,}={0,2}")
_GZIP_MAGIC: Final = b"\x1f\x8b"
_TO_STANDARD: Final = bytes.maketrans(b"-_", b"+/")
_MAX_DEPTH: Final = 3
DECODE_BUDGET_BYTES: Final = 512 * 1024 * 1024


class DecodeBudgetExceeded(AssertionError):
    """A blob needs more decoded bytes than the search budget; the sweep cannot vouch for it."""


@dataclass(slots=True)
class _Budget:
    remaining: int

    def spend(self, size: int) -> None:
        self.remaining -= size
        if self.remaining < 0:
            raise DecodeBudgetExceeded("find_canary needed more decoded bytes than its budget for one blob")


@dataclass(frozen=True, slots=True)
class Slot:
    identity: str
    description: str
    prefix: str = ""


@dataclass(frozen=True, slots=True)
class Canary:
    slot: str
    core: str
    value: str


@dataclass(frozen=True, slots=True)
class Match:
    slot: str
    encoding: str


MARKER: Final = "M0"

SLOTS: Final = MappingProxyType(
    {
        MARKER: Slot(MARKER, "Sensitivity marker in message content; must appear where prompts are stored"),
        "B1": Slot("B1", "Deployment api_key declared in the proxy config.yaml model_list"),
    }
)


def canary(slot_id: str) -> Canary:
    slot: Final = SLOTS[slot_id]
    core: Final = uuid.uuid4().hex
    return Canary(slot_id, core, f"{slot.prefix}lkc-{slot_id}-{core}")


def _decoded_runs(blob: bytes) -> Iterable[tuple[str, bytes]]:
    for text in dict.fromkeys(run.group().rstrip(b"=") for run in _BASE64_RUN.finditer(blob)):
        for offset in range(4):
            aligned = text[offset:]
            aligned = aligned[: len(aligned) - len(aligned) % 4] if len(aligned) % 4 == 1 else aligned
            padded = aligned + b"=" * (-len(aligned) % 4)
            alphabets = (("base64", b"+/"), ("base64url", b"-_"))
            for name, extra in alphabets if any(char in aligned for char in b"+/-_") else alphabets[:1]:
                try:
                    yield (
                        name,
                        binascii.a2b_base64(
                            padded.translate(_TO_STANDARD) if extra == b"-_" else padded, strict_mode=False
                        ),
                    )
                except (binascii.Error, ValueError):
                    continue


def _gunzipped(blob: bytes, budget: _Budget) -> Iterable[bytes]:
    """Inflate every gzip member in ``blob``, wherever it starts, ignoring trailing bytes."""
    start = blob.find(_GZIP_MAGIC)
    while start != -1:
        inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            inflated = inflater.decompress(blob[start:], budget.remaining + 1)
        except zlib.error:
            inflated = b""
        budget.spend(len(inflated))
        if inflated:
            yield inflated
        start = blob.find(_GZIP_MAGIC, start + 1)


def _matches(blob: bytes, canaries: Sequence[Canary], encoding: str, depth: int, budget: _Budget) -> Iterable[Match]:
    lowered: Final = blob.lower()
    for candidate in canaries:
        if candidate.core.encode() in lowered:
            yield Match(candidate.slot, encoding)
    if depth >= _MAX_DEPTH:
        return
    for inflated in _gunzipped(blob, budget):
        yield from _matches(inflated, canaries, f"{encoding}>gzip" if encoding != "raw" else "gzip", depth + 1, budget)
    for name, decoded in _decoded_runs(blob):
        budget.spend(len(decoded))
        label = f"{encoding}>{name}" if encoding != "raw" else name
        if _worth_descending(decoded):
            yield from _matches(decoded, canaries, label, depth + 1, budget)
        else:
            lowered_decoded = decoded.lower()
            yield from (Match(c.slot, label) for c in canaries if c.core.encode() in lowered_decoded)


def _worth_descending(decoded: bytes) -> bool:
    """Recursion can only find something through a gzip member or another base64 run.

    Skipping the rest is exact, not a heuristic: the core check has already run on ``decoded``.
    """
    return _GZIP_MAGIC in decoded or _BASE64_RUN.search(decoded) is not None


def find_canary(
    blob: bytes | str, canaries: Sequence[Canary], *, budget_bytes: int = DECODE_BUDGET_BYTES
) -> tuple[Match, ...]:
    """Every canary found in ``blob``, one ``Match`` per slot with the shallowest encoding seen.

    Decoding is bounded: at most ``_MAX_DEPTH`` nested layers and ``DECODE_BUDGET_BYTES`` decoded or
    inflated bytes per call (``budget_bytes``). Exceeding the byte budget raises ``DecodeBudgetExceeded`` (an
    ``AssertionError``) instead of returning a partial, possibly clean, result.
    """
    data: Final = blob.encode() if isinstance(blob, str) else blob
    found: Final[dict[str, Match]] = {}  # mutable-ok: first (shallowest) encoding per slot wins
    for match in _matches(data, canaries, "raw", 0, _Budget(budget_bytes)):
        found.setdefault(match.slot, match)
    return tuple(found.values())
