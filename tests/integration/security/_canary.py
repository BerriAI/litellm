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
- ``find_canary(blob, canaries) -> tuple[Match, ...]``: every canary whose core occurs in
  ``blob`` either raw, inside any base64-looking run after decoding it (standard and URL-safe
  alphabets, padded or not, at every 4-character alignment), or inside a gzip stream. Decoding
  is applied recursively a few levels deep, so a gzip body carrying a ``Basic`` header value is
  still searched. JSON and URL encoding leave a hex core unchanged, so the raw search covers
  them. A properly masked value such as ``sk-...e71b`` is not a match.
"""

from __future__ import annotations

import base64
import binascii
import gzip
import re
import uuid
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

_BASE64_RUN: Final = re.compile(rb"[A-Za-z0-9+/_-]{24,}={0,2}")
_GZIP_MAGIC: Final = b"\x1f\x8b"
_MAX_DEPTH: Final = 3


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
    for run in _BASE64_RUN.finditer(blob):
        text = run.group().rstrip(b"=")
        for offset in range(4):
            aligned = text[offset:]
            aligned = aligned[: len(aligned) - len(aligned) % 4] if len(aligned) % 4 == 1 else aligned
            padded = aligned + b"=" * (-len(aligned) % 4)
            for name, decode in (("base64", base64.b64decode), ("base64url", base64.urlsafe_b64decode)):
                try:
                    yield name, decode(padded)
                except (binascii.Error, ValueError):
                    continue


def _gunzipped(blob: bytes) -> bytes | None:
    if not blob.startswith(_GZIP_MAGIC):
        return None
    try:
        return gzip.decompress(blob)
    except (OSError, EOFError, zlib.error):
        return None


def _matches(blob: bytes, canaries: Sequence[Canary], encoding: str, depth: int) -> Iterable[Match]:
    lowered: Final = blob.lower()
    for candidate in canaries:
        if candidate.core.encode() in lowered:
            yield Match(candidate.slot, encoding)
    if depth >= _MAX_DEPTH:
        return
    inflated: Final = _gunzipped(blob)
    if inflated is not None:
        yield from _matches(inflated, canaries, f"{encoding}>gzip" if encoding != "raw" else "gzip", depth + 1)
    for name, decoded in _decoded_runs(blob):
        yield from _matches(decoded, canaries, f"{encoding}>{name}" if encoding != "raw" else name, depth + 1)


def find_canary(blob: bytes | str, canaries: Sequence[Canary]) -> tuple[Match, ...]:
    """Every canary found in ``blob``, one ``Match`` per slot with the shallowest encoding seen."""
    data: Final = blob.encode() if isinstance(blob, str) else blob
    found: Final[dict[str, Match]] = {}  # mutable-ok: first (shallowest) encoding per slot wins
    for match in _matches(data, canaries, "raw", 0):
        found.setdefault(match.slot, match)
    return tuple(found.values())
