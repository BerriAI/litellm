"""Mask literal PII matches after merging overlaps in the original text."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from heapq import merge
from itertools import groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.realmlabs import RealmLabsPIISpan


@dataclass(frozen=True, slots=True)
class _PIIMatch:
    start: int
    end: int
    entity_type: str


def _unique_pii_values(spans: Sequence[RealmLabsPIISpan]) -> Iterator[tuple[str, str]]:
    """Yield one label per literal value; conflicting types become ``pii``."""

    pairs: Final = (
        (value, entity_type) for span in spans if (value := span.get("text")) and (entity_type := span.get("type"))
    )
    for value, detections in groupby(sorted(frozenset(pairs)), key=lambda pair: pair[0]):
        entity_types = tuple(entity_type for _, entity_type in detections)
        yield value, entity_types[0] if len(entity_types) == 1 else "pii"


def _matches_for_value(text: str, value: str, entity_type: str) -> Iterator[_PIIMatch]:
    """Yield literal occurrences in position order, including overlapping occurrences."""

    length: Final = len(value)
    start = text.find(value)  # rebind-ok: advancing search cursor finds overlaps without rescanning earlier positions
    while start != -1:
        yield _PIIMatch(start, start + length, entity_type)
        start = text.find(value, start + 1)


def _matches_for_group(text: str, values: Sequence[str], entity_types: Mapping[str, str]) -> Iterator[_PIIMatch]:
    """Search values sharing their first character together, longest first at each position."""

    if len(values) == 1:
        yield from _matches_for_value(text, values[0], entity_types[values[0]])
        return

    alternatives: Final = "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True))
    pattern: Final = re.compile(alternatives)
    match = pattern.search(text)  # rebind-ok: advance the search cursor while retaining overlaps
    while match is not None:
        yield _PIIMatch(match.start(), match.end(), entity_types[match.group()])
        match = pattern.search(text, match.start() + 1)


def _merged_pii_matches(matches: Iterable[_PIIMatch]) -> Iterator[_PIIMatch]:
    """Merge ordered overlaps; keep an enclosing type, otherwise label the union ``pii``."""

    remaining: Final = iter(matches)
    region = next(remaining, None)  # rebind-ok: keep one pending region while consuming the ordered stream
    if region is None:
        return

    for match in remaining:
        if match.start >= region.end:
            yield region
            region = match
        elif match.end > region.end:
            region = _PIIMatch(region.start, match.end, "pii")

    yield region


def _masked_parts(text: str, regions: Iterable[_PIIMatch]) -> Iterator[str]:
    """Yield unchanged gaps and one mask per region, then the trailing text."""

    previous_end = 0  # rebind-ok: rendering cursor tracks the next unchanged slice without storing all regions
    for region in regions:
        yield text[previous_end : region.start]
        yield f"[{region.entity_type}]"
        previous_end = region.end

    yield text[previous_end:]


def mask_pii_in_text(text: str, spans: Sequence[RealmLabsPIISpan]) -> str:
    """Find original-text matches, merge overlaps, and render each masked region once.

    MLS offsets describe its rendering of the whole conversation, so local positions come from literal text.
    Group values by their first character to reduce repeated scans while preserving a literal regex prefix.
    Each group emits its longest match at a given start; shorter matches at that start are fully contained.
    With G groups and M emitted matches, ordering costs O(M log(G + 1)) time and O(G) space. Overlap merging
    is O(M). Pattern preparation, searches, and output assembly have their own costs; regex search time
    depends on the values and input text.
    """

    placeholders: Final = {f"[{label}]": label for span in spans if (label := span.get("type"))}
    entity_types: Final = MappingProxyType({"[pii]": "pii", **placeholders, **dict(_unique_pii_values(spans))})
    groups: Final = groupby(sorted(entity_types), key=lambda value: value[0])
    streams: Final = (_matches_for_group(text, tuple(values), entity_types) for _, values in groups)
    matches: Final = merge(*streams, key=lambda match: (match.start, -match.end))
    return "".join(_masked_parts(text, _merged_pii_matches(matches)))
