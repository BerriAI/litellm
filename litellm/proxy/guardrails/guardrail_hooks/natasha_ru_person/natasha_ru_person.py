"""
Natasha (Russian NLP) NER guardrail: mask person names (PER) in pre-call prompts.

Natasha is imported lazily so ``import litellm.proxy.guardrails`` works without
the optional ``natasha-ru-person`` extra installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple, Union

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks

log = logging.getLogger(__name__)

# Cyrillic blocks used for quick ASCII skip (same idea as ITMO seminar gateway).
_CYR_LO, _CYR_HI = "\u0400", "\u04ff"


def text_has_cyrillic(text: str) -> bool:
    return any(_CYR_LO <= ch <= _CYR_HI for ch in text)


def merge_overlapping_intervals(
    intervals: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int]] = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def substring_fully_covered_by_spans(
    haystack: str, needle: str, spans: List[Tuple[int, int, str]]
) -> bool:
    """
    Return True iff some occurrence of ``needle`` in ``haystack`` lies entirely
    inside the union of half-open intervals [start, stop) from PER spans.
    """
    if not needle:
        return True
    if not spans:
        return False
    idx = 0
    norm = [(int(s), int(e)) for s, e, _ in spans]
    merged = merge_overlapping_intervals(norm)
    while True:
        pos = haystack.find(needle, idx)
        if pos < 0:
            return False
        end = pos + len(needle)
        if all(any(ms <= j < me for ms, me in merged) for j in range(pos, end)):
            return True
        idx = pos + 1


def iter_russian_person_spans(
    text: str,
    *,
    segmenter: Any,
    _embedding: Any,
    ner_tagger: Any,
) -> List[Tuple[int, int, str]]:
    """
    Run Natasha NER and return (start, stop, surface) for each ``PER`` span.

    Callers must pass pre-built Natasha objects (see tests / guardrail __init__).
    """
    from natasha import Doc

    if not text or len(text) < 4:
        return []
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    out: List[Tuple[int, int, str]] = []
    for s in doc.spans:
        if s.type == "PER":
            out.append((s.start, s.stop, text[s.start : s.stop]))
    return out


def redact_russian_person_names(
    text: str,
    *,
    segmenter: Any,
    embedding: Any,
    ner_tagger: Any,
    placeholder: str = "<PER_REDACTED>",
) -> str:
    if not text or len(text) < 4:
        return text
    if not text_has_cyrillic(text):
        return text
    try:
        spans = iter_russian_person_spans(
            text, segmenter=segmenter, _embedding=embedding, ner_tagger=ner_tagger
        )
    except Exception as e:  # noqa: BLE001
        log.warning("natasha_ru_person: NER failed (%s); passing text through", e)
        return text
    if not spans:
        return text
    spans_sorted = sorted(spans, key=lambda t: t[0], reverse=True)
    out = text
    for start, stop, _ in spans_sorted:
        out = out[:start] + placeholder + out[stop:]
    return out


class NatashaRussianPersonGuardrail(CustomGuardrail):
    """Pre-call mask for Russian / Cyrillic person entities (Natasha ``PER``)."""

    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        event_hook: Optional[Union[str, GuardrailEventHooks]] = None,
        default_on: bool = False,
        redaction_placeholder: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        ph = (
            redaction_placeholder
            or os.getenv("NATASHA_RU_PERSON_PLACEHOLDER")
            or "<PER_REDACTED>"
        )
        self._placeholder = ph
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=[GuardrailEventHooks.pre_call],
            event_hook=event_hook or GuardrailEventHooks.pre_call,
            default_on=default_on,
            **kwargs,
        )
        verbose_proxy_logger.info("natasha_ru_person: loading Natasha models…")
        from natasha import NewsEmbedding, NewsNERTagger, Segmenter

        self._segmenter = Segmenter()
        self._emb = NewsEmbedding()
        self._ner_tagger = NewsNERTagger(self._emb)
        verbose_proxy_logger.info("natasha_ru_person: ready")

    def _redact(self, text: str) -> str:
        return redact_russian_person_names(
            text,
            segmenter=self._segmenter,
            embedding=self._emb,
            ner_tagger=self._ner_tagger,
            placeholder=self._placeholder,
        )

    def _process_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return self._redact(content)
        if isinstance(content, list):
            out: List[Any] = []
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    new_chunk = dict(chunk)
                    new_chunk["text"] = self._redact(chunk.get("text", "") or "")
                    out.append(new_chunk)
                else:
                    out.append(chunk)
            return out
        return content

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Optional[Union[dict, str]]:
        for msg in data.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") not in ("user", "system"):
                continue
            msg["content"] = self._process_content(msg.get("content"))
        return data
