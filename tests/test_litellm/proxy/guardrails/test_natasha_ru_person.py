"""Unit tests for the Natasha Russian person-name (PER) guardrail hook."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("natasha")

from litellm.caching.caching import DualCache
from litellm.proxy.guardrails.guardrail_hooks.natasha_ru_person.natasha_ru_person import (
    NatashaRussianPersonGuardrail,
    iter_russian_person_spans,
    redact_russian_person_names,
    substring_fully_covered_by_spans,
    text_has_cyrillic,
)


@pytest.fixture(scope="module")
def natasha_stack():
    from natasha import NewsEmbedding, NewsNERTagger, Segmenter

    segmenter = Segmenter()
    embedding = NewsEmbedding()
    ner_tagger = NewsNERTagger(embedding)
    return segmenter, embedding, ner_tagger


def test_text_has_cyrillic():
    assert text_has_cyrillic("Иванов") is True
    assert text_has_cyrillic("ascii-only") is False


def test_ascii_payload_unchanged(natasha_stack):
    segmenter, embedding, ner_tagger = natasha_stack
    text = "sk-live-012345678901234567890abcdef"
    assert (
        redact_russian_person_names(
            text,
            segmenter=segmenter,
            embedding=embedding,
            ner_tagger=ner_tagger,
        )
        == text
    )


def test_iter_spans_and_coverage(natasha_stack):
    segmenter, embedding, ner_tagger = natasha_stack
    text = "Заказчик: Иванов Иван Иванович, договор №1."
    spans = iter_russian_person_spans(
        text,
        segmenter=segmenter,
        _embedding=embedding,
        ner_tagger=ner_tagger,
    )
    assert spans, "Natasha should emit at least one PER span for a full Russian name"
    assert substring_fully_covered_by_spans(text, "Иванов", spans) or any(
        "Иванов" in surf for _, _, surf in spans
    )


def test_redact_replaces_surface(natasha_stack):
    segmenter, embedding, ner_tagger = natasha_stack
    text = "ФИО ответственного: Петрова Мария Сергеевна."
    out = redact_russian_person_names(
        text,
        segmenter=segmenter,
        embedding=embedding,
        ner_tagger=ner_tagger,
        placeholder="<PER_REDACTED>",
    )
    assert "<PER_REDACTED>" in out


@pytest.mark.asyncio
async def test_async_pre_call_hook_user_message():
    guard = NatashaRussianPersonGuardrail(
        guardrail_name="natasha-test",
        event_hook="pre_call",
        default_on=True,
    )
    data = {
        "messages": [
            {
                "role": "user",
                "content": "Ответственный: Сидоров Петр Николаевич.",
            }
        ]
    }
    out = await guard.async_pre_call_hook(
        MagicMock(),
        DualCache(),
        data,
        "acompletion",
    )
    assert isinstance(out, dict)
    content = out["messages"][0]["content"]
    assert isinstance(content, str)
    assert "Сидоров" not in content or guard._placeholder in content
