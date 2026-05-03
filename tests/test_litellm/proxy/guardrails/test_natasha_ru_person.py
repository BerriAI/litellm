"""Unit tests for the Natasha Russian person-name (PER) guardrail hook."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("natasha")

from litellm.caching.caching import DualCache
from litellm.proxy.guardrails.guardrail_hooks.natasha_ru_person.natasha_ru_person import (
    NatashaRussianPersonGuardrail,
    iter_russian_person_spans,
    merge_overlapping_intervals,
    redact_russian_person_names,
    substring_fully_covered_by_spans,
    text_has_cyrillic,
)

_NER_MODULE = (
    "litellm.proxy.guardrails.guardrail_hooks"
    ".natasha_ru_person.natasha_ru_person"
)
_INIT_MODULE = "litellm.proxy.guardrails.guardrail_hooks.natasha_ru_person"


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def natasha_stack():
    from natasha import NewsEmbedding, NewsNERTagger, Segmenter

    segmenter = Segmenter()
    embedding = NewsEmbedding()
    ner_tagger = NewsNERTagger(embedding)
    return segmenter, embedding, ner_tagger


@pytest.fixture(scope="module")
def guard():
    return NatashaRussianPersonGuardrail(
        guardrail_name="natasha-test",
        event_hook="pre_call",
        default_on=True,
    )


# ── text_has_cyrillic ─────────────────────────────────────────────────────────


def test_text_has_cyrillic():
    assert text_has_cyrillic("Иванов") is True
    assert text_has_cyrillic("ascii-only") is False


# ── merge_overlapping_intervals ───────────────────────────────────────────────


def test_merge_intervals_empty():
    assert merge_overlapping_intervals([]) == []


def test_merge_intervals_single():
    assert merge_overlapping_intervals([(2, 5)]) == [(2, 5)]


def test_merge_intervals_overlapping():
    assert merge_overlapping_intervals([(0, 5), (3, 8)]) == [(0, 8)]


def test_merge_intervals_non_overlapping():
    assert merge_overlapping_intervals([(0, 3), (5, 8)]) == [(0, 3), (5, 8)]


def test_merge_intervals_adjacent():
    # s == pe → condition s <= pe is True → fused into one span
    assert merge_overlapping_intervals([(0, 3), (3, 5)]) == [(0, 5)]


def test_merge_intervals_unsorted_input():
    assert merge_overlapping_intervals([(5, 8), (0, 3)]) == [(0, 3), (5, 8)]


# ── substring_fully_covered_by_spans ─────────────────────────────────────────


def test_covered_empty_needle():
    assert substring_fully_covered_by_spans("abc", "", [(0, 3, "abc")]) is True


def test_covered_no_spans():
    assert substring_fully_covered_by_spans("abc", "a", []) is False


def test_covered_needle_not_present():
    assert substring_fully_covered_by_spans("abc", "xyz", [(0, 3, "abc")]) is False


def test_covered_partial_overlap():
    # "ab" at [0,2), span covers [1,3) — char 0 is outside the span → False
    assert substring_fully_covered_by_spans("abcd", "ab", [(1, 3, "bc")]) is False


def test_covered_exact_span():
    assert (
        substring_fully_covered_by_spans("Иванов Иван", "Иванов", [(0, 6, "Иванов")])
        is True
    )


# ── iter_russian_person_spans edge cases ──────────────────────────────────────


def test_iter_spans_empty_text(natasha_stack):
    segmenter, _, ner_tagger = natasha_stack
    assert iter_russian_person_spans("", segmenter=segmenter, ner_tagger=ner_tagger) == []


def test_iter_spans_text_too_short(natasha_stack):
    segmenter, _, ner_tagger = natasha_stack
    assert iter_russian_person_spans("abc", segmenter=segmenter, ner_tagger=ner_tagger) == []


def test_iter_spans_and_coverage(natasha_stack):
    segmenter, _, ner_tagger = natasha_stack
    text = "Заказчик: Иванов Иван Иванович, договор №1."
    spans = iter_russian_person_spans(text, segmenter=segmenter, ner_tagger=ner_tagger)
    assert spans, "Natasha should emit at least one PER span for a full Russian name"
    assert substring_fully_covered_by_spans(text, "Иванов", spans) or any(
        "Иванов" in surf for _, _, surf in spans
    )


# ── redact_russian_person_names ───────────────────────────────────────────────


def test_ascii_payload_unchanged(natasha_stack):
    segmenter, embedding, ner_tagger = natasha_stack
    text = "sk-live-012345678901234567890abcdef"
    assert (
        redact_russian_person_names(
            text, segmenter=segmenter, embedding=embedding, ner_tagger=ner_tagger
        )
        == text
    )


def test_redact_empty_text(natasha_stack):
    segmenter, embedding, ner_tagger = natasha_stack
    assert (
        redact_russian_person_names(
            "", segmenter=segmenter, embedding=embedding, ner_tagger=ner_tagger
        )
        == ""
    )


def test_redact_short_text(natasha_stack):
    segmenter, embedding, ner_tagger = natasha_stack
    assert (
        redact_russian_person_names(
            "abc", segmenter=segmenter, embedding=embedding, ner_tagger=ner_tagger
        )
        == "abc"
    )


def test_redact_no_person_span(natasha_stack):
    segmenter, embedding, ner_tagger = natasha_stack
    with patch(f"{_NER_MODULE}.iter_russian_person_spans", return_value=[]):
        text = "Привет, мир!"
        out = redact_russian_person_names(
            text, segmenter=segmenter, embedding=embedding, ner_tagger=ner_tagger
        )
    assert out == text


def test_redact_ner_exception_returns_original(natasha_stack):
    segmenter, embedding, ner_tagger = natasha_stack
    with patch(
        f"{_NER_MODULE}.iter_russian_person_spans",
        side_effect=RuntimeError("ner error"),
    ):
        text = "Привет Иванов!"
        out = redact_russian_person_names(
            text, segmenter=segmenter, embedding=embedding, ner_tagger=ner_tagger
        )
    assert out == text


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


# ── NatashaRussianPersonGuardrail placeholder resolution ─────────────────────


def test_guardrail_custom_placeholder():
    g = NatashaRussianPersonGuardrail(
        guardrail_name="test", event_hook="pre_call", redaction_placeholder="[NAME]"
    )
    assert g._placeholder == "[NAME]"


def test_guardrail_env_var_placeholder(monkeypatch):
    monkeypatch.setenv("NATASHA_RU_PERSON_PLACEHOLDER", "<ENV_PER>")
    g = NatashaRussianPersonGuardrail(guardrail_name="test", event_hook="pre_call")
    assert g._placeholder == "<ENV_PER>"


# ── _process_content ──────────────────────────────────────────────────────────


def test_process_content_list_text_chunks(guard):
    content = [
        {"type": "text", "text": "Звонить Иванову Ивану Петровичу."},
        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
    ]
    result = guard._process_content(content)
    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert guard._placeholder in result[0]["text"]
    assert result[1] == content[1]  # non-text chunk unchanged


def test_process_content_none(guard):
    assert guard._process_content(None) is None


def test_process_content_unknown_type_passthrough(guard):
    assert guard._process_content(42) == 42


# ── async_pre_call_hook ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_pre_call_hook_user_message(guard):
    data = {
        "messages": [
            {"role": "user", "content": "Ответственный: Сидоров Петр Николаевич."}
        ]
    }
    out = await guard.async_pre_call_hook(MagicMock(), DualCache(), data, "acompletion")
    assert isinstance(out, dict)
    content = out["messages"][0]["content"]
    assert isinstance(content, str)
    assert "Сидоров" not in content and guard._placeholder in content


@pytest.mark.asyncio
async def test_async_pre_call_hook_system_message(guard):
    data = {
        "messages": [
            {
                "role": "system",
                "content": "Системный: Петров Антон Юрьевич — администратор.",
            }
        ]
    }
    out = await guard.async_pre_call_hook(MagicMock(), DualCache(), data, "acompletion")
    assert guard._placeholder in out["messages"][0]["content"]


@pytest.mark.asyncio
async def test_async_pre_call_hook_assistant_role_skipped(guard):
    original = "Иванов Иван — ассистент."
    data = {"messages": [{"role": "assistant", "content": original}]}
    out = await guard.async_pre_call_hook(MagicMock(), DualCache(), data, "acompletion")
    assert out["messages"][0]["content"] == original


@pytest.mark.asyncio
async def test_async_pre_call_hook_non_dict_message(guard):
    data = {"messages": ["not-a-dict"]}
    out = await guard.async_pre_call_hook(MagicMock(), DualCache(), data, "acompletion")
    assert out["messages"] == ["not-a-dict"]


@pytest.mark.asyncio
async def test_async_pre_call_hook_no_messages(guard):
    data: dict = {}
    out = await guard.async_pre_call_hook(MagicMock(), DualCache(), data, "acompletion")
    assert out == {}


@pytest.mark.asyncio
async def test_async_pre_call_hook_list_content(guard):
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Привет, Сидоров Петр Николаевич."},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]
    }
    out = await guard.async_pre_call_hook(MagicMock(), DualCache(), data, "acompletion")
    content = out["messages"][0]["content"]
    assert isinstance(content, list)
    assert "Сидоров" not in content[0]["text"] and guard._placeholder in content[0]["text"]
    assert content[1]["type"] == "image_url"


# ── initialize_guardrail ──────────────────────────────────────────────────────


def _make_params(mode="pre_call", default_on=True, optional_params=None):
    return SimpleNamespace(mode=mode, default_on=default_on, optional_params=optional_params)


class TestInitializeGuardrail:
    def test_missing_guardrail_name_raises(self):
        from litellm.proxy.guardrails.guardrail_hooks.natasha_ru_person import (
            initialize_guardrail,
        )

        with pytest.raises(ValueError, match="guardrail_name"):
            initialize_guardrail(_make_params(), {"guardrail_name": None})

    @patch(f"{_INIT_MODULE}.NatashaRussianPersonGuardrail")
    @patch("litellm.logging_callback_manager")
    def test_mode_as_list_uses_first_element(self, _lcm, mock_cls):
        from litellm.proxy.guardrails.guardrail_hooks.natasha_ru_person import (
            initialize_guardrail,
        )
        from litellm.types.guardrails import GuardrailEventHooks

        initialize_guardrail(_make_params(mode=["pre_call"]), {"guardrail_name": "test"})
        assert mock_cls.call_args.kwargs["event_hook"] == GuardrailEventHooks.pre_call

    @patch(f"{_INIT_MODULE}.NatashaRussianPersonGuardrail")
    @patch("litellm.logging_callback_manager")
    def test_mode_empty_list_defaults_to_pre_call(self, _lcm, mock_cls):
        from litellm.proxy.guardrails.guardrail_hooks.natasha_ru_person import (
            initialize_guardrail,
        )
        from litellm.types.guardrails import GuardrailEventHooks

        initialize_guardrail(_make_params(mode=[]), {"guardrail_name": "test"})
        assert mock_cls.call_args.kwargs["event_hook"] == GuardrailEventHooks.pre_call

    @patch(f"{_INIT_MODULE}.NatashaRussianPersonGuardrail")
    @patch("litellm.logging_callback_manager")
    def test_mode_non_string_non_list_defaults_to_pre_call(self, _lcm, mock_cls):
        from litellm.proxy.guardrails.guardrail_hooks.natasha_ru_person import (
            initialize_guardrail,
        )
        from litellm.types.guardrails import GuardrailEventHooks

        initialize_guardrail(_make_params(mode=42), {"guardrail_name": "test"})
        assert mock_cls.call_args.kwargs["event_hook"] == GuardrailEventHooks.pre_call

    @patch(f"{_INIT_MODULE}.NatashaRussianPersonGuardrail")
    @patch("litellm.logging_callback_manager")
    def test_redaction_placeholder_passed_through(self, _lcm, mock_cls):
        from litellm.proxy.guardrails.guardrail_hooks.natasha_ru_person import (
            initialize_guardrail,
        )

        optional = SimpleNamespace(
            natasha_redaction_placeholder="[MASKED]",
            natasha_ru_person_redaction_placeholder=None,
        )
        initialize_guardrail(
            _make_params(optional_params=optional), {"guardrail_name": "test"}
        )
        assert mock_cls.call_args.kwargs["redaction_placeholder"] == "[MASKED]"

    @patch(f"{_INIT_MODULE}.NatashaRussianPersonGuardrail")
    @patch("litellm.logging_callback_manager")
    def test_no_optional_params_placeholder_is_none(self, _lcm, mock_cls):
        from litellm.proxy.guardrails.guardrail_hooks.natasha_ru_person import (
            initialize_guardrail,
        )

        initialize_guardrail(_make_params(), {"guardrail_name": "test"})
        assert mock_cls.call_args.kwargs["redaction_placeholder"] is None

    def test_registries_populated(self):
        from litellm.proxy.guardrails.guardrail_hooks.natasha_ru_person import (
            guardrail_class_registry,
            guardrail_initializer_registry,
        )
        from litellm.types.guardrails import SupportedGuardrailIntegrations

        key = SupportedGuardrailIntegrations.NATASHA_RU_PERSON.value
        assert key in guardrail_initializer_registry
        assert guardrail_class_registry[key] is NatashaRussianPersonGuardrail
