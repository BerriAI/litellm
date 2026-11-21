from collections.abc import Sequence
from typing import Final

import pytest

from litellm.proxy.guardrails.guardrail_hooks.realmlabs.pii_masking import mask_pii_in_text
from litellm.types.proxy.guardrails.guardrail_hooks.realmlabs import RealmLabsPIISpan


@pytest.mark.parametrize(
    ("text", "spans", "expected"),
    [
        pytest.param(
            "Contact Ann Smith Jones today",
            [{"type": "name", "text": "Ann Smith"}, {"type": "name", "text": "Smith Jones"}],
            "Contact [pii] today",
            id="partial-overlap",
        ),
        pytest.param(
            "Contact Ann Smith Jones Lee today",
            [
                {"type": "name", "text": "Ann Smith"},
                {"type": "name", "text": "Smith Jones"},
                {"type": "name", "text": "Jones Lee"},
            ],
            "Contact [pii] today",
            id="chain-of-overlaps",
        ),
        pytest.param(
            "Contact Ann Smith at Ann.Smith@example.com",
            [
                {"type": "name", "text": "Smith"},
                {"type": "name", "text": "Ann Smith"},
                {"type": "email", "text": "Ann.Smith@example.com"},
            ],
            "Contact [name] at [email]",
            id="contained-match-starts-later",
        ),
        pytest.param(
            "Ann Smith Jones",
            [
                {"type": "name", "text": "Ann Smith"},
                {"type": "name", "text": "Smith Jones"},
                {"type": "name", "text": "Ann Smith Jones"},
            ],
            "[name]",
            id="enclosing-match-keeps-its-label",
        ),
        pytest.param(
            "AnnAnn.Smith@example.com",
            [{"type": "name", "text": "Ann"}, {"type": "email", "text": "Ann.Smith@example.com"}],
            "[name][email]",
            id="adjacent-matches-stay-separate",
        ),
        pytest.param(
            "ababa",
            [{"type": "name", "text": "aba"}],
            "[pii]",
            id="overlapping-occurrences-of-one-value",
        ),
        pytest.param(
            "ababaca",
            [{"type": "name", "text": "ababa"}, {"type": "name", "text": "abaca"}],
            "[pii]",
            id="overlap-between-values-with-the-same-first-character",
        ),
        pytest.param(
            "Contact Ann today",
            [{"type": "name", "text": "Ann"}, {"type": "name", "text": "Ann"}],
            "Contact [name] today",
            id="duplicate-detections",
        ),
        pytest.param(
            "Contact Ann today",
            [{"type": "name", "text": "Ann"}, {"type": "username", "text": "Ann"}],
            "Contact [pii] today",
            id="same-match-with-conflicting-types",
        ),
        pytest.param(
            "Ann1 Ann Annn Ann[1] Ann+",
            [{"type": "name", "text": "Ann[1]"}, {"type": "username", "text": "Ann+"}],
            "Ann1 Ann Annn [name] [username]",
            id="regex-punctuation-is-literal",
        ),
        pytest.param(
            "👋 Éva / Éva@example.com",
            [{"type": "name", "text": "Éva"}, {"type": "email", "text": "Éva@example.com"}],
            "👋 [name] / [email]",
            id="unicode-positions-and-shorter-match-fallback",
        ),
        pytest.param(
            "Ann Anna",
            [
                {"type": "name", "text": "Ann"},
                {"type": "username", "text": "Ann"},
                {"type": "name", "text": "Anna"},
            ],
            "[pii] [name]",
            id="conflicting-inner-types-do-not-change-enclosing-type",
        ),
        pytest.param(
            "[name] Alex [Alex] [unknown]",
            ({"type": "name", "text": "[name"}, {"type": "name", "text": "Alex"}),
            "[name] [name] [[name]] [unknown]",
            id="real-pii-and-arbitrary-brackets-are-not-exempt",
        ),
        pytest.param(
            "Alex[name]",
            ({"type": "name", "text": "Alex[na"},),
            "[pii]",
            id="overlap-starts-before-placeholder",
        ),
        pytest.param(
            "[name]Alex",
            ({"type": "name", "text": "me]Alex"},),
            "[pii]",
            id="overlap-ends-after-placeholder",
        ),
        pytest.param(
            "[name]Alex[name]",
            ({"type": "name", "text": "me]Alex[na"},),
            "[pii]",
            id="overlap-joins-two-placeholders",
        ),
        pytest.param(
            "Alex[name]Jones",
            ({"type": "name", "text": "Alex[name]Jones"},),
            "[name]",
            id="real-pii-encloses-placeholder",
        ),
        pytest.param(
            "[name]alex@example.com",
            ({"type": "name", "text": "[name"}, {"type": "email", "text": "alex@example.com"}),
            "[name][email]",
            id="adjacent-pii-stays-separate",
        ),
        pytest.param(
            "[name] [pii] name",
            ({"type": "name", "text": "name"}, {"type": "username", "text": "name"}),
            "[name] [pii] [pii]",
            id="conflicting-types-preserve-existing-placeholders",
        ),
        pytest.param(
            "[pii] [name",
            ({"type": "name", "text": "pii"}, {"type": "name", "text": "[name"}),
            "[pii] [name]",
            id="generic-placeholder-and-incomplete-brackets",
        ),
        pytest.param(
            "👋 [custom.type+] Éva",
            ({"type": "custom.type+", "text": "type+"}, {"type": "name", "text": "Éva"}),
            "👋 [custom.type+] [name]",
            id="free-form-types-and-unicode",
        ),
        pytest.param(
            "alex@example.com email",
            ({"type": "email", "text": "alex@example.com"}, {"type": "name", "text": "email"}),
            "[email] [name]",
            id="inserted-labels-are-not-remasked",
        ),
        pytest.param("Ann is here", (), "Ann is here", id="no-detections"),
        pytest.param(
            "Ann is here",
            (
                RealmLabsPIISpan(),
                {"text": "Ann"},
                {"type": "name", "text": None},
                {"type": "name", "text": ""},
                {"type": "", "text": "Ann"},
            ),
            "Ann is here",
            id="incomplete-detections",
        ),
        pytest.param(
            "Hello [name]! [name][name]",
            ({"type": "name", "text": "[name"},),
            "Hello [name]! [name][name]",
            id="preserve-placeholder-missing-closing-bracket",
        ),
        pytest.param(
            "Hello [name]! [name][name]",
            ({"type": "name", "text": "name"},),
            "Hello [name]! [name][name]",
            id="preserve-placeholder-label-only",
        ),
        pytest.param(
            "Hello [name]! [name][name]",
            ({"type": "name", "text": "name]"},),
            "Hello [name]! [name][name]",
            id="preserve-placeholder-missing-opening-bracket",
        ),
        pytest.param(
            "Hello [name]! [name][name]",
            ({"type": "name", "text": "[name]"},),
            "Hello [name]! [name][name]",
            id="preserve-placeholder-whole-placeholder",
        ),
        pytest.param(
            "Hello [name]! [name][name]",
            ({"type": "name", "text": "nam"},),
            "Hello [name]! [name][name]",
            id="preserve-placeholder-partial-label",
        ),
        pytest.param(
            "Hello [name]! [name][name]",
            ({"type": "name", "text": "["},),
            "Hello [name]! [name][name]",
            id="preserve-placeholder-opening-bracket-only",
        ),
        pytest.param(
            "Hello [name]! [name][name]",
            ({"type": "name", "text": "]"},),
            "Hello [name]! [name][name]",
            id="preserve-placeholder-closing-bracket-only",
        ),
    ],
)
def test_pii_masking(text: str, spans: Sequence[RealmLabsPIISpan], expected: str) -> None:
    result: Final = mask_pii_in_text(text, spans)

    assert result == expected
    assert mask_pii_in_text(text, tuple(reversed(spans))) == expected
    assert mask_pii_in_text(result, spans) == expected
