import json

import pytest

from litellm.tracing.preview import MIN_PREVIEW_BYTES, preview


def test_content_within_budget_is_unchanged() -> None:
    assert preview('[{"content":"hi"}]', 1024) == '[{"content":"hi"}]'


@pytest.mark.parametrize("unit", ["a", "é", "雪", "🦀"], ids=["ascii", "two_byte", "three_byte", "four_byte"])
def test_plain_text_is_cut_on_a_character_boundary_within_budget(unit: str) -> None:
    content = unit * 1000

    result = preview(content, 300)

    kept = result.split("…[Lens preview: ")[0]
    assert len(result.encode()) <= 300
    assert kept and content.startswith(kept)
    assert result.endswith(f"{len(content.encode()) - len(kept.encode())} bytes omitted]")


def test_json_messages_keep_their_structure_and_mark_the_cut_leaf() -> None:
    long = "雪" * 2000
    content = json.dumps([{"role": "system", "content": "short"}, {"role": "user", "content": long}])

    result = preview(content, 1024)
    parsed = json.loads(result)

    assert len(result.encode()) <= 1024
    assert parsed[0] == {"role": "system", "content": "short"}
    kept = parsed[1]["content"].split("…[Lens preview: ")[0]
    assert kept and long.startswith(kept)
    assert parsed[1]["content"].endswith(f"{len(long.encode()) - len(kept.encode())} bytes omitted]")


def test_json_whose_structure_overflows_becomes_one_marked_string() -> None:
    content = json.dumps([{"i": index} for index in range(400)])

    result = preview(content, 300)

    assert len(result.encode()) <= 300
    text = json.loads(result)
    assert isinstance(text, str)
    assert content.startswith(text.split("…[Lens preview: ")[0])


def test_budget_below_minimum_is_raised_to_the_minimum() -> None:
    result = preview("x" * 10_000, 1)
    assert MIN_PREVIEW_BYTES // 2 < len(result.encode()) <= MIN_PREVIEW_BYTES
