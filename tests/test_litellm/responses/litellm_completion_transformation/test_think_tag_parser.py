"""Tests for ``<think>...</think>`` extraction in the Responses API bridge.

Covers the pure state-machine parser plus the bridge integration that lifts
extracted reasoning into ``message.reasoning_content`` so downstream output
item extractors emit a proper Responses-API ``reasoning`` item.
"""

from litellm.responses.litellm_completion_transformation.think_tag_parser import (
    ThinkTagParser,
    extract_think_from_text,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.utils import Choices, Message


# ---------------------------------------------------------------------------
# Pure parser tests
# ---------------------------------------------------------------------------


def test_no_tags_passthrough():
    p = ThinkTagParser()
    out = p.feed("hello world")
    assert out == [("text", "hello world")]


def test_simple_think_block_complete():
    p = ThinkTagParser()
    out = p.feed("<think>thinking</think> answer")
    assert out == [
        ("reasoning_open", None),
        ("reasoning", "thinking"),
        ("reasoning_close", None),
        ("text", " answer"),
    ]


def test_text_before_think_block():
    p = ThinkTagParser()
    out = p.feed("prefix <think>mid</think> suffix")
    assert out == [
        ("text", "prefix "),
        ("reasoning_open", None),
        ("reasoning", "mid"),
        ("reasoning_close", None),
        ("text", " suffix"),
    ]


def test_only_open_tag_buffers_for_next_chunk():
    p = ThinkTagParser()
    out = p.feed("hello <think>partial")
    assert ("reasoning_open", None) in out
    assert ("text", "hello ") in out
    has_reasoning = any(kind == "reasoning" and text == "partial" for kind, text in out)
    assert has_reasoning


def test_split_open_tag_across_chunks():
    p = ThinkTagParser()
    out1 = p.feed("hello <thi")
    out2 = p.feed("nk>thinking</think> bye")
    combined = out1 + out2
    text_segs = [t for kind, t in combined if kind == "text"]
    reasoning_segs = [t for kind, t in combined if kind == "reasoning"]
    assert "".join(text_segs) == "hello  bye"
    assert "".join(reasoning_segs) == "thinking"
    assert any(kind == "reasoning_open" for kind, _ in combined)
    assert any(kind == "reasoning_close" for kind, _ in combined)


def test_split_close_tag_across_chunks():
    p = ThinkTagParser()
    out1 = p.feed("<think>thinking</thi")
    out2 = p.feed("nk> done")
    combined = out1 + out2
    text_segs = [t for kind, t in combined if kind == "text"]
    reasoning_segs = [t for kind, t in combined if kind == "reasoning"]
    assert "".join(text_segs) == " done"
    assert "".join(reasoning_segs) == "thinking"


def test_split_both_tags_byte_by_byte():
    """Worst-case fragmentation: feed one char at a time."""
    p = ThinkTagParser()
    full = "before <think>secret</think> after"
    combined: list = []
    for ch in full:
        combined.extend(p.feed(ch))
    combined.extend(p.flush())
    text_segs = [t for kind, t in combined if kind == "text"]
    reasoning_segs = [t for kind, t in combined if kind == "reasoning"]
    assert "".join(text_segs) == "before  after"
    assert "".join(reasoning_segs) == "secret"


def test_open_tag_without_close_at_stream_end():
    """If stream ends with unclosed <think>, treat remainder as reasoning."""
    p = ThinkTagParser()
    out = p.feed("<think>open forever")
    out += p.flush()
    reasoning_segs = [t for kind, t in out if kind == "reasoning"]
    assert "".join(reasoning_segs) == "open forever"
    text_segs = [t for kind, t in out if kind == "text"]
    assert "".join(text_segs) == ""


def test_close_tag_without_open_passthrough():
    p = ThinkTagParser()
    out = p.feed("</think> just text")
    out += p.flush()
    text_segs = [t for kind, t in out if kind == "text"]
    assert "".join(text_segs) == "</think> just text"
    reasoning_segs = [t for kind, t in out if kind == "reasoning"]
    assert reasoning_segs == []


def test_multiple_think_blocks_in_one_response():
    p = ThinkTagParser()
    out = p.feed("<think>a</think> x <think>b</think> y")
    out += p.flush()
    reasoning_segs = [t for kind, t in out if kind == "reasoning"]
    assert reasoning_segs == ["a", "b"]
    text_segs = [t for kind, t in out if kind == "text"]
    assert "".join(text_segs) == " x  y"


def test_empty_feed():
    p = ThinkTagParser()
    assert p.feed("") == []
    assert p.flush() == []


def test_flush_holds_pending_partial_open_as_text():
    """If buffer ends mid-`<think>` and stream ends, treat the partial as text."""
    p = ThinkTagParser()
    out = p.feed("hello <thi")
    out += p.flush()
    assert not any(kind == "reasoning_open" for kind, _ in out)
    text_segs = [t for kind, t in out if kind == "text"]
    assert "".join(text_segs) == "hello <thi"


def test_consecutive_tags_no_gap():
    p = ThinkTagParser()
    out = p.feed("<think>a</think><think>b</think>")
    out += p.flush()
    reasoning_segs = [t for kind, t in out if kind == "reasoning"]
    assert reasoning_segs == ["a", "b"]


def test_empty_think_block():
    p = ThinkTagParser()
    out = p.feed("<think></think>hello")
    out += p.flush()
    opens = [k for k, _ in out if k == "reasoning_open"]
    closes = [k for k, _ in out if k == "reasoning_close"]
    assert len(opens) == 1
    assert len(closes) == 1
    text_segs = [t for kind, t in out if kind == "text"]
    assert "".join(text_segs) == "hello"


def test_whitespace_preserved_verbatim_inside_think():
    p = ThinkTagParser()
    out = p.feed("<think>\n  multi-line\n  reasoning  \n</think>!")
    out += p.flush()
    reasoning_segs = [t for kind, t in out if kind == "reasoning"]
    assert "".join(reasoning_segs) == "\n  multi-line\n  reasoning  \n"
    text_segs = [t for kind, t in out if kind == "text"]
    assert "".join(text_segs) == "!"


def test_non_ascii_content_inside_think():
    """Non-ASCII (UTF-8 multi-byte) text inside think blocks survives intact."""
    p = ThinkTagParser()
    full = (
        "<think>User is greeting in 日本語. Simple — no tool use needed.\n</think>"
        "\n\nこんにちは!"
    )
    out = p.feed(full)
    out += p.flush()
    reasoning_segs = [t for kind, t in out if kind == "reasoning"]
    text_segs = [t for kind, t in out if kind == "text"]
    assert (
        "".join(reasoning_segs)
        == "User is greeting in 日本語. Simple — no tool use needed.\n"
    )
    assert "".join(text_segs) == "\n\nこんにちは!"


# ---------------------------------------------------------------------------
# extract_think_from_text (non-stream helper)
# ---------------------------------------------------------------------------


def test_extract_think_no_tags():
    reasoning, text = extract_think_from_text("plain text")
    assert reasoning == ""
    assert text == "plain text"


def test_extract_think_single_block():
    reasoning, text = extract_think_from_text("<think>thinking</think> answer")
    assert reasoning == "thinking"
    assert text == " answer"


def test_extract_think_multiple_blocks_concatenated():
    reasoning, text = extract_think_from_text(
        "<think>step 1</think>middle<think>step 2</think>end"
    )
    assert reasoning == "step 1step 2"
    assert text == "middleend"


# ---------------------------------------------------------------------------
# Bridge integration: _should_parse_think_tags + _apply_think_tag_split_in_place
# ---------------------------------------------------------------------------


def _build_choice(content: str, reasoning_content: str = "") -> Choices:
    return Choices(
        finish_reason="stop",
        index=0,
        message=Message(content=content, role="assistant"),
    )


def test_should_parse_think_tags_opt_in_via_model_info():
    assert (
        LiteLLMCompletionResponsesConfig._should_parse_think_tags(
            {"model_info": {"parse_think_tags": True}}
        )
        is True
    )
    assert (
        LiteLLMCompletionResponsesConfig._should_parse_think_tags(
            {"model_info": {"parse_think_tags": False}}
        )
        is False
    )
    assert (
        LiteLLMCompletionResponsesConfig._should_parse_think_tags({"model_info": {}})
        is False
    )
    assert LiteLLMCompletionResponsesConfig._should_parse_think_tags({}) is False
    assert LiteLLMCompletionResponsesConfig._should_parse_think_tags(None) is False


def test_apply_think_tag_split_lifts_reasoning_into_field():
    choice = _build_choice(content="<think>internal</think>hello")
    LiteLLMCompletionResponsesConfig._apply_think_tag_split_in_place([choice])
    assert choice.message.reasoning_content == "internal"
    assert choice.message.content == "hello"


def test_apply_think_tag_split_noop_when_no_think_tag():
    choice = _build_choice(content="just hello, no tags here")
    LiteLLMCompletionResponsesConfig._apply_think_tag_split_in_place([choice])
    assert getattr(choice.message, "reasoning_content", None) in (None, "")
    assert choice.message.content == "just hello, no tags here"


def test_apply_think_tag_split_noop_when_reasoning_already_set():
    """Native reasoning_content wins; opt-in must not overwrite."""
    choice = _build_choice(content="<think>ignored</think>hello")
    choice.message.reasoning_content = "native reasoning"
    LiteLLMCompletionResponsesConfig._apply_think_tag_split_in_place([choice])
    assert choice.message.reasoning_content == "native reasoning"
    assert choice.message.content == "<think>ignored</think>hello"


def test_apply_think_tag_split_handles_multiple_choices():
    choice_a = _build_choice(content="<think>r1</think>text1")
    choice_b = _build_choice(content="plain text, no tags")
    choice_c = _build_choice(content="<think>r3</think>text3")
    LiteLLMCompletionResponsesConfig._apply_think_tag_split_in_place(
        [choice_a, choice_b, choice_c]
    )
    assert choice_a.message.reasoning_content == "r1"
    assert choice_a.message.content == "text1"
    assert getattr(choice_b.message, "reasoning_content", None) in (None, "")
    assert choice_b.message.content == "plain text, no tags"
    assert choice_c.message.reasoning_content == "r3"
    assert choice_c.message.content == "text3"
