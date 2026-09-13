"""Regression coverage for issue #30822.

`websearch_interception` converts the `tools` array but, before this fix,
left the `tool_choice` field untouched. A Claude Code-style request that
forced the web search tool via
``tool_choice={"type": "tool", "name": "web_search"}`` was sent to the
provider after the tool entry had been renamed to ``litellm_web_search``,
and the provider returned ``Tool 'web_search' not found in provided tools``.

The rewriter is scoped to names we actually renamed in ``tools`` so a
Cowork-style request that points ``tool_choice`` at a client-side
``WebSearch`` tool (which carries an ``input_schema`` and is not
converted) is not hijacked the other way around.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME
from litellm.integrations.websearch_interception.tools import (
    collect_rewritten_tool_names,
    rewrite_web_search_tool_choice,
)


def _rewritable(tool_name: str):
    """Helper: build a ``rewritten_names`` set as if a tool named
    ``tool_name`` had been converted by the handler."""
    return {tool_name}


def test_rewrites_anthropic_style_tool_choice_naming_web_search():
    out = rewrite_web_search_tool_choice(
        {"type": "tool", "name": "web_search"}, _rewritable("web_search")
    )
    assert out == {"type": "tool", "name": LITELLM_WEB_SEARCH_TOOL_NAME}


def test_rewrites_anthropic_style_tool_choice_naming_legacy_WebSearch():
    out = rewrite_web_search_tool_choice(
        {"type": "tool", "name": "WebSearch"}, _rewritable("WebSearch")
    )
    assert out == {"type": "tool", "name": LITELLM_WEB_SEARCH_TOOL_NAME}


def test_rewrites_openai_style_tool_choice_naming_web_search():
    out = rewrite_web_search_tool_choice(
        {"type": "function", "function": {"name": "web_search"}},
        _rewritable("web_search"),
    )
    assert out == {
        "type": "function",
        "function": {"name": LITELLM_WEB_SEARCH_TOOL_NAME},
    }


def test_leaves_unrelated_tool_choice_unchanged():
    original = {"type": "tool", "name": "my_other_tool"}
    assert (
        rewrite_web_search_tool_choice(original, _rewritable("web_search")) == original
    )


def test_leaves_string_modes_unchanged():
    for mode in ("auto", "none", "any", "required"):
        assert (
            rewrite_web_search_tool_choice(mode, _rewritable("web_search")) == mode
        )


def test_leaves_none_unchanged():
    assert rewrite_web_search_tool_choice(None, _rewritable("web_search")) is None


def test_preserves_extra_fields_on_anthropic_style():
    out = rewrite_web_search_tool_choice(
        {"type": "tool", "name": "web_search", "disable_parallel_tool_use": True},
        _rewritable("web_search"),
    )
    assert out == {
        "type": "tool",
        "name": LITELLM_WEB_SEARCH_TOOL_NAME,
        "disable_parallel_tool_use": True,
    }


def test_preserves_extra_fields_on_openai_style():
    out = rewrite_web_search_tool_choice(
        {
            "type": "function",
            "function": {"name": "web_search", "extra": "x"},
            "outer": "y",
        },
        _rewritable("web_search"),
    )
    assert out == {
        "type": "function",
        "function": {"name": LITELLM_WEB_SEARCH_TOOL_NAME, "extra": "x"},
        "outer": "y",
    }


def test_does_not_mutate_input():
    original = {"type": "tool", "name": "web_search"}
    snapshot = dict(original)
    rewrite_web_search_tool_choice(original, _rewritable("web_search"))
    assert original == snapshot


def test_empty_rewritten_names_leaves_choice_unchanged():
    """If no tool was actually renamed, the rewriter is a no-op."""
    original = {"type": "tool", "name": "web_search"}
    assert rewrite_web_search_tool_choice(original, set()) is original


def test_cowork_tool_choice_pointing_at_client_side_WebSearch_not_rewritten():
    """Greptile-flagged guard for #30872: a request that ships its own
    client-side ``WebSearch`` tool (with ``input_schema``) keeps
    ``WebSearch`` in the tools array because ``is_web_search_tool`` skips
    it. The rewriter must follow the same gate and leave a forcing
    ``tool_choice={"name": "WebSearch"}`` alone — otherwise we would
    produce the inverse of the original bug.
    """
    cowork_tools = [
        {
            "name": "WebSearch",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    ]
    rewritten = collect_rewritten_tool_names(cowork_tools)
    assert rewritten == set()

    tool_choice = {"type": "tool", "name": "WebSearch"}
    out = rewrite_web_search_tool_choice(tool_choice, rewritten)
    assert out == tool_choice


def test_collect_rewritten_tool_names_picks_up_converted_tools():
    """Names that ``is_web_search_tool`` matches end up in the returned set
    so the rewriter knows which ``tool_choice`` references to rename.
    Covers Anthropic-native (``web_search_20250305`` → ``web_search``),
    Claude Code (bare ``web_search`` + a type), and legacy bare
    ``WebSearch`` (no schema) — but NOT a Cowork client-side
    ``WebSearch`` carrying an ``input_schema``.
    """
    tools = [
        {"type": "web_search_20250305", "name": "web_search"},
        {"name": "WebSearch"},  # legacy interception marker
        {"name": "calculator"},  # untouched
        {
            "name": "WebSearch",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },  # Cowork client-side tool — not rewritten
    ]
    assert collect_rewritten_tool_names(tools) == {"web_search", "WebSearch"}
