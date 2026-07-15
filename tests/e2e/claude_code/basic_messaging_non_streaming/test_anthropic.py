"""basic_messaging_non_streaming × Anthropic.

The thinnest end-to-end path through every layer of the matrix: drive the
real `claude` CLI in headless mode against a running LiteLLM proxy that
routes to Anthropic, and report the outcome via `compat_result`.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/basic_messaging_non_streaming/test_anthropic.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^
                       feature_id                         provider

Per the PRD, every cell exercises Claude Haiku 4.5, Sonnet 4.6, and Opus
4.7; the cell only goes green if all three pass. The shared
`run_basic_messaging_cell` helper fans the three model runs out in
parallel and reports one `compat_result.add(...)` entry per model so
the matrix builder still sees three rows for this (feature, provider).
"""

from __future__ import annotations

from claude_code._basic_messaging import run_basic_messaging_cell

# Per the PRD: each cell is exercised against three Claude tiers via the
# Anthropic provider. Aliases are configured in the LiteLLM proxy's
# routing config; the driver only sends the alias.
ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-7",
]


def test_basic_messaging_non_streaming_anthropic(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a reply.

    "Basic messaging" means: send a single user prompt, receive any
    non-empty assistant text reply, no tools, no streaming, no thinking.
    The whole point of this slice is to prove the path works at all —
    so the assertion is intentionally lenient on the reply contents.
    """
    run_basic_messaging_cell(
        compat_result=compat_result,
        models=ANTHROPIC_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
    )
