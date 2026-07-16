"""passthrough x Anthropic.

Drive the real `claude` CLI in its default first-party mode, but with
ANTHROPIC_BASE_URL aimed at the proxy's `/anthropic` passthrough route
instead of the `/v1/messages` translation endpoint. The proxy forwards
the request verbatim to api.anthropic.com, swapping the virtual-key
bearer for its own ANTHROPIC_API_KEY.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/passthrough/test_anthropic.py
                       ^^^^^^^^^^^     ^^^^^^^^^
                       feature_id      provider

Because nothing is translated, the model ids are the real Anthropic API
ids (which happen to equal the proxy aliases for this column). A red
cell here means the passthrough route broke forwarding itself -- auth
header swap, streaming SSE relay, or beta-header propagation -- since
no per-provider transformation is involved.
"""

from __future__ import annotations

from claude_code._passthrough import (
    ANTHROPIC_PASSTHROUGH_BASE_PATH,
    run_passthrough_cell,
)

ANTHROPIC_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]


def test_passthrough_anthropic(compat_result):
    """Drive the `claude` CLI through `{proxy}/anthropic` and assert a reply."""
    run_passthrough_cell(
        compat_result=compat_result,
        models=ANTHROPIC_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
        passthrough_base_path=ANTHROPIC_PASSTHROUGH_BASE_PATH,
    )
