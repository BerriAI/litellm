"""passthrough x Vertex AI.

Drive the real `claude` CLI in vertex mode (CLAUDE_CODE_USE_VERTEX=1)
with ANTHROPIC_VERTEX_BASE_URL aimed at the proxy's `/vertex_ai`
passthrough route. The CLI speaks the native rawPredict wire --
`POST .../projects/{p}/locations/{l}/publishers/anthropic/models/{model}:streamRawPredict`
-- with the proxy alias in the model segment; the proxy resolves the
alias through its router, replaces the placeholder project/location
path segments with the deployment's `vertex_project` /
`vertex_location`, and attaches its own Google credentials
(CLAUDE_CODE_SKIP_VERTEX_AUTH=1 keeps the CLI from minting a token).

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/passthrough/test_vertex_ai.py
                       ^^^^^^^^^^^     ^^^^^^^^^
                       feature_id      provider

This cell requires the vertex deployments in the proxy config to carry
`use_in_pass_through: true` (see test_config.yaml) -- that is what
registers their credentials with the passthrough router. Without it
the proxy forwards the CLI's own headers (the virtual-key bearer) to
Google and every tier fails with a 401.
"""

from __future__ import annotations

from claude_code._passthrough import run_passthrough_cell, vertex_extra_env

VERTEX_MODELS = [
    "claude-haiku-4-5-vertex",
    "claude-sonnet-4-6-vertex",
    "claude-opus-4-7-vertex",
]


def test_passthrough_vertex_ai(compat_result):
    """Drive the `claude` CLI through `{proxy}/vertex_ai` and assert a reply."""
    run_passthrough_cell(
        compat_result=compat_result,
        models=VERTEX_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
        build_extra_env=vertex_extra_env,
    )
