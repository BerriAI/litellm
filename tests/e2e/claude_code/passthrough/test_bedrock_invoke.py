"""passthrough x Bedrock (Invoke).

Drive the real `claude` CLI in bedrock mode (CLAUDE_CODE_USE_BEDROCK=1)
with ANTHROPIC_BEDROCK_BASE_URL aimed at the proxy's `/bedrock`
passthrough route. The CLI speaks the native InvokeModel wire --
`POST /model/{model}/invoke-with-response-stream` -- with the proxy
alias in the model segment; the proxy resolves the alias through its
router, rewrites the path to the deployment's upstream model id, and
SigV4-signs the forwarded request with its own AWS credentials
(CLAUDE_CODE_SKIP_BEDROCK_AUTH=1 keeps the CLI from signing).

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/passthrough/test_bedrock_invoke.py
                       ^^^^^^^^^^^     ^^^^^^^^^^^^^^
                       feature_id      provider

The CLI also fires a best-effort `GET /bedrock/inference-profiles`
listing at startup; its failure is non-fatal and does not gate this
cell.
"""

from __future__ import annotations

from claude_code._passthrough import bedrock_extra_env, run_passthrough_cell

BEDROCK_INVOKE_MODELS = [
    "claude-haiku-4-5-bedrock-invoke",
    "claude-sonnet-4-5-bedrock-invoke",
    "claude-opus-4-7-bedrock-invoke",
]


def test_passthrough_bedrock_invoke(compat_result):
    """Drive the `claude` CLI through `{proxy}/bedrock` and assert a reply."""
    run_passthrough_cell(
        compat_result=compat_result,
        models=BEDROCK_INVOKE_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
        build_extra_env=bedrock_extra_env,
    )
