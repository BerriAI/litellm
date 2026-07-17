"""tool_use x Azure OpenAI (GPT-5.6).

Drive the real `claude` CLI against a running LiteLLM proxy that
routes Anthropic Messages requests to Azure OpenAI deployments of the
GPT-5.6 family (Sol, Terra, Luna), ask the model to invoke a built-in
tool (`Bash`), and assert that a `tool_use` content block came back
over the wire.

Azure OpenAI serves the same function-calling wire shape as
openai.com behind per-resource deployments; LiteLLM's `azure/gpt-*`
route reuses the OpenAI tool translation on top of Azure's deployment
addressing and auth.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `claude_code/_tool_use.py` for the
security rationale.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use/test_azure_openai.py
                       ^^^^^^^^      ^^^^^^^^^^^^
                       feature_id    provider
"""

from __future__ import annotations

from claude_code._tool_use import run_tool_use_cell

AZURE_OPENAI_MODELS = [
    "gpt-5-6-sol-azure-openai",
    "gpt-5-6-terra-azure-openai",
    "gpt-5-6-luna-azure-openai",
]


def test_tool_use_azure_openai(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    tool call was emitted on the wire by each GPT-5.6 tier."""
    run_tool_use_cell(compat_result=compat_result, models=AZURE_OPENAI_MODELS)
