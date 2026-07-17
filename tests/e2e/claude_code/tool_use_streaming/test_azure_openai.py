"""tool_use_streaming x Azure OpenAI (GPT-5.6).

Drive the real `claude` CLI in headless `--output-format stream-json`
mode against a running LiteLLM proxy that routes Anthropic Messages
requests to Azure OpenAI deployments of the GPT-5.6 family (Sol,
Terra, Luna), ask the model to invoke a built-in tool (`Bash`), and
assert that the upstream (a) emitted a `tool_use` content block and
(b) streamed the tool input incrementally as `input_json_delta`
events.

Azure OpenAI streams tool arguments in the same chat-completions
fragment shape as openai.com; LiteLLM must re-emit them as Anthropic
`input_json_delta` deltas rather than buffering the full input into
one complete block.

Bash is restricted to the exact command `echo pong` plus
`--permission-mode dontAsk`; see `claude_code/_tool_use.py` for the
security rationale.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use_streaming/test_azure_openai.py
                       ^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^
                       feature_id              provider
"""

from __future__ import annotations

from claude_code._tool_use import run_tool_use_cell

AZURE_OPENAI_MODELS = [
    "gpt-5-6-sol-azure-openai",
    "gpt-5-6-terra-azure-openai",
    "gpt-5-6-luna-azure-openai",
]


def test_tool_use_streaming_azure_openai(compat_result):
    run_tool_use_cell(
        compat_result=compat_result,
        models=AZURE_OPENAI_MODELS,
        verify_streaming=True,
    )
