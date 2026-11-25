"""basic_messaging_non_streaming x Azure OpenAI (GPT-6).

Drive the real `claude` CLI in headless mode against a running LiteLLM
proxy that routes Anthropic Messages requests to Azure OpenAI
deployments of the GPT-6 family (Sol, Terra, Luna, Astra), and report the
outcome via `compat_result`.

Azure OpenAI serves the same chat-completions wire shape as
openai.com behind per-resource deployments; LiteLLM's `azure/gpt-*`
route handles the deployment addressing while reusing the OpenAI
translation, so this cell catches Azure-specific regressions
(auth headers, api-version pinning, deployment routing) that the
`openai` column cannot.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/basic_messaging_non_streaming/test_azure_openai.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^
                       feature_id                         provider

Every GPT cell exercises the four GPT-6 models; the cell only goes
green if all four pass.
"""

from __future__ import annotations

from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from claude_code._basic_messaging import run_basic_messaging_cell

AZURE_OPENAI_MODELS = [
    "gpt-6-1-sol-azure-openai",
    "gpt-5-6-terra-azure-openai",
    "gpt-6-luna-azure-openai",
    "gpt-6-astra-azure-openai",
]


@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.MESSAGES,
        providers=(Provider.AZURE,),
        models=tuple(AZURE_OPENAI_MODELS),
        mode=Mode.NONSTREAM,
    )
)
def test_basic_messaging_non_streaming_azure_openai(compat_result):
    """Drive the `claude` CLI against the LiteLLM proxy and assert a
    non-empty reply from each GPT-6 model."""
    run_basic_messaging_cell(
        compat_result=compat_result,
        models=AZURE_OPENAI_MODELS,
        prompt="Reply with the single word 'pong' and nothing else.",
    )
