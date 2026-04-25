"""prompt_caching_5m x Azure.

Azure OpenAI Service does not host Anthropic Claude models, so the
`prompt_caching_5m` × Azure cell is structurally `not_applicable` —
there is no route for the `claude` CLI to talk to Claude through Azure
via LiteLLM.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/prompt_caching_5m/test_azure.py
                       ^^^^^^^^^^^^^^^^^      ^^^^^
                       feature_id             provider
"""

from __future__ import annotations

import pytest

AZURE_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

NOT_APPLICABLE_REASON = (
    "Azure OpenAI Service does not host Anthropic Claude models. "
    "Route Claude requests through Anthropic, AWS Bedrock, or GCP Vertex AI."
)


@pytest.mark.parametrize("model", AZURE_MODELS)
def test_prompt_caching_5m_azure(compat_result, model):
    """Report `not_applicable` for every (model, Azure) combination."""
    compat_result.set({"status": "not_applicable", "reason": NOT_APPLICABLE_REASON})
