"""basic_messaging_streaming x Azure.

Azure (Azure OpenAI Service) does not host Anthropic Claude models —
the platform's first-party catalog is OpenAI models, plus a smaller set
of Microsoft and partner models. There is no supported route for the
`claude` CLI to talk to Claude through Azure via LiteLLM, so every
(model, Azure) combination for `basic_messaging_streaming` reports
`not_applicable` rather than `fail`.

The (feature, provider) for this cell is inferred from the file path by
`tests/claude_code/conftest.py`:

    tests/claude_code/basic_messaging_streaming/test_azure.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^
                       feature_id                     provider
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
def test_basic_messaging_streaming_azure(compat_result, model):
    """Report `not_applicable` for every (model, Azure) combination."""
    compat_result.set({"status": "not_applicable", "reason": NOT_APPLICABLE_REASON})
