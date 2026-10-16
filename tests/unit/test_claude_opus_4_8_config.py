"""
Validate Claude Opus 4.8 model configuration entries.

Regression coverage for the wildcard-routing failure where a bare model name
(``claude-opus-4-8``) could not match an ``anthropic/*`` deployment because
LiteLLM could not infer its provider — the model was simply missing from the
model cost map, so ``get_llm_provider`` raised and the router returned
"no healthy deployments for this model". The fix is the cost-map entries added
for Anthropic, Bedrock, Vertex AI, and Azure AI; those entries are what populate
``litellm.anthropic_models`` at import time, which is what the bare-name lookup
in ``get_llm_provider`` consumes.
"""

import os


from litellm.constants import BEDROCK_CONVERSE_MODELS

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")


def test_opus_4_8_registered_for_bedrock_converse():
    assert "anthropic.claude-opus-4-8" in BEDROCK_CONVERSE_MODELS


