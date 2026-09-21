"""
Validate Claude Sonnet 5 model configuration entries.

Sonnet 5 ships with the gen-5 adaptive-thinking profile (adaptive thinking
always on, no extended thinking, ``effort`` defaults to ``high``), so it must
mirror the sampling-param and prefill restrictions that Fable 5 / Opus 4.8 carry
rather than the older Sonnet 4.6 behavior. The cost-map entries are also what
populate ``litellm.anthropic_models`` at import, which is what lets a bare
``claude-sonnet-5`` name resolve to the ``anthropic`` provider (and match an
``anthropic/*`` wildcard deployment).
"""

import os


from litellm.constants import BEDROCK_CONVERSE_MODELS

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")

ALL_SONNET_5_VARIANTS = (
    "claude-sonnet-5",
    "anthropic.claude-sonnet-5",
    "global.anthropic.claude-sonnet-5",
    "us.anthropic.claude-sonnet-5",
    "eu.anthropic.claude-sonnet-5",
    "au.anthropic.claude-sonnet-5",
    "jp.anthropic.claude-sonnet-5",
    "vertex_ai/claude-sonnet-5",
    "vertex_ai/claude-sonnet-5@default",
    "azure_ai/claude-sonnet-5",
)


def test_sonnet_5_registered_for_bedrock_converse():
    assert "anthropic.claude-sonnet-5" in BEDROCK_CONVERSE_MODELS


