"""Shared plumbing for the GPT-5.6 (Sol / Terra / Luna) provider columns.

OpenAI shipped GPT-5.6 as a three-tier family on 2026-07-09 — Sol
(flagship), Terra (balanced), Luna (fast) — and Claude Code can drive
all three through a LiteLLM proxy that translates the Anthropic
Messages API to each provider's native shape. Four provider columns
cover "OpenAI plus the big three clouds":

    openai          OpenAI API             (openai/gpt-5.6-*)
    azure_openai    Azure OpenAI           (azure/gpt-5.6-*)
    bedrock_mantle  AWS Bedrock, Mantle    (bedrock_mantle/openai.gpt-5.6-*,
                                            Responses API)
    vertex_ai_gpt   GCP Vertex AI          not_applicable — Vertex does
                                           not offer the closed-weight
                                           GPT-5.6 family; Model Garden
                                           carries only the open-weight
                                           gpt-oss MaaS models

The openai and azure_openai columns run unconditionally, like every
other live column: the environments that run the suite carry
`OPENAI_API_KEY` and `AZURE_API_BASE` + `AZURE_API_KEY` pointing at a
resource with gpt-5.6 deployments. The bedrock_mantle column is
opt-in via `COMPAT_MANTLE_CELLS=1` because the AWS account is still
waiting on the Bedrock Mantle allowlist for the `openai.gpt-5.6-*`
models; until the flag is set each Mantle cell skips and its matrix
cell publishes as `not_tested` instead of a credential-shaped red.
The `vertex_ai_gpt` column needs no flag either way: its cells report
a static `not_applicable` and never touch the network.
"""

from __future__ import annotations

import os

import pytest

MANTLE_CELLS_ENV = "COMPAT_MANTLE_CELLS"

VERTEX_AI_GPT_NOT_APPLICABLE_REASON = (
    "GCP Vertex AI does not offer OpenAI's closed-weight GPT-5.6 family "
    "(Sol / Terra / Luna); Model Garden carries only the open-weight "
    "gpt-oss MaaS models. Convert this column's cells to live tests if "
    "Google adds the GPT-5.6 models."
)


def skip_unless_mantle_cells_enabled() -> None:
    """Skip the calling test unless `COMPAT_MANTLE_CELLS` opts the
    Bedrock Mantle cells in.

    A skipped cell is recorded as `not_tested` in the published matrix
    (see the skip handling in `tests/e2e/claude_code/conftest.py`),
    which is the honest state while the AWS account has no Mantle
    access to the GPT-5.6 models yet.
    """
    if os.environ.get(MANTLE_CELLS_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return
    pytest.skip(
        f"Bedrock Mantle GPT-5.6 cells are opt-in; set {MANTLE_CELLS_ENV}=1 "
        "once the AWS account is allowlisted for the openai.gpt-5.6-* models"
    )
