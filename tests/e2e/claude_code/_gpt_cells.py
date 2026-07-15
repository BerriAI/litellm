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

Live GPT cells are opt-in via `COMPAT_GPT_CELLS=1`. The cron VM that
runs the scheduled suite and publishes the matrix must be provisioned
with the GPT-route credentials (`OPENAI_API_KEY` with available
quota, `AZURE_OPENAI_API_BASE` + `AZURE_OPENAI_API_KEY` with gpt-5.6
deployments, and Bedrock Mantle model access) before these cells can
pass, so until the flag is set each live cell skips and its matrix
cell publishes as `not_tested` instead of a credential-shaped red.
The `vertex_ai_gpt` column ignores the flag: its cells report a
static `not_applicable` and never touch the network.
"""

from __future__ import annotations

import os

import pytest

GPT_CELLS_ENV = "COMPAT_GPT_CELLS"

VERTEX_AI_GPT_NOT_APPLICABLE_REASON = (
    "GCP Vertex AI does not offer OpenAI's closed-weight GPT-5.6 family "
    "(Sol / Terra / Luna); Model Garden carries only the open-weight "
    "gpt-oss MaaS models. Convert this column's cells to live tests if "
    "Google adds the GPT-5.6 models."
)


def skip_unless_gpt_cells_enabled() -> None:
    """Skip the calling test unless `COMPAT_GPT_CELLS` opts GPT cells in.

    A skipped cell is recorded as `not_tested` in the published matrix
    (see the skip handling in `tests/e2e/claude_code/conftest.py`),
    which is the honest state for an environment that has no GPT-route
    credentials yet.
    """
    if os.environ.get(GPT_CELLS_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return
    pytest.skip(
        f"GPT-5.6 cells are opt-in; set {GPT_CELLS_ENV}=1 once the proxy has "
        "OpenAI / Azure OpenAI / Bedrock Mantle credentials for the "
        "gpt-5-6-* aliases"
    )
