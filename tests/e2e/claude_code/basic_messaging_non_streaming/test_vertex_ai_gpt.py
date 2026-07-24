"""basic_messaging_non_streaming x Vertex AI (GPT-5.6) — not applicable.

GCP is the only one of the big-three clouds without OpenAI's
closed-weight GPT-5.6 family (Sol / Terra / Luna); Vertex AI Model
Garden carries only the open-weight gpt-oss MaaS models. The cell
reports `not_applicable` so the published matrix documents the gap
explicitly instead of leaving a `not_tested` hole.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/basic_messaging_non_streaming/test_vertex_ai_gpt.py
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^
                       feature_id                         provider
"""

from __future__ import annotations

from claude_code._gpt_cells import VERTEX_AI_GPT_NOT_APPLICABLE_REASON


def test_basic_messaging_non_streaming_vertex_ai_gpt(compat_result):
    """Record the static not_applicable outcome for this cell."""
    compat_result.set(
        {
            "status": "not_applicable",
            "reason": VERTEX_AI_GPT_NOT_APPLICABLE_REASON,
        }
    )
