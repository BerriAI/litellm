"""tool_use x Vertex AI (GPT-5.6) — not applicable.

GCP is the only one of the big-three clouds without OpenAI's
closed-weight GPT-5.6 family (Sol / Terra / Luna); Vertex AI Model
Garden carries only the open-weight gpt-oss MaaS models. The cell
reports `not_applicable` so the published matrix documents the gap
explicitly instead of leaving a `not_tested` hole.

This stub never drives the `claude` CLI, so it grants no tools and is
exempt from the Bash allow-rule pin enforced by
`_pr_gate_unit_tests/test_bash_tool_restrictions.py`.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/tool_use/test_vertex_ai_gpt.py
                       ^^^^^^^^      ^^^^^^^^^^^^^
                       feature_id    provider
"""

from __future__ import annotations

from claude_code._gpt_cells import VERTEX_AI_GPT_NOT_APPLICABLE_REASON


def test_tool_use_vertex_ai_gpt(compat_result):
    """Record the static not_applicable outcome for this cell."""
    compat_result.set(
        {
            "status": "not_applicable",
            "reason": VERTEX_AI_GPT_NOT_APPLICABLE_REASON,
        }
    )
