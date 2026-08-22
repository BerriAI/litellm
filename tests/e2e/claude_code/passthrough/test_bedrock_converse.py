"""passthrough x Bedrock (Converse).

Structurally not applicable. In bedrock mode the `claude` CLI speaks
only the InvokeModel wire (`/model/{id}/invoke-with-response-stream`);
it has no Converse-wire client, so there is no Claude Code traffic a
Converse passthrough could serve. LiteLLM's `/bedrock` route does
accept `/model/{id}/converse-stream`, but exercising it would test a
wire no Claude Code user can produce, which is out of scope for this
matrix.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/passthrough/test_bedrock_converse.py
                       ^^^^^^^^^^^     ^^^^^^^^^^^^^^^^
                       feature_id      provider
"""

from __future__ import annotations


def test_passthrough_bedrock_converse(compat_result):
    """Report not_applicable: Claude Code has no Converse-wire mode."""
    compat_result.set(
        {
            "status": "not_applicable",
            "reason": (
                "Claude Code's bedrock mode speaks only the InvokeModel wire "
                "(/model/{id}/invoke-with-response-stream); it has no "
                "Converse-wire client, so there is no Claude Code surface "
                "for Converse passthrough."
            ),
        }
    )
