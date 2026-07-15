"""long_context_1m x Bedrock (Invoke).

Drive the real `claude` CLI in headless mode with a ~210k-token padded
prompt and the `--betas context-1m-2025-08-07` beta header, route
through a LiteLLM proxy aimed at Anthropic, and assert the request
round-trips: no 400 from a stripped beta header, no 413 from a body
the proxy refused to forward, and a non-empty assistant reply at the
end.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/long_context_1m/test_bedrock_invoke.py
                       ^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^
                       feature_id           provider

Cost note (read this before scaling the prompt up):

This row genuinely exercises the long-context path -- a 210k-token
prompt is *just over* Claude's standard 200k context window, which is
the threshold that requires the `context-1m-2025-08-07` beta header
to be honored end-to-end. Anything shorter would only test whether
the proxy forwards the beta header byte-for-byte; it would not catch
provider-side regressions where the header is forwarded but the
upstream silently truncates beyond the standard context (we've seen
this on third-party gateways). Anything longer is wasted spend.

Per-cell cost at 210k input tokens: ~$0.63 Sonnet, ~$3.15 Opus.
Daily cost across all five providers (this row only): ~$19.

Haiku 4.5 is intentionally omitted: it does not support 1M context
(its window is 200k). Reporting `not_applicable` for Haiku would
flip the entire cell to `not_applicable`, hiding genuine 1M
regressions on Sonnet/Opus; instead we exclude Haiku from the model
list entirely and let the matrix's per-cell aggregator green the
cell on Sonnet + Opus passing. This is the one row where the "all
three tiers must pass" rule is relaxed; it's relaxed structurally
(via the model list), not semantically (via not_applicable), so the
matrix builder stays unmodified.

The prompt is delivered via subprocess stdin rather than a positional
argument. ARG_MAX on Linux is typically 2MB and an 840KB prompt
fits within that comfortably, but stdin is safer (no shell escaping
surprises, no surprise ARG_MAX clamp on a tightened sandbox) and
keeps the driver's `extra_args` slot free for the `--betas` flag.

`--max-budget-usd 6` is a runaway-loop guard: a misbehaving test
that loops three Claude tiers in a single cell can't accidentally
spend more than ~$18 on this cell. The cap is twice the expected
worst case (Opus @ 210k = $3.15) plus a 50% margin. Tighten it if a
provider's pricing changes and the matrix starts spending more than
$10/day on this row.
"""

from __future__ import annotations

from typing import Sequence

import pytest

from claude_code._env import require_compat_cli_credentials
from claude_code.conftest import _compat_cli_key_provider
from claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)


# Haiku 4.5 is excluded -- only Sonnet 4.6 and Opus 4.7 support the
# 1M-context beta. See module docstring for the per-cell-aggregator
# rationale.
BEDROCK_INVOKE_MODELS: Sequence[str] = (
    "claude-sonnet-4-6-bedrock-invoke",
    "claude-opus-4-7-bedrock-invoke",
)

# Beta header that opts an Anthropic-shape model into the 1M context
# window. Same string is accepted on Bedrock (Invoke + Converse) and
# Vertex per LiteLLM's transformers -- no per-provider translation is
# needed for this header, unlike `advanced-tool-use-2025-11-20` /
# `tool-search-tool-2025-10-19`.
LONG_CONTEXT_BETA = "context-1m-2025-08-07"

# Target a padded prompt that lands just above Claude's standard 200k
# context window so the request can only succeed if the
# `context-1m-2025-08-07` beta header survives all the way to the
# upstream. Below 200k the cell would silently pass even with a
# proxy-dropped beta header; above ~220k we're paying for tokens that
# don't add signal.
TARGET_INPUT_TOKENS = 210_000

# Anthropic's English tokenizer averages ~4 chars/token. We cycle
# through several benign pangrams + filler so the padding looks like a
# real document, not a repeating monolith. Identical-line padding +
# "ignore everything above" trips Opus 4.7's safety filter as a
# suspected prompt-injection attempt -- we hit that during smoke
# testing and the cell flipped red for the wrong reason. Varied prose
# with a natural document-style framing keeps the filter quiet.
_PAD_CHUNKS = (
    "The quick brown fox jumps over the lazy dog. ",
    "She sells seashells by the seashore on Sunday mornings. ",
    "Pack my box with five dozen liquor jugs for the journey. ",
    "How vexingly quick daft zebras jump over fences at dawn. ",
    "Sphinx of black quartz, judge my vow of silence and patience. ",
    "Waltz, bad nymph, for quick jigs in the moonlit meadow. ",
    "Glib jocks quiz nymph to vex dwarf with a riddle of stone. ",
    "Crazy Fredrick bought many very exquisite opal jewels lately. ",
)
_CHARS_PER_TOKEN = 4


def _build_long_prompt(target_tokens: int = TARGET_INPUT_TOKENS) -> str:
    """Build a ~target_tokens-token padded prompt with a trailing instruction.

    Framing:

      - Lead with a benign document-style preamble that justifies the
        long context (so safety filters see the prompt as "long
        document review" rather than "adversarial padding").
      - Cycle through a small set of pangrams + filler sentences for
        the bulk of the padding. Variety matters: identical repeated
        lines look like a denial-of-service or injection attempt to
        Anthropic's content filter on the larger tiers.
      - End with the actual question. Claude's instruction-following
        is stronger on recent tokens, so a 210k-token-into-the-past
        instruction would risk a false-fail where the model ignores
        it.

    `target_tokens` is an approximation: actual token count depends
    on the tokenizer, but Anthropic's English tokenizer averages
    ~4 chars/token, so 4 × target_tokens chars of padding gets us
    close enough to the 1M-beta threshold (200k) that the proxy's
    beta-header handling is the only path to success.
    """
    preamble = (
        "I'm going to share an excerpt from a long document with you. "
        "It contains a mix of practice sentences a typist might use to "
        "warm up; treat the bulk of the text as background context. "
        "I'll ask a short question at the end.\n\n"
        "Begin excerpt:\n\n"
    )
    closing = "\n\nEnd of excerpt. Please reply with the single word 'ok'."

    pad_target_chars = target_tokens * _CHARS_PER_TOKEN - len(preamble) - len(closing)
    pad_lines = []
    pad_len = 0
    idx = 0
    while pad_len < pad_target_chars:
        chunk = _PAD_CHUNKS[idx % len(_PAD_CHUNKS)]
        pad_lines.append(chunk)
        pad_len += len(chunk)
        idx += 1
    return preamble + "".join(pad_lines) + closing


def test_long_context_1m_bedrock_invoke(compat_result):
    """Drive the `claude` CLI (Bedrock (Invoke)) with a ~210k-token prompt and the
    `context-1m-2025-08-07` beta header; assert no 400 / 413 and a
    non-empty reply for Sonnet + Opus."""
    base_url, api_key = require_compat_cli_credentials(
        compat_result, cli_key_provider=_compat_cli_key_provider
    )

    long_prompt = _build_long_prompt()

    outcomes = run_claude_models_parallel(
        models=BEDROCK_INVOKE_MODELS,
        prompt=None,
        stdin_input=long_prompt,
        base_url=base_url,
        api_key=api_key,
        extra_args=[
            "--betas",
            LONG_CONTEXT_BETA,
            # Hard ceiling so a runaway test cannot blow the budget.
            # See module docstring for sizing.
            "--max-budget-usd",
            "6",
        ],
        # Long-context requests can take a couple of minutes on a
        # loaded upstream; the driver's default 120s is too tight.
        timeout=300.0,
    )

    failures = []
    for model in BEDROCK_INVOKE_MODELS:
        outcome = outcomes[model]
        if isinstance(outcome, ClaudeCLIError):
            error = f"[{model}] {outcome}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if outcome.exit_code != 0:
            error = f"[{model}] claude CLI failed: {failure_diagnostic(outcome)}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if not outcome.text.strip():
            error = f"[{model}] claude returned empty assistant text"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
