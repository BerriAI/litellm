---
title: Claude Code Compatibility
sidebar_label: Claude Code Compatibility
---

import ClaudeCodeCompatibilityTable from '@site/src/components/ClaudeCodeCompatibilityTable';

# Claude Code × LiteLLM compatibility matrix

This table is regenerated daily by an automated populator that runs the
[Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) against
the latest stable LiteLLM proxy across each supported provider, with
Haiku 4.5, Sonnet 4.6, and Opus 4.7 in parallel. A cell goes green only
if all three model tiers pass.

<ClaudeCodeCompatibilityTable />

## Legend

| Glyph | Meaning |
| --- | --- |
| ✅ | All three model tiers pass for this `(feature, provider)` cell. |
| ❌ | At least one model tier failed. Hover for the upstream error. |
| — | No test ran for this combination. |
| n/a | Not applicable (e.g. provider doesn't expose this feature). Hover for the reason. |

## Known issues

Red cells with a known root cause and a tracked fix are listed below. Each
entry stays here until the named fix has landed in a `v*-stable` release;
the next daily run after that tag is cut will flip the cells green and
the entry will be removed.

### Opus 4.7 extended thinking on Bedrock Invoke + Vertex AI

- **Affected cells**: `extended_thinking × bedrock_invoke`, `extended_thinking × vertex_ai`. Anthropic-native and Azure Foundry are unaffected on the same tier.
- **Symptom**: Claude Code's `--effort max` flag is sent to the proxy as `output_config.effort=max`. The Bedrock Invoke and Vertex AI request transformers in `v1.83.14-stable` strip `output_config.effort` for Claude 4.6+ models that aren't on a small hardcoded allow-list, so the upstream request goes out without extended thinking enabled. The response has no `thinking` content block and the cell is marked failed.
- **Status**: Fixed on `main` by [commit `a6c673e7b9`](https://github.com/BerriAI/litellm/commit/a6c673e7b9) (`fix(anthropic,bedrock,vertex): forward output_config.effort + 400 on garbage reasoning_effort`). Waiting on the next `v*-stable` cut.

### Bedrock Converse — Haiku 4.5 content-block validation

- **Affected cells**: every `* × bedrock_converse` cell (the entire Converse column).
- **Symptom**: Claude Haiku 4.5 routed through AWS Bedrock's Converse API returns `Content block is not a text block` on the first assistant message of every conversation. Because the matrix only marks a cell green when all three model tiers pass, this Haiku-only failure paints the whole Converse column red even for features that work on Sonnet 4.6 and Opus 4.7 through Converse.
- **Workaround**: Route Haiku traffic through Bedrock Invoke (column to the left), which is green for the same feature set. Sonnet 4.6 and Opus 4.7 can continue to use Converse for those features.
- **Status**: Under investigation in LiteLLM. Issue link pending.

## Source

The matrix JSON lives at
[`src/data/compatibility-matrix.json`](https://github.com/BerriAI/litellm-docs/blob/main/src/data/compatibility-matrix.json).
The populator is in
[`tests/claude_code/cron_vm/`](https://github.com/BerriAI/litellm/tree/main/tests/claude_code/cron_vm)
on the main repo.
