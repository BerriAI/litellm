#!/usr/bin/env bash
# Regenerate support/exports/customer-support-bundle.md from the source rule
# and skill files. Run after editing either source.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RULE_FILE="$REPO_ROOT/.cursor/rules/customer-support.mdc"
SKILL_FILE="$REPO_ROOT/.cursor/skills/draft-support-reply/SKILL.md"
OUT_DIR="$REPO_ROOT/support/exports"
OUT_FILE="$OUT_DIR/customer-support-bundle.md"

mkdir -p "$OUT_DIR"

for f in "$RULE_FILE" "$SKILL_FILE"; do
  if [[ ! -f "$f" ]]; then
    echo "error: missing source file $f" >&2
    exit 1
  fi
done

# Strip optional YAML frontmatter (lines between two `---` markers at the top).
strip_frontmatter() {
  awk '
    BEGIN { in_fm = 0; past_fm = 0 }
    NR == 1 && /^---$/ { in_fm = 1; next }
    in_fm && /^---$/ { in_fm = 0; past_fm = 1; next }
    in_fm { next }
    { print }
  ' "$1"
}

# Pull the description field from frontmatter (first match wins).
get_description() {
  awk -F': *' '/^description:/ { sub(/^description: */, ""); print; exit }' "$1"
}

RULE_DESC="$(get_description "$RULE_FILE")"
SKILL_DESC="$(awk -F': *' '/^description:/ { sub(/^description: */, ""); print; exit }' "$SKILL_FILE")"

RULE_BODY="$(strip_frontmatter "$RULE_FILE")"
SKILL_BODY="$(strip_frontmatter "$SKILL_FILE")"

GENERATED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

cat > "$OUT_FILE" <<EOF
# LiteLLM Customer Support Drafting — Rule + Skill (Bundle)

A single shareable document describing how the LiteLLM customer support drafting agent answers questions. It bundles two artifacts that normally live separately in [BerriAI/litellm](https://github.com/BerriAI/litellm):

- **The rule** — voice, tone, structure, what never goes in a draft. Source: \`.cursor/rules/customer-support.mdc\`
- **The skill** — workflow (classify, ground in docs, confirm in code, draft, output two sections). Source: \`.cursor/skills/draft-support-reply/SKILL.md\`

Audience: paying LiteLLM Enterprise customers and prospective enterprise customers. Default product assumption is **LiteLLM Enterprise** (proxy / LLM Gateway) on a recent stable version. Address OSS vs Enterprise only when the customer asks.

> **Generated:** ${GENERATED_AT} by \`scripts/export_support_bundle.sh\` — re-run after editing either source file to keep this bundle in sync.

## How to use this bundle

Three ways colleagues can apply it:

1. **In Cursor.** Drop the rule into \`.cursor/rules/customer-support.mdc\` and the skill into \`.cursor/skills/draft-support-reply/SKILL.md\` in any repo. Cursor auto-loads them when the chat scope or task description matches.
2. **As a system prompt elsewhere** (Claude, OpenAI, internal tools). Concatenate the rule and the skill below into one system prompt. Pass the customer question as the user message. The model will produce the same two-section output.
3. **As a writing reference.** Even without an LLM, the rule is a short style guide for a human drafting a reply.

The rule defines **voice**; the skill defines **workflow**. Edit one without touching the other.

## What "good" looks like

Every draft, whether produced by a human or an LLM following this bundle, ends in two clearly separated sections:

\`\`\`
=== CUSTOMER REPLY ===
<the reply, ready to copy-paste into the support channel>

=== INTERNAL NOTES ===
- Classification: <one of: how-to | config | error-triage | feature-availability | billing-or-licensing | oss-vs-enterprise | unclear>
- Sources checked
- Confidence: high | medium | low (one-line reason)
- Open questions for reviewer
- Suggested follow-ups (CSM ping, bug filing, doc gap)
\`\`\`

A human reviewer always edits and sends. Treat outputs as drafts, never sends.

---

## 1. The rule — voice and structure

> ${RULE_DESC}

${RULE_BODY}

---

## 2. The skill — drafting workflow

> ${SKILL_DESC}

${SKILL_BODY}

---

## License and provenance

This bundle is generated from the LiteLLM repository ([BerriAI/litellm](https://github.com/BerriAI/litellm)) and inherits its license. Source files live at \`.cursor/rules/customer-support.mdc\` and \`.cursor/skills/draft-support-reply/SKILL.md\`. Update those, then run \`./scripts/export_support_bundle.sh\` to regenerate this file.
EOF

echo "Wrote $OUT_FILE ($(wc -l < "$OUT_FILE") lines)"
