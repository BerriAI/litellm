---
name: oss-issue-triage
description: Triage open-source GitHub issues on BerriAI/litellm. Use when asked to triage, reproduce, dedupe, close, or backlog OSS issues. Covers checking whether an issue is already fixed by a recent PR (then close and link it), reproducing issues end-to-end on the SDK or the proxy with secret-safe screenshots posted to the issue, and tracking good feature requests or fixes in the backlog with a rationale for the maintainer.
---

# OSS issue triage (BerriAI/litellm)

Procedure for triaging issues on the public https://github.com/BerriAI/litellm tracker. Work **oldest first** unless the user names a specific issue or filter.

The tracker is public. Everything you post (comments, screenshots, logs) is world-readable. Never leak secrets. See "Secret safety" below and run its checklist before every post.

## When to use

- "triage the oldest litellm issues"
- "look at issue #NNNNN and reproduce it"
- "check if this issue is already fixed"
- "go through the OSS backlog"

## Prerequisites

- `gh` is authenticated for BerriAI/litellm. Confirm with `gh auth status`.
- Repo is set up. On a fresh clone run `make bootstrap` first (see CLAUDE.md).
- Provider API keys live in the session env (e.g. `ANTHROPIC_API_KEY`, `AWS_BEDROCK_TEST_*`, `AZURE_*`). Use `list_secrets` to see what is available. Reproductions should hit real provider APIs, not mocks, so the repro matches what the user sees.
- Proxy dev server (only for proxy issues):
  ```bash
  python litellm/proxy/proxy_cli.py --config litellm/proxy/dev_config.yaml --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log
  ```
  It serves on http://localhost:4000. Admin UI dev server: `npm run dev` in `ui/litellm-dashboard` (port 3000).

## Step 0: pick the next issue (oldest first)

```bash
gh issue list --repo BerriAI/litellm --state open --limit 20 \
  --search "sort:created-asc" \
  --json number,title,createdAt,labels
```

Skip issues already labeled `awaiting: user response` unless the user asks otherwise (they are blocked on the reporter). Read the full issue and its comments before acting:

```bash
gh issue view <number> --repo BerriAI/litellm --comments
```

Classify the issue: is it an **SDK** issue (someone calling `litellm.completion(...)` in Python) or a **proxy** issue (someone hitting the proxy server / Admin UI / virtual keys)? This decides how you reproduce it.

## Rule 1: check if already fixed by a recent PR

Before reproducing, determine whether a merged PR already resolves it.

- Search merged PRs that mention the issue or the symptom:
  ```bash
  gh pr list --repo BerriAI/litellm --state merged --search "<keywords or #number>" \
    --json number,title,mergedAt,url --limit 20
  gh search prs --repo BerriAI/litellm "<symptom keywords>" --merged --limit 20
  ```
- Search the code/history for the fix:
  ```bash
  git log --oneline --since="6 months ago" -S "<symbol or error string>" -- <likely path>
  ```
  Use the `grep` tool over `litellm/` to see whether the buggy code path still exists.
- If it is genuinely already fixed on the latest code, confirm by reproducing on the current branch and seeing it pass (do a quick repro per Rule 2). Then comment linking the PR and close:
  ```bash
  gh issue comment <number> --repo BerriAI/litellm \
    --body "This looks resolved by #<PR> (merged <date>). Verified on the latest main: <one line of how>. Closing; please reopen if you still hit it on the latest release."
  gh issue close <number> --repo BerriAI/litellm --reason completed
  ```
  Keep the comment factual and short. If unsure whether the PR fully covers the report, do not close; note the candidate PR and continue to Rule 2.

## Rule 2: reproduce end-to-end and attach secret-safe screenshots

Reproduce the exact scenario the reporter described. Match their model, provider, and call shape as closely as possible.

### SDK issue

Write a minimal Python repro that mirrors the report and run it against a real provider:

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True python /tmp/repro_<number>.py
```

Use the latest model in the relevant family (check `model_prices_and_context_window.json`); treat training knowledge as stale. Capture the terminal output. Do not commit the repro script.

### Proxy issue

Start the proxy (command above), then reproduce with `curl` against http://localhost:4000 so the proof is exactly what a user would run:

```bash
curl -sS http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{"model":"<model>","messages":[{"role":"user","content":"..."}]}'
```

For UI issues, drive the Admin UI at http://localhost:4000/ui/ (or the dev server on port 3000) with the computer tool and screenshot the relevant screen.

### Post the result to the issue

Capture screenshots (terminal for SDK/curl, browser for UI). Then, if reproduced:

```bash
gh issue comment <number> --repo BerriAI/litellm --body "Reproduced on <branch/commit>. Steps: ...
<paste sanitized command + output>"
```

Attach screenshots by dragging into a comment in the GitHub web UI, or upload and reference the returned URL. Label it as reproduced so the DRI can pick it up:

```bash
gh issue edit <number> --repo BerriAI/litellm --add-label bug
```

If you cannot reproduce, comment with exactly what you tried (env, model, versions, commands) and ask the reporter for the missing detail, then add `awaiting: user response`.

## Rule 3: backlog good feature requests and fixes

For issues that are legitimate feature requests or worthwhile fixes (not already fixed, not invalid), do not close them. Track them in the backlog and tell the maintainer why they are worth doing.

- Create a backlog item with the `linear` tool (`command="list_tools"` first to find `create_issue`, then `create_tool`). Title it clearly, link the GitHub issue in the description, and put it in the team's backlog. Confirm the correct team/project with the user if ambiguous.
- Message Ishaan (non-blocking) with a short rationale per issue: what it is, why it is worth fixing (impact, how many users, whether it is a quick win or unblocks a common workflow), and a link to both the GitHub issue and the Linear item.
- Route to the area DRI when relevant (see the DRI router knowledge note): UI to Ryan, LLM to Mateo, Security to Yuneng, MCP to Tin, Logging/Guardrails/Cloud/perf to Yassin, Rust to Ishaan. Ping via their Slack ID so it renders as a real mention.

## Secret safety (run before every public post)

The tracker is public. Before posting any comment, screenshot, or log:

- Scrub API keys, bearer tokens, virtual keys (`sk-...`), cloud credentials (AWS keys, Azure keys/endpoints), org identifiers, and internal URLs. Replace with placeholders like `sk-1234`, `<REDACTED>`.
- Redact `Authorization`, `api_key`, `x-api-key`, `aws_secret_access_key`, and any `os.environ/...` resolved values from `--detailed_debug` proxy logs; these logs are verbose and often echo credentials.
- Crop screenshots to hide terminal scrollback, env panes, `.env` contents, and browser tabs/history.
- Never reference internal customer or company names (CLAUDE.md rule). Use "a customer" for anything not a publicly known provider.
- Quick scan of anything you are about to paste:
  ```bash
  grep -Ei 'sk-[a-z0-9]{8,}|api[_-]?key|secret|bearer|aws_(access|secret)|AKIA[0-9A-Z]{16}' <file>
  ```
  If it matches, sanitize before posting.

## Public comment style (CLAUDE.md)

Comments are public-facing, so follow the house style: no emojis; no em dash (use `;` or `.`); avoid the "it's not X, it's Y" pattern; prefer prose over lists unless a list is clearly better; no trailing period on standalone paragraphs; use `->` not an arrow. Keep it factual and short.

## Loop

Move to the next-oldest issue and repeat. Keep a simple running checklist (e.g. `checklist.txt`) of issues seen, decision (closed / reproduced / backlogged / awaiting user), and links, so nothing is dropped across a long triage run.
