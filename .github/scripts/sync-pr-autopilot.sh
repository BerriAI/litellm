#!/usr/bin/env bash
#
# Sync PR Autopilot — verify & auto-merge the daily "chore: sync upstream" PR.
#
# Self-contained: driven entirely by env vars so the same script runs verbatim
# in every gateway fork (bifrost / litellm / new-api). It never resolves
# conflicts or edits code — CI-green is the safety gate. Conflicts or
# persistently-red CI are *held* and reported for the next sync session.
#
# Flow:
#   find open sync PR -> conflicts? hold
#                     -> checks pending/unknown? wait (a later event re-runs us)
#                     -> checks failed? re-run failed runs up to MAX_RERUNS, else hold
#                     -> all green? mark ready + merge (fires dispatch-infra-build)
#
# Required env:
#   GH_TOKEN        token for gh (PAT in SYNC_AUTOMERGE_TOKEN preferred so the
#                   post-merge pull_request:closed event fires dispatch-infra-build)
#   BASE_BRANCH     base branch the sync PR targets (e.g. main / litellm_internal_staging)
# Optional env:
#   HEAD_PREFIX     sync branch prefix (default: chore/sync-upstream-)
#   MERGE_METHOD    merge|squash|rebase (default: merge — preserves upstream history)
#   MAX_RERUNS      flaky-CI re-run cap (default: 2)
#   SLACK_WEBHOOK_URL   incoming webhook for summaries (optional)
#   DRY_RUN         "true" to evaluate without merging/re-running (default: false)
#
set -euo pipefail

HEAD_PREFIX="${HEAD_PREFIX:-chore/sync-upstream-}"
MERGE_METHOD="${MERGE_METHOD:-merge}"
MAX_RERUNS="${MAX_RERUNS:-2}"
DRY_RUN="${DRY_RUN:-false}"
HAS_AUTOMERGE_TOKEN="${HAS_AUTOMERGE_TOKEN:-true}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
REPO="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${BASE_BRANCH:?BASE_BRANCH is required}"
: "${GH_TOKEN:?GH_TOKEN is required}"

log()  { echo "::notice::$*"; }
warn() { echo "::warning::$*"; }
summary() { printf '%s\n' "$*" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"; }

# Persist outcome for later workflow steps (e.g. optional email).
emit() {
  [ -n "${GITHUB_OUTPUT:-}" ] || return 0
  printf '%s=%s\n' "$1" "$2" >> "$GITHUB_OUTPUT"
}

notify_slack() {
  [ -n "$SLACK_WEBHOOK_URL" ] || return 0
  curl -sf -X POST -H 'Content-type: application/json' \
    --data "$(jq -n --arg t "$1" '{text:$t}')" \
    "$SLACK_WEBHOOK_URL" >/dev/null || warn "Slack notification failed"
}

ensure_label() {
  gh label create "$1" --repo "$REPO" --color "$2" --description "$3" >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------------------
# 0. Stay dormant until the merge PAT is configured. Without it the bot would
#    merge with GITHUB_TOKEN, whose push does NOT fire dispatch-infra-build,
#    leaving prod silently behind. Require the PAT before acting.
# ---------------------------------------------------------------------------
if [ "$HAS_AUTOMERGE_TOKEN" != "true" ]; then
  log "SYNC_AUTOMERGE_TOKEN not set — autopilot dormant (merging without it would skip the infra build). Add the PAT to activate."
  summary "### 🛩️ Sync PR Autopilot — _${REPO}_"
  summary "**Dormant** — \`SYNC_AUTOMERGE_TOKEN\` is not configured. Add the PAT to activate."
  emit outcome "dormant"
  exit 0
fi

# ---------------------------------------------------------------------------
# 1. Locate the open sync PR (highest-numbered match on prefix + base).
# ---------------------------------------------------------------------------
PR_LIST="$(gh pr list --repo "$REPO" --state open --base "$BASE_BRANCH" \
  --json number,headRefName --limit 50)"
PR_NUM="$(jq -r --arg p "$HEAD_PREFIX" \
  '[.[] | select(.headRefName | startswith($p))] | sort_by(.number) | last | .number // empty' \
  <<<"$PR_LIST")"

if [ -z "$PR_NUM" ]; then
  log "No open sync PR (prefix '$HEAD_PREFIX', base '$BASE_BRANCH'). Nothing to do."
  summary "### 🛩️ Sync PR Autopilot — _${REPO}_"
  summary "No open \`${HEAD_PREFIX}*\` PR. Nothing to do."
  emit outcome "noop"
  exit 0
fi
log "Target sync PR: #$PR_NUM"

DETAIL="$(gh pr view "$PR_NUM" --repo "$REPO" \
  --json number,title,url,headRefName,isDraft,mergeable,mergeStateStatus,statusCheckRollup,labels)"
URL="$(jq -r .url <<<"$DETAIL")"
TITLE="$(jq -r .title <<<"$DETAIL")"
MERGEABLE="$(jq -r .mergeable <<<"$DETAIL")"
emit pr_number "$PR_NUM"
emit pr_url "$URL"
emit pr_title "$TITLE"

hold() { # $1 = human reason, $2 = slack/comment detail
  ensure_label "sync-held" "d93f0b" "Sync PR held by autopilot — needs attention"
  gh pr edit "$PR_NUM" --repo "$REPO" --add-label "sync-held" >/dev/null 2>&1 || true
  gh pr comment "$PR_NUM" --repo "$REPO" --body "🟡 **Sync autopilot: held** — $2" >/dev/null 2>&1 || true
  notify_slack ":warning: *Sync autopilot — HELD* <$URL|#$PR_NUM> \`$REPO\`: $1"
  summary "### 🛩️ Sync PR Autopilot — _${REPO}_"
  summary "**HELD** — [#$PR_NUM]($URL): $1"
  emit outcome "held"
}

# ---------------------------------------------------------------------------
# 2. Merge conflicts -> hand back to the next sync session.
# ---------------------------------------------------------------------------
if [ "$MERGEABLE" = "CONFLICTING" ]; then
  hold "merge conflicts" "this PR conflicts with \`$BASE_BRANCH\` and can't be auto-merged. The next upstream-sync session should rebuild/re-resolve it (prefer upstream where appropriate)."
  exit 0
fi
if [ "$MERGEABLE" = "UNKNOWN" ]; then
  log "Mergeability still computing; will re-evaluate on the next event."
  emit outcome "waiting"
  exit 0
fi

# ---------------------------------------------------------------------------
# 3. Evaluate the combined check rollup.
# ---------------------------------------------------------------------------
ROLLUP="$(jq -c '.statusCheckRollup // []' <<<"$DETAIL")"
read -r PENDING FAILURE SUCCESS TOTAL < <(jq -r '
  def norm:
    if (.status // "") != "" then
      ( if .status != "COMPLETED" then "PENDING"
        elif ((.conclusion // "") | . == "SUCCESS" or . == "NEUTRAL" or . == "SKIPPED") then "SUCCESS"
        else "FAILURE" end )
    else
      ( if (.state // "") == "SUCCESS" then "SUCCESS"
        elif (.state // "") == "PENDING" then "PENDING"
        else "FAILURE" end )
    end;
  [ .[] | norm ] as $s
  | "\([$s[]|select(.=="PENDING")]|length) \([$s[]|select(.=="FAILURE")]|length) \([$s[]|select(.=="SUCCESS")]|length) \($s|length)"
' <<<"$ROLLUP")
log "Checks: pending=$PENDING failure=$FAILURE success=$SUCCESS total=$TOTAL"

if [ "$TOTAL" -eq 0 ] || [ "$PENDING" -gt 0 ]; then
  log "Checks not finished yet; waiting for a later completion event."
  emit outcome "waiting"
  exit 0
fi

# ---------------------------------------------------------------------------
# 4. Failed checks -> re-run failed runs (flaky), capped by label counter.
# ---------------------------------------------------------------------------
if [ "$FAILURE" -gt 0 ]; then
  LABELS="$(jq -r '[.labels[].name] | join(",")' <<<"$DETAIL")"
  attempts=0
  for n in $(seq 1 "$MAX_RERUNS"); do
    case ",$LABELS," in *",autopilot-rerun-$n,"*) attempts=$n ;; esac
  done

  if [ "$attempts" -ge "$MAX_RERUNS" ]; then
    hold "CI still red after $MAX_RERUNS re-runs" "CI is still failing after $MAX_RERUNS automated re-runs, so this looks like a real failure rather than flake. Needs a human or the next sync session to fix."
    exit 0
  fi

  next=$((attempts + 1))
  RUN_IDS="$(jq -r '
    .[] | select((.status // "") == "COMPLETED"
      and ((.conclusion // "") | (. == "SUCCESS" or . == "NEUTRAL" or . == "SKIPPED") | not))
    | (.detailsUrl // "")' <<<"$ROLLUP" \
    | grep -oE 'actions/runs/[0-9]+' | grep -oE '[0-9]+' | sort -u)"

  if [ "$DRY_RUN" = "true" ]; then
    log "DRY_RUN: would re-run failed runs [$(echo "$RUN_IDS" | tr '\n' ' ')] (attempt $next/$MAX_RERUNS)"
    emit outcome "would-rerun"
    exit 0
  fi

  rerun_any=false
  for rid in $RUN_IDS; do
    if gh run rerun "$rid" --failed --repo "$REPO" >/dev/null 2>&1 \
       || gh run rerun "$rid" --repo "$REPO" >/dev/null 2>&1; then
      rerun_any=true
    fi
  done

  ensure_label "autopilot-rerun-$next" "fbca04" "Autopilot flaky-CI re-run attempt $next"
  gh pr edit "$PR_NUM" --repo "$REPO" --add-label "autopilot-rerun-$next" >/dev/null 2>&1 || true
  if [ "$rerun_any" = true ]; then
    log "Re-ran failed CI (attempt $next/$MAX_RERUNS)."
    notify_slack ":arrows_counterclockwise: *Sync autopilot* <$URL|#$PR_NUM> \`$REPO\`: CI red, re-running failed jobs (attempt $next/$MAX_RERUNS)."
    summary "### 🛩️ Sync PR Autopilot — _${REPO}_"
    summary "Re-running failed CI on [#$PR_NUM]($URL) — attempt $next/$MAX_RERUNS."
    emit outcome "rerun"
  else
    hold "CI failed and no re-runnable runs were found" "CI failed but the autopilot could not identify a workflow run to re-run. Needs manual attention."
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# 5. All green -> mark ready and merge (fires dispatch-infra-build downstream).
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" = "true" ]; then
  log "DRY_RUN: PR #$PR_NUM is green and would be merged (--$MERGE_METHOD)."
  summary "### 🛩️ Sync PR Autopilot — _${REPO}_"
  summary "DRY_RUN: [#$PR_NUM]($URL) is green — would merge (\`--$MERGE_METHOD\`)."
  emit outcome "would-merge"
  exit 0
fi

gh pr ready "$PR_NUM" --repo "$REPO" >/dev/null 2>&1 || true
if ! gh pr merge "$PR_NUM" --repo "$REPO" "--$MERGE_METHOD" --delete-branch; then
  hold "auto-merge command failed" "all checks were green but \`gh pr merge --$MERGE_METHOD\` failed (merge method may be disabled on this repo, or the token lacks write). Needs attention."
  exit 1
fi

log "Merged sync PR #$PR_NUM."
notify_slack ":white_check_mark: *Sync autopilot — MERGED* <$URL|#$PR_NUM> \"$TITLE\" \`$REPO\` (green CI). Infra build dispatch follows."
gh pr comment "$PR_NUM" --repo "$REPO" \
  --body "✅ **Sync autopilot: merged** — all CI green, merged via \`$MERGE_METHOD\`. The infra build dispatch (\`dispatch-infra-build.yml\`) fires on this merge." >/dev/null 2>&1 || true
summary "### 🛩️ Sync PR Autopilot — _${REPO}_"
summary "**MERGED** [#$PR_NUM]($URL) — \"$TITLE\" (green CI). Infra build dispatch follows."
emit outcome "merged"
