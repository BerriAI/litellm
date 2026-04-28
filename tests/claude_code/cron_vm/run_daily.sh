#!/usr/bin/env bash
# Daily Claude Code compatibility-matrix populator.
#
# Runs from the GCP VM `litellm-compatibility-matrix-populator` via the
# systemd timer in this directory. The flow is:
#
#   1. Resolve the latest LiteLLM v*-stable tag from the GitHub Releases API.
#   2. Update a long-lived worktree at $WORKTREE to that tag and `uv sync` it.
#   3. Boot the proxy as a background subprocess on $PROXY_PORT (default
#      4100; a separate port from the human-tended :4000 proxy).
#   4. Run `pytest tests/claude_code/` against the proxy. Test failures
#      become `fail` cells in the JSON, not script errors.
#   5. Hand the per-test results artifact + manifest to a small Python
#      CLI (`build_matrix.py`) that wraps the existing
#      `matrix_builder.build_from_paths` to produce the published
#      compatibility-matrix.json.
#   6. `gh repo clone` litellm-docs, write the JSON to a deterministic
#      branch (`compat-matrix/<litellm>-<claude>-<UTC-date>`), commit,
#      `git push --force-with-lease`, and `gh pr create`.
#
# Same-day reruns land on the same branch so they update the existing PR
# rather than spawning a new one. If the JSON is byte-identical to the
# docs branch, we skip the push entirely.
#
# Required commands on $PATH: git, uv, gh, jq, curl, claude, npm.
# Required state: ~/litellm/litellm checked out (this file lives in it),
# $WORKTREE is created on first run, gh is already authenticated.
#
# Override any default by setting the matching env var; see the systemd
# unit for the production wiring.

set -Eeuo pipefail

LITELLM_REPO="${LITELLM_REPO:-${HOME}/litellm/litellm}"
WORKTREE="${LITELLM_WORKTREE:-${HOME}/litellm-cron-worktree}"
PROXY_PORT="${PROXY_PORT:-4100}"
PROXY_API_KEY="${PROXY_API_KEY:-sk-cron-matrix}"
DOCS_REPO="${DOCS_REPO:-BerriAI/litellm-docs}"
DOCS_BRANCH="${DOCS_BRANCH:-main}"
DOCS_TARGET_PATH="${DOCS_TARGET_PATH:-src/data/compatibility-matrix.json}"
SKIP_PUBLISH="${SKIP_PUBLISH:-0}"
PYTEST_K="${PYTEST_K:-}"

POPULATOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(mktemp -d -t litellm-compat-matrix.XXXXXX)"
PROXY_PID_FILE="${WORKDIR}/proxy.pid"

# Cleanup is intentionally aggressive: it can run on normal exit, on a
# signal received by the script, or after a partial failure where the
# proxy is up but ${PROXY_PID_FILE} is stale. We try four things in
# order and stop as soon as the proxy port is free:
#
#   1. SIGTERM the pid recorded in proxy.pid.
#   2. SIGKILL anything from `pgrep -f "litellm.*--port ${PROXY_PORT}"`
#      that survived. This catches the common case where the recorded
#      pid was the sh wrapper, not the long-lived python child.
#   3. ss -K on the port (kernel kills sockets but not processes;
#      mostly useful for catching lingering CLOSE_WAITs).
#   4. wipe ${WORKDIR}.
cleanup() {
  local rc=$?
  set +e
  local proxy_pid
  if [[ -f "${PROXY_PID_FILE}" ]]; then
    proxy_pid="$(cat "${PROXY_PID_FILE}")"
    if [[ -n "${proxy_pid}" ]]; then
      kill -TERM "-${proxy_pid}" 2>/dev/null || kill -TERM "${proxy_pid}" 2>/dev/null || true
      for _ in 1 2 3 4 5; do
        kill -0 "${proxy_pid}" 2>/dev/null || break
        sleep 1
      done
    fi
  fi
  # Belt-and-braces: any python or uv talking to ${PROXY_PORT} that
  # survived the SIGTERM gets SIGKILL'd by name.
  pgrep -f "litellm.*--port[ =]?${PROXY_PORT}\b" 2>/dev/null \
    | xargs -r kill -KILL 2>/dev/null || true
  pgrep -f "${WORKTREE}/.uv-bin/uv.*run litellm" 2>/dev/null \
    | xargs -r kill -KILL 2>/dev/null || true
  rm -rf "${WORKDIR}"
  exit "${rc}"
}
trap cleanup EXIT INT TERM

log() { printf '==> %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

for cmd in git uv gh jq curl claude; do
  command -v "${cmd}" >/dev/null 2>&1 || die "missing required command: ${cmd}"
done

# ---------------------------------------------------------------------------
# 1. Resolve versions
# ---------------------------------------------------------------------------

# Newest v*-stable release on BerriAI/litellm. The `select(...)` filter
# drops drafts/non-stable, the version_key sort handles 1.10 > 1.9.
GH_AUTH_HEADER=()
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  GH_AUTH_HEADER=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
fi
LITELLM_VERSION="$(
  curl -fsS \
    -H 'Accept: application/vnd.github+json' \
    -H 'User-Agent: litellm-compat-matrix' \
    "${GH_AUTH_HEADER[@]}" \
    https://api.github.com/repos/BerriAI/litellm/releases \
    | jq -r '
        [ .[] | .tag_name // empty
          | select(test("^v[0-9]+\\.[0-9]+\\.[0-9]+-stable$"))
        ]
        | sort_by(
            capture("^v(?<a>[0-9]+)\\.(?<b>[0-9]+)\\.(?<c>[0-9]+)-stable$")
            | [(.a|tonumber), (.b|tonumber), (.c|tonumber)]
          )
        | last // empty
      '
)"
[[ -n "${LITELLM_VERSION}" ]] || die "could not resolve latest v*-stable tag"
log "resolved litellm: ${LITELLM_VERSION}"

CLAUDE_CODE_VERSION="$(claude --version 2>/dev/null | awk '{print $1}')"
[[ -n "${CLAUDE_CODE_VERSION}" ]] || die "could not read 'claude --version'"
log "local claude code: ${CLAUDE_CODE_VERSION}"

# ---------------------------------------------------------------------------
# 2. Update the worktree to that tag
# ---------------------------------------------------------------------------

if [[ ! -d "${WORKTREE}/.git" ]]; then
  log "first run: cloning litellm into ${WORKTREE}"
  mkdir -p "$(dirname "${WORKTREE}")"
  git clone https://github.com/BerriAI/litellm.git "${WORKTREE}"
fi

log "updating worktree to ${LITELLM_VERSION}"
git -C "${WORKTREE}" fetch --tags --force
git -C "${WORKTREE}" reset --hard
# Keep the venv and the .uv-bin cache around — uv sync will reconcile
# the venv and we don't want to re-download the pinned uv binary on
# every run. Drop everything else (compat-results.json, __pycache__,
# etc.) so each run starts clean.
git -C "${WORKTREE}" clean -fdx -e .venv -e .uv-bin
git -C "${WORKTREE}" checkout --force "${LITELLM_VERSION}"

# litellm pins an exact uv version in pyproject.toml's [tool.uv]
# `required-version` field, so a system uv that's newer or older
# refuses to sync. We pin our own local copy at the version the
# checked-out tag asks for, cached under .uv-bin/ inside the worktree
# so subsequent runs skip the download.
PINNED_UV_VERSION="$(
  awk -F'"' '/^required-version[[:space:]]*=/{ for (i=2; i<=NF; i++) if ($i ~ /^[=<>!~]+/) { gsub(/^[=<>!~]+/, "", $i); print $i; exit } }' \
    "${WORKTREE}/pyproject.toml"
)"
if [[ -z "${PINNED_UV_VERSION}" ]]; then
  log "no uv version pin in pyproject.toml; using system uv"
  WORKTREE_UV="$(command -v uv)"
else
  WORKTREE_UV="${WORKTREE}/.uv-bin/uv-${PINNED_UV_VERSION}"
  if [[ ! -x "${WORKTREE_UV}" ]]; then
    log "downloading uv ${PINNED_UV_VERSION} for the worktree"
    mkdir -p "${WORKTREE}/.uv-bin"
    curl -fsSL \
      "https://github.com/astral-sh/uv/releases/download/${PINNED_UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
      | tar -xzO "uv-x86_64-unknown-linux-gnu/uv" >"${WORKTREE_UV}.tmp"
    chmod +x "${WORKTREE_UV}.tmp"
    mv "${WORKTREE_UV}.tmp" "${WORKTREE_UV}"
  fi
fi
# `--extra proxy` pulls fastapi/uvicorn/etc. so `uv run litellm` can
# actually serve. `--group proxy-dev` brings in pytest and the rest of
# what tests/claude_code/ needs.
log "uv sync --frozen --group proxy-dev --extra proxy (uv ${PINNED_UV_VERSION:-system})"
(cd "${WORKTREE}" && "${WORKTREE_UV}" sync --frozen --group proxy-dev --extra proxy)

PROXY_CONFIG="${WORKTREE}/tests/claude_code/test_config.yaml"
[[ -f "${PROXY_CONFIG}" ]] || die "proxy config not found at ${PROXY_CONFIG} (does ${LITELLM_VERSION} predate the compat matrix work?)"

# ---------------------------------------------------------------------------
# 3. Boot the proxy
# ---------------------------------------------------------------------------

log "starting proxy on :${PROXY_PORT}"
# `setsid` puts the proxy in its own session+pgroup so cleanup() can
# SIGTERM the whole tree by passing the pgid as a negative pid. We
# write that pid to a file so cleanup() doesn't need to remember a
# variable that might be stale by the time the trap fires.
setsid bash -c '
  echo "$$" > "$0"
  cd "$1"
  exec "$2" run litellm --config "$3" --port "$4"
' "${PROXY_PID_FILE}" "${WORKTREE}" "${WORKTREE_UV}" "${PROXY_CONFIG}" "${PROXY_PORT}" \
  >"${WORKDIR}/proxy.log" 2>&1 &
disown

HEALTH_URL="http://127.0.0.1:${PROXY_PORT}/health/liveliness"
for _ in $(seq 1 45); do
  if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS "${HEALTH_URL}" >/dev/null \
  || { tail -50 "${WORKDIR}/proxy.log" >&2; die "proxy did not become healthy"; }

# ---------------------------------------------------------------------------
# 4. Run pytest
# ---------------------------------------------------------------------------

RESULTS_JSON="${WORKDIR}/compat-results.json"
PYTEST_ARGS=(
  tests/claude_code/
  --ignore=tests/claude_code/_driver_unit_tests
  --ignore=tests/claude_code/_builder_unit_tests
  --ignore=tests/claude_code/_publisher_unit_tests
  --ignore=tests/claude_code/_pr_gate_unit_tests
)
if [[ -n "${PYTEST_K}" ]]; then
  log "PYTEST_K set; narrowing to: ${PYTEST_K}"
  PYTEST_ARGS+=(-k "${PYTEST_K}")
fi

log "running pytest"
set +e
(
  cd "${WORKTREE}" \
    && LITELLM_PROXY_BASE_URL="http://127.0.0.1:${PROXY_PORT}" \
       LITELLM_PROXY_API_KEY="${PROXY_API_KEY}" \
       COMPAT_RESULTS_PATH="${RESULTS_JSON}" \
       "${WORKTREE_UV}" run pytest "${PYTEST_ARGS[@]}"
)
PYTEST_EXIT=$?
set -e
log "pytest exit code: ${PYTEST_EXIT} (failures become 'fail' cells, not script errors)"
[[ -f "${RESULTS_JSON}" ]] || die "pytest did not produce ${RESULTS_JSON}"

# ---------------------------------------------------------------------------
# 5. Build the matrix JSON
# ---------------------------------------------------------------------------

MATRIX_JSON="${WORKDIR}/compatibility-matrix.json"
log "building ${MATRIX_JSON}"
(
  cd "${WORKTREE}" \
    && "${WORKTREE_UV}" run python "${POPULATOR_DIR}/build_matrix.py" \
       --manifest "${WORKTREE}/tests/claude_code/manifest.yaml" \
       --results "${RESULTS_JSON}" \
       --output "${MATRIX_JSON}" \
       --litellm-version "${LITELLM_VERSION}" \
       --claude-code-version "${CLAUDE_CODE_VERSION}"
)

# ---------------------------------------------------------------------------
# 6. Open a docs-repo PR
# ---------------------------------------------------------------------------

if [[ "${SKIP_PUBLISH}" == "1" ]]; then
  cp "${MATRIX_JSON}" "${LITELLM_REPO}/compatibility-matrix.json"
  log "SKIP_PUBLISH=1; matrix written to ${LITELLM_REPO}/compatibility-matrix.json"
  exit 0
fi

DATE_UTC="$(date -u +%Y-%m-%d)"
BRANCH_NAME="compat-matrix/${LITELLM_VERSION}-${CLAUDE_CODE_VERSION}-${DATE_UTC}"
DOCS_CLONE="${WORKDIR}/litellm-docs"

log "cloning ${DOCS_REPO}@${DOCS_BRANCH}"
gh repo clone "${DOCS_REPO}" "${DOCS_CLONE}" -- --depth 1 --branch "${DOCS_BRANCH}"

cd "${DOCS_CLONE}"
git config user.email "litellm-bot@berri.ai"
git config user.name "litellm-compat-matrix-bot"
git checkout -b "${BRANCH_NAME}"

mkdir -p "$(dirname "${DOCS_TARGET_PATH}")"
cp "${MATRIX_JSON}" "${DOCS_TARGET_PATH}"
git add "${DOCS_TARGET_PATH}"

if git diff --cached --quiet; then
  log "matrix JSON unchanged from ${DOCS_BRANCH}; skipping PR"
  exit 0
fi

GENERATED_AT="$(jq -r '.generated_at' "${MATRIX_JSON}")"
COMMIT_MSG="$(cat <<EOF
Update Claude Code compatibility matrix

litellm_version: ${LITELLM_VERSION}
claude_code_version: ${CLAUDE_CODE_VERSION}
generated_at: ${GENERATED_AT}
EOF
)"
git commit -m "${COMMIT_MSG}"

# --force-with-lease so a same-day rerun fast-forwards (or rebases) the
# existing branch without clobbering a maintainer's manual fixup.
git push --force-with-lease --set-upstream origin "${BRANCH_NAME}"

# Per-feature status table for the PR body. Reviewers triage from this.
PR_FEATURE_TABLE="$(jq -r '
  .features[] as $f
  | "- **\($f.name)**: " +
    ([ .providers[] as $p
       | "\($p)=\($f.providers[$p].status // "not_tested")"
     ] | join(", "))
' "${MATRIX_JSON}")"

PR_TITLE="chore(compat-matrix): refresh for ${LITELLM_VERSION} + claude-code ${CLAUDE_CODE_VERSION}"
PR_BODY="$(cat <<EOF
Automated daily refresh of the Claude Code compatibility matrix.

| Field | Value |
| --- | --- |
| litellm_version | \`${LITELLM_VERSION}\` |
| claude_code_version | \`${CLAUDE_CODE_VERSION}\` |
| generated_at | \`${GENERATED_AT}\` |

## Per-feature results

${PR_FEATURE_TABLE}

---

Generated by \`tests/claude_code/cron_vm/run_daily.sh\`. Close without merging if the diff looks wrong; the next cron run will reopen with fresh results.
EOF
)"

log "opening PR"
set +e
PR_OUT="$(
  gh pr create \
    --repo "${DOCS_REPO}" \
    --base "${DOCS_BRANCH}" \
    --head "${BRANCH_NAME}" \
    --title "${PR_TITLE}" \
    --body "${PR_BODY}" 2>&1
)"
PR_EXIT=$?
set -e
echo "${PR_OUT}"

if [[ ${PR_EXIT} -ne 0 ]]; then
  if grep -q "a pull request for branch.*already exists" <<<"${PR_OUT}"; then
    log "PR already exists for ${BRANCH_NAME}; updated branch in place"
  else
    die "gh pr create failed (exit ${PR_EXIT})"
  fi
fi

log "done"
