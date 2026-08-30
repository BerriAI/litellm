#!/usr/bin/env bash
# Daily Claude Code compatibility-matrix populator.
#
# Runs from the GCP VM `litellm-compatibility-matrix-populator` via the
# systemd timer in this directory. The flow is:
#
#   1. Resolve the latest LiteLLM final release tag from the GitHub
#      Releases API.
#   2. Update a long-lived worktree at $WORKTREE to that tag and `uv sync` it.
#   3. Boot the proxy as a background subprocess on $PROXY_PORT (default
#      4100; a separate port from the human-tended :4000 proxy).
#   4. Run `pytest tests/e2e/claude_code/` against the proxy. Test
#      failures become `fail` cells in the JSON, not script errors.
#   5. Hand the per-test results artifact + manifest to a small Python
#      CLI (`build_matrix.py`) that wraps the existing
#      `matrix_builder.build_from_paths` to produce the published
#      compatibility-matrix.json.
#   6. `gh repo clone` litellm-docs, write the JSON to a deterministic
#      branch (`compat-matrix/<litellm>-<claude>-<UTC-date>`), commit,
#      push the branch straight to BerriAI/litellm-docs (mateo-berri has
#      write access), `gh pr create`, then — *only if no cell regressed
#      green→red versus the currently-published matrix* — enable squash
#      auto-merge so the PR merges itself once required checks pass. A
#      green→red regression leaves auto-merge off for human review; an
#      already-red cell (red→red) does not block.
#   7. Sweep stale compat-matrix PRs: once today's PR exists, close any
#      other open `compat-matrix/*` PR (and delete its bot-owned branch)
#      so at most ONE compat-matrix PR is ever open — the newest. A
#      gate-withheld PR that nobody triages is superseded by the next
#      day's run rather than accumulating in the queue.
#
# Same-day reruns land on the same branch so they update the existing PR
# rather than spawning a new one. If the JSON is byte-identical to the
# docs branch, we skip the push entirely.
#
# Required commands on $PATH: git, uv, gh, jq, curl, claude, npm.
# Required state: a litellm checkout at $LITELLM_REPO (this file lives in
# it), $WORKTREE is created on first run, gh is already authenticated.
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
# The e2e suite uses PEP 695 `type` aliases, so the venv needs Python
# >= 3.12 (also what repo CI runs) even when the VM's system python is
# older. uv fetches a managed CPython of this version on first use --
# checksum-verified against the manifest baked into the pinned uv
# binary -- and installs it under ${WORKTREE}/.uv-python (see
# UV_PYTHON_INSTALL_DIR below) so it lives inside the one tree the
# systemd sandbox lets us write to.
CRON_PYTHON_VERSION="${CRON_PYTHON_VERSION:-3.12}"
# Merge method for auto-merge. BerriAI/litellm-docs only allows squash
# merges (merge-commit and rebase are disabled at the repo level), so
# `squash` is the only valid value here unless that changes upstream.
AUTO_MERGE_METHOD="${AUTO_MERGE_METHOD:-squash}"

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
  pgrep -f "litellm.*--port[ =]?${PROXY_PORT}([^0-9]|$)" 2>/dev/null \
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

# Publishing pushes the branch straight to BerriAI/litellm-docs and opens
# the PR as mateo-berri, who has write access on the docs repo. Under
# systemd the PAT arrives as a file via LoadCredential=, NOT via the
# EnvironmentFile: several suite cells let the model-driven claude CLI
# read arbitrary files as this user, and /proc/<pid>/environ of the
# script, pytest, and the proxy would hand an env-borne token to any
# same-UID reader. Kept as an unexported shell variable and passed per
# invocation (GH_TOKEN=... / curl header / push URL), it never enters a
# child's environment. Manual runs may export GITHUB_TOKEN instead.
# Require it up front -- failing 30 minutes into a run is a waste of CI
# quota.
if [[ -z "${GITHUB_TOKEN:-}" && -n "${CREDENTIALS_DIRECTORY:-}" && -f "${CREDENTIALS_DIRECTORY}/github-token" ]]; then
  GITHUB_TOKEN="$(<"${CREDENTIALS_DIRECTORY}/github-token")"
  log "publish token source: systemd credential store"
elif [[ -n "${GITHUB_TOKEN:-}" ]]; then
  log "publish token source: process environment"
fi
if [[ "${SKIP_PUBLISH}" != "1" ]]; then
  [[ -n "${GITHUB_TOKEN:-}" ]] \
    || die "publish token required: /etc/litellm-compat-matrix-github-token via LoadCredential under systemd, or an exported GITHUB_TOKEN for manual runs (or set SKIP_PUBLISH=1)"
fi

# ---------------------------------------------------------------------------
# 1. Resolve versions
# ---------------------------------------------------------------------------

# Newest PEP 440 *final* release on BerriAI/litellm. LiteLLM moved off
# the legacy `vX.Y.Z-stable` tag convention to PEP 440: a final/stable
# release is now a bare `vX.Y.Z` tag, while pre-releases carry a
# `-rc.N` / `-dev.N` segment (and the old `…-stable` / `…-stable.patch.N`
# tags are legacy and frozen at v1.83.x). We therefore select the newest
# tag with no pre-release segment -- matching `^v[0-9]+\.[0-9]+\.[0-9]+$`
# -- and skip drafts. The numeric version_key sort handles 1.10 > 1.9.
#
# Paginate through the releases endpoint instead of grabbing only page 1
# (default page_size=30). LiteLLM ships multiple pre-releases per day, so
# it's common to need to walk past 30+ entries before hitting the most
# recent final release. We cap at 5 pages (500 releases) which is
# conservatively beyond the worst observed gap.
GH_AUTH_HEADER=()
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  GH_AUTH_HEADER=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
fi
RELEASES_JSON="${WORKDIR}/releases.json"
echo "[]" >"${RELEASES_JSON}"
for page in 1 2 3 4 5; do
  PAGE_JSON="${WORKDIR}/releases.page${page}.json"
  curl -fsS \
    -H 'Accept: application/vnd.github+json' \
    -H 'User-Agent: litellm-compat-matrix' \
    "${GH_AUTH_HEADER[@]}" \
    "https://api.github.com/repos/BerriAI/litellm/releases?per_page=100&page=${page}" \
    >"${PAGE_JSON}"
  jq -s '.[0] + .[1]' "${RELEASES_JSON}" "${PAGE_JSON}" >"${RELEASES_JSON}.merged"
  mv "${RELEASES_JSON}.merged" "${RELEASES_JSON}"
  # Stop early once we've seen at least one final release tag — no point
  # paging further for a daily script that only needs the newest.
  if jq -e '[.[] | select((.draft // false) == false) | .tag_name // "" | select(test("^v[0-9]+\\.[0-9]+\\.[0-9]+$"))] | length > 0' "${PAGE_JSON}" >/dev/null; then
    break
  fi
  # No more pages? GitHub returns an empty array past the last page.
  if [[ "$(jq 'length' "${PAGE_JSON}")" == "0" ]]; then
    break
  fi
done
LITELLM_VERSION="$(
  jq -r '
    [ .[]
      | select((.draft // false) == false)
      | .tag_name // empty
      | select(test("^v[0-9]+\\.[0-9]+\\.[0-9]+$"))
    ]
    | sort_by(
        capture("^v(?<a>[0-9]+)\\.(?<b>[0-9]+)\\.(?<c>[0-9]+)$")
        | [(.a|tonumber), (.b|tonumber), (.c|tonumber)]
      )
    | last // empty
  ' "${RELEASES_JSON}"
)"
[[ -n "${LITELLM_VERSION}" ]] || die "could not resolve latest PEP 440 final release (vX.Y.Z) in 5 pages of releases"
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
# Keep the venv, the .uv-bin cache, and the .uv-python managed
# interpreter around — uv sync will reconcile the venv on every run,
# and we don't want to re-download the pinned uv binary or the managed
# CPython each time. Drop everything else (including any prior
# tests/e2e/ shim) so each run starts clean before the shim below
# rewrites it from the dev checkout.
git -C "${WORKTREE}" clean -fdx -e .venv -e .uv-bin -e .uv-python
git -C "${WORKTREE}" checkout --force "${LITELLM_VERSION}"

# Always rebuild tests/e2e/ in the worktree from the dev checkout,
# regardless of what the resolved ${LITELLM_VERSION} tag ships. Two
# reasons:
#
#   * The matrix populator's job is to exercise *today's* tests against
#     the latest stable proxy. The dev checkout carries the most recent
#     test fixes that haven't yet rolled into a stable release, and we
#     want every cron run to pick those up the moment they land on
#     ${LITELLM_REPO}, not whenever the next stable release happens.
#   * The tag's own tests/e2e/ ships the full EKS e2e harness, whose
#     top-level conftest.py imports modules (e2e_db, lifecycle,
#     otel_client, ...) that the stable venv does not install. Copying
#     the whole tree would make pytest collection blow up on those
#     imports.
#
# So the shim is a fresh `rm -rf` of tests/e2e/ followed by copying ONLY
# the claude_code suite plus the shared transport helpers it imports.
# pytest puts tests/e2e/ itself on sys.path (it has no __init__.py, while
# claude_code/ does), which is what resolves both the `claude_code.*`
# and the bare `proxy_client` / `e2e_http` imports inside the suite.
E2E_HELPER_FILES=(proxy_client.py e2e_http.py models.py e2e_config.py transport.py)
if [[ ! -d "${LITELLM_REPO}/tests/e2e/claude_code" ]]; then
  die "no shim source at ${LITELLM_REPO}/tests/e2e/claude_code"
fi
for helper in "${E2E_HELPER_FILES[@]}"; do
  [[ -f "${LITELLM_REPO}/tests/e2e/${helper}" ]] \
    || die "missing shim helper: ${LITELLM_REPO}/tests/e2e/${helper}"
done
log "shimming tests/e2e/claude_code/ + helpers from ${LITELLM_REPO} (always-overwrite)"
rm -rf "${WORKTREE}/tests/e2e"
mkdir -p "${WORKTREE}/tests/e2e"
cp -r "${LITELLM_REPO}/tests/e2e/claude_code" "${WORKTREE}/tests/e2e/"
for helper in "${E2E_HELPER_FILES[@]}"; do
  cp "${LITELLM_REPO}/tests/e2e/${helper}" "${WORKTREE}/tests/e2e/"
done

# litellm pins an exact uv version in pyproject.toml's [tool.uv]
# `required-version` field, so a system uv that's newer or older
# refuses to sync. We pin our own local copy at the version the
# checked-out tag asks for, cached under .uv-bin/ inside the worktree
# so subsequent runs skip the download.
PINNED_UV_VERSION="$(
  awk -F'"' '
    /^required-version[[:space:]]*=/ {
      # Field 2 is the value between the quotes, e.g. ">=0.10.9" or
      # "0.10.9". Strip any leading specifier prefix so we end up with
      # the bare version string, which is what /releases/download/<v>/
      # expects.
      v = $2
      sub(/^[[:space:]=<>!~]+/, "", v)
      if (v != "") { print v; exit }
    }
  ' "${WORKTREE}/pyproject.toml"
)"
if [[ -z "${PINNED_UV_VERSION}" ]]; then
  log "no uv version pin in pyproject.toml; using system uv"
  WORKTREE_UV="$(command -v uv)"
else
  WORKTREE_UV="${WORKTREE}/.uv-bin/uv-${PINNED_UV_VERSION}"
  if [[ ! -x "${WORKTREE_UV}" ]]; then
    log "downloading uv ${PINNED_UV_VERSION} for the worktree"
    mkdir -p "${WORKTREE}/.uv-bin"
    UV_TARBALL_NAME="uv-x86_64-unknown-linux-gnu.tar.gz"
    UV_DOWNLOAD_URL="https://github.com/astral-sh/uv/releases/download/${PINNED_UV_VERSION}/${UV_TARBALL_NAME}"
    UV_TMPDIR="$(mktemp -d -t uv-download.XXXXXX)"
    # Download the tarball and Astral's official .sha256 sidecar to disk
    # and verify the digest before extracting/executing anything. This
    # closes the supply-chain trust gap of piping a remote binary
    # straight into `tar -xzO ... > file ; chmod +x` (see CLAUDE.md
    # "CI Supply-Chain Safety").
    curl -fsSL --output "${UV_TMPDIR}/${UV_TARBALL_NAME}" "${UV_DOWNLOAD_URL}"
    curl -fsSL --output "${UV_TMPDIR}/${UV_TARBALL_NAME}.sha256" "${UV_DOWNLOAD_URL}.sha256"
    (cd "${UV_TMPDIR}" && sha256sum -c "${UV_TARBALL_NAME}.sha256") \
      || { rm -rf "${UV_TMPDIR}"; die "uv ${PINNED_UV_VERSION} sha256 mismatch — refusing to install"; }
    tar -xzf "${UV_TMPDIR}/${UV_TARBALL_NAME}" -C "${UV_TMPDIR}" "uv-x86_64-unknown-linux-gnu/uv"
    mv "${UV_TMPDIR}/uv-x86_64-unknown-linux-gnu/uv" "${WORKTREE_UV}.tmp"
    chmod +x "${WORKTREE_UV}.tmp"
    mv "${WORKTREE_UV}.tmp" "${WORKTREE_UV}"
    rm -rf "${UV_TMPDIR}"
  fi
fi
# `--extra proxy` pulls fastapi/uvicorn/etc. so `uv run litellm` can
# actually serve. `--group proxy-dev` brings in pytest and the rest of
# what tests/e2e/claude_code/ needs. `--python` pins the venv to
# ${CRON_PYTHON_VERSION}; the first run after a version bump recreates
# the venv from scratch (a one-time cold sync).
export UV_PYTHON_INSTALL_DIR="${WORKTREE}/.uv-python"
log "uv sync --frozen --group proxy-dev --extra proxy --python ${CRON_PYTHON_VERSION} (uv ${PINNED_UV_VERSION:-system})"
(cd "${WORKTREE}" && "${WORKTREE_UV}" sync --frozen --group proxy-dev --extra proxy --python "${CRON_PYTHON_VERSION}")

PROXY_CONFIG="${WORKTREE}/tests/e2e/claude_code/test_config.yaml"
[[ -f "${PROXY_CONFIG}" ]] || die "proxy config not found at ${PROXY_CONFIG} (shim incomplete?)"

# ---------------------------------------------------------------------------
# 3. Boot the proxy
# ---------------------------------------------------------------------------

log "starting proxy on 127.0.0.1:${PROXY_PORT}"
# Bind the proxy to loopback only. The populator proxy is talked to
# exclusively by the pytest run on the same host (the health check and
# the test env set `LITELLM_PROXY_URL=http://127.0.0.1:...`),
# so there's no reason to expose it on the VM's external interfaces.
# Without `--host`, `litellm` defaults to 0.0.0.0, which combined with
# the predictable default `LITELLM_MASTER_KEY=sk-cron-matrix` would
# allow anything that can reach :${PROXY_PORT} on the VM to authenticate
# and burn upstream provider credentials.
#
# `setsid` puts the proxy in its own session+pgroup so cleanup() can
# SIGTERM the whole tree by passing the pgid as a negative pid. We
# write that pid to a file so cleanup() doesn't need to remember a
# variable that might be stale by the time the trap fires.
setsid env LITELLM_MASTER_KEY="${PROXY_API_KEY}" bash -c '
  echo "$$" > "$0"
  cd "$1"
  exec "$2" run litellm --config "$3" --host 127.0.0.1 --port "$4"
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
# The `_*_unit_tests` ignore is defensive: those harness-only trees are
# markerless (they run without a proxy) and don't feed matrix cells, so
# the cron skips them if/when they land in the suite.
PYTEST_ARGS=(
  tests/e2e/claude_code/
  "--ignore-glob=*_unit_tests*"
)
if [[ -n "${PYTEST_K}" ]]; then
  log "PYTEST_K set; narrowing to: ${PYTEST_K}"
  PYTEST_ARGS+=(-k "${PYTEST_K}")
fi

log "running pytest"
set +e
(
  cd "${WORKTREE}" \
    && LITELLM_PROXY_URL="http://127.0.0.1:${PROXY_PORT}" \
       LITELLM_MASTER_KEY="${PROXY_API_KEY}" \
       COMPAT_RESULTS_PATH="${RESULTS_JSON}" \
       "${WORKTREE_UV}" run pytest "${PYTEST_ARGS[@]}"
)
PYTEST_EXIT=$?
set -e
log "pytest exit code: ${PYTEST_EXIT} (failures become 'fail' cells, not script errors)"
# 0=green, 1=test failures (fail cells); >=2 = interrupted/internal/usage/no
# tests, i.e. a partial run whose missing cells would publish as not_tested.
[[ ${PYTEST_EXIT} -le 1 ]] \
  || die "pytest exited abnormally (${PYTEST_EXIT}); refusing to publish a partial matrix"
[[ -f "${RESULTS_JSON}" ]] || die "pytest did not produce ${RESULTS_JSON}"

# ---------------------------------------------------------------------------
# 5. Build the matrix JSON
# ---------------------------------------------------------------------------

MATRIX_JSON="${WORKDIR}/compatibility-matrix.json"
log "building ${MATRIX_JSON}"
(
  cd "${WORKTREE}" \
    && "${WORKTREE_UV}" run python "${POPULATOR_DIR}/build_matrix.py" \
       --manifest "${WORKTREE}/tests/e2e/claude_code/manifest.yaml" \
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

# Snapshot the currently-published matrix *before* we overwrite it, so the
# auto-merge gate below can diff old→new cell statuses. On the first-ever
# publish the file won't exist yet; we leave ${PUBLISHED_MATRIX} pointing
# at a path that doesn't exist and let check_regressions.py treat that as
# "no baseline → no regressions".
PUBLISHED_MATRIX="${WORKDIR}/published-matrix.json"
if [[ -f "${DOCS_TARGET_PATH}" ]]; then
  cp "${DOCS_TARGET_PATH}" "${PUBLISHED_MATRIX}"
fi

mkdir -p "$(dirname "${DOCS_TARGET_PATH}")"
cp "${MATRIX_JSON}" "${DOCS_TARGET_PATH}"
git add "${DOCS_TARGET_PATH}"

if git diff --cached --quiet; then
  log "matrix JSON unchanged from ${DOCS_BRANCH}; skipping PR"
  exit 0
fi

# --- Auto-merge regression gate --------------------------------------------
# Only auto-merge when the new matrix is improvement-or-equal: every cell
# transition is red→green, green→green, or red→red. If any cell flips
# green→red (a `pass` that became `fail`), we still open/refresh the PR but
# leave auto-merge OFF so a human reviews the regression before it lands on
# the public docs table. A pre-existing red cell (e.g. Anthropic out of API
# credits) is red→red and does NOT block, so the daily PR keeps flowing.
log "checking for green->red regressions vs the published matrix"
set +e
REGRESSION_REPORT="$(
  cd "${WORKTREE}" \
    && "${WORKTREE_UV}" run python "${POPULATOR_DIR}/check_regressions.py" \
       --old "${PUBLISHED_MATRIX}" \
       --new "${MATRIX_JSON}"
)"
REGRESSION_EXIT=$?
set -e
printf '%s\n' "${REGRESSION_REPORT}" | sed 's/^/  /' >&2
# Exit 0 = clean. Exit 3 = green→red regression(s) found. Any other code
# means the checker itself errored; fail *closed* (withhold auto-merge) so a
# bug in the gate can never silently auto-merge a regression.
if [[ ${REGRESSION_EXIT} -eq 0 ]]; then
  ALLOW_AUTOMERGE=1
elif [[ ${REGRESSION_EXIT} -eq 3 ]]; then
  ALLOW_AUTOMERGE=0
  log "WARN: green->red regression(s) detected; auto-merge will be left OFF for review"
else
  ALLOW_AUTOMERGE=0
  log "WARN: regression check errored (exit ${REGRESSION_EXIT}); withholding auto-merge to be safe"
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

# Push the branch straight to BerriAI/litellm-docs. mateo-berri has write
# access on the docs repo, so there's no fork hop: the PR is a same-repo
# branch PR. The temp remote carries the token in its URL, so we add it,
# push, then immediately remove it so the token never lingers in
# ${DOCS_CLONE}/.git/config. (${DOCS_CLONE} is also rm -rf'd by the
# cleanup trap on exit.)
#
# Plain --force (not --force-with-lease) is acceptable here: the
# compat-matrix/* branch is bot-owned, only this script ever writes to
# it, and runs are serialized by the systemd timer. --force-with-lease
# would require a fetch to populate the remote-tracking ref before each
# push and adds no safety in this single-writer setup.
PUBLISH_PUSH_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${DOCS_REPO}.git"
git remote remove publish 2>/dev/null || true
git remote add publish "${PUBLISH_PUSH_URL}"
git push --force --set-upstream publish "${BRANCH_NAME}"
git remote remove publish
unset PUBLISH_PUSH_URL

# Per-feature status table for the PR body. Reviewers triage from this.
PR_FEATURE_TABLE="$(jq -r '
  .features[] as $f
  | "- **\($f.name)**: " +
    ([ .providers[] as $p
       | "\($p)=\($f.providers[$p].status // "not_tested")"
     ] | join(", "))
' "${MATRIX_JSON}")"

# When the gate withheld auto-merge, call it out at the top of the PR body
# (with the offending cells) so a reviewer knows this PR needs a human and
# why. On the clean path this section is empty. Note `$(...)` strips the
# trailing newline, so the body below puts explicit blank lines *around*
# the placeholder rather than relying on the heredoc's own spacing.
if [[ "${ALLOW_AUTOMERGE}" != "1" ]]; then
  PR_REGRESSION_SECTION="$(cat <<EOF
> [!WARNING]
> **Auto-merge disabled:** one or more cells regressed green→red versus the
> currently-published matrix. Review the diff before merging.

\`\`\`
${REGRESSION_REPORT}
\`\`\`
EOF
)"
else
  PR_REGRESSION_SECTION=""
fi

PR_TITLE="chore(compat-matrix): refresh for ${LITELLM_VERSION} + claude-code ${CLAUDE_CODE_VERSION}"
PR_BODY="$(cat <<EOF
Automated daily refresh of the Claude Code compatibility matrix.

${PR_REGRESSION_SECTION}

| Field | Value |
| --- | --- |
| litellm_version | \`${LITELLM_VERSION}\` |
| claude_code_version | \`${CLAUDE_CODE_VERSION}\` |
| generated_at | \`${GENERATED_AT}\` |

## Per-feature results

${PR_FEATURE_TABLE}

---

Generated by \`tests/e2e/claude_code/cron_vm/run_daily.sh\`. Close without merging if the diff looks wrong; the next cron run will reopen with fresh results.
EOF
)"

log "opening PR from ${BRANCH_NAME} -> ${DOCS_REPO}:${DOCS_BRANCH} (as mateo-berri)"
# GH_TOKEN is mateo-berri's write-scoped token, the same identity used
# for release-listing above. The branch lives on ${DOCS_REPO} itself, so
# --head is a bare branch name (a same-repo PR), not `OWNER:BRANCH`.
set +e
PR_OUT="$(
  GH_TOKEN="${GITHUB_TOKEN}" gh pr create \
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

# Enable auto-merge so the PR merges itself once the docs repo's required
# checks pass -- we no longer gate these bot PRs on a second human
# approval. mateo-berri authors and merges them directly. The repo only
# permits squash merges and has auto-merge enabled at the repo level
# (${AUTO_MERGE_METHOD} defaults to squash accordingly).
#
# This only fires when the regression gate above is satisfied
# (${ALLOW_AUTOMERGE}==1): a green→red regression — or a gate error —
# leaves auto-merge OFF so a human triages the PR.
#
# `gh pr merge --auto` is idempotent: re-enabling auto-merge on a PR that
# already has it set is a no-op, so same-day reruns stay clean. It's
# non-fatal: if auto-merge can't be enabled (e.g. the PR is already in a
# clean/mergeable state with nothing left to wait on, or branch
# protection isn't configured), the matrix JSON has still landed on the
# PR and the worst case is a manual merge click.
if [[ "${ALLOW_AUTOMERGE}" == "1" ]]; then
  log "enabling ${AUTO_MERGE_METHOD} auto-merge on ${BRANCH_NAME}"
  set +e
  GH_TOKEN="${GITHUB_TOKEN}" gh pr merge \
    "${BRANCH_NAME}" \
    --repo "${DOCS_REPO}" \
    --auto \
    "--${AUTO_MERGE_METHOD}" 2>&1 | sed 's/^/  /'
  AUTOMERGE_EXIT=${PIPESTATUS[0]}
  set -e
  if [[ ${AUTOMERGE_EXIT} -ne 0 ]]; then
    log "WARN: gh pr merge --auto exited ${AUTOMERGE_EXIT} (non-fatal)"
  fi
else
  # Regression (or gate error): make sure auto-merge is OFF. A same-day
  # rerun may have enabled it on an earlier, clean pass, so explicitly
  # disable rather than just skipping. The disable call itself is allowed
  # to error (`--disable-auto` fails harmlessly when auto-merge was never
  # enabled), but the read-back below is authoritative: a regressed matrix
  # must never be left armed to merge, so a still-armed PR is fatal.
  log "leaving ${BRANCH_NAME} for manual review; disabling any prior auto-merge"
  set +e
  GH_TOKEN="${GITHUB_TOKEN}" gh pr merge \
    "${BRANCH_NAME}" \
    --repo "${DOCS_REPO}" \
    --disable-auto 2>&1 | sed 's/^/  /'
  set -e
  AUTOMERGE_ARMED="$(
    GH_TOKEN="${GITHUB_TOKEN}" gh pr view \
      "${BRANCH_NAME}" \
      --repo "${DOCS_REPO}" \
      --json autoMergeRequest \
      --jq '.autoMergeRequest.enabledAt // empty'
  )" || die "could not read back the auto-merge state on ${BRANCH_NAME}"
  [[ -z "${AUTOMERGE_ARMED}" ]] \
    || die "auto-merge still armed on ${BRANCH_NAME} (enabled ${AUTOMERGE_ARMED}) after --disable-auto"
fi

# --- Stale-PR sweep ----------------------------------------------------------
# Keep at most ONE compat-matrix PR open: today's. Any other open
# `compat-matrix/*` PR is a leftover from a day whose regression gate
# withheld auto-merge and nobody triaged it; the PR we just opened or
# refreshed above carries strictly fresher results, so the old one is
# pure queue noise. Closing is non-destructive — the PR record and its
# regression report stay browsable; only the bot-owned branch is
# deleted. This runs only after today's PR exists (a `die` above skips
# it), so a failed publish can never close the queue down to zero.
#
# Non-fatal: a sweep failure (rate limit, transient API error) leaves
# stale PRs for the next run to retry; it must not fail the pipeline.
log "sweeping stale compat-matrix PRs (keeping ${BRANCH_NAME})"
set +e
STALE_PRS="$(
  GH_TOKEN="${GITHUB_TOKEN}" gh pr list \
    --repo "${DOCS_REPO}" \
    --state open \
    --limit 100 \
    --json number,headRefName \
    --jq '.[] | select(.headRefName | startswith("compat-matrix/")) | "\(.number)\t\(.headRefName)"'
)"
while IFS=$'\t' read -r stale_pr stale_head; do
  [[ -z "${stale_pr}" ]] && continue
  [[ "${stale_head}" == "${BRANCH_NAME}" ]] && continue
  GH_TOKEN="${GITHUB_TOKEN}" gh pr close "${stale_pr}" \
    --repo "${DOCS_REPO}" \
    --delete-branch \
    --comment "Superseded by the newer daily compat-matrix PR from \`${BRANCH_NAME}\`; the populator keeps only the most recent compat-matrix PR open." 2>&1 | sed 's/^/  /'
  if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    log "closed stale compat-matrix PR #${stale_pr} (${stale_head})"
  else
    log "WARN: could not close stale compat-matrix PR #${stale_pr} (non-fatal)"
  fi
done <<<"${STALE_PRS}"
set -e

log "done"
