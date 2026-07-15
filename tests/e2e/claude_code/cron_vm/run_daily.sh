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
#   4. Run `pytest tests/e2e/claude_code/` against the proxy. Test failures
#      become `fail` cells in the JSON, not script errors.
#   5. Hand the per-test results artifact + manifest to a small Python
#      CLI (`build_matrix.py`) that wraps the existing
#      `matrix_builder.build_from_paths` to produce the published
#      compatibility-matrix.json.
#   6. `gh repo clone` litellm-docs, write the JSON to a deterministic
#      branch (`compat-matrix/<litellm>-<claude>-<UTC-date>`), commit,
#      `git push --force`, and `gh pr create`.
#
# Same-day reruns land on the same branch so they update the existing PR
# rather than spawning a new one. If the JSON is byte-identical to the
# docs branch, we skip the push entirely.
#
# Required commands on $PATH: git, uv, gh, jq, curl, claude.
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
# Comma-separated GitHub usernames to request a review from on every PR.
# Reviewers must have at least read access to ${DOCS_REPO}. PR-author
# (agent-shin) has implicit rights to request reviews from anyone with
# read access, so no extra token scope is needed. Set to empty to skip.
PR_REVIEWERS="${PR_REVIEWERS:-mateo-berri}"

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

# Publishing is from a fork (agent-shin/litellm-docs) so neither the cron
# host nor the bot identity needs write access to BerriAI/litellm-docs. We
# require the fork token up front -- failing 30 minutes into a run because
# the env file is missing one line is a waste of CI quota.
if [[ "${SKIP_PUBLISH}" != "1" ]]; then
  [[ -n "${AGENT_SHIN_GITHUB_TOKEN:-}" ]] \
    || die "AGENT_SHIN_GITHUB_TOKEN required to open PRs from agent-shin/litellm-docs (or set SKIP_PUBLISH=1)"
fi

# ---------------------------------------------------------------------------
# 1. Resolve versions
# ---------------------------------------------------------------------------

# Newest v*-stable release on BerriAI/litellm. The `select(...)` filter
# drops drafts/non-stable, the version_key sort handles 1.10 > 1.9.
#
# Paginate through the releases endpoint instead of grabbing only page 1
# (default page_size=30). LiteLLM ships multiple non-stable releases per
# day, so it's common to need to walk past 30+ entries before hitting
# the most recent v*-stable. We cap at 5 pages (500 releases) which is
# conservatively beyond the worst observed gap.
#
# We deliberately do NOT short-circuit on the first page that contains a
# v*-stable tag. The /releases endpoint orders by `created_at`, not by
# semver, so a backport on an older series (e.g. v1.80.1-stable cut
# today) can show up on an earlier page than a higher-versioned release
# (v1.83.0-stable cut two weeks ago). Breaking early on first-stable-seen
# would silently pin the cron to the stale tag because the
# higher-versioned release still on a later page would never make it
# into the merged set the `sort_by` below consumes. The only break we
# keep is the empty-page guard, which means a quiet period in the
# release feed doesn't waste API quota — we just always walk far enough
# to be confident we've seen the highest stable tag.
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
  # No more pages? GitHub returns an empty array past the last page.
  if [[ "$(jq 'length' "${PAGE_JSON}")" == "0" ]]; then
    break
  fi
done
LITELLM_VERSION="$(
  jq -r '
    [ .[] | .tag_name // empty
      | select(test("^v[0-9]+\\.[0-9]+\\.[0-9]+-stable$"))
    ]
    | sort_by(
        capture("^v(?<a>[0-9]+)\\.(?<b>[0-9]+)\\.(?<c>[0-9]+)-stable$")
        | [(.a|tonumber), (.b|tonumber), (.c|tonumber)]
      )
    | last // empty
  ' "${RELEASES_JSON}"
)"
[[ -n "${LITELLM_VERSION}" ]] || die "could not resolve latest v*-stable tag in 5 pages of releases"
log "resolved litellm: ${LITELLM_VERSION}"

# The systemd unit loads provider credentials and the agent-shin GitHub
# token from /etc/litellm-compat-matrix.env into this script's
# environment. Running the npm-installed `claude` binary directly here
# would hand that full env to package code -- a compromised
# @anthropic-ai/claude-code release could read ANTHROPIC_API_KEY /
# AWS_BEARER_TOKEN_BEDROCK / AZURE_AI_API_KEY /
# AGENT_SHIN_GITHUB_TOKEN from os.environ and exfiltrate them before
# the proxy or test harness ever starts. Probe under `env -i` with the
# same minimal allowlist the PR-gate uses (the matrix run itself goes
# through cli_driver.py, which already scrubs the CLI env).
#
# The probe also runs under a fresh empty HOME instead of the runtime
# user's real $HOME. `ProtectHome=read-only` in the systemd unit
# blocks *writes* to /home/mateo but still allows reads, so a
# compromised claude package invoked here with HOME=/home/mateo could
# read ~/.config/gh/hosts.yml (the gh-host token), ~/.bash_history,
# or ~/.ssh/. Pointing HOME at a per-run dir under ${WORKDIR} hides
# those entirely from the subprocess; ${WORKDIR} is rm -rf'd by the
# script-wide cleanup() trap regardless of probe outcome.
CLAUDE_PROBE_HOME="${WORKDIR}/claude-probe-home"
mkdir -p "${CLAUDE_PROBE_HOME}"
CLAUDE_CODE_VERSION="$(env -i \
  PATH="${PATH}" \
  HOME="${CLAUDE_PROBE_HOME}" \
  USER="${USER:-mateo}" \
  TERM="${TERM:-dumb}" \
  LANG="${LANG:-C.UTF-8}" \
  LC_ALL="${LC_ALL:-}" \
  TMPDIR="${TMPDIR:-/tmp}" \
  claude --version 2>/dev/null \
  | grep -oE '[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?' \
  | head -n1 || true)"
# `|| true` above keeps `set -Eeuo pipefail` from aborting silently when
# `grep` finds no match (exit 1) — without it the assignment inherits the
# pipeline's non-zero exit, `set -e` kills the script, and the operator
# never sees the helpful diagnostic below.
[[ -n "${CLAUDE_CODE_VERSION}" ]] || die "could not parse semver from 'claude --version'"
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
# the venv on every run, and we don't want to re-download the pinned
# uv binary each time. Drop everything else (including any prior
# tests/e2e/claude_code/ shim) so each run starts clean before the shim
# below rewrites it from the dev checkout.
git -C "${WORKTREE}" clean -fdx -e .venv -e .uv-bin
git -C "${WORKTREE}" checkout --force "${LITELLM_VERSION}"

# Always overwrite tests/e2e/claude_code/ in the worktree with the copy
# from the dev checkout, regardless of whether the resolved
# ${LITELLM_VERSION} tag already ships a tests/e2e/claude_code/ tree of
# its own. Rationale: the matrix populator's job is to exercise
# today's tests against the latest stable proxy. The dev checkout
# carries the most recent test fixes (e.g. the stream-json vision
# rewrite, the --effort thinking knob, the WebSearch tool_use
# assertion) that haven't yet rolled into a v*-stable, and we want
# every cron run to pick those up the moment they land on
# ${LITELLM_REPO}, not whenever the next stable release happens.
#
# Concretely this means a fresh `rm -rf` + `cp -r` every run so the
# tree is byte-identical to ${LITELLM_REPO}/tests/e2e/claude_code (no
# stale files left over from the tag's own checkout, no drift across
# runs).
if [[ ! -d "${LITELLM_REPO}/tests/e2e/claude_code" ]]; then
  die "no shim source at ${LITELLM_REPO}/tests/e2e/claude_code"
fi
log "shimming tests/e2e/claude_code/ from ${LITELLM_REPO} (always-overwrite)"
rm -rf "${WORKTREE}/tests/e2e/claude_code"
mkdir -p "${WORKTREE}/tests/e2e"
cp -r "${LITELLM_REPO}/tests/e2e/claude_code" "${WORKTREE}/tests/e2e/"

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
    # Detect host arch so the same script works on x86_64 GCP VMs and on
    # aarch64 hosts (Astral publishes both `uv-x86_64-unknown-linux-gnu`
    # and `uv-aarch64-unknown-linux-gnu` tarballs under the same release
    # tag, and `uname -m` already returns the exact token uv uses).
    UV_ARCH="$(uname -m)"
    UV_TRIPLE="uv-${UV_ARCH}-unknown-linux-gnu"
    UV_TARBALL_NAME="${UV_TRIPLE}.tar.gz"
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
    tar -xzf "${UV_TMPDIR}/${UV_TARBALL_NAME}" -C "${UV_TMPDIR}" "${UV_TRIPLE}/uv"
    mv "${UV_TMPDIR}/${UV_TRIPLE}/uv" "${WORKTREE_UV}.tmp"
    chmod +x "${WORKTREE_UV}.tmp"
    mv "${WORKTREE_UV}.tmp" "${WORKTREE_UV}"
    rm -rf "${UV_TMPDIR}"
  fi
fi
# `--extra proxy` pulls fastapi/uvicorn/etc. so `uv run litellm` can
# actually serve. `--group proxy-dev` brings in pytest and the rest of
# what tests/e2e/claude_code/ needs.
log "uv sync --frozen --group proxy-dev --extra proxy (uv ${PINNED_UV_VERSION:-system})"
(cd "${WORKTREE}" && "${WORKTREE_UV}" sync --frozen --group proxy-dev --extra proxy)

PROXY_CONFIG="${WORKTREE}/tests/e2e/claude_code/test_config.yaml"
[[ -f "${PROXY_CONFIG}" ]] || die "proxy config not found at ${PROXY_CONFIG} (does ${LITELLM_VERSION} predate the compat matrix work?)"

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
#
# Pass the master key as a shell-prefix assignment on `setsid` (inherited
# via the environment) rather than as `env KEY=VAL ...` argv. The argv
# form would land the literal key in /proc/<setsid-pid>/cmdline, where
# any local reader (a model-directed `Read` tool call, another user on
# the VM, a crash dump) could pick it up before the process execs into
# the litellm child. The shell-prefix form keeps the key out of argv at
# every layer (setsid → bash → uv → litellm).
LITELLM_MASTER_KEY="${PROXY_API_KEY}" setsid bash -c '
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
PYTEST_ARGS=(
  tests/e2e/claude_code/
  --ignore=tests/e2e/claude_code/_driver_unit_tests
  --ignore=tests/e2e/claude_code/_builder_unit_tests
  --ignore=tests/e2e/claude_code/_publisher_unit_tests
  --ignore=tests/e2e/claude_code/_pr_gate_unit_tests
)
if [[ -n "${PYTEST_K}" ]]; then
  log "PYTEST_K set; narrowing to: ${PYTEST_K}"
  PYTEST_ARGS+=(-k "${PYTEST_K}")
fi

log "running pytest"
set +e
# Pytest only needs to talk to the loopback proxy at 127.0.0.1:${PROXY_PORT}
# — it has no legitimate reason to see ANTHROPIC_API_KEY /
# AWS_BEARER_TOKEN_BEDROCK / VERTEXAI_* / AZURE_AI_* /
# AGENT_SHIN_GITHUB_TOKEN / GITHUB_TOKEN in its own env. The systemd
# unit's EnvironmentFile injects all of those into this script for the
# proxy to consume, and pytest inherits them by default. Wrap the
# invocation in `env -i` so:
#
#   1. test code under tests/e2e/claude_code/ (or anything it imports)
#      cannot read provider/agent-shin creds out of `os.environ` and
#      exfiltrate them via an outbound call from inside a conftest hook
#      or a fixture (a sibling vector to the model-controlled Bash/Read
#      concern handled by `cli_driver.py`'s own env scrub);
#   2. a model-directed `Read` tool call during a PDF/vision cell
#      cannot reach /proc/<pytest-pid>/environ and pull the creds out
#      of the parent process the way it can today;
#   3. this matches the PR-gate pytest step in `.circleci/config.yml`,
#      which already runs under `env -i` with the same minimal
#      allowlist.
#
# `cli_driver.py` re-allowlists its own subset (PATH/USER/LOGNAME/etc.)
# when spawning the `claude` binary, so the CLI still finds Node + the
# claude shim on PATH and gets a fresh isolated HOME per invocation.
(
  cd "${WORKTREE}" \
    && env -i \
       PATH="${PATH}" \
       HOME="${HOME}" \
       USER="${USER:-mateo}" \
       TERM="${TERM:-dumb}" \
       LANG="${LANG:-C.UTF-8}" \
       LC_ALL="${LC_ALL:-}" \
       TMPDIR="${TMPDIR:-/tmp}" \
       LITELLM_PROXY_URL="http://127.0.0.1:${PROXY_PORT}" \
       LITELLM_MASTER_KEY="${PROXY_API_KEY}" \
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
FORK_OWNER="${FORK_OWNER:-agent-shin}"
FORK_REPO="${FORK_REPO:-${FORK_OWNER}/litellm-docs}"

log "cloning ${DOCS_REPO}@${DOCS_BRANCH}"
# Use the agent-shin token inline rather than the host gh-cli config.
# `BerriAI/litellm-docs` is a public repo so unauthenticated clone
# would also work, but passing the token explicitly means the systemd
# unit can hide `~/.config/gh` (`InaccessiblePaths=`) without breaking
# this clone — closing the model-directed `Read("/home/mateo/.config/gh/...")`
# exfiltration path on the cron VM.
GH_TOKEN="${AGENT_SHIN_GITHUB_TOKEN}" \
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

# Push to the fork (agent-shin/litellm-docs), not to BerriAI/litellm-docs.
# The cron host has no write access to BerriAI/litellm-docs by design --
# only agent-shin's PAT does, and only over its own fork. The temp remote
# carries the token in its URL, so we add it, push, then immediately
# remove it so the token never lingers in ${DOCS_CLONE}/.git/config.
# (${DOCS_CLONE} is also rm -rf'd by the cleanup trap on exit.)
#
# Plain --force (not --force-with-lease) is acceptable here: the fork
# branch is bot-owned, only this script ever writes to it, and runs are
# serialized by the systemd timer. --force-with-lease would require a
# fetch to populate the remote-tracking ref before each push and adds
# no safety in this single-writer setup.
FORK_PUSH_URL="https://x-access-token:${AGENT_SHIN_GITHUB_TOKEN}@github.com/${FORK_REPO}.git"
git remote remove fork 2>/dev/null || true
git remote add fork "${FORK_PUSH_URL}"
git push --force --set-upstream fork "${BRANCH_NAME}"
git remote remove fork
unset FORK_PUSH_URL

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

Generated by \`tests/e2e/claude_code/cron_vm/run_daily.sh\`. Close without merging if the diff looks wrong; the next cron run will reopen with fresh results.
EOF
)"

log "opening PR from ${FORK_OWNER}:${BRANCH_NAME} -> ${DOCS_REPO}:${DOCS_BRANCH}"
# GH_TOKEN here is scoped to this single subshell so we don't bleed the
# fork token into the rest of the script (release-listing earlier uses
# ${GITHUB_TOKEN}, which may be a different identity). gh's --head accepts
# `OWNER:BRANCH` for cross-repo PRs from a fork.
#
# Reviewer assignment is done in a *separate* call below: as the PR
# author from a fork, agent-shin has no write/triage access on
# ${DOCS_REPO} and the `RequestReviewsByLogin` GraphQL mutation
# (which backs `gh pr create --reviewer` and `gh pr edit --add-reviewer`)
# rejects with "does not have the correct permissions". We use the
# collaborator-scoped ${GITHUB_TOKEN} for that instead. Don't fold
# --reviewer into `gh pr create` here -- it would fail the whole
# create on the very first cron run.
set +e
PR_OUT="$(
  GH_TOKEN="${AGENT_SHIN_GITHUB_TOKEN}" gh pr create \
    --repo "${DOCS_REPO}" \
    --base "${DOCS_BRANCH}" \
    --head "${FORK_OWNER}:${BRANCH_NAME}" \
    --title "${PR_TITLE}" \
    --body "${PR_BODY}" 2>&1
)"
PR_EXIT=$?
set -e
echo "${PR_OUT}"

if [[ ${PR_EXIT} -ne 0 ]]; then
  if grep -q "a pull request for branch.*already exists" <<<"${PR_OUT}"; then
    log "PR already exists for ${FORK_OWNER}:${BRANCH_NAME}; updated branch in place"
  else
    die "gh pr create failed (exit ${PR_EXIT})"
  fi
fi

# Request reviews from PR_REVIEWERS using the collaborator-scoped
# ${GITHUB_TOKEN} (mateo-berri's token, already provisioned for release
# listing). This is idempotent: `gh pr edit --add-reviewer` is a no-op
# on a user who's already in reviewRequests, and silently re-adds
# anyone whose prior review was dismissed -- so same-day reruns stay
# clean. Reviewer-add failures are non-fatal: the matrix JSON has
# already landed on the PR; the worst case is a manual ping.
if [[ -n "${PR_REVIEWERS}" ]]; then
  if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    log "WARN: PR_REVIEWERS set but GITHUB_TOKEN missing -- cannot request reviews; skipping"
  else
    log "requesting reviews from: ${PR_REVIEWERS}"
    set +e
    GH_TOKEN="${GITHUB_TOKEN}" gh pr edit \
      "${FORK_OWNER}:${BRANCH_NAME}" \
      --repo "${DOCS_REPO}" \
      --add-reviewer "${PR_REVIEWERS}" 2>&1 | sed 's/^/  /'
    REVIEWER_EXIT=${PIPESTATUS[0]}
    set -e
    if [[ ${REVIEWER_EXIT} -ne 0 ]]; then
      log "WARN: gh pr edit --add-reviewer exited ${REVIEWER_EXIT} (non-fatal)"
    fi
  fi
fi

log "done"
