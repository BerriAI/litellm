#!/usr/bin/env bash
#
# pre_commit_lint.sh — shift CI lint left. Run it (via `make pre-commit`) right
# before `git commit`; it inspects your staged files and runs only the matching
# gating CI checks, so a clean run means a green CI lint:
#   - litellm/ Python staged -> `make lint` (test-linting.yml's lint job)
#   - tests/e2e Python staged -> `make lint-e2e-basedpyright` (test-linting.yml's e2e type-check step)
#   - dashboard staged        -> prettier + eslint + lint budgets (test-litellm-ui-build.yml's frontend-lint)
#   - proxy/types staged      -> regenerate dashboard API types and fail on drift (check-ui-api-types.yml)
#
# Each block is skipped when no matching files are staged, so unrelated commits stay
# fast. This is intentionally not auto-installed as a git hook (see scripts/install_git_hooks.sh):
# the dashboard and basedpyright passes can take minutes, so it's run on demand rather
# than firing on every human commit. It is hook-compatible if you want that anyway:
# `ln -s ../../scripts/pre_commit_lint.sh .git/hooks/pre-commit`.

set -eu

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

staged=$(git diff --cached --name-only --diff-filter=ACMR)
staged_match() { printf '%s\n' "$staged" | grep -E "$1" || true; }

# CI's lint job (test-linting.yml) only inspects litellm/, so a tests-only or
# scripts-only commit can't turn it red; scope the trigger there to skip the slow
# make lint when it couldn't catch anything.
litellm_py_files=$(staged_match '^litellm/.*\.py$')
e2e_py_files=$(staged_match '^tests/e2e/.*\.py$')
# ruff format (and CI's format step) skip enterprise; the rest of make lint covers it.
fmt_files=$(printf '%s\n' "$litellm_py_files" | grep -v '^litellm/enterprise/' || true)
# check-ui-api-types.yml triggers on any file under litellm/proxy or litellm/types
# (Prisma schema and configs included, not just Python) plus the generator and its
# lockfiles, so match that whole trigger set rather than a Python subset.
spec_files=$(staged_match '^(litellm/(proxy|types)/.*|ui/litellm-dashboard/(scripts/gen-api-types\.mjs|package\.json|package-lock\.json|src/lib/http/schema\.d\.ts))$')
# CI's frontend-lint runs prettier over a wider extension set than eslint; keep that
# split so this flags exactly what the job would.
ui_prettier_files=$(staged_match '^ui/litellm-dashboard/.*\.(js|jsx|ts|tsx|mjs|cjs|json|css|scss|md|mdx|yml|yaml|html)$')
ui_eslint_files=$(staged_match '^ui/litellm-dashboard/.*\.(js|jsx|ts|tsx|mjs|cjs)$')

# CI lints the committed tree, so this script predicts CI for what you have STAGED
# (every trigger above reads `git diff --cached`). The tools it runs, though, read
# the working tree, so unstaged edits to tracked files and untracked files fold
# into the result and a green/red here won't match a commit of just the staged
# changes. There's no safe way to lint the index in place, so surface the gap
# instead of hiding it: stage everything you intend to commit before trusting a
# pass. This only warns; it never blocks or touches your changes.
unstaged=$(git diff --name-only)
untracked=$(git ls-files --others --exclude-standard)
if [ -n "$unstaged" ] || [ -n "$untracked" ]; then
    echo "pre-commit: NOTE - unstaged/untracked changes are included in these checks but" >&2
    echo "  won't be in a commit of only your staged changes, so this result may differ from" >&2
    echo "  CI. Stage everything you intend to commit (git add) for an accurate prediction:" >&2
    printf '%s\n' "$unstaged" "$untracked" | sed '/^$/d' | sed 's/^/    /' >&2
fi

lint_dashboard() {
    (
        rc=0
        prettier_rel=()
        eslint_rel=()
        while IFS= read -r f; do
            [ -n "$f" ] && prettier_rel+=("${f#ui/litellm-dashboard/}")
        done <<EOF
$ui_prettier_files
EOF
        while IFS= read -r f; do
            [ -n "$f" ] && eslint_rel+=("${f#ui/litellm-dashboard/}")
        done <<EOF
$ui_eslint_files
EOF
        cd ui/litellm-dashboard
        if [ ${#prettier_rel[@]} -gt 0 ]; then
            npx prettier --check "${prettier_rel[@]}" || rc=1
        fi
        if [ ${#eslint_rel[@]} -gt 0 ]; then
            npx eslint --no-warn-ignored --pass-on-unpruned-suppressions "${eslint_rel[@]}" || rc=1
        fi
        # Whole-folder lint budgets, exactly as the frontend-lint job runs them: the
        # counts are not diff-scoped, so a local pass here means the budget step will
        # pass in CI too.
        report=$(mktemp)
        npx eslint . -f json -o "$report" || true
        node scripts/check-lint-budgets.mjs "$report" eslint-budgets.json || rc=1
        rm -f "$report"
        exit $rc
    )
}

status=0

bootstrap_hint() {
    echo "  This checkout looks unprovisioned (fresh worktree or clone)." >&2
    echo "  Fix: make bootstrap" >&2
}

if [ -n "$litellm_py_files" ]; then
    echo "pre-commit: linting Python (make lint)"
    make lint || { echo "✗ Python lint failed. Fix the reds above, then re-run make pre-commit." >&2; status=1; }
    # `make lint` format-checks files in origin/base...HEAD, which at pre-commit time
    # predates the staged change, so format-check the staged litellm files directly to
    # cover a brand-new commit before it lands.
    if [ -n "$fmt_files" ]; then
        echo "pre-commit: ruff format --check (staged litellm files)"
        printf '%s\n' "$fmt_files" | xargs uv run --no-sync ruff format --check --exclude '/enterprise/' \
            || { echo "✗ Unformatted staged files. Fix with: make format, then re-stage." >&2; status=1; }
    fi
fi

if [ -n "$e2e_py_files" ] && [ -z "$litellm_py_files" ]; then
    echo "pre-commit: type-checking tests/e2e (make lint-e2e-basedpyright)"
    make lint-e2e-basedpyright || { echo "✗ tests/e2e basedpyright failed. Fix the errors above, then re-run make pre-commit." >&2; status=1; }
fi

if [ -n "$ui_prettier_files" ] || [ -n "$ui_eslint_files" ]; then
    echo "pre-commit: linting dashboard (prettier + eslint + lint budgets)"
    if [ ! -d ui/litellm-dashboard/node_modules ]; then
        echo "✗ ui/litellm-dashboard/node_modules is missing; dashboard lint cannot run." >&2
        bootstrap_hint
        status=1
    else
        lint_dashboard || { echo "✗ Dashboard lint failed. See above; format with: (cd ui/litellm-dashboard && npm run format)." >&2; status=1; }
    fi
fi

if [ -n "$spec_files" ]; then
    echo "pre-commit: checking dashboard API types are in sync (npm run gen:api)"
    # gen-api-types.mjs imports litellm.proxy.proxy_server, which needs the proxy deps
    # and an up-to-date Prisma client; check-ui-api-types.yml installs those and runs
    # prisma generate before gen:api, so mirror that here or a stale client can mask
    # drift that CI will still flag.
    if [ ! -d ui/litellm-dashboard/node_modules ]; then
        echo "✗ ui/litellm-dashboard/node_modules is missing; the gen:api sync check cannot run." >&2
        bootstrap_hint
        status=1
    elif ! uv run --no-sync python -c "import orjson, prisma" 2>/dev/null; then
        echo "✗ The Python env lacks the proxy deps (orjson/prisma) that gen:api needs." >&2
        bootstrap_hint
        status=1
    elif ! uv run --no-sync python scripts/prisma_generate_if_needed.py; then
        echo "✗ Could not regenerate Prisma client (prisma generate failed)." >&2
        status=1
    elif ( cd ui/litellm-dashboard && LITELLM_PYTHON="uv run --no-sync python" npm run gen:api ); then
        if ! git diff --quiet -- ui/litellm-dashboard/src/lib/http/schema.d.ts; then
            echo "✗ Dashboard API types are stale; regenerated src/lib/http/schema.d.ts. Stage it and re-run make pre-commit." >&2
            status=1
        fi
    else
        echo "✗ Could not regenerate API types (npm run gen:api failed)." >&2
        status=1
    fi
fi

exit $status
