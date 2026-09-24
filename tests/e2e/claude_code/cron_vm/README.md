# Render cron job for the Claude Code compatibility-matrix populator

The populator runs daily as the Render cron job `litellm-compat-matrix`
(Docker runtime, built from the `Dockerfile` in this directory) rather
than as a GitHub Action or on a dedicated VM. Trade-offs:

- ✅ No machine to keep on or patch. Render builds the image from this
  directory on every push to `main` that touches `tests/e2e/**` and
  runs it on the schedule.
- ✅ Credentials live in Render env vars and secret files, scoped to
  this one service, instead of on a VM filesystem.
- ✅ The publish token still uses the `mateo-berri` account, which is a
  collaborator on `BerriAI/litellm-docs`, so no GitHub App with
  `pull-requests: write` has to be provisioned.
- ⚠️ The disk is ephemeral, so every run starts from a fresh (blobless)
  clone of litellm plus a cold `uv sync`. That adds a few minutes on
  top of the ~10 minute test run; the job's 12 hour ceiling is nowhere
  near.
- ⚠️ The Claude Code CLI version under test is pinned in the
  `Dockerfile` (`CLAUDE_CODE_VERSION` + its checksum). Bumping it is a
  PR, see the gotchas below.

## Layout

| File | Purpose |
| --- | --- |
| `Dockerfile` | The image Render builds: Debian bookworm-slim plus pinned, checksum-verified `gh`, `uv`, and the Claude Code CLI, with this `tests/e2e/` tree copied to `/opt/litellm/tests/e2e/`. Runs as the non-root user `populator` (uid/gid 1000, which is what Render's secret files are readable by). |
| `run_daily.sh` | The actual cron job. Resolves versions, clones the worktree, boots the proxy, runs pytest, builds the JSON, opens (or updates) a docs PR, sweeps stale compat-matrix PRs. |
| `build_matrix.py` | Tiny Python CLI that wraps `claude_code.matrix_builder.build_from_paths`. Exists only because the bash script needs *some* way to render the per-cell aggregation, and the builder is already Python. |
| `check_regressions.py` | Tiny Python CLI that wraps `claude_code.matrix_builder.find_regressions`. Diffs the freshly built matrix against the currently-published one and exits `3` if any cell flipped green→red, which gates auto-merge. |
| `litellm-compat-matrix.env.example` | The service's env vars, one per line, with what each is for. |

## What `run_daily.sh` does

1. **Resolves the latest LiteLLM final release tag** (newest bare
   `vX.Y.Z`, skipping `-rc.N`/`-dev.N` pre-releases) by paging the
   GitHub Releases API (`curl | jq`).
2. **Reads the Claude Code CLI version** via `claude --version`. That
   is whatever the `Dockerfile` pins; the job never upgrades it on its
   own.
3. **Clones the worktree** at `~/litellm-cron-worktree/` (a
   `--filter=blob:none` clone, so only the checked-out tag's blobs are
   fetched), `git checkout --force <tag>`, then `uv sync --frozen
   --no-install-project` against a uv-managed CPython 3.12 followed by
   `uv pip install --no-build litellm==<version>`, so the proxy under
   test is the published PyPI wheel (what users install) rather than a
   source build: the tag builds a Rust extension through maturin, and
   the image ships no C or Rust toolchain. Then **shims the test suite**:
   `tests/e2e/` in the worktree is replaced by the image's copy of this
   whole tree, so the cron always runs *today's* tests against the
   latest stable proxy, and pytest runs with `--confcutdir` pointed at
   `claude_code/` so the tree's EKS-harness `conftest.py` (whose imports
   the stable venv doesn't install) is never loaded. The tag's own
   `tests/e2e/` is deliberately not used.
4. **Boots the proxy** as a `setsid` background process on port `4100`
   bound to loopback, then polls `/health/liveliness` until it's up.
5. **Runs pytest** on `tests/e2e/claude_code/` with `LITELLM_PROXY_URL`
   pointed at the proxy and `COMPAT_RESULTS_PATH` set so the conftest
   hook writes the per-test results artifact. Test failures become
   `fail` cells in the JSON, not script errors.
6. **Builds `compatibility-matrix.json`** by handing the artifact +
   manifest to `build_matrix.py`.
7. **Opens or updates a docs PR**: `gh repo clone` of `litellm-docs`
   into a tempdir, deterministic head branch
   (`compat-matrix/<litellm-version>-<claude-code-version>-<UTC-date>`),
   `--force` push **directly to `BerriAI/litellm-docs`** (the
   `mateo-berri` token has write access, so this is a same-repo branch,
   not a fork), `gh pr create`. A re-run on the same day fast-forwards
   the existing branch and `gh pr create` no-ops ("a pull request for
   branch ... already exists" is treated as success). If the JSON is
   byte-identical to what `main` already publishes, the push is skipped
   entirely. These PRs are not gated on a second human review.
8. **Gates auto-merge on a regression check**: before enabling
   auto-merge, `check_regressions.py` diffs the new matrix against the
   one currently on `main`. Auto-merge (`gh pr merge --auto --squash`)
   is only enabled when **no cell flipped green→red** — i.e. every
   transition is red→green, green→green, or red→red. A pre-existing red
   cell (e.g. a provider that's out of API credits) is `red→red` and
   does **not** block; only a `pass`→`fail` flip does. When a regression
   is detected the PR is still opened/updated (with a warning banner
   naming the offending cells) but auto-merge is left **off** — and any
   auto-merge a prior same-day run enabled is explicitly disabled — so a
   human reviews before it lands on the public table. The check fails
   *closed*: if it errors, auto-merge is withheld.
9. **Sweeps stale compat-matrix PRs**: once today's PR exists, every
   other open `compat-matrix/*` PR that the publishing account opened
   from a branch on the docs repo itself is closed (and its bot-owned
   branch deleted), so at most one compat-matrix PR is ever open — the
   newest. A contributor's PR under that prefix is never touched.

## The Render service

Everything below is what the live service is set to; recreate it with
the same values if it ever has to be rebuilt.

| Setting | Value |
| --- | --- |
| Workspace | Litellm (the one that already builds the other litellm services) |
| Type | Cron job, Docker runtime |
| Repo / branch | `BerriAI/litellm` @ `main` |
| Dockerfile path | `tests/e2e/claude_code/cron_vm/Dockerfile` |
| Docker build context | `tests/e2e` (the repo root `.dockerignore` excludes `tests`, so the context has to start below it) |
| Build filter | included paths `tests/e2e/**` |
| Schedule | `0 6 * * *` (06:00 UTC daily) |
| Plan / region | `4c-16g` (4 CPU, 16 GB, what the dashboard calls Pro Max; the suite fans out to ~75 concurrent CLI calls) / Oregon |
| Env vars | every key in `litellm-compat-matrix.env.example` |
| Secret files | `github-token` (the publish PAT, one line) and `vertex-service-account.json` (the Vertex service-account key) |

Render mounts secret files at `/etc/secrets/<name>`, which is where
`CREDENTIALS_DIRECTORY` and `GOOGLE_APPLICATION_CREDENTIALS` in the env
example point. Render also passes env vars to `docker build` as build
args, which is why the `Dockerfile` declares no `ARG` that could ever
be given a secret's name.

Creating it through the API looks like this (fill `envVars` and
`secretFiles` from the env example and the two secrets; `ownerId` is
the workspace id from `GET /v1/owners`):

```bash
curl -fsS https://api.render.com/v1/services \
  -H "Authorization: Bearer ${RENDER_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "cron_job",
    "name": "litellm-compat-matrix",
    "ownerId": "<workspace id>",
    "repo": "https://github.com/BerriAI/litellm",
    "branch": "main",
    "autoDeploy": "yes",
    "buildFilter": {"paths": ["tests/e2e/**"], "ignoredPaths": []},
    "envVars": [{"key": "ANTHROPIC_API_KEY", "value": "..."}],
    "secretFiles": [{"name": "github-token", "content": "..."},
                    {"name": "vertex-service-account.json", "content": "..."}],
    "serviceDetails": {
      "runtime": "docker",
      "schedule": "0 6 * * *",
      "plan": "4c-16g",
      "region": "oregon",
      "envSpecificDetails": {
        "dockerfilePath": "tests/e2e/claude_code/cron_vm/Dockerfile",
        "dockerContext": "tests/e2e"
      }
    }
  }'
```

## Operating it

```bash
# Trigger a real run right now (PRs to litellm-docs). The id is the
# service id (`crn-...`) from the dashboard URL or `GET /v1/services`.
curl -fsS -X POST "https://api.render.com/v1/cron-jobs/${CRON_ID}/runs" \
  -H "Authorization: Bearer ${RENDER_API_KEY}"

# Follow a run: the Logs tab on the service, or the API.
curl -fsS "https://api.render.com/v1/logs?ownerId=${OWNER_ID}&resource=${CRON_ID}&limit=100" \
  -H "Authorization: Bearer ${RENDER_API_KEY}"

# Rebuild the image after a merge that touches tests/e2e/** (see the
# auto-deploy gotcha below). The deploy is done once its status is
# `live`; a run triggered before that still uses the previous image.
curl -fsS -X POST "https://api.render.com/v1/services/${CRON_ID}/deploys" \
  -H "Authorization: Bearer ${RENDER_API_KEY}" \
  -H 'Content-Type: application/json' -d '{"clearCache": "do_not_clear"}'
curl -fsS "https://api.render.com/v1/services/${CRON_ID}/deploys?limit=1" \
  -H "Authorization: Bearer ${RENDER_API_KEY}"

# A run that does NOT open a PR (first-time validation, CLI bumps):
# set SKIP_PUBLISH=1 on the service, trigger a run, then remove it.
# The matrix JSON is printed at the end of the run's log (nothing on
# the container's disk outlives the run) and saved to
# ~/compatibility-matrix.json for a local docker run.
# PYTEST_K='basic_messaging_non_streaming and anthropic' narrows the
# run to one cell the same way.

# Build and run the image locally (docker on Apple silicon needs the
# platform flag; the context is tests/e2e, see the table above).
docker build --platform linux/amd64 \
  -f tests/e2e/claude_code/cron_vm/Dockerfile -t compat-matrix tests/e2e
docker run --rm --platform linux/amd64 \
  --env-file litellm-compat-matrix.env -e SKIP_PUBLISH=1 \
  -v "$PWD/secrets:/etc/secrets:ro" compat-matrix
```

## Gotchas

- **The venv is pinned to Python 3.12 (`CRON_PYTHON_VERSION`).** The
  e2e suite uses PEP 695 `type` aliases, which the image's Debian
  Python can't parse; `run_daily.sh` has uv fetch a managed CPython
  into `~/litellm-cron-worktree/.uv-python/` and syncs the venv against
  it.
- **The proxy port is `4100`, not `4000`.** Kept from the VM days so a
  developer running the script locally next to their own `:4000` proxy
  doesn't collide. Override with `PROXY_PORT=...`.
- **`uv sync --frozen` requires the resolved tag to be tagged on
  GitHub, and the wheel install requires it on PyPI.** If the latest
  stable release was made but not pushed as a git tag, the `git
  checkout` step fails; push the tag, then rerun. PyPI has had every
  stable version days before its GitHub release so far (1.102.0 was
  uploaded 2026-09-20, released on GitHub 2026-09-22), so the
  `--no-build` install failing means the wheel is genuinely missing,
  not late.
- **Pushes do not redeploy the service; deploy by hand.** `autoDeploy`
  is `yes` on the service, but Render only hears about pushes through
  its GitHub app, which is not installed on the `BerriAI` org (an org
  admin step), so no push to the branch has ever started a deploy.
  After a merge that changes anything under `tests/e2e/**`, run the
  deploy command from the operating section (or "Manual Deploy" on the
  dashboard) and wait for `live` before triggering a run, otherwise
  the next scheduled run still executes the old image.
- **Publish-token rotation is your problem.** The cron does not
  refresh the token; if `mateo-berri`'s PAT in the `github-token`
  secret file expires, the run fails at the `git push`/`gh pr create`
  step with a 401 ("Bad credentials" / "Authentication failed"). Mint
  a fresh PAT and replace the secret file on the service. The token
  needs write access to `BerriAI/litellm-docs` (classic `repo` scope,
  or fine-grained Contents:RW + Pull requests:RW). It is delivered as
  a file, not an env var, so pytest, the proxy, and the claude CLI
  never inherit it; manual runs export `GITHUB_TOKEN` instead.
- **Bumping the Claude Code CLI is a PR.** Change `CLAUDE_CODE_VERSION`
  in the `Dockerfile` and set `CLAUDE_CODE_SHA256` to the `linux-x64`
  checksum from
  `https://downloads.claude.ai/claude-code-releases/<version>/manifest.json`.
  The first run on a new CLI is the riskiest one: if the new CLI
  changes its wire format the matrix run can produce systematic
  failures, so trigger a `SKIP_PUBLISH=1` run before the next scheduled
  fire. `gh` and `uv` bump the same way, with the checksum from the
  release's `gh_<version>_checksums.txt` and the tarball's `.sha256`
  sidecar respectively.
- **A local build on Apple silicon only proves the image assembles.**
  Under QEMU the Claude Code binary (a Bun executable) dies with
  `CPU lacks AVX support` and `gh` panics in the Go runtime, so
  `claude --version` and a full run are verified with a
  `SKIP_PUBLISH=1` run on Render, not locally.
- **Nothing persists between runs.** A failed run leaves no
  half-installed venv behind, but also no cache: don't expect a rerun
  to be faster than the first one.
