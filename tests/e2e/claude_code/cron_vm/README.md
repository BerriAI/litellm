# Cron VM setup for the Claude Code compatibility-matrix populator

The populator runs daily on a dedicated GCP VM
(`litellm-compatibility-matrix-populator`) rather than as a GitHub
Action. Trade-offs:

- ✅ Real VM means we can `gh auth login` against an account that's
  already a collaborator on `BerriAI/litellm-docs`, instead of
  provisioning a GitHub App with `pull-requests: write`.
- ✅ Persistent state (a single `~/litellm-cron-worktree/` and its `.venv`)
  is reused across runs, so each daily run does a fast `git checkout` +
  incremental `uv sync` rather than a fresh clone + cold sync.
- ✅ No Docker dependency — the proxy runs directly via `uv run litellm`.
- ⚠️ The VM has to actually be on. systemd's `Persistent=true` recovers
  from short outages, but a multi-day outage means the matrix goes
  stale until the VM is back.
- ⚠️ Provider credentials live on the VM filesystem
  (`/etc/litellm-compat-matrix.env`) instead of GitHub secrets. Treat
  the VM as an environment with comparable blast radius to a CI runner.

This directory used to live at `tests/claude_code/cron_vm/` (paired with
the standalone `tests/claude_code/` suite); it now runs the maintained
`tests/e2e/claude_code/` suite instead. The pytest env interface changed
accordingly: the runner exports `LITELLM_PROXY_URL` / `LITELLM_MASTER_KEY`
(previously `LITELLM_PROXY_BASE_URL` / `LITELLM_PROXY_API_KEY`), the azure
column reads `AZURE_AI_API_KEY` / `AZURE_AI_API_BASE` (previously
`AZURE_FOUNDRY_*`), and the GPT columns need `OPENAI_API_KEY` and
`AZURE_API_BASE` / `AZURE_API_KEY` — see `litellm-compat-matrix.env.example`.

## Layout

| File | Purpose |
| --- | --- |
| `run_daily.sh` | The actual cron job. Resolves versions, updates the worktree, boots the proxy, runs pytest, builds the JSON, opens (or updates) a docs PR, sweeps stale compat-matrix PRs. |
| `build_matrix.py` | Tiny Python CLI that wraps `claude_code.matrix_builder.build_from_paths`. Exists only because the bash script needs *some* way to render the per-cell aggregation, and the builder is already Python. |
| `check_regressions.py` | Tiny Python CLI that wraps `claude_code.matrix_builder.find_regressions`. Diffs the freshly built matrix against the currently-published one and exits `3` if any cell flipped green→red, which gates auto-merge. |
| `litellm-compat-matrix.service` | systemd oneshot that invokes `run_daily.sh`. |
| `litellm-compat-matrix.timer` | `OnCalendar=*-*-* 06:00:00 UTC`, `Persistent=true`. |
| `litellm-compat-matrix.env.example` | Template for `/etc/litellm-compat-matrix.env`. |

## What `run_daily.sh` does

1. **Resolves the latest LiteLLM final release tag** (newest bare
   `vX.Y.Z`, skipping `-rc.N`/`-dev.N` pre-releases) by paging the
   GitHub Releases API (`curl | jq`).
2. **Reads the local Claude Code CLI version** via `claude --version`.
   The cron does not auto-upgrade the CLI — operators do that
   out-of-band by running `npm install -g @anthropic-ai/claude-code@latest`.
3. **Updates the persistent worktree** at `~/litellm-cron-worktree/`:
   `git fetch --tags --force`, `git reset --hard`,
   `git clean -fdx -e .venv -e .uv-bin`, `git checkout --force <tag>`.
   The `.venv` is preserved across runs so `uv sync --frozen` is
   incremental. Then **shims the test suite**: `tests/e2e/` in the
   worktree is rebuilt from the dev checkout — the `claude_code/` suite
   plus the five shared transport helpers it imports (`proxy_client.py`,
   `e2e_http.py`, `models.py`, `e2e_config.py`, `transport.py`) — so the
   cron always runs *today's* tests against the latest stable proxy. The
   tag's own `tests/e2e/` tree (including the EKS-harness `conftest.py`,
   whose imports the stable venv doesn't install) is deliberately not
   used.
4. **Boots the proxy** as a `setsid` background process on port `4100`
   (so it can't collide with a developer's `:4000`), then polls
   `/health/liveliness` until it's up.
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
   branch ... already exists" is treated as success). These PRs are no
   longer gated on a second human review.
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
   other open `compat-matrix/*` PR on the docs repo is closed (and its
   bot-owned branch deleted), so at most one compat-matrix PR is ever
   open — the newest.

## One-time VM setup

Run as `mateo` on the cron VM:

```bash
# 1. Toolchain
sudo apt-get update
sudo apt-get install -y git nodejs npm jq curl
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get install -y gh   # or follow https://cli.github.com/

# 2. Claude Code CLI (the cron does NOT auto-upgrade this; rerun this
#    line out-of-band when you want a fresh CLI to be tested)
sudo npm install -g @anthropic-ai/claude-code@latest

# 3. Litellm checkout. Used by systemd's WorkingDirectory and as the
#    source of the .service / .timer files. The cron itself runs out
#    of the separate worktree at ~/litellm-cron-worktree/.
mkdir -p ~/litellm
git clone https://github.com/BerriAI/litellm.git ~/litellm/litellm
git -C ~/litellm/litellm checkout litellm_internal_staging

# 4. gh auth — must be a collaborator on BerriAI/litellm-docs.
gh auth login   # follow prompts; pick HTTPS + token paste flow

# 5. Provider credentials + the publish token.
sudo cp ~/litellm/litellm/tests/e2e/claude_code/cron_vm/litellm-compat-matrix.env.example \
        /etc/litellm-compat-matrix.env
sudoedit /etc/litellm-compat-matrix.env   # fill in real values
sudo chmod 0600 /etc/litellm-compat-matrix.env
# The mateo-berri PAT lives in its own file, mapped into the service via
# systemd LoadCredential so it stays out of the test processes' env
# (see the env.example comment for why).
sudo install -m 0600 /dev/null /etc/litellm-compat-matrix-github-token
sudoedit /etc/litellm-compat-matrix-github-token   # single line: the PAT

# 6. systemd units.
sudo cp ~/litellm/litellm/tests/e2e/claude_code/cron_vm/litellm-compat-matrix.service /etc/systemd/system/
sudo cp ~/litellm/litellm/tests/e2e/claude_code/cron_vm/litellm-compat-matrix.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now litellm-compat-matrix.timer
```

## Operating it

```bash
# When does it run next?
systemctl list-timers litellm-compat-matrix.timer

# Trigger a real run right now (PRs to litellm-docs).
sudo systemctl start litellm-compat-matrix.service

# Trigger a run that does NOT open a PR (good for first-time validation).
SKIP_PUBLISH=1 ~/litellm/litellm/tests/e2e/claude_code/cron_vm/run_daily.sh

# Narrow to one cell while debugging.
SKIP_PUBLISH=1 PYTEST_K='basic_messaging_non_streaming and anthropic' \
  ~/litellm/litellm/tests/e2e/claude_code/cron_vm/run_daily.sh

# Watch the most recent run.
journalctl -u litellm-compat-matrix.service -f

# Read older runs.
journalctl -u litellm-compat-matrix.service --since '2 days ago'

# Disable until further notice (e.g. while debugging).
sudo systemctl disable --now litellm-compat-matrix.timer
```

## Gotchas

- **The venv is pinned to Python 3.12 (`CRON_PYTHON_VERSION`).** The
  e2e suite uses PEP 695 `type` aliases, which the VM's system Python
  (3.11) can't parse; `run_daily.sh` has uv fetch a managed CPython
  into `~/litellm-cron-worktree/.uv-python/` and syncs the venv against
  it. The first run after a version bump is a cold venv rebuild.
- **The proxy port is `4100`, not `4000`.** This is so a developer SSH'd
  into the same VM with their own `:4000` proxy doesn't collide with a
  cron run. Override with `PROXY_PORT=...` in `/etc/litellm-compat-matrix.env`
  if you need to.
- **`uv sync --frozen` requires the resolved tag to be tagged on
  GitHub.** If the latest stable release was made but not pushed as a
  git tag, the `git checkout` step fails. Push the tag, then rerun.
- **Publish-token rotation is your problem.** The cron does not
  refresh the token; if `mateo-berri`'s PAT in
  `/etc/litellm-compat-matrix-github-token` expires, the run fails at
  the `git push`/`gh pr create` step with a 401 ("Bad credentials" /
  "Authentication failed"). Mint a fresh PAT and update that file.
  The token needs write access to `BerriAI/litellm-docs` (classic
  `repo` scope, or fine-grained Contents:RW + Pull requests:RW). It is
  delivered via systemd `LoadCredential`, not the env file, so pytest,
  the proxy, and the claude CLI never inherit it; manual runs export
  `GITHUB_TOKEN` instead.
- **First run after upgrading the Claude Code CLI is the riskiest one.**
  If the new CLI changes its wire format the matrix run can produce
  systematic failures. Always run with `SKIP_PUBLISH=1` after a CLI
  upgrade before letting the next scheduled fire happen.
- **Disk:** the worktree's `.venv` is ~1.3 GB and the `.git` directory
  is ~1 GB. Plan for at least 5 GB free on the VM, otherwise
  `uv sync` will fail mid-run and leave you with a half-installed venv.
