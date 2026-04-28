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

## Layout

| File | Purpose |
| --- | --- |
| `run_daily.sh` | The actual cron job. Resolves versions, updates the worktree, boots the proxy, runs pytest, builds the JSON, opens (or updates) a docs PR. |
| `build_matrix.py` | Tiny Python CLI that wraps `tests.claude_code.matrix_builder.build_from_paths`. Exists only because the bash script needs *some* way to render the per-cell aggregation, and the builder is already Python. |
| `litellm-compat-matrix.service` | systemd oneshot that invokes `run_daily.sh`. |
| `litellm-compat-matrix.timer` | `OnCalendar=*-*-* 06:00:00 UTC`, `Persistent=true`. |
| `litellm-compat-matrix.env.example` | Template for `/etc/litellm-compat-matrix.env`. |

## What `run_daily.sh` does

1. **Resolves the latest LiteLLM `v*-stable` tag** by hitting the
   GitHub Releases API (`curl | jq`).
2. **Reads the local Claude Code CLI version** via `claude --version`.
   The cron does not auto-upgrade the CLI — operators do that
   out-of-band by running `npm install -g @anthropic-ai/claude-code@latest`.
3. **Updates the persistent worktree** at `~/litellm-cron-worktree/`:
   `git fetch --tags --force`, `git reset --hard`,
   `git clean -fdx -e .venv`, `git checkout --force <tag>`. The
   `.venv` is preserved across runs so `uv sync --frozen` is
   incremental.
4. **Boots the proxy** as a `setsid` background process on port `4100`
   (so it can't collide with a developer's `:4000`), then polls
   `/health/liveliness` until it's up.
5. **Runs pytest** with `ANTHROPIC_BASE_URL` pointed at the proxy and
   `COMPAT_RESULTS_PATH` set so the conftest hook writes the per-test
   results artifact. Test failures become `fail` cells in the JSON,
   not script errors.
6. **Builds `compatibility-matrix.json`** by handing the artifact +
   manifest to `build_matrix.py`.
7. **Opens or updates a docs PR**: `gh repo clone` of `litellm-docs`
   into a tempdir, deterministic head branch
   (`compat-matrix/<litellm-version>-<claude-code-version>-<UTC-date>`),
   `--force-with-lease` push, `gh pr create`. A re-run on the same
   day fast-forwards the existing branch and `gh pr create` no-ops
   ("a pull request for branch ... already exists" is treated as
   success).

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

# 4. gh auth — must be a collaborator on BerriAI/litellm-docs.
gh auth login   # follow prompts; pick HTTPS + token paste flow

# 5. Provider credentials.
sudo cp ~/litellm/litellm/tests/claude_code/cron_vm/litellm-compat-matrix.env.example \
        /etc/litellm-compat-matrix.env
sudoedit /etc/litellm-compat-matrix.env   # fill in real values
sudo chmod 0600 /etc/litellm-compat-matrix.env

# 6. systemd units.
sudo cp ~/litellm/litellm/tests/claude_code/cron_vm/litellm-compat-matrix.service /etc/systemd/system/
sudo cp ~/litellm/litellm/tests/claude_code/cron_vm/litellm-compat-matrix.timer   /etc/systemd/system/
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
SKIP_PUBLISH=1 ~/litellm/litellm/tests/claude_code/cron_vm/run_daily.sh

# Narrow to one cell while debugging.
SKIP_PUBLISH=1 PYTEST_K='basic_messaging_non_streaming and anthropic' \
  ~/litellm/litellm/tests/claude_code/cron_vm/run_daily.sh

# Watch the most recent run.
journalctl -u litellm-compat-matrix.service -f

# Read older runs.
journalctl -u litellm-compat-matrix.service --since '2 days ago'

# Disable until further notice (e.g. while debugging).
sudo systemctl disable --now litellm-compat-matrix.timer
```

## Gotchas

- **The proxy port is `4100`, not `4000`.** This is so a developer SSH'd
  into the same VM with their own `:4000` proxy doesn't collide with a
  cron run. Override with `PROXY_PORT=...` in `/etc/litellm-compat-matrix.env`
  if you need to.
- **`uv sync --frozen` requires the resolved tag to be tagged on
  GitHub.** If the latest stable release was made but not pushed as a
  git tag, the `git checkout` step fails. Push the tag, then rerun.
- **`gh auth` token rotation is your problem.** The cron does not
  refresh the token; if the bot account's PAT expires the run will
  fail at `gh repo clone` with a 401. Re-run `gh auth login`.
- **First run after upgrading the Claude Code CLI is the riskiest one.**
  If the new CLI changes its wire format the matrix run can produce
  systematic failures. Always run with `SKIP_PUBLISH=1` after a CLI
  upgrade before letting the next scheduled fire happen.
- **Disk:** the worktree's `.venv` is ~1.3 GB and the `.git` directory
  is ~1 GB. Plan for at least 5 GB free on the VM, otherwise
  `uv sync` will fail mid-run and leave you with a half-installed venv.
