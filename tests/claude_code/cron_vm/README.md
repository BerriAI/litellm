# Cron VM setup for the Claude Code compatibility-matrix populator

The populator runs daily on a dedicated GCP VM
(`litellm-compatibility-matrix-populator`) rather than as a GitHub
Action. Trade-offs:

- ✅ Real VM means we can `gh auth login` against a human/bot account
  that's already a collaborator on `BerriAI/litellm-docs`, instead of
  provisioning a GitHub App with `pull-requests: write`.
- ✅ Persistent state (a single `~/litellm-cron-worktree/` and its `.venv`)
  is reused across runs, so each daily run does a fast `git checkout` +
  incremental `uv sync` rather than a fresh clone + cold sync.
- ✅ No Docker dependency — proxy is run directly via `uv run litellm`.
- ⚠️ The VM has to actually be on. systemd's `Persistent=true` recovers
  from short outages, but a multi-day outage means the matrix goes
  stale until the VM is back.
- ⚠️ Provider credentials live on the VM filesystem
  (`/etc/litellm-compat-matrix.env`) instead of GitHub secrets. Treat
  the VM as an environment with comparable blast radius to a CI runner.

## What the populator does, end to end

`tests/claude_code/publisher.py` (`python -m tests.claude_code.publisher`):

1. Resolves the latest `v*-stable` tag of `BerriAI/litellm` via the
   GitHub Releases API (`tests/claude_code/resolver.py`).
2. Reads the locally installed Claude Code CLI version
   (`claude --version`).
3. Updates the persistent worktree at `~/litellm-cron-worktree/` to
   that tag, and `uv sync --frozen`s its `.venv`. `git clean -fdx -e .venv`
   wipes any cruft from previous runs while keeping the venv around.
4. Boots the LiteLLM proxy as a subprocess on port `4100` (override
   with `PROXY_PORT`), using
   `tests/claude_code/test_config.yaml` from the checked-out tag.
5. Runs `pytest tests/claude_code/` with `ANTHROPIC_BASE_URL` pointed
   at the proxy and `COMPAT_RESULTS_PATH` set so the conftest hook
   writes the per-test results artifact.
6. Builds `compatibility-matrix.json` from the artifact via
   `tests.claude_code.matrix_builder.build_from_paths`.
7. Clones the docs repo (`gh repo clone BerriAI/litellm-docs`) into a
   temp dir, checks out a deterministic head branch
   (`compat-matrix/<litellm-version>-<claude-code-version>-<UTC-date>`),
   commits the JSON, force-with-lease pushes, and opens a PR with
   `gh pr create`.

Re-running on the same day with the same versions is idempotent: the
branch name collides, the force-with-lease updates the existing branch,
and `gh pr create` no-ops because the PR already exists.

## One-time VM setup

Run as `mateo` on the cron VM:

```bash
# 1. Toolchain
sudo apt-get update
sudo apt-get install -y git nodejs npm
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get install -y gh   # or follow https://cli.github.com/

# 2. Claude Code CLI (the cron does NOT auto-upgrade this; rerun this
#    line out-of-band when you want a fresh CLI to be tested)
sudo npm install -g @anthropic-ai/claude-code@latest

# 3. Litellm dev checkout. Used as the launcher for the publisher
#    module; the populator mutates a separate worktree under
#    ~/litellm-cron-worktree/.
mkdir -p ~/litellm
git clone https://github.com/BerriAI/litellm.git ~/litellm/litellm
cd ~/litellm/litellm && uv sync --frozen

# 4. gh auth — must be a collaborator on BerriAI/litellm-docs.
gh auth login   # follow prompts; pick HTTPS + token paste flow

# 5. Provider credentials.
sudo cp tests/claude_code/cron_vm/litellm-compat-matrix.env.example \
        /etc/litellm-compat-matrix.env
sudoedit /etc/litellm-compat-matrix.env   # fill in real values
sudo chmod 0600 /etc/litellm-compat-matrix.env

# 6. systemd units.
sudo cp tests/claude_code/cron_vm/litellm-compat-matrix.service /etc/systemd/system/
sudo cp tests/claude_code/cron_vm/litellm-compat-matrix.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now litellm-compat-matrix.timer
```

## Operating it

```bash
# When does it run next?
systemctl list-timers litellm-compat-matrix.timer

# Trigger a run right now (still PRs to litellm-docs).
sudo systemctl start litellm-compat-matrix.service

# Trigger a run that does NOT open a PR (good for first-time validation).
cd ~/litellm/litellm
uv run python -m tests.claude_code.publisher --skip-publish

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
  cron run. Override with `PROXY_PORT=...` in
  `/etc/litellm-compat-matrix.env` if you need to.
- **`uv sync --frozen` requires the resolved tag to be tagged on
  GitHub.** If the latest stable release was made but not pushed as a
  git tag, the run will `git checkout` fail. Push the tag, then rerun.
- **`gh auth` token rotation is your problem.** The cron does not
  refresh the token; if the bot account's PAT expires the run will
  fail at `gh repo clone` with a 401. Re-run `gh auth login`.
- **First run after upgrading the Claude Code CLI is the riskiest one.**
  If the new CLI changes its wire format the matrix run can produce
  systematic failures. Always run `--skip-publish` after a CLI upgrade
  to inspect the JSON before the next scheduled fire.
