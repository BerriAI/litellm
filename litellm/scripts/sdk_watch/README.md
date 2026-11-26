# Provider SDK watch

An agent that reads new releases of the OpenAI, Anthropic and Google GenAI Python SDKs, decides which ones need a LiteLLM change, makes the change with a test, and opens a PR with a video of the test failing on main and passing with the fix. It runs every day from `.github/workflows/provider-sdk-watch.yml` and calls its models through a LiteLLM proxy

## How it works

The run is a LangGraph graph with four stages, and each stage only runs when the one before it found something

1. `fetch` reads each SDK's CHANGELOG.md and keeps the releases from the chosen day, or by default the releases of the last 7 days that no earlier scheduled run checked
2. `triage` drops chores, docs and other sections that never need a LiteLLM change, then asks the model which entries do (a new model, request param, response or usage field, endpoint, or behavior change)
3. `dedupe` skips a change when its branch already exists, or an open PR or a PR merged since the release names it in its title
4. `implement` hands each remaining change to a deepagents agent that works in the checkout. It can read and edit files, read the SDK commit diff with `fetch_sdk_diff`, and run test files with `run_tests`. It may only write under `litellm/` (not `litellm/scripts/`), `tests/` and the cost map, and it returns `not_needed` instead of guessing a price or limit it cannot see. It takes at most `--max-implementations` changes per run, and a scheduled run hands the rest to the next one

The workflow then runs three more jobs

1. `watch` runs the graph and uploads a PR draft (patch, title, walkthrough, test files) for each change
2. `verify` runs with no secrets. It records the draft's tests on main and with the fix under asciinema, and turns the recording into a GIF with agg
3. `open-pr` never runs the patch. It applies it, refuses any path outside the allowed ones, pushes the branch as github-actions[bot], opens the PR with the walkthrough and the GIF, and posts a Slack message when `SDK_WATCH_SLACK_WEBHOOK_URL` is set

The agent writes code and runs tests in the same job as the provider key, so that job starts the proxy from a pinned LiteLLM image in Docker and then removes the runner user's sudo and Docker access. Code the agent writes can call the proxy but cannot read the key from the container. The jobs that run the agent's code never save a cache, so that code cannot reach a later main build. The only cache they write is the scheduled run's history, which lists the releases it checked and the changes it left for the next run

## Secrets

| Secret | Used by | Without it |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | `watch` | The run fails |
| `SDK_WATCH_SLACK_WEBHOOK_URL` | `open-pr` | No Slack ping |

## Run it locally

1. Make a throwaway checkout: `git worktree add --detach ../litellm-sdk-watch origin/main && cd ../litellm-sdk-watch && make bootstrap`. The agent resets and cleans its checkout between changes and refuses one with uncommitted work, so never point it at the checkout you work in
2. `cd litellm/scripts/sdk_watch && uv sync --group dev --group demo`
3. In another terminal at the repo root, start a proxy: `LITELLM_MASTER_KEY=sk-local ANTHROPIC_API_KEY=<key> uv run litellm --config litellm/scripts/sdk_watch/proxy_config.yaml --port 4000`
4. Check one release: `LITELLM_PROXY_URL=http://127.0.0.1:4000 LITELLM_PROXY_KEY=sk-local GITHUB_TOKEN=$(gh auth token) uv run python -m sdk_watch watch --repo ../../.. --out /tmp/sdk-watch --sdk anthropic --version 0.105.0`
5. Record a draft's tests: `uv run asciinema rec -c "uv run python -m sdk_watch demo --repo ../../.. --change /tmp/sdk-watch/<branch> --python ../../../.venv/bin/python --verdict /tmp/sdk-watch/<branch>/verdict.json" /tmp/sdk-watch/<branch>/demo.cast`

Use `--day YYYY-MM-DD` to check one day, and `--history <file>` to skip what an earlier run checked. To replay an old release, check out a commit from before LiteLLM supported it in a separate worktree and pass `--skip-dedupe`

## Limits

PRs opened with the workflow's `GITHUB_TOKEN` start no `pull_request` workflows, so a maintainer closes and reopens the PR to run CI. The GIF is committed in the PR branch and deleted in the next commit, so the PR body can show it while the merged tree never holds it. A new model whose price the agent cannot find in the SDK diff or the repo comes back as `not_needed`

## Friction found while building it

The repo variable `LITELLM_API_BASE` still points at gateway.litellm-sandbox.ai, which no longer resolves, so `issue_classifier.yml` and `duplicate_issue_check.yml` fail before calling a model. The gateway that replaced it is on a private network a GitHub runner cannot reach, so this workflow starts its own proxy from the LiteLLM image instead

The repo has no `ANTHROPIC_API_KEY` secret. `e2e_record_replay.yml` already reads one and its record job gets an empty value

claude-opus-5-5 rejects forced tool use (`tool_choice` of `any` or `tool`). LangChain's default `with_structured_output` and deepagents' default `response_format` both force a tool call, so the agent asks for JSON schema output instead (`method="json_schema"` and `ProviderStrategy`). deepagents only picks that strategy on its own when it recognizes the model name, and a LiteLLM alias like `sdk-watch` is not one it knows

GitHub has no API for the image uploads a PR body normally uses, which is why the GIF goes through a commit

agg publishes no checksum file next to its release binaries, so the workflow pins a SHA-256 computed from the v1.9.0 download
