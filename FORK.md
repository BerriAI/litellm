# Bitovi LiteLLM Fork

Bitovi fork of [BerriAI/litellm](https://github.com/BerriAI/litellm). We keep upstream LiteLLM stable releases while carrying Bitovi-specific features and platform deploy config.

## Remotes and branches

| Remote / branch | Purpose |
|-----------------|---------|
| `origin` | `https://github.com/bitovi/litellm.git` |
| `upstream` | `https://github.com/BerriAI/litellm.git` |
| `litellm_internal_staging` (ours) | Working base for all Bitovi work and PRs |
| BerriAI `litellm_internal_staging` | Their internal integration branch; **do not sync from it** (name collision is coincidental) |
| `upstream/main` | Their development tip (nightlies / `dev`); **do not sync from it** |

One-time setup:

```bash
git remote add upstream https://github.com/BerriAI/litellm.git
git fetch upstream --tags
```

## Sync source: stable tags only

Sync from the latest upstream **stable** git tag matching `^v[0-9]+\.[0-9]+\.[0-9]+$` (for example `v1.92.0`, `v1.91.3`).

Do **not** sync from:

- `upstream/main` (currently tracks nightlies such as `v1.94.0-dev.N`)
- BerriAI’s `litellm_internal_staging`
- Tags with `-dev`, `-rc`, or `nightly`

BerriAI’s cycle is `dev` → `rc` → stable ([release cycle](https://docs.litellm.ai/docs/proxy/release_cycle)). Stables are weekly (often Sunday) with PATCH tags for hotfixes. Prefer both weekly minors and patches.

Check status locally:

```bash
make fork-sync-status
```

## Fork-owned changes

Keep these when resolving sync conflicts; everything else should stay close to upstream.

| Area | What | Where |
|------|------|-------|
| Bedrock Mantle | Provider with bearer / SigV4 auth, chat + Responses transforms | `litellm/llms/bedrock_mantle/`, `tests/test_litellm/llms/bedrock_mantle/` |
| Team / virtual keys | Auto-assign team on VK generation, default team for personal keys, VK perms fixes, per-model budgets / spend tracing, admin budget UI | proxy key/team paths and related UI |
| Usage / budgets | Windowing, config-based budgets, usage data | proxy spend / budget code |
| Platform deploy | Proxy URL, email allowlist, MCP tool prefix, Valkey `-primary` hostname, OnePassword/ESO, publish workflows | `deploy/values.yaml`, `.github/workflows/publish-*.yml` |

Auth contract to preserve for Mantle: `BedrockMantleAuthMixin`, `_resolve_bearer_token`, `_resolve_region`.

If upstream moves provider transforms into Rust, Mantle will need a Rust port; watch provider/auth RFCs when reviewing sync PRs.

## Day-to-day feature work

Branch from **current** `litellm_internal_staging`. Open PRs **into** `litellm_internal_staging` (not `main`).

```bash
make fork-branch NAME=litellm_my_change
# implement, push, open PR with base litellm_internal_staging
```

Branch names: `litellm_<short_description>`, no `/` in the name.

Do not branch from `upstream/main` or a stable tag for Bitovi-only work; that drops Mantle, team/VK, and deploy changes from your starting point.

## Upstream sync

### Automated (preferred)

`.github/workflows/fork-compatibility-check.yml` runs Mondays and Thursdays (and on `workflow_dispatch`). When a newer stable tag exists than our staging tip contains, it:

1. Creates `sync/upstream-vX.Y.Z` from `litellm_internal_staging`
2. Merges the stable tag onto that branch (so the PR Files changed tab shows the real diff)
3. If there are conflicts, commits the conflict markers on the branch and labels the PR `needs-conflict-resolution`
4. Opens (or refreshes) a PR into `litellm_internal_staging` labeled `upstream-sync`
5. Runs Bedrock Mantle tests on staging and on the sync branch when there are no conflicts
6. Notifies via the PR and optional Slack (`SLACK_WEBHOOK_URL` secret)

**The Action never merges into `litellm_internal_staging`.** A person reviews the PR, resolves conflicts on the sync branch if needed, then merges the PR when ready.

### Required: `FORK_SYNC_TOKEN` (PR create)

`GITHUB_TOKEN` often cannot open PRs even when the repo checkbox **Allow GitHub Actions to create and approve pull requests** is enabled. Common reasons:

- Org (or Enterprise) Actions settings still disallow PR create, which overrides the repo
- Workflow permissions are still **Read** only (the create-PR checkbox is separate from Read and write)
- `gh pr create` uses GraphQL `createPullRequest`, which fails with “Resource not accessible by integration”

This Action creates PRs via the **REST** API and expects a PAT:

1. Create a **classic** PAT with `repo` scope (simplest), or a fine-grained PAT on `bitovi/litellm` with **Contents**, **Pull requests**, and **Issues** read/write
2. Add it as repo secret `FORK_SYNC_TOKEN` (Settings → Secrets and variables → Actions)
3. Re-run the workflow

Optional (only helps `GITHUB_TOKEN` fallback; PAT is still recommended):

- Repo **and** org: Settings → Actions → General → Workflow permissions → **Read and write permissions**
- Same page: enable **Allow GitHub Actions to create and approve pull requests** (must be allowed at org/enterprise first if the repo checkbox is grayed or ineffective)

Optional: `SLACK_WEBHOOK_URL` for Slack alerts (incoming webhook URL).

### Manual fallback

```bash
git fetch upstream --tags
git checkout litellm_internal_staging
git pull origin litellm_internal_staging
git merge vX.Y.Z   # or: git rebase vX.Y.Z && git push --force-with-lease
pytest tests/test_litellm/llms/bedrock_mantle/ -v
# smoke team/VK paths if those files conflicted
git push origin litellm_internal_staging
```

After the sync PR (or manual merge) lands on staging, deploy via the existing Bitovi publish/tag workflows.

### Conflict expectations

Conflicts should concentrate in Mantle, team/VK/budget code, and `deploy/`. Widespread unrelated conflicts usually mean upstream refactored shared surfaces; stop and inspect before forcing.

## Contribute back to BerriAI (rare)

Only when opening a PR against **BerriAI/litellm**. Branch from a clean `upstream/main` so the PR does not include Bitovi-only commits:

```bash
make upstream-branch NAME=litellm_upstream_fix
```

If the change is useful both places, land it on BerriAI first, then pick it up here via the next stable-tag sync (or cherry-pick onto staging sooner if needed).

## Keeping current

1. Ensure `FORK_SYNC_TOKEN` is set so the Action can open sync PRs
2. Rely on the Monday/Thursday Action and review open `upstream-sync` PRs (diff + conflicts)
3. Resolve conflicts on the sync branch if labeled `needs-conflict-resolution`, then merge the PR into staging
4. Optionally set `SLACK_WEBHOOK_URL` for alerts when a new stable appears or Mantle checks fail
5. Watch upstream release notes and breaking signals in provider auth, proxy team/key APIs, and Helm/deploy
6. After each sync: Mantle pytest; smoke team/VK if those areas changed
