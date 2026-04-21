# LiteLLM v1.83.3 Upgrade — Working Artifacts

Scratch directory for the v1.81.3 → v1.83.3 upgrade. Delete or archive after merge.

Strategy spec: `docs/v1.83.3-upgrade-strategy.md`

## Layout

```
.upgrade/
├── replay-matrix.csv        # 87 custom commits (chronological), SHA|date|subject
├── batches/                 # domain-grouped cherry-pick lists
│   ├── 01-claude-anthropic-compat.txt     (7 commits)
│   ├── 02-gcs-gcp-logging.txt             (13 commits)
│   ├── 03-routing-vision.txt              (17 commits)
│   ├── 04-rate-limit-concurrency.txt      (6 commits)
│   ├── 05-budgets.txt                     (5 commits)
│   ├── 06-analytics-spend-failure-logs.txt (13 commits)
│   ├── 07-ui.txt                          (10 commits)
│   ├── 08-admin-user-mgmt.txt             (4 commits)
│   └── 09-build-ci-playground-misc.txt    (12 commits)
├── audit/                   # one .md per batch (Phase 2 output)
└── verification/            # MUST-SURVIVE evidence (Phase 4 output)
```

## Batch order for replay (Phase 3)

Recommended — low to high risk, each batch independently verifiable:

| # | Batch | Rationale for order |
|---|---|---|
| 1 | 01-claude-anthropic-compat | Small, isolated header/shape changes. Low conflict surface. |
| 2 | 09-build-ci-playground-misc | Mostly build files + model config + playground. Isolated. |
| 3 | 02-gcs-gcp-logging | Mostly our own files (gcp stdout logger); GCS touches shared logger. |
| 4 | 05-budgets | FREE_MODELS + user budget; auth_checks.py — high-priority MUST-SURVIVE. |
| 5 | 08-admin-user-mgmt | Mostly new endpoints; small conflict surface. |
| 6 | 06-analytics-spend-failure-logs | Analytics endpoints + schema changes; revert pair #111/#115. |
| 7 | 07-ui | Heavy UI churn expected from upstream. |
| 8 | 04-rate-limit-concurrency | Upstream rewrote MPR limiter to v2. Highest REWORK probability. |
| 9 | 03-routing-vision | Routing is our most-customized area; upstream also heavily reworked. |

Batches 1–5 should mostly be KEEP-AS-IS. Batches 8–9 need the deepest audit.

## Key revert pairs to watch in Phase 2

- **#46 / #48** (`f7f8141e` / `0930b8d7`) — "temp logging change" + revert. Both in batch 07. Candidate for DROP-both.
- **#111 / #115** (`20caa0ae` / `596a3a3a`) — "schema migration for aggregated spend logs" + revert. Both in batch 06. Candidate for DROP-both.

Dropping a revert pair requires: both commits marked DROP with justification "reverted in place — net zero change".

## Exclusions

Two commits intentionally excluded from replay (they are upstream squash-merges, not our work):
- `ed2b8e9fae` — Feature/upgrade to v1.81.0 stable (#52)
- `7fc45ccc1e` — Release/sandbox v1.81.3-stable (#68)

## Commands

Generate replay matrix:
```bash
git log --reverse --no-merges --format="%H|%ad|%s" --date=short 88c78df665^..main \
  | grep -v -E "^(ed2b8e9fae532292e05af3927ef6da55dc24191e|7fc45ccc1e150728d9b02f37d0ba8d7b2ca0785a)\|" \
  > .upgrade/replay-matrix.csv
```

Audit a single commit:
```bash
sha=<commit-sha>
git show --stat $sha
git log --oneline v1.81.3-stable..v1.83.3-stable -- $(git show --name-only --format= $sha)
```

Cherry-pick a batch:
```bash
while read sha; do
  git cherry-pick -x "$sha" || { echo "CONFLICT on $sha — resolve and continue"; break; }
done < .upgrade/batches/01-claude-anthropic-compat.txt
```
