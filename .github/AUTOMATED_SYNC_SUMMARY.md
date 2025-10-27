# CARTO Automated Upstream Sync - Claude Code

## What We Built

An AI-powered automated workflow for syncing CARTO's fork with upstream LiteLLM stable releases using Claude Code.

## The Solution

**File:** `.github/workflows/carto_claude_sync.yaml`

Claude Code (AI) intelligently handles the entire sync process:
- Detects new upstream `-stable` releases nightly
- Merges changes and resolves conflicts automatically
- Attempts to fix test failures
- Creates detailed, context-aware PRs
- Preserves CARTO-specific modifications

## Why Claude Code?

**Traditional scripted workflows fail at:**
- Complex merge conflicts requiring context
- Test failures needing code changes
- Understanding CARTO-specific conventions
- Providing helpful explanations

**Claude Code succeeds because it:**
- ✅ Reads and understands `CARTO_CLAUDE.md` documentation
- ✅ Analyzes conflict context and makes smart decisions
- ✅ Can modify code to fix linting/test issues
- ✅ Explains its reasoning in detail
- ✅ Knows when to ask for human review

## How It Works

### Every Night at 3 AM UTC

```
1. Detect new -stable release
   ↓
2. Run detection script
   ↓ [New release found]
3. Create branch: automated-sync/v1.78.5-stable
   ↓
4. Merge upstream tag
   ↓
5. Analyze conflicts
   ↓
6. Resolve conflicts intelligently
   - Keep carto_*.yaml workflows
   - Preserve CARTO Dockerfile/Makefile mods
   - Update pyproject.toml version
   ↓
7. Run tests (make lint && make test-unit)
   ↓ [Tests fail?]
8. Attempt to fix issues
   ↓
9. Create detailed PR
   - Link to upstream changes
   - Explanation of conflicts resolved
   - Test results
   - Recommendation: ready or needs-review
   ↓
10. Slack notification 📢
```

### Intelligent Decision Making

**Example 1: Workflow Conflict**
```
Conflict in: .github/workflows/ghcr_deploy.yml
Claude's thought process:
"Upstream added new workflow features. But CARTO uses carto_ghcr_deploy.yaml
instead. This file should stay disabled (.yml.txt). Keeping CARTO version."
→ Keeps CARTO's approach
```

**Example 2: Test Failure**
```
Test failure: test_vertex_ai_labels()
Claude's thought process:
"Upstream changed API to require 'labels' parameter. Our mock doesn't include it.
Let me update the mock to match new API signature."
→ Fixes the test automatically
```

**Example 3: Complex Change**
```
Upstream refactored authentication system
Claude's thought process:
"Major architectural change. High risk. I can merge cleanly but can't verify
all CARTO integrations. Marking for review with detailed notes."
→ Creates PR with needs-review label
```

## Setup

### Prerequisites

1. **Anthropic API Key** (Required)
   ```bash
   # Get key from: https://console.anthropic.com/settings/keys
   # Add to: Repository Settings → Secrets → New secret
   # Name: ANTHROPIC_API_KEY
   # Value: sk-ant-...
   ```

2. **Slack Webhook** (Optional)
   ```bash
   # Add to: Repository Settings → Variables → New variable
   # Name: SLACK_WEBHOOK_URL
   # Value: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
   ```

### Installation

```bash
# Files are already created:
.github/workflows/carto_claude_sync.yaml  # Main workflow
.github/scripts/detect_stable_release.py  # Detection script

# Just commit them:
git add .github/workflows/carto_claude_sync.yaml
git add .github/scripts/detect_stable_release.py
git add CARTO_CLAUDE.md
git add .github/workflows/README_CLAUDE_SYNC.md
git commit -m "feat: add Claude Code automated upstream sync"
git push origin carto/main
```

### Testing

```bash
# Manual trigger (test before relying on nightly schedule)
gh workflow run carto_claude_sync.yaml

# Watch it run
gh run list --workflow=carto_claude_sync.yaml --limit 5

# View logs
gh run view --log
```

## What Happens Nightly

### Scenario 1: No New Release (Most Common - ~90%)
- Detection script finds no new `-stable` release
- Workflow exits early
- **Cost: $0.01** (just detection)

### Scenario 2: Clean Merge (~5%)
- New release detected
- Merge has no conflicts
- Tests all pass
- PR created with `automated-sync,ready` label
- **Cost: ~$0.50**

### Scenario 3: Conflicts (~4%)
- New release detected
- Merge has conflicts
- Claude resolves them following CARTO conventions
- Tests pass
- PR created with detailed explanation
- **Cost: ~$1.50**

### Scenario 4: Test Failures (~0.9%)
- New release detected
- Merge successful (with or without conflicts)
- Tests fail
- Claude attempts fixes
- PR created with `needs-review` and explanation
- **Cost: ~$1.00**

### Scenario 5: Major Changes (~0.1%)
- Significant upstream refactoring
- Claude merges but uncertain about implications
- PR created with `needs-review` and detailed analysis
- **Cost: ~$2.00**

## Cost Analysis

### Monthly Estimates

**Typical upstream release frequency:** 2-4 stable releases per month

**Scenario A: Quiet Month (2 releases)**
- 28 no-ops × $0.01 = $0.28
- 1 clean merge × $0.50 = $0.50
- 1 with conflicts × $1.50 = $1.50
- **Total: ~$2.30/month**

**Scenario B: Average Month (3-4 releases)**
- 26 no-ops × $0.01 = $0.26
- 2 clean merges × $0.50 = $1.00
- 2 with conflicts × $1.50 = $3.00
- **Total: ~$4.30/month**

**Scenario C: Active Month (6-8 releases)**
- 23 no-ops × $0.01 = $0.23
- 3 clean merges × $0.50 = $1.50
- 4 with conflicts × $1.50 = $6.00
- 1 with test fixes × $1.00 = $1.00
- **Total: ~$8.70/month**

**Reality:** Expect $3-$5/month on average.

## Monitoring

### GitHub Actions
View workflow runs: https://github.com/CartoDB/litellm/actions/workflows/carto_claude_sync.yaml

**What to look for:**
- ✅ Green checkmarks = successful run
- 🟡 Yellow = running
- ❌ Red = failed (check logs)

### Pull Requests
Filter for automated PRs: https://github.com/CartoDB/litellm/pulls?q=label%3Aautomated-sync

**Labels:**
- `automated-sync` = All automated PRs
- `automated-sync,ready` = Confident, ready to merge
- `automated-sync,needs-review` = Needs human review

### Slack Notifications
You'll receive messages for:
- ✅ New sync PR created (with link)
- ❌ Workflow failures (with logs link)

## Best Practices

### 1. Review Claude's Work
Even when labeled `ready`, briefly review:
- PR description (Claude's summary)
- Changed files (what was modified)
- Commit messages (Claude's reasoning)

This helps you:
- Catch any issues
- Learn what changed upstream
- Provide feedback to improve future syncs

### 2. Provide Feedback
When Claude makes mistakes:
1. Comment on the PR explaining the issue
2. Update `CARTO_CLAUDE.md` to clarify expectations
3. Fix the issue and merge

Claude learns from documentation, so better docs = better syncs!

### 3. Monitor Costs
Check monthly usage: https://console.anthropic.com/settings/billing

If costs are higher than expected:
- Review workflow runs for inefficiencies
- Adjust max_iterations in workflow (default: 30)
- Consider reducing to weekly instead of nightly

### 4. Test After Major Upstream Changes
When Claude syncs a major version:
- Review the PR carefully
- Test beta release in staging
- Check CARTO-specific features
- Verify Docker image builds correctly

## Troubleshooting

### Claude Creates PRs Needing Review

**This is normal!** Claude is conservative and asks for review when uncertain.

**Common reasons:**
- Major upstream refactoring
- Complex conflicts
- Test failures it couldn't fix
- First time seeing a pattern

**What to do:** Review Claude's detailed explanation in PR description

### Workflow Fails

**Check:**
1. API key is valid: Settings → Secrets
2. API credits available: https://console.anthropic.com/
3. Workflow logs for error message
4. GitHub Actions permissions

### No PRs Created

**Reasons:**
- No new upstream `-stable` releases (most likely)
- Detection script errored (check logs)
- PR already exists for that version
- Workflow disabled/paused

### Unexpected Changes in PR

**Remember:** Claude makes decisions based on:
- `CARTO_CLAUDE.md` documentation
- Code context and comments
- Pattern recognition from codebase

If changes are wrong:
- Update documentation to be clearer
- Add comments to code explaining conventions
- Provide feedback in PR comments

## File Structure

```
.github/
├── workflows/
│   ├── carto_claude_sync.yaml       # Claude Code sync workflow
│   ├── carto_release.yaml           # Manual release creation
│   ├── carto_ghcr_deploy.yaml       # Docker image builds
│   └── README_CLAUDE_SYNC.md        # Setup guide
├── scripts/
│   └── detect_stable_release.py     # Upstream detection
└── AUTOMATED_SYNC_SUMMARY.md        # This file

CARTO_CLAUDE.md                       # Main documentation (Claude reads this!)
docs/
└── CARTO_RELEASE_PROCESS.md         # Release process
```

## Next Steps

1. ✅ **Files created** - All workflow files ready
2. ⏳ **Add `ANTHROPIC_API_KEY`** - Required for workflow
3. ⏳ **Add `SLACK_WEBHOOK_URL`** - Optional but recommended
4. ⏳ **Commit files** - Push to `carto/main`
5. ⏳ **Test manually** - Trigger workflow to verify
6. ⏳ **Wait for first nightly run** - Monitor tomorrow morning
7. ⏳ **Review first PR** - See Claude Code in action!

## Documentation

- **Setup Guide:** [.github/workflows/README_CLAUDE_SYNC.md](.github/workflows/README_CLAUDE_SYNC.md)
- **Main Docs:** [CARTO_CLAUDE.md](../CARTO_CLAUDE.md)
- **Release Process:** [docs/CARTO_RELEASE_PROCESS.md](../docs/CARTO_RELEASE_PROCESS.md)

---

**Created:** 2025-10-27
**Status:** Ready for deployment
**Technology:** Claude Code (Anthropic)
**Maintained by:** CARTO Engineering Team
