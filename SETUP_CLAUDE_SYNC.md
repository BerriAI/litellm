# Setup: Claude Code Automated Upstream Sync

Quick setup guide for CARTO's AI-powered upstream sync workflow.

## What You're Setting Up

An automated workflow that runs nightly, checking for new upstream LiteLLM releases and using Claude Code (AI) to:
- Merge changes automatically
- Resolve conflicts intelligently
- Fix test failures
- Create detailed PRs

**Cost:** ~$3-$5/month

---

## Step 1: Add Anthropic API Key (Required)

1. Get API key from: https://console.anthropic.com/settings/keys

2. Add to GitHub:
   ```
   Go to: https://github.com/CartoDB/litellm/settings/secrets/actions
   Click: "New repository secret"
   Name: ANTHROPIC_API_KEY
   Value: sk-ant-... (your key)
   ```

---

## Step 2: Add Slack Webhook (Optional)

1. Get webhook URL from your Slack workspace

2. Add to GitHub:
   ```
   Go to: https://github.com/CartoDB/litellm/settings/variables/actions
   Click: "New repository variable"
   Name: SLACK_WEBHOOK_URL
   Value: https://hooks.slack.com/services/...
   ```

---

## Step 3: Commit the Files

```bash
# Add all the new files
git add .github/workflows/carto_claude_sync.yaml
git add .github/workflows/README_CLAUDE_SYNC.md
git add .github/scripts/detect_stable_release.py
git add .github/AUTOMATED_SYNC_SUMMARY.md
git add CARTO_CLAUDE.md
git add SETUP_CLAUDE_SYNC.md

# Commit
git commit -m "feat: add Claude Code automated upstream sync

- AI-powered nightly sync with upstream LiteLLM
- Intelligent conflict resolution
- Automatic test failure fixes
- Detailed PR creation with context
- Preserves CARTO-specific modifications

Workflow runs nightly at 3 AM UTC and can be triggered manually.
Cost: ~\$3-5/month
"

# Push
git push origin carto/main
```

---

## Step 4: Test It

```bash
# Manually trigger the workflow
gh workflow run carto_claude_sync.yaml

# Watch it run (this may take 5-10 minutes)
gh run watch

# Or view status
gh run list --workflow=carto_claude_sync.yaml --limit 1
```

---

## Step 5: Review First PR

When Claude creates a PR:

1. Check the PR description - Claude explains:
   - What was merged
   - How conflicts were resolved
   - Test results
   - Whether it's ready or needs review

2. Look at the files changed

3. Read Claude's commit messages to understand reasoning

4. Merge if everything looks good!

---

## What Happens Next

### Every Night at 3 AM UTC:
- Workflow checks for new upstream `-stable` releases
- If found: Claude Code creates a PR
- You review and merge in the morning

### Most Nights (~90%):
- No new release
- Workflow exits early
- No cost

### When New Release (~10%):
- PR created automatically
- Check your PRs or Slack
- Review and merge

---

## Monitoring

**GitHub Actions:**
https://github.com/CartoDB/litellm/actions/workflows/carto_claude_sync.yaml

**Pull Requests:**
https://github.com/CartoDB/litellm/pulls?q=label%3Aautomated-sync

**Costs:**
https://console.anthropic.com/settings/billing

---

## Files Created

```
.github/
├── workflows/
│   ├── carto_claude_sync.yaml       ← Main workflow
│   └── README_CLAUDE_SYNC.md        ← Detailed setup guide
├── scripts/
│   └── detect_stable_release.py     ← Detection script
└── AUTOMATED_SYNC_SUMMARY.md        ← Full documentation

CARTO_CLAUDE.md                       ← Updated with sync info
SETUP_CLAUDE_SYNC.md                  ← This file
```

---

## Need Help?

**Documentation:**
- [CARTO_CLAUDE.md](CARTO_CLAUDE.md) - Main guide
- [AUTOMATED_SYNC_SUMMARY.md](.github/AUTOMATED_SYNC_SUMMARY.md) - Detailed overview
- [README_CLAUDE_SYNC.md](.github/workflows/README_CLAUDE_SYNC.md) - Setup guide

**Troubleshooting:**
- Check workflow logs for Claude's reasoning
- Review the troubleshooting section in CARTO_CLAUDE.md
- Claude's PR descriptions explain any issues

**Common Issues:**
- **No PRs?** → Probably no new upstream releases (normal!)
- **PR needs review?** → Claude found something complex (read description!)
- **Workflow fails?** → Check API key and credits

---

## Success Looks Like

✅ You wake up to PRs instead of manual sync work
✅ Conflicts are already resolved
✅ Test failures are fixed or explained
✅ Detailed context for every change
✅ Less than $5/month in costs

---

**Ready to start?** Just add the API key and commit! 🚀
