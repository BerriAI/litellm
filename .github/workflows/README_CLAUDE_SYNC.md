# Claude Code Automated Upstream Sync

This workflow uses [Claude Code GitHub Action](https://github.com/anthropics/claude-code-action) to automatically handle upstream syncs from BerriAI/litellm.

## How It Works

Every night at 3 AM UTC, Claude Code:
1. Checks for new upstream `-stable` releases
2. Creates a branch and merges the new release
3. Intelligently resolves merge conflicts (preserving CARTO changes)
4. Updates `pyproject.toml` version
5. Runs tests (`make lint` and `make test-unit`)
6. Creates a PR with detailed summary and recommendations

## Setup

### 1. Install Anthropic API Key

The workflow requires an Anthropic API key to use Claude Code.

**Option A: Organization-level (Recommended for CARTO)**
```bash
# Go to GitHub Organization Settings
# → Secrets and variables → Actions → Secrets
# Add organization secret: ANTHROPIC_API_KEY
```

**Option B: Repository-level**
```bash
# Go to Repository Settings
# → Secrets and variables → Actions → Secrets
# Add repository secret: ANTHROPIC_API_KEY
```

Get your API key from: https://console.anthropic.com/settings/keys

### 2. Configure Slack Notifications (Optional)

Add Slack webhook as a repository variable (NOT secret):
```bash
# Repository Settings → Secrets and variables → Actions → Variables
# Name: SLACK_WEBHOOK_URL
# Value: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### 3. Verify Workflow Permissions

Ensure the workflow has required permissions (should be automatic):
- ✅ Contents: write
- ✅ Pull requests: write
- ✅ Issues: write
- ✅ ID token: write

## Testing

### Manual Trigger
```bash
# Trigger via GitHub CLI
gh workflow run carto_claude_sync.yaml

# Or via GitHub UI:
# Actions → CARTO Claude Code Upstream Sync → Run workflow
```

### First Run
After setup:
1. Manually trigger the workflow to test
2. Check the workflow logs to see Claude Code in action
3. Review any PRs created
4. Verify Slack notifications (if configured)

## What Claude Code Does

### Conflict Resolution
Claude Code intelligently handles conflicts:
- **Workflows**: Always keeps CARTO versions (`carto_*.yaml`)
- **Dockerfile/Makefile**: Preserves CARTO modifications
- **pyproject.toml**: Updates version but keeps CARTO dependencies
- **Other files**: Analyzes and resolves based on context

### Testing
Runs comprehensive tests:
```bash
make lint      # Ruff, MyPy, Black, circular imports
make test-unit # Unit tests with 4 parallel workers
```

### PR Creation
Creates detailed PRs with:
- Link to upstream changes
- Summary of merged changes
- Conflict resolution details
- Test results
- Recommendation (ready to merge / needs review)

## Monitoring

### GitHub Actions
- View runs: https://github.com/CartoDB/litellm/actions/workflows/carto_claude_sync.yaml
- Check workflow logs to see Claude Code's reasoning

### Pull Requests
- Automated PRs labeled: `automated-sync`
- Ready PRs: `automated-sync,ready`
- Review needed: `automated-sync,needs-review`

### Slack (if configured)
- Success notifications
- Failure alerts with workflow run link

## Why Claude Code?

Traditional scripted approaches fail when:
- Merge conflicts need contextual understanding
- Tests break and need code fixes
- Changes require judgment calls
- Detailed explanations are needed

Claude Code succeeds because it:
- **Reads documentation** - Understands CARTO_CLAUDE.md conventions
- **Makes smart decisions** - Analyzes conflict context
- **Fixes issues** - Can modify code to resolve problems
- **Explains reasoning** - Provides detailed PR descriptions
- **Asks for help** - Knows when human review is needed

## Troubleshooting

### Workflow Fails with "Anthropic API Key Missing"
1. Verify `ANTHROPIC_API_KEY` is set in secrets
2. Check it's available at organization or repository level
3. Ensure it hasn't expired

### Claude Code Creates PRs Needing Review
This is normal! Claude Code is conservative when:
- Conflicts are complex
- Tests fail after resolution attempts
- Upstream changes are significant

Review the PR and Claude's reasoning in comments.

### No PRs Created
Check workflow logs:
- May be no new stable upstream releases
- Detection script may have found issues
- Claude Code may have determined no action needed

## Cost Estimation

Claude Code uses API credits based on:
- Tokens processed (code read/written)
- Typical sync: ~50k-200k tokens
- Approximate cost: $0.15-$1.50 per sync (Sonnet 3.5)

With nightly runs:
- Monthly cost: ~$5-$45
- Most nights will be no-ops (no new releases)
- Actual cost likely $10-$20/month

## Resources

- [Claude Code Action Docs](https://github.com/anthropics/claude-code-action)
- [CARTO Sync Documentation](../../CARTO_CLAUDE.md)
- [Release Process](../../docs/CARTO_RELEASE_PROCESS.md)
- [Anthropic API Docs](https://docs.anthropic.com/)

---

**Last Updated:** 2025-10-27
