# CARTO_CLAUDE.md

> **AI Assistant Guide for CARTO's LiteLLM Fork**
>
> This document provides instructions specifically for AI assistants (like Claude Code) and developers working on CARTO's fork of LiteLLM. It documents the branching strategy, upstream sync process, CARTO-specific modifications, and common troubleshooting steps.

---

## Table of Contents

1. [Overview](#overview)
2. [Branch Strategy](#branch-strategy)
3. [Upstream Sync Process](#upstream-sync-process)
   - [Automated Nightly Sync](#automated-nightly-sync)
   - [Manual Sync](#manual-sync)
4. [CARTO-Specific Changes](#carto-specific-changes)
5. [Development Workflow](#development-workflow)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Quick Reference](#quick-reference)

---

## Overview

CARTO maintains a fork of [BerriAI/litellm](https://github.com/BerriAI/litellm) to:
- Deploy custom fixes and features faster
- Ensure greater stability for CARTO's AI infrastructure
- Build on stable upstream releases (not bleeding-edge main)

**Key Principle:**
> Production deployments (`carto/main`) are always based on **stable upstream release tags** (e.g., `v1.75.2`), NOT upstream's main branch.

---

## Branch Strategy

```
upstream/main          → BerriAI's development branch (may be unstable)
         ↓
       main            → Mirrors upstream/main (tracking/reference only)

upstream/v1.75.2       → Stable upstream release tag
         ↓
    carto/main         → Production branch (stable tag + CARTO mods)
         ↓
    feature/*          → Development branches
```

### Branch Purposes

| Branch | Purpose | Base | Stability |
|--------|---------|------|-----------|
| `main` | Track upstream development | `upstream/main` | Unstable (reference only) |
| `carto/main` | CARTO production deployments | Stable upstream tags | Stable |
| `feature/*` | Development work | `carto/main` | Development |

### Critical Rules

⚠️ **NEVER merge `main` into `carto/main`** - `main` may contain unstable upstream commits

✅ **ALWAYS merge stable upstream release tags** (e.g., `v1.76.5`) into `carto/main`

---

## Upstream Sync Process

CARTO uses an AI-powered automated workflow to sync with upstream LiteLLM stable releases.

### Automated Claude Code Sync

🤖 **New in 2025:** CARTO uses Claude Code (AI) to automatically handle upstream syncing - from detection to conflict resolution to PR creation.

#### How It Works

Every night at 3 AM UTC, the `carto_claude_sync.yaml` workflow:

1. **Detects** new upstream stable releases (e.g., `v1.78.5-stable`)
   - Skips nightlies and release candidates
   - Only looks for `-stable` tagged releases
   - Uses detection script: `.github/scripts/detect_stable_release.py`

2. **Creates branch and merges** the stable tag
   - Automatically handles merge conflicts intelligently
   - Preserves CARTO-specific workflows (`carto_*.yaml`)
   - Keeps CARTO modifications in Dockerfile/Makefile
   - Updates `pyproject.toml` version

3. **Runs tests and fixes issues** (if possible)
   - Runs `make lint` and `make test-unit`
   - Attempts to fix linting errors automatically
   - Attempts to fix test failures
   - Analyzes and reports issues it can't fix

4. **Creates detailed PR** with:
   - Link to upstream changes
   - Summary of what was merged
   - Conflict resolution details (how and why)
   - Test results and recommendations
   - Context-aware suggestions for review

#### What Makes It Intelligent

Claude Code reads and understands:
- **CARTO_CLAUDE.md** - This documentation
- **CARTO conventions** - Workflow naming, file structure
- **Conflict context** - Why conflicts occurred and how to resolve them
- **Test failures** - Root causes and potential fixes

This means it can make smart decisions like:
- "This upstream workflow conflicts with our custom one - keep ours"
- "This test fails because the upstream API changed - update our mock"
- "This dependency version conflict needs manual review"

#### Setup

**Required Secret:**
```bash
# Add to GitHub repository secrets:
# Settings → Secrets and variables → Actions → Secrets
# Name: ANTHROPIC_API_KEY
# Value: (get from https://console.anthropic.com/settings/keys)
```

**Optional Variable (for Slack notifications):**
```bash
# Settings → Secrets and variables → Actions → Variables
# Name: SLACK_WEBHOOK_URL
# Value: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

**See:** [.github/workflows/README_CLAUDE_SYNC.md](.github/workflows/README_CLAUDE_SYNC.md) for full setup guide.

#### Cost
- Typical sync: $0.15-$1.50 per run
- Most nights are no-ops (no new releases detected)
- Upstream releases ~2-4 per month
- **Estimated: $3-$5/month**

#### Monitoring

**GitHub Actions:**
- View runs: https://github.com/CartoDB/litellm/actions/workflows/carto_claude_sync.yaml
- Check workflow logs to see Claude's reasoning

**Pull Requests:**
- Automated PRs labeled: `automated-sync`
- Ready PRs: `automated-sync,ready`
- Review needed: `automated-sync,needs-review`

**Slack (if configured):**
- Success notifications with PR link
- Failure alerts with workflow run link

#### Manual Triggering

```bash
# Via GitHub CLI
gh workflow run carto_claude_sync.yaml

# Via GitHub UI:
# Actions → CARTO Claude Code Upstream Sync → Run workflow
```

---

### Manual Sync (When Needed)

If Claude Code needs help or you want to sync manually:

#### A. Regular Monitoring (Keep `main` Updated)

**Purpose:** Track what upstream is working on (for awareness)

```bash
# Fetch latest upstream changes
git fetch upstream

# Update local main
git checkout main
git merge upstream/main
git push origin main
```

**Frequency:** Weekly or as needed for monitoring

**Note:** This does NOT affect `carto/main` - it's purely for tracking upstream development.

---

### B. Production Upgrade (Update `carto/main` to New Stable Release)

**Purpose:** Upgrade CARTO's production branch to a new stable upstream version

#### Step 1: Identify Stable Upstream Release

```bash
# List available upstream releases
git fetch upstream --tags
git tag -l | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -10

# Or check GitHub releases
# https://github.com/BerriAI/litellm/releases
```

**Choose a stable release tag** (e.g., `v1.76.5`)

#### Step 2: Merge Stable Tag into `carto/main`

```bash
# Ensure carto/main is clean
git checkout carto/main
git pull origin carto/main
git status  # Should be clean

# Merge the specific stable tag (NOT main!)
git fetch upstream --tags
git merge v1.76.5

# If conflicts occur, see Troubleshooting Guide below
```

#### Step 3: Resolve Conflicts (if any)

Common conflict areas:
- `.github/workflows/` - Keep CARTO workflows, discard upstream's
- `Dockerfile` - Preserve CARTO modifications
- `Makefile` - Keep CARTO-specific commands
- `pyproject.toml` - Update version to match upstream tag

```bash
# After resolving conflicts
git add .
git commit -m "chore: merge upstream v1.76.5 into carto/main"
```

#### Step 4: Update Version and Test

```bash
# Update pyproject.toml version to match upstream
vim pyproject.toml  # Change version = "1.76.5"

# Run tests to ensure everything works
make install-dev
make test-unit
make lint

# Commit version update
git commit -am "chore: update version to 1.76.5"
git push origin carto/main
```

#### Step 5: Create CARTO Release

Once merged and tested, create a new CARTO release:

1. Go to: https://github.com/CartoDB/litellm/actions/workflows/carto_release.yaml
2. Click "Run workflow"
3. Select `carto/main` branch
4. Choose bump type: `patch` (for first release on new upstream version)
5. This creates: `carto-v1.76.5-0.1.0`

**See:** [docs/CARTO_RELEASE_PROCESS.md](docs/CARTO_RELEASE_PROCESS.md) for detailed release instructions.

---

## CARTO-Specific Changes

These modifications exist in `carto/main` but NOT in upstream. Be careful to preserve them during merges.

### 1. Custom GitHub Workflows

**Added:**
- `.github/workflows/carto_ghcr_deploy.yaml` - CI/CD for CARTO Docker images
- `.github/workflows/carto_release.yaml` - Automated release creation
- `.github/workflows/carto_claude_sync.yaml` - **NEW:** AI-powered automated nightly upstream sync
- `.github/scripts/detect_stable_release.py` - Script to detect new stable upstream releases

**Disabled/Modified:**
- `.github/workflows/ghcr_deploy.yml` → `.github/workflows/ghcr_deploy.yml.txt` (disabled)
- `.github/workflows/ghcr_helm_deploy.yml` → `.github/workflows/ghcr_helm_deploy.yml.txt` (disabled)
- `.github/workflows/helm_unit_test.yml` → `.github/workflows/helm_unit_test.yml.txt` (disabled)

**Reason:** CARTO uses custom workflows with different Docker registry and tagging strategy.

### 2. Custom Documentation

**Added:**
- `CARTO_CLAUDE.md` (this file) - AI assistant guide
- `docs/CARTO_RELEASE_PROCESS.md` - Release workflow documentation
- `APSCHEDULER_MEMORY_LEAK_FIX.md` - Documents APScheduler memory fix
- `REDIS_SESSION_PATCH.md` - Redis session handling improvements
- `RESPONSES_API_TEST_README.md` - Testing guide for Responses API

### 3. Dockerfile Modifications

- Modified base image or build steps for CARTO infrastructure
- Check `Dockerfile` for CARTO-specific comments

### 4. Makefile Changes

- Custom development commands
- CARTO-specific test configurations

### 5. Security & Infrastructure

- Removed upstream security scanning workflows
- Custom secret management setup

---

## Development Workflow

### Initial Setup

```bash
# Clone the fork
git clone https://github.com/CartoDB/litellm.git
cd litellm

# Verify remotes
git remote -v
# Should show:
# origin    https://github.com/CartoDB/litellm.git
# upstream  https://github.com/BerriAI/litellm.git

# If upstream is missing, add it:
git remote add upstream https://github.com/BerriAI/litellm.git

# Checkout carto/main
git checkout carto/main
git pull origin carto/main

# Install dependencies
make install-dev
```

### Creating a Feature Branch

```bash
# Always branch from carto/main
git checkout carto/main
git pull origin carto/main

# Create feature branch
git checkout -b feature/my-awesome-fix

# Make changes, commit using conventional commits
git commit -m "fix: resolve authentication bug"
git commit -m "feat: add new caching layer"

# Push and create PR to carto/main
git push origin feature/my-awesome-fix
```

### Testing

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests
make test-integration

# Linting
make lint

# Format code
make format
```

### Creating a Pull Request

1. Push feature branch to origin
2. Open PR against `carto/main` (NOT main!)
3. Ensure CI passes
4. Request review from team
5. Merge when approved

---

## Troubleshooting Guide

### Merge Conflicts During Upstream Sync

**Problem:** Conflicts when merging upstream tag into `carto/main`

**Solution:**

```bash
# Check which files have conflicts
git status

# For workflow files (.github/workflows/*):
# - Keep CARTO versions (carto_*.yaml)
# - Discard or rename upstream versions

# For Dockerfile, Makefile:
# - Carefully preserve CARTO modifications
# - Look for comments like "# CARTO:" in code

# For pyproject.toml:
# - Accept upstream version number
# - Keep CARTO-specific dependencies if any

# After resolving each file:
git add <file>

# Complete merge
git commit
```

### Claude Code Sync Issues

**Problem:** AI-powered sync workflow encounters issues or makes unexpected decisions

#### Issue: Claude Code Workflow Fails

**Symptoms:** Workflow fails to complete or exits with error.

**Solution:**

1. Check workflow logs for Claude's reasoning:
   - https://github.com/CartoDB/litellm/actions/workflows/carto_claude_sync.yaml
   - Look for error messages in the "Run Claude Code Sync" step
2. Verify API key is set:
   - Settings → Secrets → Check `ANTHROPIC_API_KEY` exists
3. Check API credits at https://console.anthropic.com/
4. If persistent, try manual trigger:
   ```bash
   gh workflow run carto_claude_sync.yaml
   ```

#### Issue: No New Release Detected

**Symptoms:** Workflow runs but reports "No new stable release found" despite upstream having a `-stable` release.

**Solution:**

```bash
# Check if tag exists locally and might be hiding new release
git tag -l | grep -stable

# Check the detection script manually:
python .github/scripts/detect_stable_release.py

# If new release should exist, manually trigger:
gh workflow run carto_claude_sync.yaml
```

#### Issue: Claude Creates PR Needing Review

**Symptoms:** PR is created but labeled `needs-review` instead of `ready`.

**Solution:**

This is normal and intentional! Claude Code is conservative and requests review when:
- Conflicts were complex
- Tests failed despite attempts to fix
- Upstream changes are significant
- Uncertain about best resolution approach

**What to do:**
1. Read Claude's PR description carefully - it explains why review is needed
2. Review the changes and Claude's reasoning
3. Check workflow logs to see what Claude tried
4. Make any needed adjustments
5. Merge when satisfied

This is actually a feature - Claude won't merge changes it's not confident about!

#### Issue: Conflicts Not Fully Resolved

**Symptoms:** PR has conflicts that Claude didn't resolve correctly.

**Solution:**

1. Checkout the PR branch:
   ```bash
   gh pr checkout <PR_NUMBER>
   ```
2. Review Claude's conflict resolution in the commits
3. Make corrections if needed:
   ```bash
   # Fix the conflicts
   git add <files>
   git commit -m "fix: correct conflict resolution"
   git push
   ```
4. Comment on PR explaining what you changed and why
5. This helps Claude learn for future syncs!

#### Issue: Tests Fail After Claude's Changes

**Symptoms:** PR created but tests are failing.

**Solution:**

1. Review Claude's attempts to fix tests in the PR description
2. Check workflow logs to see what Claude tried
3. Checkout branch and fix locally:
   ```bash
   gh pr checkout <PR_NUMBER>
   make lint
   make test-unit
   # Fix issues
   git push
   ```
4. Claude may have left helpful comments in the code explaining issues it couldn't fix

#### Issue: Slack Notifications Not Received

**Symptoms:** Workflow runs but no Slack messages appear.

**Solution:**

1. Check if `SLACK_WEBHOOK_URL` is set in repository variables (NOT secrets):
   - Settings → Secrets and variables → Actions → Variables
2. Verify webhook URL is valid:
   ```bash
   curl -X POST "$SLACK_WEBHOOK_URL" \
     -H "Content-Type: application/json" \
     -d '{"text": "Test message from CARTO LiteLLM"}'
   ```
3. If missing, add the variable and re-run workflow

#### Issue: Claude Made Unexpected Changes

**Symptoms:** Changes in PR don't match expectations.

**Solution:**

1. Read Claude's reasoning in:
   - PR description (explains what and why)
   - Commit messages (step-by-step reasoning)
   - Workflow logs (detailed thought process)
2. If changes are incorrect:
   - Comment on the PR explaining the issue
   - Update `CARTO_CLAUDE.md` to clarify expectations
   - This helps Claude make better decisions next time
3. Fix the changes and merge

Remember: Claude learns from the documentation and context - improve docs to improve future syncs!

### Version Mismatch Errors

**Problem:** `pyproject.toml` version doesn't match expected release

**Solution:**

```bash
# Check current version
grep '^version = ' pyproject.toml

# Update to match upstream tag you merged
# If you merged v1.76.5, set version = "1.76.5"
vim pyproject.toml

git commit -am "chore: update version to match upstream"
```

### Docker Build Failures

**Problem:** CI fails to build Docker image after upstream merge

**Solution:**

1. Check if upstream changed Dockerfile structure
2. Compare `git diff v1.75.2..v1.76.5 -- Dockerfile`
3. Re-apply CARTO modifications if needed
4. Test locally: `docker build -t test .`

### Prisma Migration Issues

**Problem:** Database schema conflicts after upstream merge

**Solution:**

```bash
# Generate new migration
poetry run prisma migrate dev --name sync_upstream_changes

# Review migration files in litellm/proxy/prisma/migrations/
# Test against both PostgreSQL and SQLite

# Commit migration
git add litellm/proxy/prisma/migrations/
git commit -m "fix: update Prisma schema after upstream sync"
```

### Test Failures After Merge

**Problem:** Tests fail after merging new upstream version

**Solution:**

```bash
# Run tests with verbose output
poetry run pytest tests/ -v -s

# Check if upstream changed test requirements
git diff v1.75.2..v1.76.5 -- tests/

# Update CARTO-specific tests if needed
# Look for tests in tests/ that reference CARTO modifications

# Re-run specific failing test
poetry run pytest tests/path/to/test_file.py::test_function -v
```

### "Detached HEAD" State

**Problem:** Accidentally checked out a tag directly

**Solution:**

```bash
# Check current state
git status

# Return to carto/main
git checkout carto/main

# If you made commits in detached HEAD, create a branch:
git checkout -b recovery-branch <commit-sha>
git checkout carto/main
git merge recovery-branch
```

---

## Quick Reference

### Essential Commands

```bash
# Sync main with upstream (monitoring only)
git fetch upstream && git checkout main && git merge upstream/main && git push origin main

# Upgrade carto/main to new stable release
git checkout carto/main && git merge v1.76.5

# Create feature branch
git checkout carto/main && git pull && git checkout -b feature/my-fix

# Run tests
make test-unit

# Lint code
make lint

# Install dependencies
make install-dev
```

### Important Links

- **Release Process:** [docs/CARTO_RELEASE_PROCESS.md](docs/CARTO_RELEASE_PROCESS.md)
- **CARTO Releases:** https://github.com/CartoDB/litellm/releases
- **Docker Images:** https://github.com/CartoDB/litellm/pkgs/container/litellm-non_root
- **Upstream Repo:** https://github.com/BerriAI/litellm
- **Upstream Releases:** https://github.com/BerriAI/litellm/releases

### Docker Image Tags

**Development (auto-built on push to carto/main):**
- `ghcr.io/cartodb/litellm-non_root:carto-main-latest`
- `ghcr.io/cartodb/litellm-non_root:carto-main-<sha>`

**Production (created via release workflow):**
- `ghcr.io/cartodb/litellm-non_root:carto-v1.75.2-0.1.0` (specific version)
- `ghcr.io/cartodb/litellm-non_root:carto-stable` (latest release)
- `ghcr.io/cartodb/litellm-non_root:carto-v1.75.2-latest` (latest for upstream v1.75.2)

### Git Remotes

```bash
origin    → https://github.com/CartoDB/litellm.git (CARTO fork)
upstream  → https://github.com/BerriAI/litellm.git (BerriAI original)
```

### Current Status Check

```bash
# What branch am I on?
git branch --show-current

# What's the current version?
grep '^version = ' pyproject.toml

# What's the latest CARTO release?
git describe --tags --match "carto-*" --abbrev=0

# What's the latest upstream release?
git ls-remote --tags upstream | grep -E 'refs/tags/v[0-9]+\.[0-9]+\.[0-9]+$' | tail -5
```

---

## Best Practices for AI Assistants

When working on this codebase:

1. **Always check which branch you're on** before making changes
2. **Never merge main into carto/main** - only merge stable upstream tags
3. **Preserve CARTO-specific files** during upstream syncs:
   - `carto_*.yaml` workflows
   - `CARTO_*.md` documentation
   - Modified `Dockerfile` and `Makefile`
4. **Test thoroughly after upstream merges** - run full test suite
5. **Document changes** - update this file if you discover new patterns
6. **Use conventional commits** for clear history:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `chore:` for maintenance
   - `docs:` for documentation

---

## Support

For questions about this fork:
1. Check this documentation
2. Review [docs/CARTO_RELEASE_PROCESS.md](docs/CARTO_RELEASE_PROCESS.md)
3. Check existing GitHub issues: https://github.com/CartoDB/litellm/issues
4. Contact the CARTO AI team

---

**Last Updated:** 2025-10-27
**Maintained By:** CARTO Engineering Team
**For:** AI Assistants & Developers working on CARTO's LiteLLM fork
