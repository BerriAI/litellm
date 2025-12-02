# CARTO_CLAUDE.md

> **AI Assistant Guide for CARTO's LiteLLM Fork**
>
> This document provides instructions specifically for AI assistants (like Claude Code) and developers working on CARTO's fork of LiteLLM. It documents the branching strategy, upstream sync process, CARTO-specific modifications, and common troubleshooting steps.

---

## Table of Contents

1. [Overview](#overview)
2. [Branch Strategy](#branch-strategy)
3. [Versioning Strategy](#versioning-strategy)
   - [Version Format](#version-format)
   - [Commit Format Guidelines](#commit-format-guidelines)
   - [How Versions are Calculated](#how-versions-are-calculated)
4. [Upstream Sync Process](#upstream-sync-process)
   - [Automated Nightly Sync](#automated-nightly-sync)
   - [Manual Sync](#manual-sync)
5. [CARTO-Specific Changes](#carto-specific-changes)
6. [Upstream-First Development Policy](#upstream-first-development-policy)
7. [Development Workflow](#development-workflow)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Quick Reference](#quick-reference)

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

## Versioning Strategy

CARTO uses a **hybrid semantic versioning** strategy that tracks both upstream LiteLLM versions and CARTO-specific customizations.

### Version Format

**Format:** `v{upstream}-carto.{MAJOR}.{MINOR}.{PATCH}`

**Components:**
- `{upstream}` - Upstream LiteLLM version (e.g., `1.79.1`)
- `{MAJOR}.{MINOR}.{PATCH}` - CARTO semantic version (cumulative, never resets)

**Example progression:**
```
v1.75.2-carto.1.0.0   ← First CARTO release on upstream 1.75.2
v1.75.2-carto.1.1.0   ← CARTO feature added
v1.75.2-carto.1.1.3   ← After 3 CARTO fixes
v1.79.1-carto.1.1.3   ← Upstream sync (CARTO version unchanged)
v1.79.1-carto.1.2.0   ← New CARTO feature
v1.79.1-carto.2.0.0   ← CARTO breaking change
```

**Key properties:**
- ✅ Upstream version shows which LiteLLM base is used
- ✅ CARTO version accumulates over time (never resets)
- ✅ Upstream syncs keep CARTO version, update upstream version
- ✅ Easy to see total CARTO divergence from upstream

---

### Commit Format Guidelines

**CRITICAL:** Proper commit messages are essential for correct version calculation.

CARTO uses **Conventional Commits** format to automatically calculate semantic versions:

#### Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Commit Types (affects versioning)

| Type | Version Bump | Use When | Example |
|------|-------------|----------|---------|
| `feat:` | **MINOR** | Adding new functionality | `feat: add Vertex AI labels support` |
| `fix:` | **PATCH** | Bug fixes | `fix: resolve Prisma CLI path issue` |
| `BREAKING CHANGE:` | **MAJOR** | Breaking API changes | `feat!: redesign authentication API`<br/>`BREAKING CHANGE: auth config format changed` |

#### Other Types (no version bump, but good practice)

- `docs:` - Documentation changes
- `style:` - Code style/formatting (no logic change)
- `refactor:` - Code refactoring (no behavior change)
- `perf:` - Performance improvements
- `test:` - Adding/updating tests
- `build:` - Build system changes
- `ci:` - CI/CD changes
- `chore:` - Maintenance tasks

#### Examples

✅ **GOOD:**
```
feat: add support for custom Vertex AI labels

Allows passing labels directly or converting from metadata
fields for better request tracking in Google Cloud.
```

```
fix: correct Prisma CLI path in Docker offline mode

The proxy now correctly locates pre-cached Prisma binaries,
eliminating npm download failures in air-gapped environments.
```

```
feat!: redesign authentication API

BREAKING CHANGE: Authentication configuration format has changed.
Users must update their config files to use new schema.

Migration guide: docs/migration-v2.md
```

❌ **BAD:**
```
updated stuff
```

```
fixes
```

```
WIP: testing things
```

#### Why This Matters

1. **Automatic version calculation** - Commits are analyzed to determine version bumps
2. **Clear release notes** - Well-formatted commits generate better release notes
3. **Change tracking** - Easy to see what changed and why
4. **Team communication** - Commit messages document decision-making

---

### How Versions are Calculated

Versions are calculated **automatically** during the release workflow using `.github/scripts/calculate_carto_version.sh`.

#### Calculation Process

1. **Find commits** since last CARTO release
   - Only counts commits from `@carto.com` or `@cartodb.com` authors
   - Excludes upstream sync merges (`sync:`, `Merge` commits)

2. **Analyze commit messages**
   - `BREAKING CHANGE:` / `breaking:` / `major:` → MAJOR bump
   - `feat:` / `feature:` → MINOR bump (resets PATCH to 0)
   - `fix:` / `bugfix:` → PATCH bump

3. **Apply bumps chronologically**
   - Commits are processed in order
   - Each bump follows semantic versioning rules

4. **Generate version tag**
   - Combines upstream version + CARTO version
   - Format: `v{upstream}-carto.{MAJOR}.{MINOR}.{PATCH}`

#### Example Calculation

**Starting point:** `v1.79.1-carto.1.7.1`

**New commits:**
```
- fix: resolve metadata null check         → PATCH bump: 1.7.1 → 1.7.2
- feat: add streaming improvements        → MINOR bump: 1.7.2 → 1.8.0
- fix: correct error handling             → PATCH bump: 1.8.0 → 1.8.1
```

**Result:** `v1.79.1-carto.1.8.1`

#### Upstream Sync Behavior

When upstream version changes (e.g., `1.79.1` → `1.79.3`):
- CARTO version **stays the same**
- Only upstream version changes
- `v1.79.1-carto.1.8.1` → `v1.79.3-carto.1.8.1`

This shows that:
- No new CARTO customizations were added
- Just syncing with newer upstream base

---

### Best Practices

1. **Always use conventional commits**
   - Helps with automatic versioning
   - Makes release notes better
   - Documents your intent

2. **Choose the right type**
   - `feat:` for new capabilities
   - `fix:` for bug fixes
   - `feat!:` or `BREAKING CHANGE:` for breaking changes

3. **Write clear descriptions**
   - Explain WHAT changed and WHY
   - Use present tense ("add" not "added")
   - Keep first line under 72 characters

4. **Test before committing**
   - Ensure your change works
   - Run `make lint` and `make test-unit`
   - Verify no unintended side effects

5. **Group related changes**
   - One logical change per commit
   - Don't mix features and fixes
   - Makes version bumps more meaningful

---

## Upstream Sync Process

CARTO uses an automated workflow to sync with upstream LiteLLM stable releases.

### Automated Upstream Sync

🔄 **Automated Sync:** CARTO runs an automated workflow that detects new upstream stable releases and creates sync PRs for team review.

#### How It Works

Every 8 hours, the `carto-upstream-sync.yml` workflow:

1. **Detects** new upstream stable releases (e.g., `v1.78.5-stable`)
   - Uses `gh CLI` to fetch releases from BerriAI/litellm
   - Skips nightlies, pre-releases, and release candidates
   - Only processes `-stable` tagged releases
   - All detection logic in bash (no Python dependencies)

2. **Syncs main branch** with upstream
   - Merges `BerriAI/litellm:main` → `CartoDB/litellm:main`
   - Pushes updated main branch automatically

3. **Creates PR** from main to carto/main
   - PR: `CartoDB/litellm:main` → `CartoDB/litellm:carto/main`
   - Detects if conflicts exist
   - Creates detailed PR with resolution guidelines
   - Labels PR appropriately (`upstream-sync`, `automated`)

4. **Provides comprehensive PR** with:
   - Link to upstream changes and release notes
   - Branch flow diagram
   - Summary of commits and files changed
   - Detailed conflict resolution guidelines
   - Testing checklist
   - Step-by-step resolution instructions

#### Conflict Handling Strategy

The workflow **detects but does not automatically resolve** conflicts. This ensures:
- ✅ No silent breaking changes
- ✅ Human review of important conflicts
- ✅ Clear documentation of what needs resolution
- ✅ Safe, conservative approach

When conflicts are detected, the PR includes detailed guidelines on which files to:
- **Keep CARTO versions:** `carto_*.yaml`, `CARTO_*.md`
- **Accept upstream:** Core `litellm/` code, `tests/`
- **Manually review:** `Dockerfile`, `Makefile` (check `# CARTO:` comments)

#### Setup

**Optional Variable (for Slack notifications):**
```bash
# Settings → Secrets and variables → Actions → Variables
# Name: SLACK_WEBHOOK_URL
# Value: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

**Note:** Slack notifications are optional. The workflow functions without them.

#### Monitoring

**GitHub Actions:**
- View runs: https://github.com/CartoDB/litellm/actions/workflows/carto-upstream-sync.yml
- Check workflow logs for detailed execution steps

**Pull Requests:**
- Automated PRs labeled: `upstream-sync`, `automated`
- Clean PRs: `upstream-sync`, `clean-merge`
- Conflict PRs: `upstream-sync`, `conflicts`

**Slack (if configured):**
- Success notifications with PR link
- Conflict alerts requiring attention
- Workflow run links for debugging

#### Manual Triggering

```bash
# Via GitHub CLI
gh workflow run carto-upstream-sync.yml

# Via GitHub UI:
# Actions → CARTO - Upstream Sync → Run workflow
```

**Note:** Schedule is currently disabled for initial testing. Will be enabled after successful manual testing.

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
- `.github/workflows/carto-upstream-sync.yml` - Automated upstream sync (runs every 8 hours, all bash)
- `.github/workflows/carto_ghcr_deploy.yaml` - CI/CD for CARTO Docker images
- `.github/workflows/carto_release.yaml` - Automated release creation

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

## Upstream-First Development Policy

**CRITICAL: Before implementing ANY fix or feature, check upstream first!**

This policy minimizes divergence from upstream, reduces duplicate work, and keeps the fork maintainable.

### 1. Check Upstream First

**Before implementing any fix or feature, ALWAYS:**

```bash
# 1. Search for related commits in upstream
git fetch upstream
git log upstream/main --grep="<keyword>" --oneline -50

# Example: For Vertex AI streaming issue
git log upstream/main --grep="vertex" --grep="streaming" --grep="chunk" --oneline -50

# 2. Check if specific file has been modified
git diff HEAD..upstream/main -- <file-path>

# Example: Check Vertex AI Gemini handler
git diff HEAD..upstream/main -- litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py

# 3. Search GitHub issues for related problems
gh issue list --search "<keywords>" --repo BerriAI/litellm --limit 20

# Example: Search for streaming issues
gh issue list --search "vertex streaming json" --repo BerriAI/litellm --limit 20

# 4. Search GitHub PRs (including closed ones)
gh pr list --search "<keywords>" --repo BerriAI/litellm --state all --limit 20

# Example: Search for related fixes
gh pr list --search "vertex json chunk" --repo BerriAI/litellm --state all --limit 20
```

### 2. Use Upstream Fix If Available

**If a fix exists in upstream:**

```bash
# Option A: Cherry-pick specific commit
git cherry-pick <commit-hash>

# Option B: Merge upstream changes
git fetch upstream
git merge upstream/main

# Option C: Cherry-pick range of commits
git cherry-pick <start-commit>..<end-commit>

# Always test thoroughly after applying upstream fix
make test-unit
make lint
```

### 3. Develop Custom Fix Only If

Proceed with a custom fix ONLY when:

- ✅ No upstream fix exists (verified via search above)
- ✅ Upstream fix doesn't apply to our fork
- ✅ Urgent CARTO-specific requirements
- ✅ Fix is specific to CARTO infrastructure

### 4. Document the Decision

**Always document why you're implementing a custom fix:**

```bash
# In commit message, include:
git commit -m "fix: resolve vertex streaming json parsing

No upstream fix available as of 2025-11-12.
Related upstream issues: #16037, #14747, #10410, #5650

Removes sent_first_chunk guard to allow JSON accumulation
on any partial chunk, not just the first one.

May submit PR to upstream to help community."
```

**In PR description, include:**
- Link to related upstream issues (if any)
- Confirmation that no upstream fix exists
- Date when upstream was last checked
- Consider submitting fix to upstream

### 5. Example: Vertex AI Gemini Streaming Bug

**Issue:** JSON parsing fails when Vertex AI sends partial chunks like `{` during streaming

**Upstream Check (2025-11-12):**
- ❌ No fix in upstream/main
- 📋 Related upstream issues: #16037, #14747, #10410, #5650
- 🔍 Verified file unchanged: `vertex_and_google_ai_studio_gemini.py`
- ⏳ Status: Bug exists in upstream, affecting multiple users

**Decision:** Implement custom fix in CARTO fork
**Rationale:** No upstream fix available, blocking production use
**Future:** Monitor upstream, consider submitting PR

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

### Upstream Sync Workflow Issues

**Problem:** Automated sync workflow encounters issues

#### Issue: Workflow Fails

**Symptoms:** Workflow fails to complete or exits with error.

**Solution:**

1. Check workflow logs:
   - https://github.com/CartoDB/litellm/actions/workflows/carto-upstream-sync.yml
   - Look for error messages in each job step
2. Try manual trigger:
   ```bash
   gh workflow run carto-upstream-sync.yml
   ```
3. If persistent, check for:
   - Network issues with GitHub API
   - Issues with upstream repository access
   - Malformed tags or unexpected release formats

#### Issue: No New Release Detected

**Symptoms:** Workflow runs but reports "No new stable release found" despite upstream having a `-stable` release.

**Solution:**

```bash
# Check if tag exists locally
git fetch upstream --tags
git tag -l "*-stable" | sort -V | tail -5

# Check latest upstream releases manually:
gh release list --repo BerriAI/litellm --limit 10 | grep stable

# Verify pyproject.toml version
grep '^version = ' pyproject.toml

# If new release should exist, manually trigger:
gh workflow run carto-upstream-sync.yml
```

#### Issue: PR Created with Conflicts

**Symptoms:** Sync PR is labeled with `conflicts`.

**Solution:**

This is expected when upstream changes conflict with CARTO modifications. Follow the resolution guide in the PR:

1. Checkout the PR branch:
   ```bash
   gh pr checkout <PR_NUMBER>
   ```
2. Review conflicts using the guidelines in the PR body
3. Resolve conflicts:
   ```bash
   # Fix the conflicts
   git add <files>
   git commit -m "resolve: conflicts from upstream sync"
   git push
   ```
4. Run tests to verify:
   ```bash
   make lint
   make test-unit
   ```

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

**Note:** Slack notifications are optional - the workflow works without them.

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

1. **CHECK UPSTREAM FIRST** - Before implementing any fix, follow the [Upstream-First Development Policy](#upstream-first-development-policy)
2. **Always check which branch you're on** before making changes
3. **Never merge main into carto/main** - only merge stable upstream tags
4. **Preserve CARTO-specific files** during upstream syncs:
   - `carto_*.yaml` workflows
   - `CARTO_*.md` documentation
   - Modified `Dockerfile` and `Makefile`
5. **Test thoroughly after upstream merges** - run full test suite
6. **Document changes** - update this file if you discover new patterns
7. **Use conventional commits** for clear history:
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

**Last Updated:** 2025-11-12
**Maintained By:** CARTO Engineering Team
**For:** AI Assistants & Developers working on CARTO's LiteLLM fork
