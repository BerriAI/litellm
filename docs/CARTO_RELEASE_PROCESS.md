# CARTO LiteLLM Fork - Release Process

This document describes the release workflow for the CARTO fork of LiteLLM.

## 📋 Table of Contents
- [Overview](#overview)
- [Branch Structure](#branch-structure)
- [Release Naming Convention](#release-naming-convention)
- [Development Workflow](#development-workflow)
- [Creating a Release](#creating-a-release)
- [Docker Image Tags](#docker-image-tags)
- [Syncing with Upstream](#syncing-with-upstream)
- [Branch Protection Rules](#branch-protection-rules)

---

## Overview

Our fork uses a **hybrid upstream-friendly workflow** that balances:
- ✅ Easy deployment of CARTO-specific fixes and features
- ✅ Clear separation from upstream changes
- ✅ Simple release process with semantic versioning
- ✅ Future-proof for upstream syncs

## Branch Structure

```
main              → Tracks upstream BerriAI/litellm (read-only for CARTO changes)
carto/main        → CARTO production branch (protected)
feature/*         → Feature/fix branches (created from carto/main)
```

### Branch Purposes

| Branch | Purpose | Protection |
|--------|---------|------------|
| `main` | Tracks upstream, no CARTO commits | Read-only for CARTO |
| `carto/main` | CARTO production, all releases | Protected, requires PR + approval |
| `feature/*` | Development branches | None |

---

## Release Naming Convention

```
carto-v{UPSTREAM_VERSION}-{CARTO_SEMVER}
```

### Examples:
- `carto-v1.75.2-0.1.0` - First CARTO release on upstream 1.75.2
- `carto-v1.75.2-0.2.0` - Second release (minor feature added)
- `carto-v1.75.2-0.2.1` - Patch/hotfix release
- `carto-v1.77.4-0.1.0` - First release after syncing to upstream 1.77.4

### CARTO Semver (X.Y.Z):
- **Major (X)**: Breaking changes in CARTO-specific code
- **Minor (Y)**: New features, non-breaking changes
- **Patch (Z)**: Bug fixes, small improvements

---

## Development Workflow

### 1. Create a Feature Branch

```bash
# Make sure you're on carto/main
git checkout carto/main
git pull origin carto/main

# Create feature branch
git checkout -b feature/my-awesome-fix
```

### 2. Make Changes & Commit

Follow conventional commit format:

```bash
git commit -m "fix: resolve MCP migration issue"
git commit -m "feat: add prisma binary caching"
git commit -m "chore: update dependencies"
```

### 3. Push & Create PR

```bash
git push origin feature/my-awesome-fix
```

Create a PR to `carto/main` via GitHub UI.

### 4. Merge to carto/main

Once approved:
- PR is merged to `carto/main`
- CI automatically builds Docker image with tags:
  - `carto-main-latest`
  - `carto-main-<short-sha>`

---

## Creating a Release

### When to Create a Release

- After merging one or more fixes/features to `carto/main`
- Ready to deploy to production
- Want a stable version tag

### Release Process

1. **Go to GitHub Actions**
   - Navigate to: `https://github.com/CartoDB/litellm/actions/workflows/carto_release.yaml`

2. **Click "Run workflow"**

3. **Fill in the inputs:**
   - **Use workflow from:** `carto/main`
   - **Version bump type:** `patch`, `minor`, or `major`
   - **Upstream version:** Current LiteLLM version (e.g., `1.75.2`)

4. **Run the workflow**

The workflow will automatically:
- ✅ Calculate the next semantic version
- ✅ Generate release notes from commits
- ✅ Create a Git tag (e.g., `carto-v1.75.2-0.1.0`)
- ✅ Create a GitHub Release
- ✅ Build and push Docker images with multiple tags

### Example

If the latest tag is `carto-v1.75.2-0.1.0` and you select **patch**:
- New tag: `carto-v1.75.2-0.1.1`
- Docker tags:
  - `carto-v1.75.2-0.1.1`
  - `carto-stable` (always points to latest release)
  - `carto-v1.75.2-latest` (latest for this upstream version)

---

## Docker Image Tags

### CI Tags (auto-deployed on every push to carto/main)
```
ghcr.io/cartodb/litellm-non_root:carto-main-latest
ghcr.io/cartodb/litellm-non_root:carto-main-<sha>
```

### Release Tags (created via manual release workflow)
```
ghcr.io/cartodb/litellm-non_root:carto-v1.75.2-0.1.0  # Specific version
ghcr.io/cartodb/litellm-non_root:carto-stable         # Latest release
ghcr.io/cartodb/litellm-non_root:carto-v1.75.2-latest # Latest for upstream v1.75.2
```

### Usage in Kubernetes/Docker

**Development:**
```yaml
image: ghcr.io/cartodb/litellm-non_root:carto-main-latest
```

**Production (recommended):**
```yaml
image: ghcr.io/cartodb/litellm-non_root:carto-v1.75.2-0.1.0  # Pin to specific version
```

**Production (auto-update to latest stable):**
```yaml
image: ghcr.io/cartodb/litellm-non_root:carto-stable
```

---

## Syncing with Upstream

### Automatic Sync (Recommended)

CARTO maintains an **automated upstream sync workflow** that runs every 8 hours to detect new stable releases from BerriAI/litellm.

#### How It Works

1. **Detection**: Workflow checks for new `-stable` tags from upstream
2. **PR Creation**: Automatically creates a sync PR with:
   - Merge attempt from upstream stable tag
   - Conflict detection and resolution guidelines
   - Comprehensive testing checklist
3. **Review**: Team reviews and tests the sync PR
4. **Merge**: Once approved, PR is merged to `main`

#### Workflow Status

Check the latest sync status:
- **Workflow:** [`carto-upstream-sync.yml`](../.github/workflows/carto-upstream-sync.yml)
- **Actions:** https://github.com/CartoDB/litellm/actions/workflows/carto-upstream-sync.yml

#### Manual Trigger

You can manually trigger the sync workflow:

```bash
# Via GitHub UI
1. Go to Actions → CARTO - Upstream Sync
2. Click "Run workflow"
3. Select branch: main
4. Click "Run workflow"

# Via GitHub CLI
gh workflow run carto-upstream-sync.yml
```

#### Handling Sync PRs

When a sync PR is created:

**If Clean Merge (✅ `clean-merge` label):**
1. Review changes in the PR
2. Run tests locally or wait for CI
3. Approve and merge

**If Conflicts (⚠️ `conflicts` label):**
1. Pull the sync branch locally:
   ```bash
   git fetch origin
   git checkout upstream-sync/v{version}
   ```

2. Resolve conflicts following PR guidelines:
   - **Keep CARTO versions:** `carto_*.yaml`, `CARTO_*.md`
   - **Accept upstream:** Core `litellm/` code, `tests/`
   - **Manual review:** `Dockerfile`, `Makefile` (check `# CARTO:` comments)

3. Test your changes:
   ```bash
   make lint
   make test-unit
   docker build -f docker/Dockerfile.non_root .
   ```

4. Push resolved changes:
   ```bash
   git add .
   git commit -m "resolve: conflicts from upstream sync"
   git push origin upstream-sync/v{version}
   ```

5. Get approval and merge

#### After Syncing to main

Once the sync PR is merged to `main`:

1. **Sync main → carto/main** (currently manual):
   ```bash
   git checkout carto/main
   git merge main
   # Resolve any conflicts
   git push origin carto/main
   ```

2. **Create first release for new upstream version**:
   - Run release workflow
   - Set upstream_version to match new version (e.g., `1.78.5`)
   - Select bump type: `patch`
   - This creates: `carto-v1.78.5-0.1.0`

---

### Manual Sync (Fallback)

If you need to sync manually (e.g., automation is broken):

1. **Fetch upstream changes**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git push origin main
   ```

2. **Merge upstream into carto/main**
   ```bash
   git checkout carto/main
   git merge main
   # Resolve any conflicts
   git push origin carto/main
   ```

3. **Update pyproject.toml version**
   ```bash
   # Update version in pyproject.toml to match new upstream version
   vim pyproject.toml  # Change version = "1.78.5"
   git commit -am "chore: sync to upstream v1.78.5"
   git push origin carto/main
   ```

4. **Create first release for new upstream version**
   - Run release workflow
   - Set upstream_version to `1.78.5`
   - Select bump type: `patch`
   - This creates: `carto-v1.78.5-0.1.0`

---

## Branch Protection Rules

### Settings for `carto/main` (via GitHub UI)

Navigate to: `Settings → Branches → Branch protection rules → Add rule`

**Branch name pattern:** `carto/main`

**Protections:**
- ✅ Require a pull request before merging
  - Require approvals: 1
  - Dismiss stale pull request approvals when new commits are pushed
- ✅ Require status checks to pass before merging
  - Require branches to be up to date before merging
  - Status checks: (add your CI checks here)
- ✅ Do not allow bypassing the above settings
- ✅ Restrict who can push to matching branches
  - Add: Repository admins only

### Settings for `main` (optional, for upstream tracking)

**Branch name pattern:** `main`

**Protections:**
- ✅ Require a pull request before merging
  - Note: Only merge upstream changes here, no CARTO commits
- ✅ Lock branch (to prevent accidental CARTO commits)

---

## FAQ

### Q: How do I know what version to use for a release?
**A:** The workflow auto-calculates it! Just choose patch/minor/major.

### Q: What if I need to hotfix a production release?
**A:** Create a feature branch, merge to `carto/main`, then create a **patch** release.

### Q: Can I skip versions?
**A:** No, versions are sequential. The workflow always increments from the latest tag.

### Q: What if the upstream version changes?
**A:** The automated sync workflow will detect new stable releases and create a PR. Review, test, and merge the PR. Then update `carto/main` and create a new release with the upstream version.

### Q: How do I see what changed between releases?
**A:** Check the GitHub Releases page - release notes are auto-generated from commits.

### Q: How often does the auto-sync check for updates?
**A:** Every 8 hours. You can also trigger it manually via GitHub Actions.

### Q: What if the auto-sync PR has conflicts?
**A:** The PR will be labeled with `conflicts` and include detailed resolution guidelines. Follow the instructions in the PR to resolve conflicts locally, test, and push.

---

## Support

For questions or issues with the release process:
1. Check this documentation
2. Review existing releases: https://github.com/CartoDB/litellm/releases
3. Contact the team via #litellm channel

---

**Last Updated:** 2025-10-27
