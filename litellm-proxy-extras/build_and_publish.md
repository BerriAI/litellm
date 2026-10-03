# Build & Publish `litellm-proxy-extras`

This runbook covers building and publishing a new version of the `litellm-proxy-extras` PyPI package. For use by litellm engineers only.

## Dashboard assets

Before building, run `./build_ui.sh` from `ui/litellm-dashboard` with the pinned Node environment. It writes the dashboard into `litellm-proxy-extras/litellm_proxy_extras/ui/`

Check that the extras wheel contains `ui/index.html` and `ui/_next/`, and that the core LiteLLM wheel contains no dashboard assets. Publish the matching extras version before releasing the core SDK that pins it. Validate a fresh `litellm[proxy]` installation against both built wheels before publication

The root `pyproject.toml` declares `[tool.litellm.ui] package = "litellm-proxy-extras"` for the release pipeline. Generated dashboard assets are not tracked in Git. Automated releases must use project-releaser support for this layout before publishing; source-only extras builds do not contain a dashboard.

## Prerequisites

- All `schema.prisma` files are in sync (see [migration_runbook.md](./migration_runbook.md) Step 0)
- Migration has been generated and committed
- You are in the `litellm-proxy-extras/` directory

## Step 1: Bump the Version

### Option A: Automatic Version Bump (Recommended)

Use commitizen to automatically bump the version across all files:

```bash
cd litellm-proxy-extras
cz bump --increment patch
```

This will automatically:
- Bump the version in `pyproject.toml` (both `[project].version` and `[tool.commitizen].version`)
- Update the version in `../pyproject.toml` (root)
- Create a git commit with the version bump

Then skip to Step 3 (Clean Old Artifacts).

### Option B: Manual Version Bump

Update the version in `pyproject.toml`:

```bash
cd litellm-proxy-extras

# Check current version
grep 'version' pyproject.toml
```

Edit `pyproject.toml` and bump the version (both `[project].version` and `[tool.commitizen].version`).

#### Step 2: Update Version in the Root Package Metadata (Manual Only)

After bumping the version in `litellm-proxy-extras/pyproject.toml`, you **must** also update the version reference in the root `pyproject.toml`:

| File | Line to update |
|------|---------------|
| `pyproject.toml` (root) | `litellm-proxy-extras==X.Y.Z` in `[project.optional-dependencies].proxy` |

```bash
# From the repo root — replace OLD with NEW version
sed -i '' 's/litellm-proxy-extras==OLD/litellm-proxy-extras==NEW/' pyproject.toml
```

> **Do NOT skip this step.** The main `litellm` package pins the extras version — if you don't update these, users will install the old version.

## Step 3: Clean Old Artifacts

```bash
rm -rf dist/ build/ *.egg-info
```

## Step 4: Build the Package

```bash
uv build
```

This creates `.tar.gz` and `.whl` files in the `dist/` directory.

Verify the build output:

```bash
ls -la dist/
```

## Step 5: Upload to PyPI

```bash
uv tool run --from 'twine==6.2.0' twine upload dist/*
```

You will be prompted for your PyPI API token:

```
Enter your API token: pypi-...
```

> Use `__token__` as the username and your PyPI API token as the password.

## Quick Reference (Copy-Paste)

```bash
cd litellm-proxy-extras
rm -rf dist/ build/ *.egg-info
uv build
uv tool run --from 'twine==6.2.0' twine upload dist/*
```

---

## Do you want to build and publish a new `litellm-proxy-extras` package? (y/n)

If **yes**, run the following commands in order:

```bash
cd litellm-proxy-extras
rm -rf dist/ build/ *.egg-info
uv build
uv tool run --from 'twine==6.2.0' twine upload dist/*
```

When `twine upload` runs, enter your PyPI credentials:
- **Username:** `__token__`
- **Password:** *(paste your PyPI API key)*

If **no**, you're done — no package publish needed.
