# Build & Publish `litellm-proxy-extras`

This runbook covers building and publishing a new version of the `litellm-proxy-extras` PyPI package. For use by litellm engineers only.

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
- Bump the version in `pyproject.toml` (both `[tool.poetry].version` and `[tool.commitizen].version`)
- Update the version in `../requirements.txt`
- Update the version in `../pyproject.toml` (root)
- Create a git commit with the version bump

Then skip to Step 3 (Install Build Dependencies).

### Option B: Manual Version Bump

Update the version in `pyproject.toml`:

```bash
cd litellm-proxy-extras

# Check current version
grep 'version' pyproject.toml
```

Edit `pyproject.toml` and bump the version (both `[tool.poetry].version` and `[tool.commitizen].version`).

#### Step 2: Update Version in Root Package Files (Manual Only)

After bumping the version in `litellm-proxy-extras/pyproject.toml`, you **must** also update the version reference in the root-level files:

| File | Line to update |
|------|---------------|
| `requirements.txt` | `litellm-proxy-extras==X.Y.Z` |
| `pyproject.toml` (root) | `litellm-proxy-extras = {version = "X.Y.Z", optional = true}` |

```bash
# From the repo root — replace OLD with NEW version
sed -i '' 's/litellm-proxy-extras==OLD/litellm-proxy-extras==NEW/' requirements.txt
sed -i '' 's/litellm-proxy-extras = {version = "OLD"/litellm-proxy-extras = {version = "NEW"/' pyproject.toml
```

> **Do NOT skip this step.** The main `litellm` package pins the extras version — if you don't update these, users will install the old version.

## Step 3: Install Build Dependencies

```bash
pip install build twine
```

## Step 4: Clean Old Artifacts

```bash
rm -rf dist/ build/ *.egg-info
```

## Step 5: Build the Package

```bash
python3 -m build
```

This creates `.tar.gz` and `.whl` files in the `dist/` directory.

Verify the build output:

```bash
ls -la dist/
```

## Step 6: Upload to PyPI

```bash
twine upload dist/*
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
python3 -m build
twine upload dist/*
```

---

## Do you want to build and publish a new `litellm-proxy-extras` package? (y/n)

If **yes**, run the following commands in order:

```bash
cd litellm-proxy-extras
pip install build twine
rm -rf dist/ build/ *.egg-info
python3 -m build
twine upload dist/*
```

When `twine upload` runs, enter your PyPI credentials:
- **Username:** `__token__`
- **Password:** *(paste your PyPI API key)*

If **no**, you're done — no package publish needed.
