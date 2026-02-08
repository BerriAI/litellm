# PR Strategy: Fix Gemini Function Response Role (Revised)

This document outlines the step-by-step plan to prepare and submit the Pull Request, incorporating feedback to ensure best practices and code freshness.

## 1. Preserve Uncommitted Changes (Safety First)

Before switching branches, we must handle any work currently in progress.

**Option A: Stash (Recommended for small, temporary changes)**
This saves your changes to a stack and reverts the working directory to the last commit.
```bash
git stash save "WIP: Pre-Gemini fix work"
```
*To restore later:* `git stash pop`

**Option B: Temporary Branch (Recommended for significant work)**
This commits your changes to a separate branch so they are safe and part of the history.
```bash
git checkout -b temp/wip-backup
git add .
git commit -m "WIP: Backup before Gemini fix"
git checkout - # Switch back to previous branch
```

## 2. Configure Remote Repositories

We need to ensure we have both `origin` (your fork) and `upstream` (official repo) configured.

```bash
# Check current remotes
git remote -v

# If upstream is missing:
# git remote add upstream https://github.com/BerriAI/litellm.git

# If origin is missing or incorrect:
# git remote add origin https://github.com/ZqinKing/litellm.git
```

## 3. Create and Switch to Feature Branch

We will start from the official `upstream/main` to ensure our PR is up-to-date and minimize conflicts.

```bash
# Fetch latest changes from upstream
git fetch upstream

# Create new branch from upstream/main
git checkout -b fix/gemini-function-response-role upstream/main
```

## 4. Apply the Fix

Modify `litellm/llms/vertex_ai/gemini/transformation.py`.

**Change 1:**
Find:
```python
                if len(tool_call_responses) > 0:
                    contents.append(ContentType(parts=tool_call_responses))
                    tool_call_responses = []
```
Replace with:
```python
                if len(tool_call_responses) > 0:
                    contents.append(ContentType(role="user", parts=tool_call_responses))
                    tool_call_responses = []
```

**Change 2:**
Find:
```python
        if len(tool_call_responses) > 0:
            contents.append(ContentType(parts=tool_call_responses))
```
Replace with:
```python
        if len(tool_call_responses) > 0:
            contents.append(ContentType(role="user", parts=tool_call_responses))
```

## 5. Verification

Since we are modifying core transformation logic, we should verify the changes.
- **Manual Check:** Verify the code structure matches the fix.
- **Unit Tests:** Attempt to run relevant tests if environment permits.
  ```bash
  # Example: Run tests for vertex/gemini
  # poetry run pytest tests/unified_google_tests/test_vertex_ai_native.py
  ```
- **Documentation:** If tests are not run, explicitly state "Tests not run" in the PR description, but provide the reproduction evidence.

## 6. Commit and Push

```bash
git add litellm/llms/vertex_ai/gemini/transformation.py
git commit -m "fix(gemini): add missing role=\"user\" to function_response to resolve INVALID_ARGUMENT"
git push -u origin fix/gemini-function-response-role
```

## 7. Create PR Description

We will generate a `plans/pr_description.md` file containing:
- **Summary:** Fix for missing `role="user"` in `function_response`.
- **Issue Reference:** Link to the issue or describe the bug.
- **Evidence:**
    - Reference `error-v1beta-models-gemini-3-pro-preview-streamGenerateContent-2026-02-08T114448-d2e51a48.log`
    - Reference `request.json` (if relevant/sanitized)
- **Reproduction Steps:** How to trigger the `INVALID_ARGUMENT` error.
