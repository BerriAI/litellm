# Plan: Fix `/converse/` + AWS Bedrock Inference Profile ARNs

**Branch:** `fix/model_id_parsing` (work may also cherry-pick from `fix/bedrock-converse-model-id-prefix`)  
**Related issues:** GitHub #8911, LIT-3274  
**Status:** Ready to implement  
**Last updated:** 2026-06-09 (Phase 5 PR workflow added)

---

## Background

Users force tool/function calling on AWS Bedrock Inference Profiles by prepending or appending `/converse/` to the model string (e.g. `bedrock/converse/arn:aws:bedrock:us-east-1:…:application-inference-profile/xyz`).

Issue #8911 (PR #9123) fixed URL-encoding of the model ID but did not address structural ARN parsing or prefix synchronization in `converse_handler.py`.

---

## Verification Summary (codebase research)

### Confirmed


| Finding                                                      | Detail                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **#8911 scope was narrow**                                   | PR #9123 only changed `encode_model_id(model_id=model)` vs raw model. No ARN-aware parsing.                                                                                                                                                                |
| **Slash-based region extraction fails for ARNs**             | Lines 288–291 in `converse_handler.py` split on `/`. For ARNs, `_potential_region` becomes `arn:aws:bedrock:…:application-inference-profile`, which is not a valid region → block skipped.                                                                 |
| `**/converse/` required for inference profile tool routing** | `BedrockModelInfo.get_bedrock_route()` returns `"invoke"` for bare ARNs; `"converse"` only when `converse/` appears in the model string.                                                                                                                   |
| **Real bug: model ID encoding when prefixes leak**           | After prefix stripping, `_stripped` is updated but `_model_for_id` lags unless an embedded region matches. Produces malformed URLs like `bedrock%2Fconverse%2Farn%3A…`. Fix exists on branch `fix/bedrock-converse-model-id-prefix` (commit `fc90f4c0d7`). |


### Partially incorrect (original theory)


| Claim                                                          | Actual behavior                                                                                                                                                                     |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**_region_from_model` staying `None` causes `NoRegionError`** | `_get_aws_region_name()` falls back to `_get_aws_region_from_model_arn(model)` which extracts region from ARN colon structure (`parts[3]`), even when routing prefixes are present. |
| **Wrong regional endpoint from env default**                   | ARN fallback runs **before** env vars (`AWS_REGION`, `AWS_REGION_NAME`). If all sources fail, code defaults to `us-west-2` — does not raise `NoRegionError`.                        |
| **Standard `completion()` path is broken**                     | `main.py` strips `converse/` before calling the handler. Verified URL: `https://bedrock-runtime.eu-central-1.amazonaws.com/model/arn%3Aaws%3A…/converse` ✓                          |


### Empirical test results (2026-06-09)

```
ARN slash extraction:     skipped (expected)
ARN fallback region:      us-east-1 ✓
Prefixed model fallback:  eu-west-1 ✓  (bedrock/converse/<arn>)
completion() end-to-end:  eu-central-1 ✓, correct encoded modelId ✓

Direct handler call WITHOUT sync fix:
  modelId = bedrock%2Fconverse%2Farn%3A…  ❌
Direct handler call WITH sync fix (fc90f4c0d7):
  modelId = arn%3Aaws%3Abedrock%3A…       ✓
```

---

## Root Cause

Two separate parsing mechanisms exist in `converse_handler.py`:

1. **Slash-based** (lines 288–291) — works for `bedrock/us-east-1/model-name`, fails for ARNs.
2. **ARN colon-based** (`_get_aws_region_from_model_arn` in `base_aws_llm.py`) — works for regions but is only invoked later in `_get_aws_region_name()`, not during model ID preparation.

Additionally, `_model_for_id` is not synchronized with `_stripped` after routing-prefix removal, so encoded model IDs can retain `bedrock/converse/` segments when the handler is called directly (bypassing `main.py` stripping).

---

## Affected Code


| File                                            | Lines / symbol                                                                   | Role                                              |
| ----------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------- |
| `litellm/llms/bedrock/chat/converse_handler.py` | 278–303                                                                          | Prefix strip, region extract, modelId encode      |
| `litellm/llms/bedrock/base_aws_llm.py`          | `_get_aws_region_from_model_arn`, `_get_aws_region_name`, `get_bedrock_model_id` | Region fallback; invoke-path model ID             |
| `litellm/llms/bedrock/common_utils.py`          | `get_bedrock_base_model`, `get_bedrock_route`                                    | Shared parsing; route detection                   |
| `litellm/main.py`                               | 3891–3893                                                                        | Strips `converse/` before handler (standard path) |


---

## Implementation Plan

### Phase 1 — Ship model ID prefix sync (low risk, immediate) ⭐

**Goal:** Fix malformed Bedrock URLs when prefixed models reach the handler.

**Change:** After routing-prefix stripping in `converse_handler.py`, sync `_model_for_id`:

```python
for rp in ["bedrock/converse/", "bedrock/", "converse/"]:
    if _stripped.startswith(rp):
        _stripped = _stripped[len(rp):]
        break
_model_for_id = _stripped  # ← add this line (fc90f4c0d7)
```

**Source:** Cherry-pick commit `fc90f4c0d7` from `fix/bedrock-converse-model-id-prefix`.

**Tests to keep/add** (`tests/test_litellm/llms/bedrock/test_base_aws_llm.py`):

- `test_converse_handler_strips_bedrock_prefix_for_inference_profile_arn` (already on branch)
- Parametrize: `bedrock/<arn>`, `bedrock/converse/<arn>`, `converse/<arn>`
- Assert posted URL segment equals `encode_model_id(arn)` — no `bedrock%2F` prefix
- Assert URL host region matches ARN region (e.g. `bedrock-runtime.eu-west-1.amazonaws.com`)

**Acceptance criteria:**

- Direct `BedrockConverseLLM.completion(model="bedrock/converse/<arn>", …)` builds correct URL
- Existing `test_bedrock_application_inference_profile` in `tests/llm_translation/test_bedrock_completion.py` still passes

---

### Phase 2 — Consolidate region + model ID parsing (structural fix)

**Goal:** Single helper for both path-format regions and ARN regions; remove slash-only trap.

**New helper** in `litellm/llms/bedrock/common_utils.py`:

```python
def extract_bedrock_region_and_model_id(model: str) -> tuple[Optional[str], str]:
    """
    Strip LiteLLM routing prefixes, then extract region + cleaned model ID.

    Handles:
      - bedrock/converse/us-east-1/anthropic.claude-…  → ("us-east-1", "anthropic.claude-…")
      - arn:aws:bedrock:us-east-1:…:application-inference-profile/xyz
          → ("us-east-1", full ARN string)
    """
```

**Logic order:**

1. Strip routing prefixes: `bedrock/converse/`, `bedrock/`, `converse/`, `invoke/`, `nova-2/`, `nova/`
2. If string contains `arn:aws:bedrock` (or gov/cn variants): extract region via colon split (`parts[3]`), validate with `_VALID_AWS_REGION_PATTERN`
3. Else if first `/`-segment ∈ `_get_all_bedrock_regions()`: strip region prefix (existing behavior from commit `5864317d92`)
4. Return `(region | None, cleaned_model_id)`

**Refactor call sites:**

- `converse_handler.py` lines 278–303 → call helper; inject `optional_params["aws_region_name"]` when region returned
- `get_bedrock_base_model()` in `common_utils.py` (lines 557–566) — same slash-only limitation today
- Unit tests in `tests/test_litellm/llms/chat/test_converse_handler.py::TestBedrockRegionInModelPath` — extend with ARN cases

**Acceptance criteria:**

- ARN models inject region via `optional_params` (not only via later fallback)
- Path-format models (`us-east-1/model-name`) unchanged
- No regression for cross-region inference prefixes (`us.anthropic.…`)

---

### Phase 3 — Harden invoke path (related, not on main)

**Goal:** Parity between converse and invoke routes for ARN + prefix handling.

**Source:** Commit `b5cfbb0935` (LIT-3274) — on remote branch, not merged to `main`.

**Change** in `get_bedrock_model_id()` (`base_aws_llm.py`):

```python
else:
    model_id = model
    for _prefix in ("bedrock/", "converse/", "invoke/"):
        if model_id.startswith(_prefix):
            model_id = model_id[len(_prefix):]
            break
    if model_id.startswith("arn:"):
        return BaseAWSLLM.encode_model_id(model_id=model_id)
```

**Tests:**

- `bedrock/arn:aws:…:inference-profile/…` invoke URL has no `bedrock/` segment
- ARN is URL-encoded in invoke path

---

### Phase 4 — Full test matrix


| Scenario                                           | Expected modelId       | Expected region     | Route    |
| -------------------------------------------------- | ---------------------- | ------------------- | -------- |
| `bedrock/converse/<arn>` via `completion()`        | URL-encoded ARN        | From ARN            | converse |
| `bedrock/converse/<arn>` direct handler call       | URL-encoded ARN        | From ARN            | converse |
| `converse/<arn>` direct handler call               | URL-encoded ARN        | From ARN            | converse |
| `bedrock/converse/us-east-1/model-name`            | `model-name` (encoded) | `us-east-1`         | converse |
| `bedrock/<arn>` without `/converse/`               | URL-encoded ARN        | From ARN            | invoke   |
| Proxy: alias model + separate `model_id` ARN param | From `model_id`        | From `model_id` ARN | varies   |
| `bedrock/converse/<arn>` + tools                   | URL-encoded ARN        | From ARN            | converse |


**Test file targets:**

- `tests/test_litellm/llms/chat/test_converse_handler.py`
- `tests/test_litellm/llms/bedrock/test_base_aws_llm.py`
- `tests/llm_translation/test_bedrock_completion.py` (`test_bedrock_application_inference_profile`)

---

### Phase 5 — Create the PR

**Goal:** Open an upstream-ready pull request against `BerriAI/litellm` with a clear summary and test plan.

**Prerequisite:** Phases 1–4 complete; all targeted tests passing locally.

#### 5.1 Pre-flight checks

Run in parallel to understand branch state:

```bash
git status
git diff
git branch -vv                          # confirm tracking / ahead-of-remote
git log --oneline main..HEAD              # commits included in PR
git diff main...HEAD                      # full diff vs base branch
```

Review:

- Only intended files changed (no secrets, `.env`, credentials)
- Commit messages follow repo style (fix/feat prefix, concise why)
- All commits that belong in this PR are on the branch (not just latest)

#### 5.2 Decide PR scope


| Option                                                    | When to use                                                              |
| --------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Single PR (recommended for first upstream submission)** | Phases 1–3 shipped together; one cohesive Bedrock inference-profile fix  |
| **Stacked PRs**                                           | Split if reviewers prefer smaller diffs — see optional split table below |


**Suggested single-PR title:**

> `fix(bedrock): strip routing prefixes from inference profile ARNs on converse path`

**Optional split (if needed later):**


| PR   | Scope                                                 | Files                                                                |
| ---- | ----------------------------------------------------- | -------------------------------------------------------------------- |
| PR A | Model ID prefix sync (`fc90f4c0d7`) + tests           | `converse_handler.py`, `test_base_aws_llm.py`                        |
| PR B | Shared `extract_bedrock_region_and_model_id()` helper | `common_utils.py`, `converse_handler.py`, `test_converse_handler.py` |
| PR C | Invoke path parity (`b5cfbb0935`)                     | `base_aws_llm.py`, `test_base_aws_llm.py`                            |


#### 5.3 Push branch

```bash
git push -u origin HEAD
```

Use branch name aligned with the fix, e.g. `fix/bedrock-converse-inference-profile-arn`.

#### 5.4 Create PR via GitHub CLI

```bash
gh pr create --title "fix(bedrock): strip routing prefixes from inference profile ARNs on converse path" --body "$(cat <<'EOF'
## Summary
- Sync `_model_for_id` with stripped routing prefixes in `converse_handler.py` so inference profile ARNs are URL-encoded without leftover `bedrock/converse/` segments
- Add consolidated region + model ID extraction for ARN and path-format models
- Align invoke-path `get_bedrock_model_id()` with converse-path prefix stripping and ARN encoding

Fixes malformed Bedrock URLs when using `bedrock/converse/<inference-profile-arn>` for tool calling (follow-up to #8911).

## Test plan
- [ ] `pytest tests/test_litellm/llms/bedrock/test_base_aws_llm.py -k converse_handler_strips_bedrock_prefix`
- [ ] `pytest tests/test_litellm/llms/chat/test_converse_handler.py -k region`
- [ ] `pytest tests/llm_translation/test_bedrock_completion.py -k test_bedrock_application_inference_profile`
- [ ] Manual: `completion(model="bedrock/converse/arn:aws:bedrock:…:application-inference-profile/…", tools=[…])` → correct regional endpoint and encoded modelId in URL

EOF
)"
```

Adjust title/body if scope is split across multiple PRs.

#### 5.5 Post-create checklist

- PR URL recorded below in **Tracking**
- Link to GitHub #8911 (and LIT-3274 if applicable) in PR description
- CI green on first run; triage failures if any
- Respond to review comments; re-run tests after changes

#### 5.6 Tracking


| Item           | Value |
| -------------- | ----- |
| Branch         |       |
| PR URL         |       |
| Upstream issue | #8911 |
| CI status      |       |


---

### Phase 6 — Optional UX improvement (product decision, post-PR)

Auto-route inference profile ARNs to Converse when tools are present, removing the `/converse/` workaround:

```python
# Candidate location: get_bedrock_route() or main.py bedrock dispatch
if "inference-profile" in model and tools_present:
    return "converse"
```

**Tradeoffs:**


| Pro                                    | Con                                      |
| -------------------------------------- | ---------------------------------------- |
| Simpler config for users               | Behavior change for existing deployments |
| Matches user intent when passing tools | May surprise users expecting invoke API  |


**Decision needed before implementing.** Ship as a separate follow-up PR after Phase 5 merges.

---

## Work Checklist

### Phase 1 (start here)

- Cherry-pick `fc90f4c0d7` onto working branch
- Run `test_converse_handler_strips_bedrock_prefix_for_inference_profile_arn`
- Run `test_bedrock_application_inference_profile`

### Phase 2

- Implement `extract_bedrock_region_and_model_id()` in `common_utils.py`
- Refactor `converse_handler.py` to use helper
- Add ARN parametrized cases to `TestBedrockRegionInModelPath`

### Phase 3

- Cherry-pick or rebase `b5cfbb0935`
- Verify invoke-path tests

### Phase 4

- Complete test matrix above
- Manual smoke test against real Bedrock inference profile (if credentials available)

### Phase 5 (create PR)

- Run pre-flight git checks (5.1)
- Confirm PR scope — single vs stacked (5.2)
- Push branch to origin (5.3)
- Create PR with `gh pr create` (5.4)
- Fill in Tracking table with PR URL (5.6)
- Monitor CI and address review feedback (5.5)

### Phase 6 (optional, after merge)

- Decide on auto-route inference profiles with tools
- Open separate follow-up PR if approved

---

## References

- Issue #8911 fix: merge commit `2c011d9a93` (PR #9123)
- Region path fix: commit `5864317d92` (`bedrock/{region}/{model}`)
- Model ID sync fix: commit `fc90f4c0d7` on `fix/bedrock-converse-model-id-prefix`
- Invoke path fix: commit `b5cfbb0935` (LIT-3274, not on main)
- Existing integration test: `tests/llm_translation/test_bedrock_completion.py::test_bedrock_application_inference_profile`

