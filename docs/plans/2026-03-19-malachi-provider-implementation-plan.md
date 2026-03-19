# Malachi Provider Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `malachi` as a first-class LiteLLM provider that appears in provider catalogs and resolves `malachi/<model>` through the existing OpenAI-compatible execution path.

**Architecture:** The patch will introduce `malachi` as an official provider identity in LiteLLM while intentionally reusing existing OpenAI-compatible routing and config machinery. The implementation will touch the provider enum, provider metadata manifests, provider resolution logic, and focused tests, while avoiding a custom transport stack.

**Tech Stack:** Python, Poetry, pytest, LiteLLM provider manifests, OpenAI-compatible provider config logic

---

### Task 1: Document the approved design in-repo

**Files:**
- Create: `docs/plans/2026-03-19-malachi-provider-design.md`

**Step 1: Confirm the design doc exists**

Run: `dir docs\plans`
Expected: `2026-03-19-malachi-provider-design.md` is present

**Step 2: Commit the design document**

```bash
git add docs/plans/2026-03-19-malachi-provider-design.md
git commit -m "docs: add malachi provider design"
```

### Task 2: Add the provider to LiteLLM's canonical provider enum

**Files:**
- Modify: `litellm/types/utils.py`
- Test: `tests/documentation_tests/test_readme_providers.py`

**Step 1: Write the failing test**

Add a focused assertion in an existing lightweight provider metadata test or create a new small test that expects `malachi` to exist in `LlmProviders`.

Example:

```python
from litellm.types.utils import LlmProviders


def test_malachi_in_llm_providers():
    assert LlmProviders.MALACHI.value == "malachi"
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/path/to/new_or_updated_test.py -q`
Expected: FAIL because `MALACHI` does not exist yet

**Step 3: Write minimal implementation**

Add:

```python
MALACHI = "malachi"
```

to `LlmProviders` in alphabetical/established style.

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/path/to/new_or_updated_test.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add litellm/types/utils.py tests/path/to/new_or_updated_test.py
git commit -m "feat: add malachi provider enum"
```

### Task 3: Add Malachi to the provider endpoints support manifest

**Files:**
- Modify: `provider_endpoints_support.json`
- Test: `tests/path/to/provider_manifest_test.py` or create one under `tests/test_litellm/`

**Step 1: Write the failing test**

Create a test that loads `provider_endpoints_support.json` and asserts:
- `malachi` exists
- `display_name` is set
- `chat_completions` is true
- `responses` is true

Example:

```python
import json
from pathlib import Path


def test_provider_endpoints_support_includes_malachi():
    payload = json.loads(Path("provider_endpoints_support.json").read_text())
    malachi = payload["providers"]["malachi"]
    assert malachi["display_name"] == "Malachi (`malachi`)"
    assert malachi["endpoints"]["chat_completions"] is True
    assert malachi["endpoints"]["responses"] is True
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/path/to/provider_manifest_test.py -q`
Expected: FAIL because `malachi` is missing

**Step 3: Write minimal implementation**

Add a `malachi` entry to `provider_endpoints_support.json` with conservative endpoint flags.

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/path/to/provider_manifest_test.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add provider_endpoints_support.json tests/path/to/provider_manifest_test.py
git commit -m "feat: add malachi provider support metadata"
```

### Task 4: Add Malachi to provider create fields metadata

**Files:**
- Modify: `litellm/proxy/public_endpoints/provider_create_fields.json`
- Test: `tests/path/to/provider_field_manifest_test.py`

**Step 1: Write the failing test**

Create a test that loads `provider_create_fields.json` and asserts a `malachi` entry exists with:
- `api_base`
- `api_key`
- a reasonable `default_model_placeholder`

Example:

```python
import json
from pathlib import Path


def test_provider_create_fields_includes_malachi():
    items = json.loads(
        Path("litellm/proxy/public_endpoints/provider_create_fields.json").read_text()
    )
    malachi = next(item for item in items if item["litellm_provider"] == "malachi")
    keys = [field["key"] for field in malachi["credential_fields"]]
    assert "api_base" in keys
    assert "api_key" in keys
    assert malachi["default_model_placeholder"] == "gpt-5.4"
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/path/to/provider_field_manifest_test.py -q`
Expected: FAIL because `malachi` is missing

**Step 3: Write minimal implementation**

Add a `malachi` entry modeled after `custom_openai` / `litellm_proxy`, but with Malachi naming.

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/path/to/provider_field_manifest_test.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add litellm/proxy/public_endpoints/provider_create_fields.json tests/path/to/provider_field_manifest_test.py
git commit -m "feat: add malachi provider field metadata"
```

### Task 5: Add provider resolution for `malachi/<model>`

**Files:**
- Modify: `litellm/litellm_core_utils/get_llm_provider_logic.py`
- Modify: `litellm/constants.py` if provider-specific allowlists need updating
- Test: `tests/path/to/get_llm_provider_malachi_test.py`

**Step 1: Write the failing test**

Create a routing test that verifies:

```python
model, provider, dynamic_api_key, api_base = get_llm_provider(
    model="malachi/gpt-5.4"
)
```

returns:
- `model == "gpt-5.4"`
- `provider == "malachi"`

and, in a separate test with env vars set:
- `api_base == MALACHI_API_BASE`
- `dynamic_api_key == MALACHI_API_KEY`

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/path/to/get_llm_provider_malachi_test.py -q`
Expected: FAIL because `malachi` is not recognized yet

**Step 3: Write minimal implementation**

Update `get_llm_provider_logic.py` so:
- prefixed `malachi/<model>` resolves as provider `malachi`
- env resolution uses `MALACHI_API_BASE` and `MALACHI_API_KEY`
- the implementation mirrors existing OpenAI-compatible provider patterns without duplicating unrelated logic

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/path/to/get_llm_provider_malachi_test.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add litellm/litellm_core_utils/get_llm_provider_logic.py litellm/constants.py tests/path/to/get_llm_provider_malachi_test.py
git commit -m "feat: resolve malachi provider models"
```

### Task 6: Wire Malachi into provider config resolution

**Files:**
- Modify: `litellm/utils.py`
- Test: `tests/path/to/provider_config_manager_malachi_test.py`

**Step 1: Write the failing test**

Create a test proving `ProviderConfigManager.get_provider_chat_config(..., provider=LlmProviders.MALACHI)` returns an OpenAI-compatible config implementation.

Example:

```python
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_provider_config_manager_returns_openai_compatible_config_for_malachi():
    config = ProviderConfigManager.get_provider_chat_config(
        model="gpt-5.4",
        provider=LlmProviders.MALACHI,
    )
    assert config is not None
```

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/path/to/provider_config_manager_malachi_test.py -q`
Expected: FAIL because the provider config map does not know `MALACHI`

**Step 3: Write minimal implementation**

Add `LlmProviders.MALACHI` to the provider config map in `litellm/utils.py`, pointing at the same OpenAI-compatible configuration family used for OpenAI-like providers.

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/path/to/provider_config_manager_malachi_test.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add litellm/utils.py tests/path/to/provider_config_manager_malachi_test.py
git commit -m "feat: add malachi provider config mapping"
```

### Task 7: Add README / provider docs coverage required by existing tests

**Files:**
- Modify: `README.md`
- Optionally create: `docs/my-website/docs/providers/malachi.md`
- Test: `tests/documentation_tests/test_readme_providers.py`

**Step 1: Write the failing test**

If the existing README provider test is still the canonical enforcement, use it directly as the failing test after adding `MALACHI` to `LlmProviders`.

**Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/documentation_tests/test_readme_providers.py -q`
Expected: FAIL if `malachi` is not documented in README

**Step 3: Write minimal implementation**

Add `Malachi (`malachi`)` to the supported providers table in `README.md`, keeping the ordering conventions intact.

If the provider docs site expects a provider page for table links, add a small `malachi.md` page as well.

**Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/documentation_tests/test_readme_providers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add README.md docs/my-website/docs/providers/malachi.md
git commit -m "docs: add malachi provider documentation"
```

### Task 8: Run focused verification

**Files:**
- No code changes unless verification reveals gaps

**Step 1: Run focused provider tests**

Run:

```bash
poetry run pytest tests/path/to/provider_manifest_test.py tests/path/to/provider_field_manifest_test.py tests/path/to/get_llm_provider_malachi_test.py tests/path/to/provider_config_manager_malachi_test.py -q
```

Expected: PASS

**Step 2: Run documentation coverage test**

Run:

```bash
poetry run pytest tests/documentation_tests/test_readme_providers.py -q
```

Expected: PASS

**Step 3: Note environment-specific blockers if they remain**

If pytest startup still fails in this Windows environment because `psycopg` cannot find `libpq`, record that clearly in your status instead of claiming the suite passed.

**Step 4: Commit final polish if needed**

```bash
git add .
git commit -m "test: verify malachi provider integration"
```
