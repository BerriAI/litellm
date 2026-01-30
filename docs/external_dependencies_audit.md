# External Dependencies Audit: LiteLLM Runtime Dependencies

This document identifies all external dependencies that could impact LLM calling logic at runtime, and provides mitigation strategies.

## Executive Summary

LiteLLM has **one critical external dependency** that affects LLM calling at import time:

| Dependency | Impact Level | Affects LLM Calls | Mitigation Available |
|------------|-------------|-------------------|---------------------|
| **GitHub Model Cost Map** | **CRITICAL** | Yes - can break model lookups | `LITELLM_LOCAL_MODEL_COST_MAP=True` |
| HuggingFace API | Low | No - only for provider mapping | Fails gracefully |
| Together AI API | Low | No - only for prompt templates | Fails gracefully |
| Ollama local server | Low | No - only for model listing | User's own infrastructure |

---

## Critical Dependency: Model Cost Map

### What It Does
At **import time**, LiteLLM fetches the model cost map from GitHub:

```python
# litellm/__init__.py (line 429-431)
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
model_cost = get_model_cost_map(url=model_cost_map_url)
```

Default URL:
```
https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
```

### Why It's Critical
1. **Import-time execution**: Runs when `import litellm` is called
2. **Model validation**: Used to validate model names and look up providers
3. **Cost calculation**: Required for accurate cost tracking
4. **Error messages**: Can cause "model isn't mapped" errors

### Failure Modes

| Scenario | Current Behavior | Impact |
|----------|------------------|--------|
| Network error | Falls back to local backup | ✅ Safe |
| HTTP 4xx/5xx | Falls back to local backup | ✅ Safe |
| JSON parse error | Falls back to local backup | ✅ Safe |
| **Valid JSON, invalid data** | **Uses bad data** | ❌ **BREAKS CALLS** |

### Mitigation

**Option 1: Use Local Model Cost Map (RECOMMENDED for production)**
```bash
export LITELLM_LOCAL_MODEL_COST_MAP=True
```

**Option 2: Self-host the model cost map**
```bash
export LITELLM_MODEL_COST_MAP_URL="https://your-cdn.com/model_prices.json"
```

---

## Low-Risk External Dependencies

### 1. HuggingFace Hub API

**File**: `litellm/llms/huggingface/common_utils.py`

**Purpose**: Fetch inference provider mappings for HuggingFace models

**URL**: `https://huggingface.co/api/models/{model}`

**When called**: Only when using HuggingFace provider with `inferenceProviderMapping`

**Failure behavior**: Raises `HuggingFaceError` - fails the specific request, not all LLM calls

**Risk**: LOW - only affects HuggingFace-specific functionality

---

### 2. Together AI API

**File**: `litellm/litellm_core_utils/prompt_templates/factory.py`

**Purpose**: Fetch model-specific prompt format and chat templates

**URL**: `https://api.together.xyz/models/info`

**When called**: Only when using Together AI provider and prompt formatting is needed

**Failure behavior**: Returns `None, None` - gracefully falls back to default prompt template

**Risk**: LOW - fails gracefully, only affects Together AI

---

### 3. Ollama Local Server

**File**: `litellm/llms/ollama/common_utils.py`

**Purpose**: List available models on user's Ollama instance

**URL**: `http://localhost:11434/api/tags` (or user-configured)

**When called**: Only when listing Ollama models

**Failure behavior**: Raises exception - only affects Ollama model listing

**Risk**: LOW - this is user's own infrastructure, not external

---

### 4. Databricks OAuth

**File**: `litellm/llms/databricks/common_utils.py`

**Purpose**: Exchange M2M credentials for access token

**When called**: Only when using Databricks with M2M auth

**Failure behavior**: Auth failure - affects only Databricks calls

**Risk**: LOW - expected provider interaction

---

### 5. Assembly AI API (Passthrough)

**File**: `litellm/proxy/pass_through_endpoints/llm_provider_handlers/assembly_passthrough_logging_handler.py`

**Purpose**: Fetch transcript status for logging

**When called**: Only in passthrough mode for Assembly AI

**Failure behavior**: Logging failure, doesn't affect the actual API call

**Risk**: LOW - only affects logging/observability

---

## Dynamic Code Loading Patterns

### Files Using Dynamic Import

These files use `importlib`, `eval`, or `exec`, but are **not loading external code**:

1. `litellm/proxy/guardrails/guardrail_registry.py` - Loading configured guardrail modules
2. `litellm/secret_managers/main.py` - Loading secret manager backends
3. `litellm/caching/caching.py` - Loading cache backends
4. `litellm/proxy/hooks/litellm_skills/main.py` - Loading skill plugins

**Risk**: LOW - these load local Python modules, not external code

---

## GitHub-Connected Features (Non-Critical)

These features connect to GitHub but don't affect core LLM calling:

1. **GitHub Copilot Provider** - Only if configured
2. **MCP Server (GitHub MCP)** - Only if configured
3. **Documentation links in error messages** - Just static strings

---

## Recommended Production Configuration

For maximum stability, use these environment variables:

```bash
# CRITICAL: Disable remote model cost map fetching
export LITELLM_LOCAL_MODEL_COST_MAP=True

# OPTIONAL: If you need day-0 model support, self-host the map
# export LITELLM_MODEL_COST_MAP_URL="https://your-internal-cdn.com/model_prices.json"
```

---

## Proposed Code Changes for Enhanced Safety

### 1. Add Validation to `get_model_cost_map.py`

```python
def validate_model_cost_map(data: dict) -> bool:
    """Validate that fetched data is a valid model cost map."""
    if not isinstance(data, dict):
        return False
    if len(data) < 100:  # Sanity check - should have many models
        return False
    # Check a sample of entries have required fields
    for key, value in list(data.items())[:10]:
        if key == "sample_spec":
            continue
        if not isinstance(value, dict):
            return False
        if "litellm_provider" not in value:
            return False
    return True

def get_model_cost_map(url: str) -> dict:
    # ... existing code ...
    try:
        response = httpx.get(url, timeout=5)
        response.raise_for_status()
        content = response.json()
        
        # NEW: Validate before accepting
        if not validate_model_cost_map(content):
            verbose_logger.warning(
                "Remote model cost map failed validation, using local backup"
            )
            return _load_local_backup()
        
        return content
    except Exception:
        return _load_local_backup()
```

### 2. Enhanced CI Validation

Add comprehensive validation to `.github/workflows/test-model-map.yaml`:

```yaml
- name: Validate model_prices_and_context_window.json
  run: |
    python scripts/validate_model_cost_map.py
```

### 3. Add Monitoring/Alerting

Log when falling back to local backup:

```python
if using_local_backup:
    verbose_logger.warning(
        "Using local model cost map backup. "
        "Remote fetch failed or returned invalid data."
    )
```

---

## Audit Checklist for New Features

When adding new features, verify:

- [ ] No import-time external HTTP calls
- [ ] External API calls have proper error handling
- [ ] Fallback behavior doesn't break unrelated functionality
- [ ] Timeouts are set on all HTTP calls
- [ ] Failures are logged appropriately

---

## Summary

| Concern | Status |
|---------|--------|
| External dependencies affecting LLM calls | 1 critical (model cost map), mitigable |
| Dynamic code loading from external sources | None |
| Hidden GitHub dependencies | None beyond model cost map |
| Graceful degradation | Needs improvement for model cost map |

**Recommendation**: Set `LITELLM_LOCAL_MODEL_COST_MAP=True` for all production deployments until validation improvements are implemented.
