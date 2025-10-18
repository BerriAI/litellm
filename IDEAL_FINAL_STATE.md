# Dispatcher Refactoring: Ideal State

**Goal:** Replace 47-provider `if/elif` chain with an O(1) dispatcher lookup.

---

## Impact

**Performance**

- Lookup: O(n) → O(1)
- Average speedup: 24x
- Worst case: 47x
- The if-else chain essentially becomes a linear search loop through all providers, and adding a new provider increases lookup time proportionally

---

## Current State

```python
def completion(...):
    # 1,416 lines: setup, validation (KEEP)

    # 2,300 lines: provider routing (REPLACE)
    if custom_llm_provider == "azure":
        # 120 lines
    elif custom_llm_provider == "anthropic":
        # 58 lines
    # ... 45 more elif blocks ...
```

---

## Target State

```python
def completion(...):
    # Setup, validation (unchanged)

    # Single dispatcher call (replaces all if/elif)
    response = ProviderDispatcher.dispatch(
        custom_llm_provider=custom_llm_provider,
        model=model,
        messages=messages,
        # ... pass all params ...
    )
    return response
```

---

## Progress

**Current (POC)**

- OpenAI migrated
- 99 lines removed
- All tests passing
---

## Detailed Final Structure

### main.py Structure (After Full Migration)

```python
# ========================================
# ENDPOINT FUNCTIONS (~2,800 lines total)
# ========================================

def completion(...):  # ~500 lines
    # Setup (400 lines)
    # Dispatch (30 lines)
    # Error handling (70 lines)

def embedding(...):  # ~150 lines
    # Setup (100 lines)
    # Dispatch (20 lines)
    # Error handling (30 lines)

def image_generation(...):  # ~100 lines
    # Setup (70 lines)
    # Dispatch (20 lines)
    # Error handling (10 lines)

def transcription(...):  # ~150 lines
    # Simpler - fewer providers

def speech(...):  # ~150 lines
    # Simpler - fewer providers

# Other helper functions (1,750 lines)
# ========================================
# TOTAL: ~2,800 lines (from 6,272)
# ========================================
```

### provider_dispatcher.py Structure

```python
# ========================================
# PROVIDER DISPATCHER (~3,500 lines total)
# ========================================

class ProviderDispatcher:
    """Unified dispatcher for all endpoints"""

    # COMPLETION HANDLERS (~2,000 lines)
    _completion_dispatch = {
        "openai": _handle_openai_completion,  # DONE
        "azure": _handle_azure_completion,
        "anthropic": _handle_anthropic_completion,
        # ... 44 more
    }

    # EMBEDDING HANDLERS (~800 lines)
    _embedding_dispatch = {
        "openai": _handle_openai_embedding,
        "azure": _handle_azure_embedding,
        "vertex_ai": _handle_vertex_embedding,
        # ... 21 more
    }

    # IMAGE GENERATION HANDLERS (~400 lines)
    _image_dispatch = {
        "openai": _handle_openai_image,
        "azure": _handle_azure_image,
        # ... 13 more
    }

    # SHARED UTILITIES (~300 lines)
    @staticmethod
    def _get_openai_credentials(**ctx):
        """Shared across completion, embedding, image_gen"""
        pass

    @staticmethod
    def _get_azure_credentials(**ctx):
        """Shared across completion, embedding, image_gen"""
        pass
```
