# LiteLLM Patches

Patches for LiteLLM `main-latest` (as of 2025-12-19).

---

## 1. File Deletion Fails for Batch Output Files

### Broken Feature

`DELETE /files/{file_id}` - Deleting batch output files fails with a Pydantic validation error.

### Error Message

```
openai.InternalServerError: Error code: 500 - {
  'error': {
    'message': '1 validation error for LiteLLM_ManagedFileTable\nfile_object\n  Input should be a valid dictionary or instance of OpenAIFileObject [type=model_type, input_value=None, input_type=NoneType]'
  }
}
```

### Root Cause

When LiteLLM stores batch output files in `LiteLLM_ManagedFileTable`, it sets `file_object=None`. However, the Pydantic model requires this field to be a valid `OpenAIFileObject`.

### Code Change (`_types.py`)

```python
# Before
class LiteLLM_ManagedFileTable(LiteLLMPydanticObjectBase):
    file_object: OpenAIFileObject

# After
class LiteLLM_ManagedFileTable(LiteLLMPydanticObjectBase):
    file_object: Optional[OpenAIFileObject] = None  # PATCHED
```

---

## 2. File Deletion Returns Wrong Response

### Broken Feature

`DELETE /files/{file_id}` - Even after fixing patch #1, the method returns `None` instead of the delete confirmation.

### Error Message

```
Exception: LiteLLM Managed File object with id=... not found
```

### Root Cause

`afile_delete` in `managed_files.py` calls `llm_router.afile_delete` (which deletes the file at the provider) but discards the response.

### Code Change (`managed_files.py`)

```python
# Before
async def afile_delete(self, file_id, ...):
    for model_id, model_file_id in mapping.items():
        await llm_router.afile_delete(model=model_id, file_id=model_file_id, **data)
    # Returns None

# After
async def afile_delete(self, file_id, ...):
    delete_response = None
    for model_id, model_file_id in mapping.items():
        delete_response = await llm_router.afile_delete(...)  # PATCHED: Capture response
    if delete_response:
        delete_response.id = file_id  # PATCHED: Replace with unified ID
        return delete_response
```

---

## 3. Batch Listing Fails with Duplicate Argument

### Broken Feature

`GET /batches?target_model_names=...` - Listing batches fails when using `target_model_names` query parameter.

### Error Message

```
openai.InternalServerError: Error code: 500 - {
  'error': {
    'message': "alist_batches() got multiple values for keyword argument 'model'"
  }
}
```

### Root Cause

The code passes `model` explicitly AND includes it in `**data`:

```python
model = target_model_names.split(",")[0]
response = await llm_router.alist_batches(
    model=model,        # Passed explicitly
    **data,             # Also contains 'model' key
)
```

### Code Change (`batches_endpoints.py`)

```python
# Before
model = target_model_names.split(",")[0]
response = await llm_router.alist_batches(model=model, **data)

# After
model = target_model_names.split(",")[0]
data.pop("model", None)              # PATCHED: Remove duplicate
data.pop("target_model_names", None) # PATCHED: Remove to avoid passing to downstream
response = await llm_router.alist_batches(model=model, **data)
```

---

## 4. File Retrieve Returns None for Batch Output Files

### Broken Feature

`GET /files/{file_id}` - Retrieving batch output file metadata returns `None`.

### Error Message

```
AttributeError: 'NoneType' object has no attribute 'id'
```

### Root Cause

`afile_retrieve` returns `stored_file_object.file_object` which is `None` for batch output files. It should fetch from the provider.

### Code Change (`managed_files.py` + `files_endpoints.py`)

```python
# managed_files.py - After
async def afile_retrieve(self, file_id, litellm_parent_otel_span, llm_router=None):
    stored = await self.get_unified_file_id(file_id, ...)
    if stored.file_object:
        return stored.file_object
    # PATCHED: Fetch from provider when file_object is None
    for model_id, model_file_id in stored.model_mappings.items():
        response = await llm_router.afile_retrieve(model=model_id, file_id=model_file_id)
        response.id = file_id
        return response
```

```python
# files_endpoints.py - After
response = await managed_files_obj.afile_retrieve(
    file_id=file_id,
    litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
    llm_router=llm_router,  # PATCHED: Pass router
)
```

---

## Known Issues (Not Bugs)

### Azure Batch Creation Response Missing `endpoint`

**Behavior:** Azure's batch creation response returns `endpoint=''` (empty string).

**Expected:** When you call `batches.retrieve()` or `batches.list()`, Azure returns `endpoint='/v1/chat/completions'` correctly.

**Workaround:** If you need the endpoint immediately after creation, retrieve the batch to get the correct value.

### Azure Batch Listing Returns Raw IDs

**Behavior:** `batches.list()` returns raw Azure batch IDs (e.g., `batch_abc123`) instead of LiteLLM unified IDs.

**Root Cause:** LiteLLM routes `batches.list()` directly to Azure instead of querying its internal managed batches database.

**Workaround:** Use `batches.retrieve(unified_batch_id)` instead of relying on list.

### Config Fix: Azure Batch Listing 404

**Behavior:** `batches.list()` returns empty results because Azure returns 404.

**Root Cause:** LiteLLM defaults to OpenAI handler instead of Azure handler for batch operations.

**Fix (config):** Add `custom_llm_provider: azure` to your model's `litellm_params`:

```yaml
litellm_params:
  model: azure/gpt-5-batch
  custom_llm_provider: azure  # Required for Azure batch operations
```

---

## Patch Files

| Patch File | Container Path |
|------------|----------------|
| `_types.py` | `/usr/lib/python3.13/site-packages/litellm/proxy/_types.py` |
| `managed_files.py` | `/usr/lib/python3.13/site-packages/litellm_enterprise/proxy/hooks/managed_files.py` |
| `batches_endpoints.py` | `/usr/lib/python3.13/site-packages/litellm/proxy/batches_endpoints/endpoints.py` |
| `files_endpoints.py` | `/usr/lib/python3.13/site-packages/litellm/proxy/openai_files_endpoints/files_endpoints.py` |

---

## Usage

### With Patches

```bash
./start-patched.sh
```

### Without Patches

```bash
./start-unpatched.sh
```

### Docker Compose Volumes

```yaml
volumes:
  - ./patches/_types.py:/usr/lib/python3.13/site-packages/litellm/proxy/_types.py
  - ./patches/managed_files.py:/usr/lib/python3.13/site-packages/litellm_enterprise/proxy/hooks/managed_files.py
  - ./patches/batches_endpoints.py:/usr/lib/python3.13/site-packages/litellm/proxy/batches_endpoints/endpoints.py
  - ./patches/files_endpoints.py:/usr/lib/python3.13/site-packages/litellm/proxy/openai_files_endpoints/files_endpoints.py
```
