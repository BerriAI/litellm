# LiteLLM Batch API Fixes

This document describes bugs found in LiteLLM's managed batch/files functionality and the patches applied to fix them. It also provides step-by-step instructions to reproduce the tests from a clean slate.

## Table of Contents

1. [Bug 1: File Deletion Fails for Batch Output Files](#bug-1-file-deletion-fails-for-batch-output-files)
2. [Bug 2: File Deletion Returns Wrong Response](#bug-2-file-deletion-returns-wrong-response)
3. [Bug 3: Batch Listing Fails with Duplicate Argument](#bug-3-batch-listing-fails-with-duplicate-argument)
4. [Bug 4: File Retrieve Returns None for Batch Output Files](#bug-4-file-retrieve-returns-none-for-batch-output-files)
5. [Mock Server: Azure-like Credential Validation](#mock-server-azure-like-credential-validation)
6. [Test Setup Instructions](#test-setup-instructions)

---

## Bug 1: File Deletion Fails for Batch Output Files

### Description

**Broken Feature:** `DELETE /files/{file_id}` - Deleting batch output files fails with a Pydantic validation error.

**Error Message:**
```
openai.InternalServerError: Error code: 500 - {
  'error': {
    'message': '1 validation error for LiteLLM_ManagedFileTable\nfile_object\n  Input should be a valid dictionary or instance of OpenAIFileObject [type=model_type, input_value=None, input_type=NoneType]'
  }
}
```

**Root Cause:** When LiteLLM stores batch output files in `LiteLLM_ManagedFileTable`, it sets `file_object=None`. However, the Pydantic model requires this field to be a valid `OpenAIFileObject`.

### Patch

**File:** `litellm/proxy/_types.py`, line ~3759

```python
# Before
class LiteLLM_ManagedFileTable(LiteLLMPydanticObjectBase):
    file_object: OpenAIFileObject

# After
class LiteLLM_ManagedFileTable(LiteLLMPydanticObjectBase):
    file_object: Optional[OpenAIFileObject] = None  # PATCHED
```

---

## Bug 2: File Deletion Returns Wrong Response

### Description

**Broken Feature:** `DELETE /files/{file_id}` - Even after fixing Bug #1, the method returns `None` instead of the delete confirmation.

**Error Message:**
```
Exception: LiteLLM Managed File object with id=... not found
```

**Root Cause:** `afile_delete` in `managed_files.py` calls `llm_router.afile_delete()` (which deletes the file at the provider) but discards the response.

### Patch

**File:** `enterprise/litellm_enterprise/proxy/hooks/managed_files.py`, line ~879

```python
# Before
async def afile_delete(self, file_id, ...):
    for model_id, model_file_id in mapping.items():
        await llm_router.afile_delete(model=model_id, file_id=model_file_id, **data)
    # Returns None when stored_file_object is None

# After
async def afile_delete(self, file_id, ...):
    delete_response = None  # PATCHED: Capture response
    for model_id, model_file_id in mapping.items():
        delete_response = await llm_router.afile_delete(model=model_id, file_id=model_file_id, **data)
    
    stored_file_object = await self.delete_unified_file_id(file_id, ...)
    if stored_file_object:
        return stored_file_object
    elif delete_response:  # PATCHED: Return provider response
        delete_response.id = file_id  # Replace with unified ID
        return delete_response
    else:
        raise Exception(...)
```

---

## Bug 3: Batch Listing Fails with Duplicate Argument

### Description

**Broken Feature:** `GET /batches?target_model_names=...` - Listing batches fails when using `target_model_names` query parameter.

**Error Message:**
```
openai.InternalServerError: Error code: 500 - {
  'error': {
    'message': "alist_batches() got multiple values for keyword argument 'model'"
  }
}
```

**Root Cause:** The code passes `model` explicitly AND includes it in `**data`:
```python
model = target_model_names.split(",")[0]
response = await llm_router.alist_batches(
    model=model,        # Passed explicitly
    **data,             # Also contains 'model' and 'target_model_names' keys
)
```

### Patch

**File:** `litellm/proxy/batches_endpoints/endpoints.py`, line ~576-577

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

## Bug 4: File Retrieve Returns None for Batch Output Files

### Description

**Broken Feature:** `GET /files/{file_id}` - Retrieving batch output file metadata returns `None`.

**Error Message:**
```
AttributeError: 'NoneType' object has no attribute 'id'
```

**Root Cause:** `afile_retrieve` returns `stored_file_object.file_object` which is `None` for batch output files. It should fetch the file metadata from the provider instead.

### Patch (Part A)

**File:** `enterprise/litellm_enterprise/proxy/hooks/managed_files.py`, line ~839-868

Add `import litellm` at the top of the file, then modify `afile_retrieve`:

```python
# Before
async def afile_retrieve(self, file_id, litellm_parent_otel_span):
    stored = await self.get_unified_file_id(file_id, ...)
    return stored.file_object  # Returns None for batch output files!

# After
import litellm  # Added at top of file

async def afile_retrieve(self, file_id, litellm_parent_otel_span, llm_router=None):  # PATCHED: Added llm_router
    stored = await self.get_unified_file_id(file_id, ...)
    if stored:
        if stored.file_object:
            return stored.file_object
        # PATCHED: Fetch from provider when file_object is None
        elif stored.model_mappings and llm_router:
            for model_id, model_file_id in stored.model_mappings.items():
                deployment = llm_router.get_deployment(model_id=model_id)
                if deployment:
                    credentials = llm_router.get_deployment_credentials(model_id=model_id) or {}
                    # Extract custom_llm_provider - afile_retrieve needs it as explicit param
                    custom_llm_provider = credentials.pop("custom_llm_provider", None)
                    if not custom_llm_provider:
                        # Infer from model name (e.g., "azure/gpt-5" -> "azure")
                        model_name = deployment.litellm_params.model or ""
                        if "/" in model_name:
                            custom_llm_provider = model_name.split("/")[0]
                        else:
                            custom_llm_provider = "openai"
                    response = await litellm.afile_retrieve(
                        file_id=model_file_id,
                        custom_llm_provider=custom_llm_provider,  # Explicit param for Azure
                        **credentials
                    )
                    response.id = file_id  # Replace with unified ID
                    return response
```

### Patch (Part B)

**File:** `litellm/proxy/openai_files_endpoints/files_endpoints.py`, line ~888

```python
# Before
response = await managed_files_obj.afile_retrieve(
    file_id=file_id,
    litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
)

# After
response = await managed_files_obj.afile_retrieve(
    file_id=file_id,
    litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
    llm_router=llm_router,  # PATCHED: Pass router to fetch from provider
)
```

---

## Test Setup Instructions

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Poetry (Python package manager)

### Step 1: Clone and Setup Environment

```bash
# Install dependencies
poetry install --extras "proxy extra_proxy"

# Install enterprise package in editable mode (required for patches to work)
poetry run pip install -e enterprise
```

### Step 2: Terminal 1 - Start Database and Mock Server

```bash
cd tests/batches_tests/local-litellm

# Build and start PostgreSQL and Mock Azure Server
docker compose -f docker-compose.dev.yml up --build
```

Wait until you see both services are healthy:
- `litellm_dev_db` - PostgreSQL database
- `mock-server` - Mock Azure OpenAI server (with credential validation enabled by default)

**Note:** The mock server now validates credentials like real Azure. Use `--build` to ensure you have the latest mock server with credential validation.

### Step 3: Terminal 2 - Start LiteLLM Proxy

```bash
cd /path/to/litellm

# Set environment variables
export DATABASE_URL="postgresql://llmproxy:dbpassword9090@localhost:5432/litellm"
export LITELLM_MASTER_KEY="sk-1234"
export LITELLM_SALT_KEY="mock-salt-key-12345"

# For real Azure testing (optional):
# export OPENAI_API_KEY="your-azure-api-key"
# export OPENAI_API_BASE=https://your azure endpoint"

# Generate Prisma client (first time only)
poetry run python -m prisma generate

# Start the proxy server
poetry run litellm --config tests/batches_tests/local-litellm/litellm-config.yaml --detailed_debug --port 4000
```

Wait until you see:
```
INFO:     Uvicorn running on http://0.0.0.0:4000
```

### Step 4: Terminal 3 - Run Tests

```bash
cd /path/to/litellm

# Run the end-to-end managed files test with mock server
USE_MOCK_SERVER=true poetry run pytest tests/batches_tests/test_managed_files_endtoend.py -s -vvv
```

### Expected Output

The test should pass with output similar to:

```
tests/batches_tests/test_managed_files_endtoend.py::TestManagedFilesAPI::test_e2e_managed_batch[gpt] 
Creating batch input file...
Created batch input file: bGl0ZWxs...

Creating batch...
Created batch: bGl0ZWxs...

Waiting for batch to reach completed state...
Batch status: completed

Retrieving batch output file metadata...
Output file metadata: ...

Fetching batch output file content...
Output file content: ...

Deleting input file...
Deleting output file...

PASSED
```

---

## Configuration Files

### `tests/batches_tests/local-litellm/litellm-config-local.yaml`

This config file sets up models for local testing:
- Mock OpenAI models pointing to `http://localhost:8090`
- Mock Azure batch model pointing to `http://localhost:8090`
- (Optional) Real Azure batch model with API key from environment

### `tests/batches_tests/local-litellm/docker-compose.dev.yml`

Docker Compose file that runs:
- PostgreSQL 16 database on port 5432
- Mock Azure OpenAI server on port 8090

---

## Troubleshooting

### "No module named prisma"

```bash
poetry run pip install prisma==0.11.0
poetry run python -m prisma generate
```

### Database connection error

Ensure PostgreSQL is running and the DATABASE_URL is correct:
```bash
docker ps | grep postgres
# Should show litellm_dev_db running on port 5432
```

### Patches not being picked up/

1. Clear Python cache:
   ```bash
   find enterprise -name "__pycache__" -type d -exec rm -rf {} +
   find litellm -name "__pycache__" -type d -exec rm -rf {} +
   ```

2. Verify editable install:
   ```bash
   poetry run pip show litellm-enterprise | grep "Editable"
   # Should show: Editable project location: /path/to/litellm/enterprise
   ```

3. Restart the proxy server

### Azure credentials error when testing with real Azure

Set the environment variable before starting the proxy:
```bash
export OPENAI_API_KEY="your-actual-azure-api-key"
```

---
