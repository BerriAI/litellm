# LiteLLM Batch API Fixes

This document describes bugs found in LiteLLM's managed batch/files functionality and the patches applied to fix them. It also provides step-by-step instructions to reproduce the tests from a clean slate.

## Table of Contents

1. [Bug 1: File Deletion Fails for Batch Output Files](#bug-1-file-deletion-fails-for-batch-output-files)
2. [Bug 2: File Deletion Returns Wrong Response](#bug-2-file-deletion-returns-wrong-response)
3. [Bug 3: File Retrieve Returns None for Batch Output Files](#bug-3-file-retrieve-returns-none-for-batch-output-files)
4. [Known Limitation: Error Files Not Retrievable](#known-limitation-error-files-not-retrievable)
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

---

## Bug 2: File Deletion Returns Wrong Response

### Description

**Broken Feature:** `DELETE /files/{file_id}` - Even after fixing Bug #1, the method returns `None` instead of the delete confirmation.

**Error Message:**
```
Exception: LiteLLM Managed File object with id=... not found
```

**Root Cause:** `afile_delete` in `managed_files.py` calls `llm_router.afile_delete()` (which deletes the file at the provider) but discards the response.

---

## Bug 3: File Retrieve Returns None for Batch Output Files

### Description

**Broken Feature:** `GET /files/{file_id}` - Retrieving batch output file metadata returns `None`.

**Error Message:**
```
AttributeError: 'NoneType' object has no attribute 'id'
```

**Root Cause:** `afile_retrieve` returns `stored_file_object.file_object` which is `None` for batch output files. It should fetch the file metadata from the provider instead.

---

## Known Limitation: Error Files Not Retrievable

### Description

When a batch fails, the provider returns an `error_file_id` containing details about failed requests. Currently, **error files are NOT retrievable** through the managed files API (`GET /files/{file_id}`).

### Root Cause

Only `output_file_id` is stored in `LiteLLM_ManagedFileTable` when a batch completes. The `error_file_id` is encoded in the batch response but never stored in the managed files table.

**In `async_post_call_success_hook`:**
```python
# Only output_file_id is handled:
if response.output_file_id and model_id:
    await self.store_unified_file_id(
        file_id=response.output_file_id,
        ...
    )
# error_file_id is NOT stored
```

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
