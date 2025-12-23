# GCS (Google Cloud Storage) Bucket Logging on LiteLLM Gateway 

This folder contains the GCS Bucket Logging integration for LiteLLM Gateway. 

## Folder Structure 

- `gcs_bucket.py`: This is the main file that handles failure/success logging to GCS Bucket
- `gcs_bucket_base.py`: This file contains the GCSBucketBase class which handles Authentication for GCS Buckets

## Further Reading
- [Doc setting up GCS Bucket Logging on LiteLLM Proxy (Gateway)](https://docs.litellm.ai/docs/proxy/bucket)
- [Doc on Key / Team Based logging with GCS](https://docs.litellm.ai/docs/proxy/team_logging)


# GCS Logger for LiteLLM Using Callbacks

Logger that saves logs to Google Cloud Storage with organized folder structure.

## Quick Start

### 1. Install Dependencies
```bash
pip install google-cloud-storage
```

### 2. Set Environment Variables
```bash
export GCS_SUCCESS_BUCKET_NAME="litellm-success-logs"
export GCS_ERROR_BUCKET_NAME="litellm-error-logs"
# Optional: export GCS_PATH_SERVICE_ACCOUNT="/path/to/service-account.json"
```

### 3. Configure LiteLLM
```yaml
# config.yaml
litellm_settings:
  callbacks: logging.gcs_logger.logger_instance # path

environment_variables:
  GCS_SUCCESS_BUCKET_NAME: "litellm-success-logs"
  GCS_ERROR_BUCKET_NAME: "litellm-error-logs"
```

### 4. Run
```bash
litellm --config config.yaml
```

## GCS Folder Structure

**Success Logs:**
```
gs://litellm-success-logs/
  └── xyne/xyne-training/siraj/2025-12-09_chatcmpl-abc123.json
```
Path: `department/team/user/{date}_{correlation_id}.json`

**Error Logs:**
```
gs://litellm-error-logs/
  └── gpt-4/2025-12-09_uuid-error.json
```
Path: `model/{date}_{correlation_id}.json`

## What's Logged

**Success:** Full conversation, response, usage, cost, user info, timing  
**Error:** Error details, user info, model info, request context

Done! 🎉
