# GCS (Google Cloud Storage) Bucket Logging on LiteLLM Gateway 

This folder contains the GCS Bucket Logging integration for LiteLLM Gateway. 

## Folder Structure 

- `gcs_bucket.py`: This is the main file that handles failure/success logging to GCS Bucket
- `gcs_bucket_base.py`: This file contains the GCSBucketBase class which handles Authentication for GCS Buckets

## Further Reading
- [Doc setting up GCS Bucket Logging on LiteLLM Proxy (Gateway)](https://docs.litellm.ai/docs/observability/gcs_bucket_integration)
- [Doc on Key / Team Based logging with GCS](https://docs.litellm.ai/docs/proxy/team_logging)