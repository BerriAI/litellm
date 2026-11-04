# S3 storage

`litellm-storage-s3` builds an `aws_sdk_s3` client config whose requests go through `litellm_http::Client` and whose credentials come from `litellm-auth-aws`. It re-exports `aws_sdk_s3` so every S3 user shares one SDK version

It knows no buckets, keys or object formats. `litellm-cache-s3` stores cache entries and `litellm-model-catalog-refresh` reads the cost map object on top of it
