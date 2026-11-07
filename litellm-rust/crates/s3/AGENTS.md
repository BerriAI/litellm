`litellm-s3` builds `aws-sdk-s3` clients that send through `litellm_http::Client`, and owns the S3 concepts the SDK lacks, such as the `s3://bucket/key` URI. It reads no environment and picks no credentials: the host passes the region, the endpoint and a credentials provider, usually `litellm_auth_aws::AwsCredentialsProvider`

Operation policy such as retries, timeouts and checksum settings stays with the caller, which adjusts the returned config builder
