# Bedrock managed file content retrieval

Issue: BerriAI/litellm#26335 — "File Retrieval always fail for bedrock"

## Problem

`GET /v1/files/{id}/content` returns a 500 for every Bedrock-backed managed file.
`BedrockFilesConfig.transform_file_content_request` and
`transform_file_content_response` both unconditionally raise
`NotImplementedError("BedrockFilesConfig does not support file content retrieval")`.

The issue reporter traced the full path: `files_endpoints.get_file_content` ->
enterprise `managed_files.afile_content` -> `router.afile_content` ->
`litellm.afile_content` -> `ProviderConfigManager.get_provider_files_config`
returns a non-None `BedrockFilesConfig` ->
`base_llm_http_handler.retrieve_file_content` ->
`transform_file_content_request` raises. Because the config is non-None, the path
short-circuits before reaching the older `bedrock_files_instance.file_content`
branch in `files/main.py`, which is therefore dead code.

A working S3 downloader already exists in `litellm/llms/bedrock/files/handler.py`
(`BedrockFilesHandler`), complete with bucket-confusion and path-traversal
validation and covered by `tests/test_litellm/llms/bedrock/files/test_bedrock_files_handler.py`.
It is never reached.

## Scope

File content retrieval only. Not in scope: file deletion, file listing, or
transforming Bedrock batch output into OpenAI batch output shape (a possible
follow-up). Retrieval returns the raw S3 object bytes.

## Approach

Implement the two transform methods on `BedrockFilesConfig` so Bedrock uses the
same generic `retrieve_file_content` path as Vertex AI, Anthropic, and Manus.

The generic handler (`llm_http_handler.retrieve_file_content`) computes the
request URL and params via `transform_file_content_request`, computes headers
separately via `validate_environment` (a no-op for Bedrock), then issues a plain
`httpx` GET and hands the response to `transform_file_content_response`. Headers
and URL are produced independently, so SigV4 cannot sign them together as a pair.
The fit for this contract is a presigned S3 GET URL, where the authentication
lives entirely in the query string. This mirrors how `VertexAIFilesConfig`
returns a plain HTTPS URL and lets `validate_environment` attach auth.

Presigning is pure-local botocore (no network round-trip), works identically for
the sync and async paths, and uses the official SDK rather than hand-rolled
signing. Verified that a botocore presigned URL survives the handler's
`params.update(extract_query_params(url))` re-parsing with the `X-Amz-Signature`,
`X-Amz-Credential`, and full query string preserved byte-for-byte. Path-style
addressing against a regional endpoint produces
`https://s3.{region}.amazonaws.com/{bucket}/{key}?...`, matching the URL shape
that `transform_create_file_request` already builds and avoiding the 307 redirect
that breaks SigV4 for non-`us-east-1` buckets.

Alternatives rejected:
- Signing request headers in the config: the handler computes headers via
  `validate_environment`, which never sees the URL or file_id, so a header-based
  SigV4 signature cannot be produced in the right place.
- Special-casing bedrock to call the boto3 downloader: bypasses the generic
  handler and recreates the exact dead-code split the issue complains about.

## Data flow (after fix)

```
GET /v1/files/{id}/content  (custom_llm_provider=bedrock)
  -> litellm.afile_content -> retrieve_file_content (generic handler)
     -> BedrockFilesConfig.transform_file_content_request(file_content_request, {}, litellm_params)
          extract s3:// URI from file_id (decode base64 unified id or accept s3:// directly)
          resolve configured bucket (trusted credentials / AWS_S3_BUCKET_NAME)
          validate_managed_cloud_file_id (bucket match + managed prefix + no traversal)
          resolve region (s3_region_name > aws_region_name > _get_aws_region_name)
          resolve AWS credentials from litellm_params
          -> (presigned_s3_get_url, {})
     -> validate_environment  (Bedrock no-op)
     -> httpx GET presigned_s3_get_url        # signature preserved through handler
     -> BedrockFilesConfig.transform_file_content_response(raw_response, ...)
          -> HttpxBinaryResponseContent(response=raw_response)   # raw S3 bytes
```

The file_id at the config boundary is an `s3://bucket/key` URI: this is what
`transform_create_file_response` returns for uploads and what the Bedrock batch
handler surfaces as `output_file_id` (`<prefix>/<job-id>/<basename>.out`).

## Components

### `litellm/llms/bedrock/files/transformation.py`

`BedrockFilesConfig` already extends `BaseAWSLLM`, so `get_credentials`,
`_get_aws_region_name`, and `_get_ssl_verify` are available.

Move these validated helpers from `BedrockFilesHandler` into the config:
- `_extract_s3_uri_from_file_id(file_id: str) -> str`
- `_get_configured_s3_bucket_name(litellm_params: dict) -> str`

Reuse `validate_managed_cloud_file_id` and `should_allow_legacy_cloud_file_ids`
directly (the handler's `_parse_s3_uri` was a thin wrapper over the former).

Replace the two raising methods:

```python
def transform_file_content_request(self, file_content_request, optional_params, litellm_params) -> tuple[str, dict]:
    file_id = file_content_request.get("file_id") or ""
    s3_uri = self._extract_s3_uri_from_file_id(file_id)
    configured_bucket = self._get_configured_s3_bucket_name(litellm_params)
    bucket_name, object_key = validate_managed_cloud_file_id(
        file_id=s3_uri,
        scheme="s3://",
        configured_bucket_name=configured_bucket,
        allowed_object_prefixes=BEDROCK_MANAGED_S3_PREFIXES,
        allow_legacy_cloud_file_ids=should_allow_legacy_cloud_file_ids(litellm_params),
    )
    url = self._generate_presigned_s3_get_url(bucket_name, object_key, litellm_params)
    return url, {}

def transform_file_content_response(self, raw_response, logging_obj, litellm_params) -> HttpxBinaryResponseContent:
    return HttpxBinaryResponseContent(response=raw_response)
```

`_generate_presigned_s3_get_url` resolves the region with the same precedence
`transform_create_file_request` uses (`s3_region_name` from litellm_params or
optional_params, then `_get_aws_region_name`), resolves credentials via
`get_credentials`, and calls botocore `generate_presigned_url("get_object", ...)`
with `Config(signature_version="s3v4", s3={"addressing_style": "path"})` against
a regional endpoint. Credentials are read out of `litellm_params` and passed as
typed arguments into `get_credentials` so no `Any`/`dict` leaks into the signing
call.

### `litellm/llms/bedrock/files/handler.py`

Delete. The download logic is superseded by the presigned-URL path through the
generic handler; the validated helpers move to the config.

### `litellm/files/main.py`

Remove the `BedrockFilesHandler` import, the `bedrock_files_instance` module
global, and the unreachable `elif custom_llm_provider == "bedrock":` branch in
`file_content`. The generic `provider_config is not None` branch already routes
Bedrock through `retrieve_file_content`.

## Error handling

The presign step performs only local validation, which already raises clear
`ValueError`s for a bad scheme, wrong bucket, unmanaged prefix / path traversal,
or missing bucket configuration. Those propagate unchanged. The S3 GET goes
through the shared httpx client; a 403/404 from S3 flows into
`transform_file_content_response` and is wrapped as-is, so the caller sees the
real S3 status and body instead of a 500. No speculative handling is added for
cases that cannot occur.

## Tests

`tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py`
(new `TestBedrockFilesContentRetrieval`), AWS/S3 boundary mocked, real transform
logic exercised:

1. `transform_file_content_request` for an `s3://configured-bucket/litellm-batch-outputs/job/in.jsonl.out`
   file_id returns a presigned GET URL: asserts `X-Amz-Algorithm=AWS4-HMAC-SHA256`
   and `X-Amz-Signature` are present and the URL targets the right bucket and key.
2. Region precedence: `s3_region_name` wins over `aws_region_name`; the chosen
   region appears in the presigned `X-Amz-Credential` scope. (create_file parity;
   prime mutation target.)
3. A base64-encoded unified file_id is decoded to its underlying `s3://` URI and
   presigned correctly.
4. Security: file_id bucket != configured bucket raises `ValueError`; an
   unmanaged-prefix key in the configured bucket raises `ValueError`.
5. `transform_file_content_response` returns the raw bytes unchanged: an
   `httpx.Response` carrying known JSONL bytes round-trips byte-for-byte through
   `HttpxBinaryResponseContent.content` (proves raw bytes, not a transform).
6. Wiring: drive `base_llm_http_handler.retrieve_file_content` with
   `BedrockFilesConfig` and a fake client returning canned bytes; assert the GET
   uses the presigned URL and the returned content matches. Proves the path from
   the issue's 28-step trace no longer raises.

`tests/test_litellm/llms/bedrock/files/test_bedrock_files_handler.py`: re-point
the existing `_parse_s3_uri`, `_extract_s3_uri_from_file_id`, and
`_get_configured_s3_bucket_name` tests at `BedrockFilesConfig`, preserving the
SSRF / path-traversal regression coverage as the methods move.

## Proof of fix

Live proxy (`localhost:4000`) against real AWS S3. Upload a managed batch JSONL
via `POST /v1/files` (real S3 write), then `GET /v1/files/{id}/content` and show
the curl command and returned bytes. A full batch end-to-end (create batch, wait
for `Completed`, fetch output) is shown if it completes in the available window;
otherwise the limitation is stated and the closest real-S3-backed retrieval is
shown.
