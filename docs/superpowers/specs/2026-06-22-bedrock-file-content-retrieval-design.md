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
`X-Amz-Credential`, and full query string preserved byte-for-byte.

The boto3 S3 client is built with `Config(signature_version="s3v4",
s3={"addressing_style": "path"})` and NO explicit `endpoint_url`. `s3v4` is
required because botocore's default emits deprecated SigV2 query params.
Omitting `endpoint_url` lets botocore derive the correct host per partition
from `region_name` — `s3.{region}.amazonaws.com` for standard regions,
`s3.{region}.amazonaws.com.cn` for China, and the GovCloud/ISO hosts for those
partitions — while staying regional (avoids the `us-east-1` 307 redirect that
breaks SigV4). A hardcoded regional endpoint string would misroute China and
GovCloud. STS/role credentials flow through as `X-Amz-Security-Token`.
`ExpiresIn` is 300s: the URL is consumed synchronously and is logged by the
handler as `api_base`, so a short window bounds replay of a leaked URL. All
verified across us-west-2, us-east-1, eu-central-1, ap-southeast-2, cn-north-1,
and us-gov-west-1.

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
          resolve allowed buckets from the trusted snapshot (input + output)
          validate_managed_cloud_file_id against each candidate (bucket match + managed prefix + no traversal)
          resolve region from trusted snapshot (s3_region_name) then aws_region_name
          resolve AWS credentials (incl. aws_external_id)
          -> (presigned_s3_get_url, {})
     -> validate_environment  (Bedrock no-op)
     -> httpx GET presigned_s3_get_url        # signature preserved through handler
     -> BedrockFilesConfig.transform_file_content_response(raw_response, ...)
          -> HttpxBinaryResponseContent(response=raw_response)   # raw S3 bytes
```

The file_id at the config boundary is an `s3://bucket/key` URI: this is what
`transform_create_file_response` returns for uploads and what the Bedrock batch
handler surfaces as `output_file_id` (`<output-bucket>/litellm-batch-outputs/<job-id>/<basename>.out`).

## Credential, region, and bucket resolution

The production `litellm.file_content` path builds `litellm_params` via
`get_litellm_params`, which keeps `aws_region_name` and `aws_external_id` but
**drops** `s3_bucket_name`, `s3_region_name`, and `s3_output_bucket_name`. The
proxy instead forwards the model's deployment credentials as an immutable
`MappingProxyType` snapshot under `_litellm_internal_model_credentials`
(`prepare_data_with_credentials(..., include_internal_credentials=True)`). That
snapshot is the only trustworthy source of the S3 bucket/region config, and
using it (rather than request-supplied keys) is what prevents a caller from
redirecting the presign at an arbitrary bucket. Therefore:

- Bucket(s) and `s3_region_name` are read from the trusted snapshot, falling
  back to env (`AWS_S3_BUCKET_NAME`) / `aws_region_name` respectively.
- Validation accepts the file_id if it matches **either** the configured input
  bucket (`s3_bucket_name`, honoring its optional prefix) **or** the configured
  output bucket (`s3_output_bucket_name`, if set). Bedrock batch creation writes
  results to `s3://{s3_output_bucket_name or input_bucket}/litellm-batch-outputs/<job>/`
  at bucket root, ignoring any input prefix, so retrieval validation must allow
  that exact shape or it would reject the very outputs the feature exists to
  serve.
- `get_credentials` is called with the full AWS arg set including
  `aws_external_id`, matching the Bedrock batches handler, so cross-account
  AssumeRole-with-ExternalId deployments work.

## Components

### `litellm/llms/bedrock/files/transformation.py`

`BedrockFilesConfig` already extends `BaseAWSLLM`, so `get_credentials`,
`_get_aws_region_name`, and `_get_ssl_verify` are available.

Move/add these helpers to the config:
- `_extract_s3_uri_from_file_id(file_id: str) -> str` (moved from the handler)
- `_get_trusted_credentials(litellm_params: dict) -> Mapping[str, Any]` — return
  the `_litellm_internal_model_credentials` `MappingProxyType` snapshot, or an
  empty mapping when absent.
- `_get_configured_s3_buckets(litellm_params: dict) -> tuple[str, ...]` — the
  ordered, de-duplicated set of trusted buckets to validate against: the input
  bucket (trusted `s3_bucket_name` or `AWS_S3_BUCKET_NAME`) plus the output
  bucket (trusted `s3_output_bucket_name` or `AWS_S3_OUTPUT_BUCKET_NAME`) when
  configured. Raises `ValueError` if none resolve.
- `_resolve_s3_region(litellm_params: dict) -> str` — trusted `s3_region_name`,
  else `self._get_aws_region_name(optional_params=litellm_params, model="")`
  (which reads the surviving `aws_region_name` and env, with a safe default).

Reuse `validate_managed_cloud_file_id` and `should_allow_legacy_cloud_file_ids`
directly (the handler's `_parse_s3_uri` was a thin wrapper over the former).

Replace the two raising methods:

```python
def transform_file_content_request(self, file_content_request, optional_params, litellm_params) -> tuple[str, dict]:
    file_id = file_content_request.get("file_id") or ""
    s3_uri = self._extract_s3_uri_from_file_id(file_id)
    bucket_name, object_key = self._validate_against_configured_buckets(
        s3_uri=s3_uri, litellm_params=litellm_params
    )
    url = self._generate_presigned_s3_get_url(bucket_name, object_key, litellm_params)
    return url, {}

def transform_file_content_response(self, raw_response, logging_obj, litellm_params) -> HttpxBinaryResponseContent:
    return HttpxBinaryResponseContent(response=raw_response)
```

`_validate_against_configured_buckets` tries `validate_managed_cloud_file_id`
once per configured bucket (input then output) with
`allowed_object_prefixes=BEDROCK_MANAGED_S3_PREFIXES` and
`allow_legacy_cloud_file_ids=should_allow_legacy_cloud_file_ids(litellm_params)`,
returning the first `(bucket, key)` that validates and raising the last
`ValueError` if none do. The output bucket is validated with no configured
prefix because Bedrock writes batch outputs at bucket root.

`_generate_presigned_s3_get_url` resolves the region via `_resolve_s3_region`,
resolves credentials via `get_credentials` passing the full AWS arg set
including `aws_external_id`, and calls botocore
`generate_presigned_url("get_object", ...)` with
`Config(signature_version="s3v4", s3={"addressing_style": "path"})` and no
explicit `endpoint_url` (botocore derives the correct per-partition host).
Credentials are read from the trusted snapshot / `litellm_params` and passed as
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

## Known limitations (documented, not fixed here)

- The shared `retrieve_file_content` GET in `llm_http_handler` does not forward
  the `timeout` argument and does not `raise_for_status`, so an S3 403/404 is
  returned to the client but the proxy may record the request as `"success"`.
  Both behaviors are in the generic handler, affect every provider on that path,
  and predate this work; fixing them is a separate PR, not part of this scope.
- Web-identity (OIDC) auth uses an STS session-policy ceiling
  (`BaseAWSLLM._auth_with_web_identity_token`) that grants only Bedrock /
  Anthropic actions, not `s3:GetObject`. OIDC-authenticated deployments cannot
  retrieve S3 content until that ceiling adds an S3 statement. This affects any
  S3-backed retrieval regardless of presign-vs-download, so it is noted rather
  than solved here.

## Security note on the presigned URL

The presigned URL carries the access key id (`X-Amz-Credential`, a public
identifier), an `X-Amz-Signature` (a non-reversible HMAC scoped to this one GET
on this one object), and, for STS credentials, `X-Amz-Security-Token` (unusable
without the secret key). The generic handler logs the URL as `api_base`, and the
log masker only scrubs `key=`, so the URL can appear in debug logs. The worst
case is a single-object read within the expiry window; the secret access key is
never in the URL. `ExpiresIn` is set to 300s to bound that window. This is the
same exposure profile as any presigned-URL flow and is an accepted tradeoff.

## Tests

`tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py`
(new `TestBedrockFilesContentRetrieval`), AWS/S3 boundary mocked, real transform
logic exercised:

All bucket/region config is supplied via the trusted
`_litellm_internal_model_credentials` `MappingProxyType` snapshot, exactly as the
proxy delivers it in production (not as plain `litellm_params` keys, which
`get_litellm_params` would strip).

1. `transform_file_content_request` for an `s3://input-bucket/litellm-batch-outputs/job/in.jsonl.out`
   file_id returns a presigned GET URL: asserts `X-Amz-Algorithm=AWS4-HMAC-SHA256`
   and `X-Amz-Signature` are present and the URL targets the right bucket and key.
2. Region precedence: trusted `s3_region_name` wins over `aws_region_name`; the
   chosen region appears in the presigned `X-Amz-Credential` scope. (Prime
   mutation target; also guards the get_litellm_params-strips-s3_region_name bug.)
2a. Partition correctness: a `cn-north-1` region presigns against
   `s3.cn-north-1.amazonaws.com.cn` (guards against re-introducing a hardcoded
   `amazonaws.com` endpoint).
2b. Signature version: the URL carries `X-Amz-Algorithm=AWS4-HMAC-SHA256` and no
   SigV2 `Signature`/`AWSAccessKeyId` params (guards the `s3v4` config).
3. A base64-encoded unified file_id is decoded to its underlying `s3://` URI and
   presigned correctly.
4. Security: a file_id whose bucket is neither the input nor the output bucket
   raises `ValueError`; an unmanaged-prefix key in the configured bucket raises
   `ValueError`; only the trusted snapshot is honored (a plain request-level
   `s3_bucket_name` is ignored).
4a. Output bucket: a file_id in the trusted `s3_output_bucket_name` (a *different*
   bucket than `s3_bucket_name`) under `litellm-batch-outputs/` validates and
   presigns. Guards the "separate output bucket rejected" regression.
4b. Prefix-scoped input bucket: a `litellm-batch-outputs/...` output key at bucket
   root validates even when the input-bucket config carries a prefix. Guards the
   "prefix-scoped config rejects its own outputs" regression.
4c. `aws_external_id` from the trusted snapshot is forwarded into
   `get_credentials` (assert via a mocked `get_credentials` capturing kwargs).
5. `transform_file_content_response` returns the raw bytes unchanged: an
   `httpx.Response` carrying known JSONL bytes round-trips byte-for-byte through
   `HttpxBinaryResponseContent.content` (proves raw bytes, not a transform).
6. Wiring: drive `litellm.files.main.base_llm_http_handler.retrieve_file_content`
   (the real symbol the production path uses) with `BedrockFilesConfig`, patching
   the underlying httpx client's transport so the real `HTTPHandler.get`
   query-param reconstruction runs; assert the GET reaches S3 with the presigned
   signature intact and the returned content matches. Proves the path from the
   issue's 28-step trace no longer raises and that the presigned URL survives the
   handler's `params.update(extract_query_params(url))`.

`tests/test_litellm/llms/bedrock/files/test_bedrock_files_handler.py`: re-point
the existing `_parse_s3_uri` / `_extract_s3_uri_from_file_id` /
`_get_configured_s3_bucket_name` regression tests at `BedrockFilesConfig` (or the
shared `validate_managed_cloud_file_id`), preserving the SSRF / path-traversal
coverage as the methods move.

## Proof of fix

Live proxy (`localhost:4000`) against real AWS, full Bedrock batch end-to-end,
captured for the PR:

1. `POST /v1/files` with a batch JSONL (real S3 write) -> managed file id
2. `POST /v1/batches` against a real Bedrock model -> batch id
3. poll `GET /v1/batches/{id}` until status `completed` -> `output_file_id`
4. `GET /v1/files/{output_file_id}/content` -> the batch output bytes returned

Each step shown as the curl command plus its output. Before/after contrast: the
same step 4 against `main` returns the 500
`BedrockFilesConfig does not support file content retrieval`; after the fix it
returns the S3 object content.
