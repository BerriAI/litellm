# Bedrock Managed File Content Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `GET /v1/files/{id}/content` return the S3 object bytes for a Bedrock-backed managed file instead of a 500.

**Architecture:** Implement `transform_file_content_request` / `transform_file_content_response` on `BedrockFilesConfig` so Bedrock flows through the same generic `retrieve_file_content` handler as Vertex/Anthropic/Manus. The request transform validates the file_id against the configured input and output buckets and returns a botocore presigned S3 GET URL (auth in the query string, the only shape that fits a handler which computes URL and headers independently); the response transform wraps the raw bytes. All bucket/region config is read from the trusted `_litellm_internal_model_credentials` snapshot, because the production `file_content` path strips plain `s3_*` keys via `get_litellm_params`. The validated S3-id helpers move out of the now-dead `BedrockFilesHandler`, which is deleted along with its unreachable branch in `files/main.py`.

**Tech Stack:** Python, botocore (S3 SigV4 presigning, already a hard dependency `boto3>=1.43.1`), httpx, pytest.

## Global Constraints

- No comments unless they explain genuinely complex business logic; do not add comments that restate the code.
- Fully typed; no `Any` or bare `dict`/`dict[str, Any]`. Validate untyped inputs and pass typed values.
- No mutation of variables where avoidable; prefer building values in one shot.
- Tests must fail if the feature is mutated/broken (mutation kill rate > 90%); mock only the AWS/S3 boundary, never the transform logic under test.
- All bucket/region config in tests must be supplied via the `_litellm_internal_model_credentials` `MappingProxyType` snapshot, never as plain `litellm_params` keys — that is what the production proxy path delivers (`get_litellm_params` strips `s3_bucket_name`/`s3_region_name`/`s3_output_bucket_name`).
- Run `make format` and `make lint` before each commit; run `make lint-budget-update` and commit lowered baselines if `ruff-strict-budget.json` / `basedpyright-code-budget.json` trip.
- Branch is `litellm_bedrock_file_content`, based on `litellm_internal_staging`. Commit messages follow conventional commits; no Claude attribution.
- `git push` always uses `--no-verify` (only when explicitly asked to push).

## File Structure

- `litellm/llms/bedrock/files/transformation.py` (modify): add `_get_trusted_credentials`, `_extract_s3_uri_from_file_id`, `_get_configured_s3_buckets`, `_validate_against_configured_buckets`, `_resolve_s3_region`, `_generate_presigned_s3_get_url` to `BedrockFilesConfig`; replace the two `NotImplementedError` content methods with real implementations.
- `litellm/llms/bedrock/files/handler.py` (delete): superseded; logic moved to the config.
- `litellm/files/main.py` (modify): drop the `BedrockFilesHandler` import, the `bedrock_files_instance` global, and the unreachable `elif custom_llm_provider == "bedrock":` branch in `file_content`.
- `tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py` (modify): new `TestBedrockFilesContentRetrieval` class.
- `tests/test_litellm/llms/bedrock/files/test_bedrock_files_handler.py` (modify): re-point the helper tests at `BedrockFilesConfig`.

## Background facts (verified)

- `get_litellm_params(**kwargs)` keeps `aws_region_name` and `aws_external_id` but **drops** `s3_bucket_name`, `s3_region_name`, `s3_output_bucket_name`, `s3_endpoint_url`. So `transform_file_content_request` must not rely on those plain keys.
- The proxy forwards the model deployment credentials as `litellm_params["_litellm_internal_model_credentials"] = MappingProxyType(dict(credentials))`. That snapshot carries `s3_bucket_name`, `s3_region_name`, `s3_output_bucket_name`, `aws_external_id`, etc. It is immutable and server-side, so it is the trustworthy source (a plain request-level `s3_bucket_name` must be ignored to prevent SSRF/bucket redirection).
- Bedrock batch creation writes outputs to `s3://{s3_output_bucket_name or input_bucket}/litellm-batch-outputs/{job}/...` — at bucket root, ignoring any input-bucket prefix. Retrieval validation must allow that exact shape.
- `validate_managed_cloud_file_id(file_id, scheme, configured_bucket_name, allowed_object_prefixes, allow_legacy_cloud_file_ids)` returns `(bucket, raw_object_key)`; when `configured_bucket_name` includes a prefix (`bucket/prefix`), it prepends that prefix to each allowed prefix.
- botocore `generate_presigned_url("get_object", ...)` with `Config(signature_version="s3v4", s3={"addressing_style": "path"})` and no `endpoint_url`: emits the correct per-partition host (`s3.{region}.amazonaws.com`, `s3.amazonaws.com` for us-east-1, `s3.{region}.amazonaws.com.cn` for China), puts STS tokens in `X-Amz-Security-Token`, and survives the handler's `params.update(extract_query_params(url))` re-parse byte-for-byte.

---

### Task 1: Trusted-credential, bucket, and region helpers on BedrockFilesConfig

Add the config-owned helpers that read the trusted snapshot and resolve buckets/region. These are pure functions over `litellm_params`, easy to unit-test in isolation before any presigning.

**Files:**
- Modify: `litellm/llms/bedrock/files/transformation.py`
- Test: `tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py`

**Interfaces:**
- Consumes: `validate_managed_cloud_file_id`, `should_allow_legacy_cloud_file_ids`, `BEDROCK_MANAGED_S3_PREFIXES` (from `litellm.litellm_core_utils.cloud_storage_security`); `SpecialEnums` (from `litellm.types.utils`).
- Produces:
  - `BedrockFilesConfig._get_trusted_credentials(self, litellm_params: dict) -> Mapping[str, Any]`
  - `BedrockFilesConfig._extract_s3_uri_from_file_id(self, file_id: str) -> str`
  - `BedrockFilesConfig._get_configured_s3_buckets(self, litellm_params: dict) -> tuple[str, ...]`
  - `BedrockFilesConfig._resolve_s3_region(self, litellm_params: dict) -> str`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py`:

```python
import base64
import os
from types import MappingProxyType
from unittest.mock import patch

import pytest

from litellm.llms.bedrock.files.transformation import BedrockFilesConfig
from litellm.types.utils import SpecialEnums


def _encode_unified_file_id(s3_uri: str) -> str:
    unified_file_id = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
        "application/json",
        "unified-id",
        "",
        s3_uri,
        "model-id",
    )
    return base64.urlsafe_b64encode(unified_file_id.encode()).decode().rstrip("=")


def _trusted(**creds) -> dict:
    return {"_litellm_internal_model_credentials": MappingProxyType(dict(creds))}


class TestBedrockFilesConfigResolution:
    def setup_method(self):
        self.config = BedrockFilesConfig()

    def test_extract_direct_s3_uri(self):
        assert (
            self.config._extract_s3_uri_from_file_id(
                "s3://b/litellm-batch-outputs/job/output.jsonl"
            )
            == "s3://b/litellm-batch-outputs/job/output.jsonl"
        )

    def test_extract_unified_managed_s3_uri(self):
        file_id = _encode_unified_file_id("s3://b/litellm-batch-outputs/job/output.jsonl")
        assert (
            self.config._extract_s3_uri_from_file_id(file_id)
            == "s3://b/litellm-batch-outputs/job/output.jsonl"
        )

    def test_extract_rejects_non_s3_file_id(self):
        with pytest.raises(ValueError, match="managed LiteLLM S3 file id"):
            self.config._extract_s3_uri_from_file_id("b/private.jsonl")

    def test_buckets_prefer_trusted_snapshot_over_request(self):
        params = {
            "s3_bucket_name": "attacker-bucket",
            **_trusted(s3_bucket_name="safe-bucket"),
        }
        with patch.dict(os.environ, {}, clear=True):
            assert self.config._get_configured_s3_buckets(params) == ("safe-bucket",)

    def test_buckets_include_output_bucket(self):
        params = _trusted(s3_bucket_name="in-bucket", s3_output_bucket_name="out-bucket")
        with patch.dict(os.environ, {}, clear=True):
            assert self.config._get_configured_s3_buckets(params) == (
                "in-bucket",
                "out-bucket",
            )

    def test_buckets_fall_back_to_env(self):
        with patch.dict(
            os.environ,
            {"AWS_S3_BUCKET_NAME": "env-in", "AWS_S3_OUTPUT_BUCKET_NAME": "env-out"},
            clear=True,
        ):
            assert self.config._get_configured_s3_buckets({}) == ("env-in", "env-out")

    def test_buckets_require_at_least_one(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="S3 bucket_name is required"):
                self.config._get_configured_s3_buckets({"s3_bucket_name": "ignored"})

    def test_region_prefers_trusted_s3_region_name(self):
        params = {"aws_region_name": "us-east-1", **_trusted(s3_region_name="eu-central-1")}
        assert self.config._resolve_s3_region(params) == "eu-central-1"

    def test_region_falls_back_to_aws_region_name(self):
        assert self.config._resolve_s3_region({"aws_region_name": "ap-southeast-2"}) == (
            "ap-southeast-2"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -m pytest tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py::TestBedrockFilesConfigResolution -v`
Expected: FAIL with `AttributeError: 'BedrockFilesConfig' object has no attribute '_get_trusted_credentials'` (or the first missing helper).

- [ ] **Step 3: Add imports and helpers**

In `litellm/llms/bedrock/files/transformation.py`, add to the top-of-file imports:

```python
import base64
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast
```

Extend the existing `cloud_storage_security` import to add `BEDROCK_MANAGED_S3_PREFIXES`, `should_allow_legacy_cloud_file_ids`, and `validate_managed_cloud_file_id`; extend the `litellm.types.utils` import to add `SpecialEnums`.

Add these methods to `BedrockFilesConfig` (place above the content-transform methods):

```python
    def _get_trusted_credentials(self, litellm_params: dict) -> Mapping[str, Any]:
        snapshot = litellm_params.get("_litellm_internal_model_credentials")
        if isinstance(snapshot, type(MappingProxyType({}))):
            return cast(Mapping[str, Any], snapshot)
        return MappingProxyType({})

    def _extract_s3_uri_from_file_id(self, file_id: str) -> str:
        try:
            padded = file_id + "=" * (-len(file_id) % 4)
            decoded = base64.urlsafe_b64decode(padded).decode()
            if decoded.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
                if "llm_output_file_id," in decoded:
                    return decoded.split("llm_output_file_id,")[1].split(";")[0]
        except Exception:
            pass

        if file_id.startswith("s3://"):
            return file_id

        raise ValueError("file_id must be a managed LiteLLM S3 file id")

    def _get_configured_s3_buckets(self, litellm_params: dict) -> Tuple[str, ...]:
        trusted = self._get_trusted_credentials(litellm_params)

        input_candidate = trusted.get("s3_bucket_name")
        input_bucket = (
            input_candidate
            if isinstance(input_candidate, str) and input_candidate
            else os.getenv("AWS_S3_BUCKET_NAME")
        )

        output_candidate = trusted.get("s3_output_bucket_name")
        output_bucket = (
            output_candidate
            if isinstance(output_candidate, str) and output_candidate
            else os.getenv("AWS_S3_OUTPUT_BUCKET_NAME")
        )

        buckets = tuple(
            dict.fromkeys(
                bucket for bucket in (input_bucket, output_bucket) if bucket
            )
        )
        if not buckets:
            raise ValueError(
                "S3 bucket_name is required. Set 's3_bucket_name' in proxy config or "
                "AWS_S3_BUCKET_NAME for Bedrock file content retrieval."
            )
        return buckets

    def _resolve_s3_region(self, litellm_params: dict) -> str:
        trusted = self._get_trusted_credentials(litellm_params)
        trusted_region = trusted.get("s3_region_name")
        if isinstance(trusted_region, str) and trusted_region:
            return trusted_region
        return self._get_aws_region_name(optional_params=litellm_params, model="")
```

Notes:
- `_get_configured_s3_buckets` reads only the trusted snapshot or env, never a plain `litellm_params["s3_bucket_name"]`, preserving SSRF protection.
- Buckets are built with `dict.fromkeys(...)` to de-duplicate while preserving order (input first), with no mutation — per the functional-style constraint.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -m pytest tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py::TestBedrockFilesConfigResolution -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content
make format && make lint
git add litellm/llms/bedrock/files/transformation.py tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py
git commit --no-verify -m "refactor(bedrock): add trusted-snapshot S3 bucket/region resolution to files config"
```

---

### Task 2: Multi-bucket validation + presigned-URL content transforms

Replace the two `NotImplementedError` methods. Validation tries each configured bucket (input then output) so batch outputs in a separate bucket or at bucket root validate; presigning resolves region + credentials (incl. `aws_external_id`) and signs locally.

**Files:**
- Modify: `litellm/llms/bedrock/files/transformation.py`
- Test: `tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py`

**Interfaces:**
- Consumes: Task 1 helpers; `self.get_credentials(...)`, `self._get_aws_region_name(...)`, `self._get_ssl_verify()` (from `BaseAWSLLM`); `HttpxBinaryResponseContent` (already imported).
- Produces:
  - `BedrockFilesConfig._validate_against_configured_buckets(self, s3_uri: str, litellm_params: dict) -> tuple[str, str]`
  - `BedrockFilesConfig._generate_presigned_s3_get_url(self, bucket_name: str, object_key: str, litellm_params: dict) -> str`
  - `BedrockFilesConfig.transform_file_content_request(self, file_content_request, optional_params: dict, litellm_params: dict) -> tuple[str, dict]`
  - `BedrockFilesConfig.transform_file_content_response(self, raw_response, logging_obj, litellm_params: dict) -> HttpxBinaryResponseContent`

- [ ] **Step 1: Write the failing tests**

Add this class to `tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py`:

```python
from urllib.parse import parse_qs, urlsplit

import httpx
from botocore.credentials import Credentials

from litellm.types.llms.openai import HttpxBinaryResponseContent


class TestBedrockFilesContentRetrieval:
    def setup_method(self):
        self.config = BedrockFilesConfig()
        self.fake_creds = Credentials(
            access_key="AKIAIOSFODNN7EXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            token=None,
        )

    def _request(self, file_id, litellm_params, capture=None):
        real_get_credentials = self.config.get_credentials

        def spy(**kwargs):
            if capture is not None:
                capture.update(kwargs)
            return self.fake_creds

        with patch.object(self.config, "get_credentials", side_effect=spy):
            return self.config.transform_file_content_request(
                file_content_request={"file_id": file_id},
                optional_params={},
                litellm_params=litellm_params,
            )

    def test_returns_presigned_get_url_for_input_bucket(self):
        url, params = self._request(
            "s3://in-bucket/litellm-batch-outputs/job/in.jsonl.out",
            _trusted(s3_bucket_name="in-bucket", aws_region_name="us-west-2"),
        )
        assert params == {}
        parts = urlsplit(url)
        query = parse_qs(parts.query)
        assert parts.path == "/in-bucket/litellm-batch-outputs/job/in.jsonl.out"
        assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
        assert "X-Amz-Signature" in query

    def test_region_from_trusted_snapshot_wins(self):
        url, _ = self._request(
            "s3://in-bucket/litellm-batch-outputs/job/in.jsonl.out",
            {
                "aws_region_name": "us-east-1",
                **_trusted(
                    s3_bucket_name="in-bucket",
                    s3_region_name="eu-central-1",
                    aws_region_name="us-east-1",
                ),
            },
        )
        credential = parse_qs(urlsplit(url).query)["X-Amz-Credential"][0]
        assert "/eu-central-1/s3/aws4_request" in credential
        assert urlsplit(url).netloc == "s3.eu-central-1.amazonaws.com"

    def test_china_partition_uses_correct_endpoint_suffix(self):
        url, _ = self._request(
            "s3://in-bucket/litellm-batch-outputs/job/in.jsonl.out",
            _trusted(s3_bucket_name="in-bucket", aws_region_name="cn-north-1"),
        )
        assert urlsplit(url).netloc == "s3.cn-north-1.amazonaws.com.cn"
        assert "X-Amz-Signature" in parse_qs(urlsplit(url).query)

    def test_uses_sigv4_not_deprecated_sigv2(self):
        url, _ = self._request(
            "s3://in-bucket/litellm-batch-outputs/job/in.jsonl.out",
            _trusted(s3_bucket_name="in-bucket", aws_region_name="us-west-2"),
        )
        query = parse_qs(urlsplit(url).query)
        assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
        assert "Signature" not in query and "AWSAccessKeyId" not in query

    def test_output_bucket_distinct_from_input_validates(self):
        url, _ = self._request(
            "s3://out-bucket/litellm-batch-outputs/job/in.jsonl.out",
            _trusted(
                s3_bucket_name="in-bucket",
                s3_output_bucket_name="out-bucket",
                aws_region_name="us-west-2",
            ),
        )
        assert urlsplit(url).path == "/out-bucket/litellm-batch-outputs/job/in.jsonl.out"

    def test_batch_output_at_root_validates_with_prefixed_input_bucket(self):
        # input bucket is prefix-scoped; batch output lands at bucket root
        url, _ = self._request(
            "s3://shared/litellm-batch-outputs/job/in.jsonl.out",
            _trusted(
                s3_bucket_name="shared/team-a",
                s3_output_bucket_name="shared",
                aws_region_name="us-west-2",
            ),
        )
        assert urlsplit(url).path == "/shared/litellm-batch-outputs/job/in.jsonl.out"

    def test_unified_file_id_is_decoded_and_presigned(self):
        file_id = _encode_unified_file_id("s3://in-bucket/litellm-batch-outputs/job/in.jsonl.out")
        url, _ = self._request(
            file_id, _trusted(s3_bucket_name="in-bucket", aws_region_name="us-west-2")
        )
        assert urlsplit(url).path == "/in-bucket/litellm-batch-outputs/job/in.jsonl.out"
        assert "X-Amz-Signature" in parse_qs(urlsplit(url).query)

    def test_rejects_bucket_outside_configured_set(self):
        with pytest.raises(ValueError):
            self._request(
                "s3://other-bucket/litellm-batch-outputs/job/in.jsonl.out",
                _trusted(
                    s3_bucket_name="in-bucket",
                    s3_output_bucket_name="out-bucket",
                    aws_region_name="us-west-2",
                ),
            )

    def test_rejects_unmanaged_prefix_in_configured_bucket(self):
        with pytest.raises(ValueError):
            self._request(
                "s3://in-bucket/private/secret.jsonl",
                _trusted(s3_bucket_name="in-bucket", aws_region_name="us-west-2"),
            )

    def test_ignores_plain_request_bucket(self):
        with pytest.raises(ValueError, match="S3 bucket_name is required"):
            self._request(
                "s3://attacker-bucket/litellm-batch-outputs/job/in.jsonl.out",
                {"s3_bucket_name": "attacker-bucket", "aws_region_name": "us-west-2"},
            )

    def test_forwards_aws_external_id_to_get_credentials(self):
        capture: dict = {}
        self._request(
            "s3://in-bucket/litellm-batch-outputs/job/in.jsonl.out",
            {
                "aws_external_id": "ext-123",
                **_trusted(s3_bucket_name="in-bucket", aws_region_name="us-west-2"),
            },
            capture=capture,
        )
        assert capture.get("aws_external_id") == "ext-123"

    def test_response_returns_raw_bytes_unchanged(self):
        body = b'{"recordId":"req-1","modelOutput":{"foo":"bar"}}\n'
        raw_response = httpx.Response(
            status_code=200,
            content=body,
            headers={"content-type": "application/octet-stream"},
            request=httpx.Request("GET", "https://s3.us-west-2.amazonaws.com/in-bucket/x"),
        )
        result = self.config.transform_file_content_response(
            raw_response=raw_response, logging_obj=None, litellm_params={}
        )
        assert isinstance(result, HttpxBinaryResponseContent)
        assert result.content == body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -m pytest tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py::TestBedrockFilesContentRetrieval -v`
Expected: FAIL — `transform_file_content_request` raises `NotImplementedError`.

- [ ] **Step 3: Implement the methods**

In `litellm/llms/bedrock/files/transformation.py`, replace the two file-content methods that currently raise `NotImplementedError` with:

```python
    def _validate_against_configured_buckets(
        self, s3_uri: str, litellm_params: dict
    ) -> Tuple[str, str]:
        configured_buckets = self._get_configured_s3_buckets(litellm_params)
        allow_legacy = should_allow_legacy_cloud_file_ids(litellm_params)
        last_error: Optional[ValueError] = None
        for configured_bucket in configured_buckets:
            try:
                return validate_managed_cloud_file_id(
                    file_id=s3_uri,
                    scheme="s3://",
                    configured_bucket_name=configured_bucket,
                    allowed_object_prefixes=BEDROCK_MANAGED_S3_PREFIXES,
                    allow_legacy_cloud_file_ids=allow_legacy,
                )
            except ValueError as e:
                last_error = e
        raise last_error or ValueError(
            "file_id must reference a LiteLLM-managed storage object"
        )

    def _generate_presigned_s3_get_url(
        self, bucket_name: str, object_key: str, litellm_params: dict
    ) -> str:
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        region = self._resolve_s3_region(litellm_params)
        credentials = self.get_credentials(
            aws_access_key_id=litellm_params.get("aws_access_key_id"),
            aws_secret_access_key=litellm_params.get("aws_secret_access_key"),
            aws_session_token=litellm_params.get("aws_session_token"),
            aws_region_name=region,
            aws_session_name=litellm_params.get("aws_session_name"),
            aws_profile_name=litellm_params.get("aws_profile_name"),
            aws_role_name=litellm_params.get("aws_role_name"),
            aws_web_identity_token=litellm_params.get("aws_web_identity_token"),
            aws_sts_endpoint=litellm_params.get("aws_sts_endpoint"),
            aws_external_id=litellm_params.get("aws_external_id"),
        )

        s3_client = boto3.client(
            "s3",
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
            region_name=region,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
            verify=self._get_ssl_verify(),
        )

        return s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=300,
        )

    def transform_file_content_request(
        self,
        file_content_request,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        file_id = file_content_request.get("file_id") or ""
        s3_uri = self._extract_s3_uri_from_file_id(file_id)
        bucket_name, object_key = self._validate_against_configured_buckets(
            s3_uri=s3_uri, litellm_params=litellm_params
        )
        url = self._generate_presigned_s3_get_url(
            bucket_name=bucket_name,
            object_key=object_key,
            litellm_params=litellm_params,
        )
        return url, {}

    def transform_file_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> HttpxBinaryResponseContent:
        return HttpxBinaryResponseContent(response=raw_response)
```

Notes:
- `validate_managed_cloud_file_id` returns the raw (not URL-encoded) object key; botocore URL-encodes it. Do not pre-encode.
- No `endpoint_url` is passed: botocore derives the correct per-partition host while staying regional.
- `aws_external_id` is forwarded to match the Bedrock batches handler.
- Output-bucket configs are validated by trying each bucket; because `s3_output_bucket_name` is passed to `_get_configured_s3_buckets` without a prefix, batch outputs at bucket root validate even when the input bucket is prefix-scoped.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -m pytest tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py::TestBedrockFilesContentRetrieval -v`
Expected: PASS (12 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content
make format && make lint
git add litellm/llms/bedrock/files/transformation.py tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py
git commit --no-verify -m "feat(bedrock): retrieve managed file content via presigned S3 GET (#26335)"
```

---

### Task 3: Real wiring test through the generic handler

Prove the issue's 28-step path works end to end and that the presigned URL survives `HTTPHandler.get`'s query-param reconstruction. Use the real `litellm.files.main.base_llm_http_handler` symbol and patch the httpx transport (not the handler's `.get`), so the real param-rebuild runs.

**Files:**
- Test: `tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py`

**Interfaces:**
- Consumes: `litellm.files.main.base_llm_http_handler.retrieve_file_content`; `BedrockFilesConfig` (Task 2); `litellm.llms.custom_httpx.http_handler.HTTPHandler`.

- [ ] **Step 1: Write the test**

Add to `TestBedrockFilesContentRetrieval`:

```python
    def test_retrieve_file_content_through_generic_handler(self):
        import time as _time

        from litellm.files.main import base_llm_http_handler
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        body = b'{"recordId":"req-1","modelOutput":{"ok":true}}\n'
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(status_code=200, content=body)

        sync_client = HTTPHandler()
        sync_client.client = httpx.Client(transport=httpx.MockTransport(handler))

        logging_obj = Logging(
            model="",
            messages=[],
            stream=False,
            call_type="file_content",
            start_time=_time.time(),
            litellm_call_id="test-call",
            function_id="",
        )

        with patch.object(self.config, "get_credentials", return_value=self.fake_creds):
            result = base_llm_http_handler.retrieve_file_content(
                file_content_request={
                    "file_id": "s3://in-bucket/litellm-batch-outputs/job/in.jsonl.out"
                },
                provider_config=self.config,
                litellm_params=_trusted(
                    s3_bucket_name="in-bucket", aws_region_name="us-west-2"
                ),
                headers={},
                logging_obj=logging_obj,
                _is_async=False,
                client=sync_client,
            )

        assert "X-Amz-Signature=" in seen["url"]
        assert "/in-bucket/litellm-batch-outputs/job/in.jsonl.out" in seen["url"]
        assert result.content == body
```

- [ ] **Step 2: Run the test**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -m pytest "tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py::TestBedrockFilesContentRetrieval::test_retrieve_file_content_through_generic_handler" -v`
Expected: PASS. The signature reaching the MockTransport proves it survived `params.update(extract_query_params(url))`.

- [ ] **Step 3: Format, lint, commit**

```bash
cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content
make format && make lint
git add tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py
git commit --no-verify -m "test(bedrock): cover file content retrieval through generic handler"
```

---

### Task 4: Delete dead BedrockFilesHandler and re-point its tests

Remove the superseded handler and migrate its SSRF/path-traversal regression coverage onto the shared validator / config.

**Files:**
- Delete: `litellm/llms/bedrock/files/handler.py`
- Modify: `tests/test_litellm/llms/bedrock/files/test_bedrock_files_handler.py`

**Interfaces:**
- Consumes: `validate_managed_cloud_file_id`, `BEDROCK_MANAGED_S3_PREFIXES` (shared); `BedrockFilesConfig._extract_s3_uri_from_file_id` (Task 1).

- [ ] **Step 1: Rewrite the handler test file to target the shared validator + config**

Replace `TestBedrockFilesHandler` and its import in `tests/test_litellm/llms/bedrock/files/test_bedrock_files_handler.py`. Keep the two module-level `files_main` forwarding tests at the bottom unchanged.

```python
import base64
from types import MappingProxyType

import pytest

from litellm.litellm_core_utils.cloud_storage_security import (
    BEDROCK_MANAGED_S3_PREFIXES,
    validate_managed_cloud_file_id,
)
from litellm.llms.bedrock.files.transformation import BedrockFilesConfig
from litellm.types.utils import SpecialEnums


def _encode_unified_file_id(s3_uri: str) -> str:
    unified_file_id = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
        "application/json", "unified-id", "", s3_uri, "model-id"
    )
    return base64.urlsafe_b64encode(unified_file_id.encode()).decode().rstrip("=")


def _parse(s3_uri, configured_bucket_name, allow_legacy_cloud_file_ids=False):
    return validate_managed_cloud_file_id(
        file_id=s3_uri,
        scheme="s3://",
        configured_bucket_name=configured_bucket_name,
        allowed_object_prefixes=BEDROCK_MANAGED_S3_PREFIXES,
        allow_legacy_cloud_file_ids=allow_legacy_cloud_file_ids,
    )


class TestBedrockFilesConfigS3Validation:
    def setup_method(self):
        self.config = BedrockFilesConfig()

    def test_should_parse_direct_managed_s3_uri(self):
        bucket, key = _parse(
            "s3://safe-bucket/litellm-bedrock-files-model-id-abc.jsonl", "safe-bucket"
        )
        assert bucket == "safe-bucket"
        assert key == "litellm-bedrock-files-model-id-abc.jsonl"

    def test_should_parse_managed_batch_output_uri(self):
        bucket, key = _parse("s3://safe-bucket/litellm-batch-outputs/job/", "safe-bucket")
        assert bucket == "safe-bucket"
        assert key == "litellm-batch-outputs/job/"

    def test_should_reject_arbitrary_bucket(self):
        with pytest.raises(ValueError, match="configured storage bucket"):
            _parse(
                "s3://other-bucket/litellm-bedrock-files-model-id-abc.jsonl", "safe-bucket"
            )

    def test_should_reject_unmanaged_same_bucket_key(self):
        with pytest.raises(ValueError, match="LiteLLM-managed"):
            _parse("s3://safe-bucket/private/output.jsonl", "safe-bucket")

    def test_should_allow_legacy_same_bucket_key_when_server_flag_enabled(self):
        bucket, key = _parse(
            "s3://safe-bucket/private/output.jsonl",
            "safe-bucket",
            allow_legacy_cloud_file_ids=True,
        )
        assert bucket == "safe-bucket"
        assert key == "private/output.jsonl"

    def test_should_keep_configured_prefix_for_legacy_keys(self):
        bucket, key = _parse(
            "s3://safe-bucket/team-a/private/output.jsonl",
            "safe-bucket/team-a",
            allow_legacy_cloud_file_ids=True,
        )
        assert bucket == "safe-bucket"
        assert key == "team-a/private/output.jsonl"

    def test_should_reject_legacy_key_outside_configured_prefix(self):
        with pytest.raises(ValueError, match="configured storage prefix"):
            _parse(
                "s3://safe-bucket/team-b/private/output.jsonl",
                "safe-bucket/team-a",
                allow_legacy_cloud_file_ids=True,
            )

    def test_should_reject_dot_segment_key(self):
        with pytest.raises(ValueError, match="invalid path segment"):
            _parse("s3://safe-bucket/litellm-bedrock-files/../secret.jsonl", "safe-bucket")

    def test_should_reject_empty_middle_path_segment(self):
        with pytest.raises(ValueError, match="invalid path segment"):
            _parse("s3://safe-bucket/litellm-bedrock-files//secret.jsonl", "safe-bucket")

    def test_should_extract_unified_managed_s3_uri(self):
        file_id = _encode_unified_file_id(
            "s3://safe-bucket/litellm-batch-outputs/job/output.jsonl"
        )
        assert (
            self.config._extract_s3_uri_from_file_id(file_id)
            == "s3://safe-bucket/litellm-batch-outputs/job/output.jsonl"
        )

    def test_should_reject_file_id_without_s3_scheme(self):
        with pytest.raises(ValueError, match="managed LiteLLM S3 file id"):
            self.config._extract_s3_uri_from_file_id("safe-bucket/private.jsonl")

    def test_should_reject_unified_unmanaged_s3_uri(self):
        file_id = _encode_unified_file_id("s3://safe-bucket/private/output.jsonl")
        s3_uri = self.config._extract_s3_uri_from_file_id(file_id)
        with pytest.raises(ValueError, match="LiteLLM-managed"):
            _parse(s3_uri, "safe-bucket")
```

The bucket-resolution / trusted-credential tests now live in `TestBedrockFilesConfigResolution` (Task 1), so they are not duplicated here.

- [ ] **Step 2: Delete the handler file**

```bash
cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content
git rm litellm/llms/bedrock/files/handler.py
```

- [ ] **Step 3: Note — collection breaks until Task 5**

Deleting `handler.py` breaks `files/main.py`'s import, so the test module won't collect until Task 5 removes that import. Commit Tasks 4 and 5 together; run the suite after Task 5.

---

### Task 5: Remove the dead bedrock branch from files/main.py

**Files:**
- Modify: `litellm/files/main.py`

**Interfaces:**
- Consumes: nothing new. The generic `provider_config is not None` branch already routes Bedrock through `retrieve_file_content`.

- [ ] **Step 1: Remove the import (currently line 42)**

```python
from litellm.llms.bedrock.files.handler import BedrockFilesHandler
```

- [ ] **Step 2: Remove the module global (currently line 85)**

```python
bedrock_files_instance = BedrockFilesHandler()
```

- [ ] **Step 3: Remove the unreachable branch in `file_content` (currently lines 1021-1029)**

```python
        elif custom_llm_provider == "bedrock":
            response = bedrock_files_instance.file_content(
                _is_async=_is_async,
                file_content_request=_file_content_request,
                api_base=optional_params.api_base,
                optional_params=litellm_params_dict,
                timeout=timeout,
                max_retries=optional_params.max_retries,
            )
```

Leave the surrounding `if/elif/else` and the `else` BadRequestError message (which still lists `'bedrock'` as supported — accurate, since bedrock is served by the generic `provider_config` branch above).

- [ ] **Step 4: Verify the module imports cleanly**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -c "import litellm.files.main"`
Expected: no output, exit 0.

- [ ] **Step 5: Run the full bedrock files test dir**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -m pytest tests/test_litellm/llms/bedrock/files/ -v`
Expected: PASS (all three test files).

- [ ] **Step 6: Format, lint, commit (Tasks 4 + 5 together)**

```bash
cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content
make format && make lint
git add litellm/files/main.py litellm/llms/bedrock/files/handler.py tests/test_litellm/llms/bedrock/files/test_bedrock_files_handler.py
git commit --no-verify -m "refactor(bedrock): delete dead BedrockFilesHandler and unreachable file_content branch (#26335)"
```

---

### Task 6: Full suite sanity + budget check

**Files:** none (verification only).

- [ ] **Step 1: Run the bedrock files + integration suites**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -m pytest tests/test_litellm/llms/bedrock/files/ -v`
Expected: PASS. The existing `test_bedrock_files_integration.py` mocks `retrieve_file_content`, so it stays green.

- [ ] **Step 2: Lint budget check**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && make lint`
Expected: PASS. If a budget file trips (likely *down*, since a file is deleted), run `make lint-budget-update` and commit only if the budget files actually changed:

```bash
make lint-budget-update
git add ruff-strict-budget.json basedpyright-code-budget.json
git commit --no-verify -m "chore: ratchet lint budgets after bedrock file content change"
```

- [ ] **Step 3: grep for stragglers**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && grep -rn "BedrockFilesHandler\|bedrock_files_instance" litellm/ tests/ enterprise/`
Expected: no matches.

---

## Proof of Fix (manual, for PR — after implementation)

Live proxy with real AWS credentials and an S3 bucket configured (config defines a bedrock batch model + `s3_bucket_name`, optionally `s3_output_bucket_name`). Capture each command and its output.

1. Start proxy: `python litellm/proxy/proxy_cli.py --config litellm/proxy/dev_config.yaml --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log`
2. `POST /v1/files` with a small batch JSONL (`purpose=batch`) -> managed file id
3. `POST /v1/batches` against a real Bedrock model -> batch id
4. Poll `GET /v1/batches/{id}` until `status == completed` -> `output_file_id`
5. `GET /v1/files/{output_file_id}/content` -> batch output bytes returned

Show before (500 on `main`) vs after (bytes returned) for step 5.

## Out of scope (documented, not implemented)

- Shared `retrieve_file_content` GET ignores `timeout` and does not `raise_for_status`; the proxy may record an S3 403/404 as `"success"`. Generic-handler behavior, all providers, pre-existing; separate PR.
- Web-identity (OIDC) STS session-policy ceiling grants only Bedrock/Anthropic actions, not `s3:GetObject`; OIDC deployments can't retrieve S3 content until that ceiling adds an S3 statement. Affects any S3-backed retrieval regardless of approach.

## Self-Review

**Spec coverage:**
- presigned content request/response -> Task 2.
- trusted-snapshot bucket/region resolution (get_litellm_params strips s3_* keys) -> Task 1 + Task 2 region test.
- input + output bucket validation, batch-output-at-root with prefixed input -> Task 2 tests `test_output_bucket_distinct_from_input_validates`, `test_batch_output_at_root_validates_with_prefixed_input_bucket`.
- `aws_external_id` forwarded -> Task 2 test `test_forwards_aws_external_id_to_get_credentials`.
- ignore plain request bucket (SSRF) -> Task 1 `test_buckets_prefer_trusted_snapshot_over_request` + Task 2 `test_ignores_plain_request_bucket`.
- partition correctness + sigv4 -> Task 2 tests.
- raw bytes (no OpenAI transform) -> Task 2 `test_response_returns_raw_bytes_unchanged`.
- real wiring (correct module, real param reconstruction) -> Task 3.
- delete dead handler + unreachable branch -> Tasks 4, 5.
- known limitations -> Out of scope section.

**Placeholder scan:** none; every code step has complete code and expected output.

**Type consistency:** `_get_trusted_credentials(litellm_params) -> Mapping[str, Any]`, `_extract_s3_uri_from_file_id(file_id) -> str`, `_get_configured_s3_buckets(litellm_params) -> Tuple[str, ...]`, `_resolve_s3_region(litellm_params) -> str`, `_validate_against_configured_buckets(s3_uri, litellm_params) -> Tuple[str, str]`, `_generate_presigned_s3_get_url(bucket_name, object_key, litellm_params) -> str`, `transform_file_content_request(file_content_request, optional_params, litellm_params) -> tuple[str, dict]`, `transform_file_content_response(raw_response, logging_obj, litellm_params) -> HttpxBinaryResponseContent`. Consistent across tasks and with the generic handler's `url, params = transform_file_content_request(...)` call.
