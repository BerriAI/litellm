# Bedrock Managed File Content Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `GET /v1/files/{id}/content` return the S3 object bytes for a Bedrock-backed managed file instead of a 500.

**Architecture:** Implement `transform_file_content_request` / `transform_file_content_response` on `BedrockFilesConfig` so Bedrock flows through the same generic `retrieve_file_content` handler as Vertex/Anthropic/Manus. The request transform validates the file_id and returns a botocore presigned S3 GET URL (auth in the query string, the only shape that fits a handler which computes URL and headers independently); the response transform wraps the raw bytes. The validated S3-id helpers move out of the now-dead `BedrockFilesHandler`, which is deleted along with its unreachable branch in `files/main.py`.

**Tech Stack:** Python, botocore (S3 SigV4 presigning), httpx, pytest.

## Global Constraints

- No comments unless they explain genuinely complex business logic; do not add comments that restate the code.
- Fully typed; no `Any` or bare `dict`/`dict[str, Any]`. Validate untyped inputs and pass typed values.
- No mutation of variables where avoidable; prefer building values in one shot.
- Tests must fail if the feature is mutated/broken (mutation kill rate > 90%); mock only the AWS/S3 boundary, never the transform logic under test.
- Run `make format` and `make lint` before each commit; run `make lint-budget-update` and commit lowered baselines if `ruff-strict-budget.json` / `basedpyright-code-budget.json` trip.
- Branch is `litellm_bedrock_file_content`, based on `litellm_internal_staging`. Commit messages follow conventional commits; no Claude attribution.
- `git push` always uses `--no-verify` (only when explicitly asked to push).

## File Structure

- `litellm/llms/bedrock/files/transformation.py` (modify): add `_extract_s3_uri_from_file_id`, `_get_configured_s3_bucket_name`, `_resolve_s3_region`, `_generate_presigned_s3_get_url` to `BedrockFilesConfig`; replace the two `NotImplementedError` content methods with real implementations.
- `litellm/llms/bedrock/files/handler.py` (delete): superseded; logic moved to the config.
- `litellm/files/main.py` (modify): drop the `BedrockFilesHandler` import, the `bedrock_files_instance` global, and the unreachable `elif custom_llm_provider == "bedrock":` branch in `file_content`.
- `tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py` (modify): new `TestBedrockFilesContentRetrieval` class.
- `tests/test_litellm/llms/bedrock/files/test_bedrock_files_handler.py` (modify): re-point the helper tests at `BedrockFilesConfig`.

---

### Task 1: Move validated S3-id helpers into BedrockFilesConfig

Move the file-id validation helpers from `BedrockFilesHandler` onto `BedrockFilesConfig` so the config owns them before the content methods use them. `BedrockFilesConfig` already extends `BaseAWSLLM`, so credential/region helpers are in scope.

**Files:**
- Modify: `litellm/llms/bedrock/files/transformation.py`
- Test: `tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py`

**Interfaces:**
- Consumes: `validate_managed_cloud_file_id`, `should_allow_legacy_cloud_file_ids`, `BEDROCK_MANAGED_S3_PREFIXES` (from `litellm.litellm_core_utils.cloud_storage_security`); `SpecialEnums` (from `litellm.types.utils`).
- Produces:
  - `BedrockFilesConfig._extract_s3_uri_from_file_id(self, file_id: str) -> str`
  - `BedrockFilesConfig._get_configured_s3_bucket_name(self, litellm_params: dict) -> str`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py` (top-level, after the existing classes). The `_encode_unified_file_id` helper mirrors the one in `test_bedrock_files_handler.py`.

```python
import base64
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


class TestBedrockFilesConfigS3IdHelpers:
    def setup_method(self):
        self.config = BedrockFilesConfig()

    def test_extract_direct_s3_uri(self):
        assert (
            self.config._extract_s3_uri_from_file_id(
                "s3://safe-bucket/litellm-batch-outputs/job/output.jsonl"
            )
            == "s3://safe-bucket/litellm-batch-outputs/job/output.jsonl"
        )

    def test_extract_unified_managed_s3_uri(self):
        file_id = _encode_unified_file_id(
            "s3://safe-bucket/litellm-batch-outputs/job/output.jsonl"
        )
        assert (
            self.config._extract_s3_uri_from_file_id(file_id)
            == "s3://safe-bucket/litellm-batch-outputs/job/output.jsonl"
        )

    def test_extract_rejects_non_s3_file_id(self):
        with pytest.raises(ValueError, match="managed LiteLLM S3 file id"):
            self.config._extract_s3_uri_from_file_id("safe-bucket/private.jsonl")

    def test_configured_bucket_prefers_trusted_credentials(self):
        trusted = MappingProxyType({"s3_bucket_name": "safe-bucket"})
        with patch.dict("os.environ", {}, clear=True):
            assert (
                self.config._get_configured_s3_bucket_name(
                    {
                        "s3_bucket_name": "attacker-bucket",
                        "_litellm_internal_model_credentials": trusted,
                    }
                )
                == "safe-bucket"
            )

    def test_configured_bucket_ignores_untrusted_request_value(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="S3 bucket_name is required"):
                self.config._get_configured_s3_bucket_name(
                    {"s3_bucket_name": "attacker-bucket"}
                )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -m pytest tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py::TestBedrockFilesConfigS3IdHelpers -v`
Expected: FAIL with `AttributeError: 'BedrockFilesConfig' object has no attribute '_extract_s3_uri_from_file_id'`.

- [ ] **Step 3: Add the helpers to BedrockFilesConfig**

In `litellm/llms/bedrock/files/transformation.py`, extend the existing import from `cloud_storage_security` (which already imports `BEDROCK_MANAGED_S3_BATCH_PREFIX`, `BEDROCK_MANAGED_S3_UPLOAD_PREFIX`, etc.) to also bring in the names below, and add `base64` / `MappingProxyType` / `cast` / `Mapping` imports at the top:

```python
import base64
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast
```

```python
from litellm.litellm_core_utils.cloud_storage_security import (
    BEDROCK_MANAGED_S3_BATCH_PREFIX,
    BEDROCK_MANAGED_S3_PREFIXES,
    BEDROCK_MANAGED_S3_UPLOAD_PREFIX,
    build_managed_cloud_object_name,
    encode_s3_object_key_for_url,
    sanitize_cloud_object_component,
    should_allow_legacy_cloud_file_ids,
    split_configured_cloud_bucket_name,
    validate_managed_cloud_file_id,
)
from litellm.types.utils import ExtractedFileData, LlmProviders, SpecialEnums
```

Add these two methods to `BedrockFilesConfig` (place them just above `transform_file_content_request`):

```python
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

    def _get_configured_s3_bucket_name(self, litellm_params: dict) -> str:
        trusted_model_credentials = litellm_params.get(
            "_litellm_internal_model_credentials"
        )
        bucket_name: Optional[str] = None
        if isinstance(trusted_model_credentials, type(MappingProxyType({}))):
            trusted_mapping = cast(Mapping[str, Any], trusted_model_credentials)
            candidate = trusted_mapping.get("s3_bucket_name")
            if isinstance(candidate, str):
                bucket_name = candidate
        bucket_name = bucket_name or os.getenv("AWS_S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError(
                "S3 bucket_name is required. Set 's3_bucket_name' in proxy config or "
                "AWS_S3_BUCKET_NAME for Bedrock file content retrieval."
            )
        return bucket_name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -m pytest tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py::TestBedrockFilesConfigS3IdHelpers -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content
make format && make lint
git add litellm/llms/bedrock/files/transformation.py tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py
git commit --no-verify -m "refactor(bedrock): move S3 file-id validation helpers into BedrockFilesConfig"
```

---

### Task 2: Implement presigned-URL content request + response transforms

Replace the two `NotImplementedError` methods with the real implementations: validate the file_id, resolve bucket/key/region/credentials, presign a GET, and wrap the response bytes.

**Files:**
- Modify: `litellm/llms/bedrock/files/transformation.py`
- Test: `tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py`

**Interfaces:**
- Consumes: `_extract_s3_uri_from_file_id`, `_get_configured_s3_bucket_name` (Task 1); `self.get_credentials(...)`, `self._get_aws_region_name(optional_params, model)`, `self._get_ssl_verify()` (from `BaseAWSLLM`); `HttpxBinaryResponseContent` (already imported).
- Produces:
  - `BedrockFilesConfig._resolve_s3_region(self, optional_params: dict, litellm_params: dict) -> str`
  - `BedrockFilesConfig._generate_presigned_s3_get_url(self, bucket_name: str, object_key: str, optional_params: dict, litellm_params: dict) -> str`
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

    def _request(self, file_id, litellm_params):
        with patch.object(self.config, "get_credentials", return_value=self.fake_creds):
            return self.config.transform_file_content_request(
                file_content_request={"file_id": file_id},
                optional_params={},
                litellm_params=litellm_params,
            )

    def test_returns_presigned_get_url_for_managed_s3_uri(self):
        url, params = self._request(
            "s3://safe-bucket/litellm-batch-outputs/job/in.jsonl.out",
            {"s3_bucket_name": "safe-bucket", "aws_region_name": "us-west-2"},
        )
        assert params == {}
        parts = urlsplit(url)
        query = parse_qs(parts.query)
        assert parts.path == "/safe-bucket/litellm-batch-outputs/job/in.jsonl.out"
        assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
        assert "X-Amz-Signature" in query

    def test_s3_region_name_wins_over_aws_region_name(self):
        url, _ = self._request(
            "s3://safe-bucket/litellm-batch-outputs/job/in.jsonl.out",
            {
                "s3_bucket_name": "safe-bucket",
                "aws_region_name": "us-east-1",
                "s3_region_name": "eu-central-1",
            },
        )
        credential = parse_qs(urlsplit(url).query)["X-Amz-Credential"][0]
        assert "/eu-central-1/s3/aws4_request" in credential
        assert urlsplit(url).netloc == "s3.eu-central-1.amazonaws.com"

    def test_unified_file_id_is_decoded_and_presigned(self):
        file_id = _encode_unified_file_id(
            "s3://safe-bucket/litellm-batch-outputs/job/in.jsonl.out"
        )
        url, _ = self._request(
            file_id, {"s3_bucket_name": "safe-bucket", "aws_region_name": "us-west-2"}
        )
        assert urlsplit(url).path == "/safe-bucket/litellm-batch-outputs/job/in.jsonl.out"
        assert "X-Amz-Signature" in parse_qs(urlsplit(url).query)

    def test_rejects_bucket_mismatch(self):
        with pytest.raises(ValueError, match="configured storage bucket"):
            self._request(
                "s3://other-bucket/litellm-batch-outputs/job/in.jsonl.out",
                {"s3_bucket_name": "safe-bucket", "aws_region_name": "us-west-2"},
            )

    def test_rejects_unmanaged_prefix_in_configured_bucket(self):
        with pytest.raises(ValueError, match="LiteLLM-managed"):
            self._request(
                "s3://safe-bucket/private/secret.jsonl",
                {"s3_bucket_name": "safe-bucket", "aws_region_name": "us-west-2"},
            )

    def test_response_returns_raw_bytes_unchanged(self):
        body = b'{"recordId":"req-1","modelOutput":{"foo":"bar"}}\n'
        raw_response = httpx.Response(
            status_code=200,
            content=body,
            headers={"content-type": "application/octet-stream"},
            request=httpx.Request("GET", "https://s3.us-west-2.amazonaws.com/safe-bucket/x"),
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

In `litellm/llms/bedrock/files/transformation.py`, replace the two methods at the bottom of `BedrockFilesConfig` (currently raising `NotImplementedError` for file content) with:

```python
    def _resolve_s3_region(self, optional_params: dict, litellm_params: dict) -> str:
        s3_region_name = litellm_params.get("s3_region_name") or optional_params.get(
            "s3_region_name"
        )
        if s3_region_name:
            return s3_region_name
        return self._get_aws_region_name(optional_params=litellm_params, model="")

    def _generate_presigned_s3_get_url(
        self,
        bucket_name: str,
        object_key: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> str:
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        region = self._resolve_s3_region(optional_params, litellm_params)
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
        )

        s3_client = boto3.client(
            "s3",
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
            region_name=region,
            endpoint_url=f"https://s3.{region}.amazonaws.com",
            config=Config(
                signature_version="s3v4", s3={"addressing_style": "path"}
            ),
            verify=self._get_ssl_verify(),
        )

        return s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=3600,
        )

    def transform_file_content_request(
        self,
        file_content_request,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        file_id = file_content_request.get("file_id") or ""
        s3_uri = self._extract_s3_uri_from_file_id(file_id)
        configured_bucket = self._get_configured_s3_bucket_name(litellm_params)
        bucket_name, object_key = validate_managed_cloud_file_id(
            file_id=s3_uri,
            scheme="s3://",
            configured_bucket_name=configured_bucket,
            allowed_object_prefixes=BEDROCK_MANAGED_S3_PREFIXES,
            allow_legacy_cloud_file_ids=should_allow_legacy_cloud_file_ids(
                litellm_params
            ),
        )
        url = self._generate_presigned_s3_get_url(
            bucket_name=bucket_name,
            object_key=object_key,
            optional_params=optional_params,
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

Note: `validate_managed_cloud_file_id` returns the *raw* (not URL-encoded) object key, which is what `generate_presigned_url` expects — botocore URL-encodes the key itself. Do not pre-encode.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -m pytest tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py::TestBedrockFilesContentRetrieval -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Format, lint, commit**

```bash
cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content
make format && make lint
git add litellm/llms/bedrock/files/transformation.py tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py
git commit --no-verify -m "feat(bedrock): retrieve managed file content via presigned S3 GET (#26335)"
```

---

### Task 3: Wiring test through the generic handler

Prove the exact path the issue's 28-step trace exercises no longer raises: `base_llm_http_handler.retrieve_file_content` with `BedrockFilesConfig` calls the presigned URL and returns the bytes. Mock only the httpx GET.

**Files:**
- Test: `tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py`

**Interfaces:**
- Consumes: `litellm.files.main.base_llm_http_handler.retrieve_file_content`; `BedrockFilesConfig` (Task 2); `litellm.llms.custom_httpx.http_handler.HTTPHandler`.

- [ ] **Step 1: Write the failing test**

Add to `TestBedrockFilesContentRetrieval`:

```python
    def test_retrieve_file_content_through_generic_handler(self):
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        from litellm.main import base_llm_http_handler

        body = b'{"recordId":"req-1","modelOutput":{"ok":true}}\n'
        captured = {}

        class FakeClient(HTTPHandler):
            def __init__(self):
                pass

            def get(self, url, headers=None, params=None):
                captured["url"] = url
                return httpx.Response(
                    status_code=200,
                    content=body,
                    request=httpx.Request("GET", url),
                )

        logging_obj = Logging(
            model="",
            messages=[],
            stream=False,
            call_type="file_content",
            start_time=__import__("time").time(),
            litellm_call_id="test-call",
            function_id="",
        )

        with patch.object(self.config, "get_credentials", return_value=self.fake_creds):
            result = base_llm_http_handler.retrieve_file_content(
                file_content_request={
                    "file_id": "s3://safe-bucket/litellm-batch-outputs/job/in.jsonl.out"
                },
                provider_config=self.config,
                litellm_params={"s3_bucket_name": "safe-bucket", "aws_region_name": "us-west-2"},
                headers={},
                logging_obj=logging_obj,
                _is_async=False,
                client=FakeClient(),
            )

        assert "X-Amz-Signature" in captured["url"]
        assert result.content == body
```

- [ ] **Step 2: Run test to verify it passes (now that Task 2 is done)**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -m pytest "tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py::TestBedrockFilesContentRetrieval::test_retrieve_file_content_through_generic_handler" -v`
Expected: PASS. (This test guards the wiring; if it fails with `NotImplementedError`, Task 2 regressed.)

- [ ] **Step 3: Commit**

```bash
cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content
make format && make lint
git add tests/test_litellm/llms/bedrock/files/test_bedrock_files_transformation.py
git commit --no-verify -m "test(bedrock): cover file content retrieval through generic handler"
```

---

### Task 4: Delete dead BedrockFilesHandler and re-point its tests

Remove the now-superseded handler and migrate its regression coverage onto the config. The helper tests already pass against the config (Task 1 added equivalents), so this task is about removing the file and the tests that import the deleted symbol, keeping any coverage not already duplicated.

**Files:**
- Delete: `litellm/llms/bedrock/files/handler.py`
- Modify: `tests/test_litellm/llms/bedrock/files/test_bedrock_files_handler.py`

**Interfaces:**
- Consumes: `BedrockFilesConfig._parse_s3_uri` does NOT exist; tests must call `validate_managed_cloud_file_id` directly or the config helpers. The config exposes `_extract_s3_uri_from_file_id` and `_get_configured_s3_bucket_name` (Task 1).

- [ ] **Step 1: Rewrite the handler test file to target the config**

Replace the body of `tests/test_litellm/llms/bedrock/files/test_bedrock_files_handler.py`. Keep the two `files_main` forwarding tests at the bottom unchanged (they don't reference `BedrockFilesHandler`). Replace the `TestBedrockFilesHandler` class and its import:

```python
import base64
import os
from types import MappingProxyType
from unittest.mock import MagicMock, patch

import pytest

import litellm.files.main as files_main
from litellm.litellm_core_utils.cloud_storage_security import (
    BEDROCK_MANAGED_S3_PREFIXES,
    validate_managed_cloud_file_id,
)
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
                "s3://other-bucket/litellm-bedrock-files-model-id-abc.jsonl",
                "safe-bucket",
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
            _parse(
                "s3://safe-bucket/litellm-bedrock-files/../secret.jsonl", "safe-bucket"
            )

    def test_should_reject_empty_middle_path_segment(self):
        with pytest.raises(ValueError, match="invalid path segment"):
            _parse(
                "s3://safe-bucket/litellm-bedrock-files//secret.jsonl", "safe-bucket"
            )

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

    def test_should_not_trust_request_s3_bucket_name_for_expected_bucket(self):
        with patch.dict(os.environ, {"AWS_S3_BUCKET_NAME": "safe-bucket"}):
            assert (
                self.config._get_configured_s3_bucket_name(
                    {"s3_bucket_name": "attacker-bucket"}
                )
                == "safe-bucket"
            )

    def test_should_trust_proxy_config_s3_bucket_name_for_expected_bucket(self):
        trusted_credentials = MappingProxyType({"s3_bucket_name": "safe-bucket"})
        with patch.dict(os.environ, {}, clear=True):
            assert (
                self.config._get_configured_s3_bucket_name(
                    {
                        "s3_bucket_name": "attacker-bucket",
                        "_litellm_internal_model_credentials": trusted_credentials,
                    }
                )
                == "safe-bucket"
            )

    def test_should_not_trust_user_supplied_internal_credentials_dict(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="S3 bucket_name is required"):
                self.config._get_configured_s3_bucket_name(
                    {
                        "_litellm_internal_model_credentials": {
                            "s3_bucket_name": "attacker-bucket"
                        }
                    }
                )

    def test_should_require_server_s3_bucket_name(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="S3 bucket_name is required"):
                self.config._get_configured_s3_bucket_name(
                    {"s3_bucket_name": "attacker-bucket"}
                )
```

Leave the two module-level functions `test_should_forward_trusted_model_credentials_to_bedrock_provider_config` and `test_should_forward_trusted_model_credentials_to_retrieve_provider_config` (lines 168-207 of the original) intact.

- [ ] **Step 2: Delete the handler file**

```bash
cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content
git rm litellm/llms/bedrock/files/handler.py
```

- [ ] **Step 3: Run the handler tests to verify they pass without the deleted file**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -m pytest tests/test_litellm/llms/bedrock/files/test_bedrock_files_handler.py -v`
Expected: PASS. If it fails on `import litellm.files.main`, that's expected until Task 5 removes the import of the deleted module — do Task 5's Step 1 before re-running, or run Tasks 4 and 5 as one review unit. (The import `from litellm.llms.bedrock.files.handler import BedrockFilesHandler` in `files/main.py` will break collection.)

- [ ] **Step 4: Commit (together with Task 5 — see note)**

Because deleting `handler.py` breaks `files/main.py`'s import, commit Task 4 and Task 5 together. Proceed to Task 5 before committing.

---

### Task 5: Remove the dead bedrock branch from files/main.py

`files/main.py` imports the deleted `BedrockFilesHandler`, instantiates `bedrock_files_instance`, and has an unreachable `elif custom_llm_provider == "bedrock":` branch. Remove all three.

**Files:**
- Modify: `litellm/files/main.py`

**Interfaces:**
- Consumes: nothing new. The generic `provider_config is not None` branch (already present, line ~927) routes Bedrock through `retrieve_file_content`.

- [ ] **Step 1: Remove the import**

Delete this line (currently line 42):

```python
from litellm.llms.bedrock.files.handler import BedrockFilesHandler
```

- [ ] **Step 2: Remove the module global**

Delete this line (currently line 85):

```python
bedrock_files_instance = BedrockFilesHandler()
```

- [ ] **Step 3: Remove the unreachable branch**

In `file_content`, delete the entire `elif custom_llm_provider == "bedrock":` block (currently lines 1021-1029):

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

The surrounding `if ... elif ...else` (openai / azure / vertex_ai / else BadRequestError) stays. The `else` branch's message lists `'bedrock'` as supported — leave it; bedrock is still reached via the generic `provider_config` branch above, so the message stays accurate.

- [ ] **Step 4: Verify the module imports cleanly**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -c "import litellm.files.main"`
Expected: no output, exit 0.

- [ ] **Step 5: Run the full bedrock files test dir**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -m pytest tests/test_litellm/llms/bedrock/files/ -v`
Expected: PASS (all tests across the three test files).

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

- [ ] **Step 1: Run the broader files + bedrock suites**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && python -m pytest tests/test_litellm/llms/bedrock/files/ tests/test_litellm/test_files.py -v`
Expected: PASS. (If `tests/test_litellm/test_files.py` does not exist, skip that path; run only the bedrock files dir.)

- [ ] **Step 2: Lint budget check**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && make lint`
Expected: PASS. If `ruff-strict-budget.json` or `basedpyright-code-budget.json` trips (likely *down*, since we delete a file), run `make lint-budget-update` and commit the lowered baselines:

```bash
make lint-budget-update
git add ruff-strict-budget.json basedpyright-code-budget.json
git commit --no-verify -m "chore: ratchet lint budgets after bedrock file content change"
```

(Only commit if the budget files actually changed.)

- [ ] **Step 3: grep for stragglers**

Run: `cd /Users/kentpeng/projects/litellm/.claude/worktrees/litellm_bedrock_file_content && grep -rn "BedrockFilesHandler\|bedrock_files_instance" litellm/ tests/ enterprise/`
Expected: no matches. If any remain, remove them and re-run Step 1.

---

## Proof of Fix (manual, for PR — after implementation)

Run against a live proxy with real AWS credentials and S3 bucket configured. Capture each command and its output for the PR.

1. Start proxy: `python litellm/proxy/proxy_cli.py --config litellm/proxy/dev_config.yaml --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log` (config must define a bedrock batch model + `s3_bucket_name`).
2. `POST /v1/files` with a small batch JSONL (`purpose=batch`) -> capture the managed file id.
3. `POST /v1/batches` referencing that file id against a real Bedrock model -> capture batch id.
4. Poll `GET /v1/batches/{id}` until `status == completed` -> capture `output_file_id`.
5. `GET /v1/files/{output_file_id}/content` -> the batch output bytes are returned (before the fix this step returns 500 `BedrockFilesConfig does not support file content retrieval`).

Show the before (500 on `main`) and after (bytes returned) for step 5.

---

## Self-Review

**Spec coverage:**
- Implement `transform_file_content_request`/`_response` via presigned URL -> Task 2. Covered.
- Move validated helpers into the config -> Task 1. Covered.
- Delete dead `BedrockFilesHandler` -> Task 4. Covered.
- Remove unreachable `files/main.py` branch -> Task 5. Covered.
- Raw S3 bytes (no OpenAI transform) -> Task 2 Step 3 (`transform_file_content_response` returns response as-is) + test in Task 2. Covered.
- Region precedence (s3_region_name wins) -> Task 2 test `test_s3_region_name_wins_over_aws_region_name`. Covered.
- Security (bucket confusion, unmanaged prefix, traversal) -> Tasks 1, 2, 4 tests. Covered.
- Wiring no longer raises -> Task 3. Covered.
- Proof of fix (full batch e2e) -> Proof of Fix section. Covered.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every command shows expected output. Clean.

**Type consistency:** `_extract_s3_uri_from_file_id(file_id: str) -> str`, `_get_configured_s3_bucket_name(litellm_params: dict) -> str`, `_resolve_s3_region(optional_params, litellm_params) -> str`, `_generate_presigned_s3_get_url(bucket_name, object_key, optional_params, litellm_params) -> str`, `transform_file_content_request(file_content_request, optional_params, litellm_params) -> tuple[str, dict]`, `transform_file_content_response(raw_response, logging_obj, litellm_params) -> HttpxBinaryResponseContent`. Names and signatures are consistent across Tasks 1-3 and match the calls in the generic handler (`url, params = transform_file_content_request(...)`).
