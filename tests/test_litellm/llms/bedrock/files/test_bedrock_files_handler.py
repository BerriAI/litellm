import base64

import pytest

from litellm.llms.bedrock.files.handler import BedrockFilesHandler
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


class TestBedrockFilesHandler:
    def setup_method(self):
        self.handler = BedrockFilesHandler()

    def test_should_parse_direct_managed_s3_uri(self):
        bucket, key = self.handler._parse_s3_uri(
            s3_uri="s3://safe-bucket/litellm-bedrock-files-model-id-abc.jsonl",
            configured_bucket_name="safe-bucket",
        )

        assert bucket == "safe-bucket"
        assert key == "litellm-bedrock-files-model-id-abc.jsonl"

    def test_should_parse_managed_batch_output_uri(self):
        bucket, key = self.handler._parse_s3_uri(
            s3_uri="s3://safe-bucket/litellm-batch-outputs/job/",
            configured_bucket_name="safe-bucket",
        )

        assert bucket == "safe-bucket"
        assert key == "litellm-batch-outputs/job/"

    def test_should_reject_arbitrary_bucket(self):
        with pytest.raises(ValueError, match="configured storage bucket"):
            self.handler._parse_s3_uri(
                s3_uri="s3://other-bucket/litellm-bedrock-files-model-id-abc.jsonl",
                configured_bucket_name="safe-bucket",
            )

    def test_should_reject_unmanaged_same_bucket_key(self):
        with pytest.raises(ValueError, match="LiteLLM-managed"):
            self.handler._parse_s3_uri(
                s3_uri="s3://safe-bucket/private/output.jsonl",
                configured_bucket_name="safe-bucket",
            )

    def test_should_reject_dot_segment_key(self):
        with pytest.raises(ValueError, match="invalid path segment"):
            self.handler._parse_s3_uri(
                s3_uri="s3://safe-bucket/litellm-bedrock-files/../secret.jsonl",
                configured_bucket_name="safe-bucket",
            )

    def test_should_extract_unified_managed_s3_uri(self):
        file_id = _encode_unified_file_id(
            "s3://safe-bucket/litellm-batch-outputs/job/output.jsonl"
        )

        assert (
            self.handler._extract_s3_uri_from_file_id(file_id)
            == "s3://safe-bucket/litellm-batch-outputs/job/output.jsonl"
        )

    def test_should_reject_file_id_without_s3_scheme(self):
        with pytest.raises(ValueError, match="managed LiteLLM S3 file id"):
            self.handler._extract_s3_uri_from_file_id("safe-bucket/private.jsonl")

    def test_should_reject_unified_unmanaged_s3_uri(self):
        file_id = _encode_unified_file_id("s3://safe-bucket/private/output.jsonl")
        s3_uri = self.handler._extract_s3_uri_from_file_id(file_id)

        with pytest.raises(ValueError, match="LiteLLM-managed"):
            self.handler._parse_s3_uri(
                s3_uri=s3_uri,
                configured_bucket_name="safe-bucket",
            )
