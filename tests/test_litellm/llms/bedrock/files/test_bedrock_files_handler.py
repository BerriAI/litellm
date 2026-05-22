import base64
import os
from types import MappingProxyType
from unittest.mock import MagicMock, patch

import pytest

import litellm.files.main as files_main
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

    def test_should_allow_legacy_same_bucket_key_when_server_flag_enabled(self):
        bucket, key = self.handler._parse_s3_uri(
            s3_uri="s3://safe-bucket/private/output.jsonl",
            configured_bucket_name="safe-bucket",
            allow_legacy_cloud_file_ids=True,
        )

        assert bucket == "safe-bucket"
        assert key == "private/output.jsonl"

    def test_should_keep_configured_prefix_for_legacy_keys(self):
        bucket, key = self.handler._parse_s3_uri(
            s3_uri="s3://safe-bucket/team-a/private/output.jsonl",
            configured_bucket_name="safe-bucket/team-a",
            allow_legacy_cloud_file_ids=True,
        )

        assert bucket == "safe-bucket"
        assert key == "team-a/private/output.jsonl"

    def test_should_reject_legacy_key_outside_configured_prefix(self):
        with pytest.raises(ValueError, match="configured storage prefix"):
            self.handler._parse_s3_uri(
                s3_uri="s3://safe-bucket/team-b/private/output.jsonl",
                configured_bucket_name="safe-bucket/team-a",
                allow_legacy_cloud_file_ids=True,
            )

    def test_should_reject_dot_segment_key(self):
        with pytest.raises(ValueError, match="invalid path segment"):
            self.handler._parse_s3_uri(
                s3_uri="s3://safe-bucket/litellm-bedrock-files/../secret.jsonl",
                configured_bucket_name="safe-bucket",
            )

    def test_should_reject_empty_middle_path_segment(self):
        with pytest.raises(ValueError, match="invalid path segment"):
            self.handler._parse_s3_uri(
                s3_uri="s3://safe-bucket/litellm-bedrock-files//secret.jsonl",
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

    def test_should_not_trust_request_s3_bucket_name_for_expected_bucket(self):
        with patch.dict(os.environ, {"AWS_S3_BUCKET_NAME": "safe-bucket"}):
            assert (
                self.handler._get_configured_s3_bucket_name(
                    {"s3_bucket_name": "attacker-bucket"}
                )
                == "safe-bucket"
            )

    def test_should_trust_proxy_config_s3_bucket_name_for_expected_bucket(self):
        trusted_credentials = MappingProxyType({"s3_bucket_name": "safe-bucket"})

        with patch.dict(os.environ, {}, clear=True):
            assert (
                self.handler._get_configured_s3_bucket_name(
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
                self.handler._get_configured_s3_bucket_name(
                    {
                        "_litellm_internal_model_credentials": {
                            "s3_bucket_name": "attacker-bucket"
                        }
                    }
                )

    def test_should_require_server_s3_bucket_name(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="S3 bucket_name is required"):
                self.handler._get_configured_s3_bucket_name(
                    {"s3_bucket_name": "attacker-bucket"}
                )


def test_should_forward_trusted_model_credentials_to_bedrock_provider_config():
    trusted_credentials = MappingProxyType({"s3_bucket_name": "safe-bucket"})
    mock_response = MagicMock()

    with patch.object(
        files_main.base_llm_http_handler,
        "retrieve_file_content",
        return_value=mock_response,
    ) as mock_retrieve_file_content:
        response = files_main.file_content(
            file_id="s3://safe-bucket/litellm-bedrock-files/file.jsonl",
            custom_llm_provider="bedrock",
            _litellm_internal_model_credentials=trusted_credentials,
        )

    assert response is mock_response
    litellm_params = mock_retrieve_file_content.call_args.kwargs["litellm_params"]
    assert litellm_params["_litellm_internal_model_credentials"] is trusted_credentials
    assert "s3_bucket_name" not in litellm_params


def test_should_forward_trusted_model_credentials_to_retrieve_provider_config():
    trusted_credentials = MappingProxyType({"allow_legacy_cloud_file_ids": True})
    mock_response = MagicMock()

    with patch.object(
        files_main.base_llm_http_handler,
        "retrieve_file",
        return_value=mock_response,
    ) as mock_retrieve_file:
        response = files_main.file_retrieve(
            file_id="gs://safe-bucket/private/file.jsonl",
            custom_llm_provider="vertex_ai",
            _litellm_internal_model_credentials=trusted_credentials,
        )

    assert response is mock_response
    litellm_params = mock_retrieve_file.call_args.kwargs["litellm_params"]
    assert litellm_params["_litellm_internal_model_credentials"] is trusted_credentials
