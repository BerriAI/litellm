import re
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from litellm.proxy.common_utils.load_config_utils import (
    get_config_from_bucket,
    get_file_contents_from_s3,
    resolve_bucket_includes,
)


class TestGetFileContentsFromS3:
    """Test suite for S3 config loading functionality."""

    @patch("boto3.client")
    @patch("litellm.main.bedrock_converse_chat_completion")
    @patch("yaml.safe_load")
    def test_get_file_contents_from_s3_no_temp_file_creation(
        self, mock_yaml_load, mock_bedrock, mock_boto3_client
    ):
        """
        Test that get_file_contents_from_s3 doesn't create temporary files
        and uses yaml.safe_load directly on the S3 response content.

        Note: It's critical that yaml.safe_load is used

        Relevant issue/PR: https://github.com/BerriAI/litellm/pull/12078
        """
        # Mock credentials
        mock_credentials = MagicMock()
        mock_credentials.access_key = "test_access_key"
        mock_credentials.secret_key = "test_secret_key"
        mock_credentials.token = "test_token"
        mock_bedrock.get_credentials.return_value = mock_credentials

        # Mock S3 client and response
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        # Mock S3 response with YAML content
        yaml_content = """
        model_list:
          - model_name: gpt-3.5-turbo
            litellm_params:
              model: gpt-3.5-turbo
        """
        mock_response_body = MagicMock()
        mock_response_body.read.return_value = yaml_content.encode("utf-8")
        mock_s3_response = {"Body": mock_response_body}
        mock_s3_client.get_object.return_value = mock_s3_response

        # Mock yaml.safe_load to return parsed config
        expected_config = {
            "model_list": [
                {
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {"model": "gpt-3.5-turbo"},
                }
            ]
        }
        mock_yaml_load.return_value = expected_config

        # Call the function
        bucket_name = "test-bucket"
        object_key = "config.yaml"
        result = get_file_contents_from_s3(bucket_name, object_key)

        # Assertions
        assert result == expected_config

        # Verify S3 client was created with correct credentials
        mock_boto3_client.assert_called_once_with(
            "s3",
            aws_access_key_id="test_access_key",
            aws_secret_access_key="test_secret_key",
            aws_session_token="test_token",
        )

        # Verify S3 get_object was called with correct parameters
        mock_s3_client.get_object.assert_called_once_with(
            Bucket=bucket_name, Key=object_key
        )

        # Verify the response body was read and decoded
        mock_response_body.read.assert_called_once()

        # Verify yaml.safe_load was called with the decoded content
        mock_yaml_load.assert_called_once_with(yaml_content)


class TestBucketConfigIncludes:
    """`include:` directives in a bucket-hosted config.yaml (LIT-6982).

    They used to be dropped silently: the proxy booted with the root config applied and everything
    the included objects declared missing, with nothing logged.
    """

    @staticmethod
    def _bucket(objects):
        async def fetch(object_key):
            return objects.get(object_key)

        return fetch

    @pytest.mark.asyncio
    async def test_include_resolves_against_the_config_objects_prefix(self):
        merged = await resolve_bucket_includes(
            config={"include": ["model_config.yaml"], "general_settings": {"master_key": "sk-1234"}},
            object_key="configs/prod/config.yaml",
            fetch=self._bucket(
                {"configs/prod/model_config.yaml": {"model_list": [{"model_name": "gpt-4o-mini"}]}}
            ),
        )

        assert merged == {
            "general_settings": {"master_key": "sk-1234"},
            "model_list": [{"model_name": "gpt-4o-mini"}],
        }

    @pytest.mark.asyncio
    async def test_include_with_a_leading_slash_reads_from_the_bucket_root(self):
        merged = await resolve_bucket_includes(
            config={"include": ["/shared/models.yaml"]},
            object_key="configs/prod/config.yaml",
            fetch=self._bucket({"shared/models.yaml": {"model_list": [{"model_name": "shared"}]}}),
        )

        assert merged == {"model_list": [{"model_name": "shared"}]}

    @pytest.mark.asyncio
    async def test_include_walks_out_of_the_prefix_with_dot_dot(self):
        merged = await resolve_bucket_includes(
            config={"include": ["../shared/models.yaml"]},
            object_key="configs/prod/config.yaml",
            fetch=self._bucket({"configs/shared/models.yaml": {"model_list": [{"model_name": "shared"}]}}),
        )

        assert merged == {"model_list": [{"model_name": "shared"}]}

    @pytest.mark.asyncio
    async def test_included_configs_may_declare_further_includes(self):
        merged = await resolve_bucket_includes(
            config={"include": ["models.yaml"]},
            object_key="configs/config.yaml",
            fetch=self._bucket(
                {
                    "configs/models.yaml": {
                        "include": ["extra/more_models.yaml"],
                        "model_list": [{"model_name": "first"}],
                    },
                    "configs/extra/more_models.yaml": {"model_list": [{"model_name": "second"}]},
                }
            ),
        )

        assert merged == {"model_list": [{"model_name": "first"}, {"model_name": "second"}]}

    @pytest.mark.asyncio
    async def test_list_values_are_extended_and_other_values_are_overridden(self):
        merged = await resolve_bucket_includes(
            config={
                "include": ["models.yaml"],
                "model_list": [{"model_name": "from-root"}],
                "litellm_settings": {"drop_params": True},
            },
            object_key="config.yaml",
            fetch=self._bucket(
                {
                    "models.yaml": {
                        "model_list": [{"model_name": "from-include"}],
                        "litellm_settings": {"num_retries": 3},
                    }
                }
            ),
        )

        assert merged == {
            "model_list": [{"model_name": "from-root"}, {"model_name": "from-include"}],
            "litellm_settings": {"num_retries": 3},
        }

    @pytest.mark.asyncio
    async def test_a_missing_included_object_fails_loudly_with_its_key(self):
        with pytest.raises(FileNotFoundError, match=re.escape("configs/prod/model_config.yaml")):
            await resolve_bucket_includes(
                config={"include": ["model_config.yaml"]},
                object_key="configs/prod/config.yaml",
                fetch=self._bucket({}),
            )

    @pytest.mark.asyncio
    async def test_a_non_list_include_fails_loudly(self):
        with pytest.raises(ValueError, match="'include' must be a list of file paths"):
            await resolve_bucket_includes(
                config={"include": "model_config.yaml"},
                object_key="config.yaml",
                fetch=self._bucket({}),
            )

    @pytest.mark.asyncio
    async def test_get_config_from_bucket_merges_includes_over_s3(self, monkeypatch):
        objects = {
            "lit6982/config.yaml": {
                "include": ["model_config.yaml"],
                "general_settings": {"master_key": "sk-1234"},
            },
            "lit6982/model_config.yaml": {"model_list": [{"model_name": "included-model"}]},
        }
        monkeypatch.setattr(
            "litellm.proxy.common_utils.load_config_utils.get_file_contents_from_s3",
            lambda bucket_name, object_key: objects.get(object_key),
        )

        config = await get_config_from_bucket(
            bucket_type="s3", bucket_name="litellm-configs", object_key="lit6982/config.yaml"
        )

        assert config == {
            "general_settings": {"master_key": "sk-1234"},
            "model_list": [{"model_name": "included-model"}],
        }

    @pytest.mark.asyncio
    async def test_get_config_from_bucket_merges_includes_over_gcs(self, monkeypatch):
        objects = {
            "lit6982/config.yaml": {
                "include": ["model_config.yaml"],
                "general_settings": {"master_key": "sk-1234"},
            },
            "lit6982/model_config.yaml": {"model_list": [{"model_name": "included-model"}]},
        }

        async def fake_gcs(bucket_name, object_key):
            return objects.get(object_key)

        monkeypatch.setattr(
            "litellm.proxy.common_utils.load_config_utils.get_config_file_contents_from_gcs", fake_gcs
        )

        config = await get_config_from_bucket(
            bucket_type="gcs", bucket_name="litellm-configs", object_key="lit6982/config.yaml"
        )

        assert config == {
            "general_settings": {"master_key": "sk-1234"},
            "model_list": [{"model_name": "included-model"}],
        }

    @pytest.mark.asyncio
    async def test_get_config_from_bucket_returns_none_when_the_root_object_is_missing(self, monkeypatch):
        monkeypatch.setattr(
            "litellm.proxy.common_utils.load_config_utils.get_file_contents_from_s3",
            lambda bucket_name, object_key: None,
        )

        assert (
            await get_config_from_bucket(
                bucket_type="s3", bucket_name="litellm-configs", object_key="missing.yaml"
            )
            is None
        )
