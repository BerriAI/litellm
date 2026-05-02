"""
Tests for VertexAIFilesConfig transformation methods (Issues 5-7).
"""

import urllib.parse
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from unittest.mock import MagicMock

from litellm.llms.vertex_ai.files.transformation import VertexAIFilesConfig
from litellm.types.llms.openai import OpenAIFileObject, HttpxBinaryResponseContent
from openai.types.file_deleted import FileDeleted


@pytest.fixture
def config():
    return VertexAIFilesConfig()


class TestParseGcsUri:
    """Tests for the _parse_gcs_uri helper used by retrieve / content / delete."""

    def test_should_parse_standard_gs_uri(self, config):
        file_id = "gs://my-bucket/litellm-vertex-files/path/to/object.jsonl"
        bucket, encoded = config._parse_gcs_uri(
            file_id, litellm_params={"bucket_name": "my-bucket"}
        )
        assert bucket == "my-bucket"
        assert encoded == urllib.parse.quote(
            "litellm-vertex-files/path/to/object.jsonl", safe=""
        )

    def test_should_parse_uri_with_nested_publisher_path(self, config):
        uri = "gs://litellm-local/litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc-123"
        bucket, encoded = config._parse_gcs_uri(
            uri, litellm_params={"bucket_name": "litellm-local"}
        )
        assert bucket == "litellm-local"
        expected_path = (
            "litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc-123"
        )
        assert encoded == urllib.parse.quote(expected_path, safe="")

    def test_should_handle_url_encoded_input(self, config):
        encoded_uri = urllib.parse.quote(
            "gs://my-bucket/litellm-vertex-files/some/path", safe=""
        )
        bucket, encoded = config._parse_gcs_uri(
            encoded_uri, litellm_params={"bucket_name": "my-bucket"}
        )
        assert bucket == "my-bucket"
        assert encoded == urllib.parse.quote("litellm-vertex-files/some/path", safe="")

    def test_should_reject_bucket_only(self, config):
        with pytest.raises(ValueError, match="object name"):
            config._parse_gcs_uri(
                "gs://my-bucket", litellm_params={"bucket_name": "my-bucket"}
            )

    def test_should_reject_no_gs_prefix(self, config):
        with pytest.raises(ValueError, match="gs://"):
            config._parse_gcs_uri(
                "my-bucket/litellm-vertex-files/object.txt",
                litellm_params={"bucket_name": "my-bucket"},
            )

    def test_should_reject_unmanaged_object_path(self, config):
        with pytest.raises(ValueError, match="LiteLLM-managed"):
            config._parse_gcs_uri(
                "gs://my-bucket/private/object.txt",
                litellm_params={"bucket_name": "my-bucket"},
            )

    def test_should_allow_legacy_object_path_when_server_flag_enabled(self, config):
        bucket, encoded = config._parse_gcs_uri(
            "gs://my-bucket/private/object.txt",
            litellm_params={
                "bucket_name": "my-bucket",
                "allow_legacy_cloud_file_ids": True,
            },
        )

        assert bucket == "my-bucket"
        assert encoded == urllib.parse.quote("private/object.txt", safe="")

    def test_should_keep_configured_prefix_for_legacy_object_path(self, config):
        bucket, encoded = config._parse_gcs_uri(
            "gs://my-bucket/team-a/private/object.txt",
            litellm_params={
                "bucket_name": "my-bucket/team-a",
                "allow_legacy_cloud_file_ids": True,
            },
        )

        assert bucket == "my-bucket"
        assert encoded == urllib.parse.quote("team-a/private/object.txt", safe="")

    def test_should_reject_legacy_object_outside_configured_prefix(self, config):
        with pytest.raises(ValueError, match="configured storage prefix"):
            config._parse_gcs_uri(
                "gs://my-bucket/team-b/private/object.txt",
                litellm_params={
                    "bucket_name": "my-bucket/team-a",
                    "allow_legacy_cloud_file_ids": True,
                },
            )

    def test_should_reject_unconfigured_bucket(self, config):
        with pytest.raises(ValueError, match="configured storage bucket"):
            config._parse_gcs_uri(
                "gs://other-bucket/litellm-vertex-files/object.txt",
                litellm_params={"bucket_name": "my-bucket"},
            )


class TestCreateFileUrl:
    def test_should_ignore_request_metadata_bucket_and_sanitize_filename(self, config):
        url = config.get_complete_file_url(
            api_base=None,
            api_key=None,
            model="",
            optional_params={},
            litellm_params={
                "bucket_name": "safe-bucket",
                "litellm_metadata": {"gcs_bucket_name": "attacker-bucket"},
            },
            data={
                "file": ("../../owned.jsonl?alt=media", b"{}", "application/jsonl"),
                "purpose": "assistants",
            },
        )

        parsed_url = urlparse(url)
        object_name = parse_qs(parsed_url.query)["name"][0]
        assert "/b/safe-bucket/" in parsed_url.path
        assert "attacker-bucket" not in url
        assert object_name.startswith("litellm-vertex-files/uploads/")
        assert object_name.endswith("-owned.jsonl_alt_media")
        assert ".." not in object_name
        assert "?" not in object_name


class TestTransformRetrieveFile:

    def test_should_build_correct_gcs_metadata_url(self, config):
        file_id = "gs://my-bucket/litellm-vertex-files/path/to/file.jsonl"
        url, params = config.transform_retrieve_file_request(
            file_id=file_id,
            optional_params={},
            litellm_params={"bucket_name": "my-bucket"},
        )
        expected_encoded = urllib.parse.quote(
            "litellm-vertex-files/path/to/file.jsonl", safe=""
        )
        assert (
            url
            == f"https://storage.googleapis.com/storage/v1/b/my-bucket/o/{expected_encoded}"
        )
        assert params == {}

    def test_should_return_openai_file_object_from_gcs_response(self, config):
        gcs_json = {
            "id": "my-bucket/path/to/file.jsonl/123456",
            "name": "path/to/file.jsonl",
            "size": "4096",
            "timeCreated": "2025-02-15T10:00:00.000Z",
            "metadata": {"purpose": "batch"},
        }
        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = gcs_json

        result = config.transform_retrieve_file_response(
            raw_response=raw_response,
            logging_obj=MagicMock(),
            litellm_params={},
        )

        assert isinstance(result, OpenAIFileObject)
        assert result.id == "gs://my-bucket/path/to/file.jsonl"
        assert result.filename == "path/to/file.jsonl"
        assert result.bytes == 4096
        assert result.object == "file"
        assert result.status == "processed"
        assert result.purpose == "batch"

    def test_should_default_purpose_to_batch_when_metadata_missing(self, config):
        gcs_json = {
            "id": "bucket/obj/999",
            "name": "obj",
            "size": "0",
            "timeCreated": "2025-01-01T00:00:00.000Z",
        }
        raw_response = MagicMock(spec=httpx.Response)
        raw_response.json.return_value = gcs_json

        result = config.transform_retrieve_file_response(
            raw_response=raw_response,
            logging_obj=MagicMock(),
            litellm_params={},
        )
        assert result.purpose == "batch"


class TestTransformFileContent:

    def test_should_build_gcs_media_download_url(self, config):
        file_id = "gs://my-bucket/litellm-vertex-files/path/to/file.jsonl"
        url, params = config.transform_file_content_request(
            file_content_request={"file_id": file_id},
            optional_params={},
            litellm_params={"bucket_name": "my-bucket"},
        )
        encoded = urllib.parse.quote("litellm-vertex-files/path/to/file.jsonl", safe="")
        assert (
            url
            == f"https://storage.googleapis.com/storage/v1/b/my-bucket/o/{encoded}?alt=media"
        )
        assert params == {}

    def test_should_return_binary_response_content(self, config):
        raw_response = httpx.Response(
            status_code=200,
            content=b'{"line": 1}\n{"line": 2}\n',
            headers={"content-type": "application/octet-stream"},
            request=httpx.Request("GET", "https://example.com"),
        )

        result = config.transform_file_content_response(
            raw_response=raw_response,
            logging_obj=MagicMock(),
            litellm_params={},
        )

        assert isinstance(result, HttpxBinaryResponseContent)
        assert result.response.content == b'{"line": 1}\n{"line": 2}\n'


class TestTransformDeleteFile:
    def test_should_build_correct_gcs_delete_url(self, config):
        file_id = "gs://my-bucket/litellm-vertex-files/path/to/file.jsonl"
        url, params = config.transform_delete_file_request(
            file_id=file_id,
            optional_params={},
            litellm_params={"bucket_name": "my-bucket"},
        )
        encoded = urllib.parse.quote("litellm-vertex-files/path/to/file.jsonl", safe="")
        assert (
            url == f"https://storage.googleapis.com/storage/v1/b/my-bucket/o/{encoded}"
        )
        assert params == {}

    def test_should_return_file_deleted_with_reconstructed_id(self, config):
        raw_response = MagicMock(spec=httpx.Response)
        mock_request = MagicMock()
        encoded_name = urllib.parse.quote(
            "litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc",
            safe="",
        )
        mock_request.url = (
            f"https://storage.googleapis.com/storage/v1/b/my-bucket/o/{encoded_name}"
        )
        raw_response.request = mock_request

        result = config.transform_delete_file_response(
            raw_response=raw_response,
            logging_obj=MagicMock(),
            litellm_params={},
        )

        assert isinstance(result, FileDeleted)
        assert result.deleted is True
        assert result.object == "file"
        assert (
            result.id
            == "gs://my-bucket/litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc"
        )

    def test_should_fallback_to_deleted_id_when_no_request(self, config):
        raw_response = MagicMock(spec=httpx.Response)
        raw_response.request = None

        result = config.transform_delete_file_response(
            raw_response=raw_response,
            logging_obj=MagicMock(),
            litellm_params={},
        )

        assert isinstance(result, FileDeleted)
        assert result.id == "deleted"
        assert result.deleted is True

    def test_should_include_bucket_name_in_reconstructed_delete_id(self, config):
        """
        Regression: the old code split on /o/ only, dropping the bucket from
        the reconstructed gs:// URI. e.g. gs://path/to/file instead of
        gs://my-bucket/path/to/file.
        """
        raw_response = MagicMock(spec=httpx.Response)
        mock_request = MagicMock()
        encoded_object = urllib.parse.quote("path/to/file.jsonl", safe="")
        mock_request.url = (
            f"https://storage.googleapis.com/storage/v1/b/my-bucket/o/{encoded_object}"
        )
        raw_response.request = mock_request

        result = config.transform_delete_file_response(
            raw_response=raw_response,
            logging_obj=MagicMock(),
            litellm_params={},
        )

        assert result.id == "gs://my-bucket/path/to/file.jsonl"

    def test_should_include_bucket_in_nested_object_path(self, config):
        """Verify bucket extraction works with deeply nested GCS object paths."""
        raw_response = MagicMock(spec=httpx.Response)
        mock_request = MagicMock()
        encoded_object = urllib.parse.quote(
            "litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc-123",
            safe="",
        )
        mock_request.url = f"https://storage.googleapis.com/storage/v1/b/prod-bucket/o/{encoded_object}"
        raw_response.request = mock_request

        result = config.transform_delete_file_response(
            raw_response=raw_response,
            logging_obj=MagicMock(),
            litellm_params={},
        )

        assert result.id == (
            "gs://prod-bucket/litellm-vertex-files/publishers/google/"
            "models/gemini-2.0-flash-001/abc-123"
        )
