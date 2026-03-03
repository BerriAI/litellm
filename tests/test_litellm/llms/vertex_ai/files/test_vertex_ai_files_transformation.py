"""
Tests for VertexAIFilesConfig transformation methods (Issues 5-7).
"""

import json
import urllib.parse

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
        bucket, encoded = config._parse_gcs_uri(
            "gs://my-bucket/path/to/object.jsonl"
        )
        assert bucket == "my-bucket"
        assert encoded == urllib.parse.quote("path/to/object.jsonl", safe="")

    def test_should_parse_uri_with_nested_publisher_path(self, config):
        uri = "gs://litellm-local/litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc-123"
        bucket, encoded = config._parse_gcs_uri(uri)
        assert bucket == "litellm-local"
        expected_path = "litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc-123"
        assert encoded == urllib.parse.quote(expected_path, safe="")

    def test_should_handle_url_encoded_input(self, config):
        encoded_uri = urllib.parse.quote("gs://my-bucket/some/path", safe="")
        bucket, encoded = config._parse_gcs_uri(encoded_uri)
        assert bucket == "my-bucket"
        assert encoded == urllib.parse.quote("some/path", safe="")

    def test_should_handle_bucket_only(self, config):
        bucket, encoded = config._parse_gcs_uri("gs://my-bucket")
        assert bucket == "my-bucket"
        assert encoded == ""

    def test_should_handle_no_gs_prefix(self, config):
        bucket, encoded = config._parse_gcs_uri("my-bucket/object.txt")
        assert bucket == "my-bucket"
        assert encoded == "object.txt"

class TestTransformRetrieveFile:

    def test_should_build_correct_gcs_metadata_url(self, config):
        file_id = "gs://my-bucket/path/to/file.jsonl"
        url, params = config.transform_retrieve_file_request(
            file_id=file_id, optional_params={}, litellm_params={}
        )
        expected_encoded = urllib.parse.quote("path/to/file.jsonl", safe="")
        assert url == f"https://storage.googleapis.com/storage/v1/b/my-bucket/o/{expected_encoded}"
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
        file_id = "gs://my-bucket/path/to/file.jsonl"
        url, params = config.transform_file_content_request(
            file_content_request={"file_id": file_id},
            optional_params={},
            litellm_params={},
        )
        encoded = urllib.parse.quote("path/to/file.jsonl", safe="")
        assert url == f"https://storage.googleapis.com/storage/v1/b/my-bucket/o/{encoded}?alt=media"
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
        file_id = "gs://my-bucket/path/to/file.jsonl"
        url, params = config.transform_delete_file_request(
            file_id=file_id, optional_params={}, litellm_params={}
        )
        encoded = urllib.parse.quote("path/to/file.jsonl", safe="")
        assert url == f"https://storage.googleapis.com/storage/v1/b/my-bucket/o/{encoded}"
        assert params == {}

    def test_should_return_file_deleted_with_reconstructed_id(self, config):
        raw_response = MagicMock(spec=httpx.Response)
        mock_request = MagicMock()
        encoded_name = urllib.parse.quote(
            "litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc", safe=""
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
        assert "litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc" in result.id

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
