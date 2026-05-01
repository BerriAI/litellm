"""
Tests for VertexAIFilesConfig transformation methods (Issues 5-7).
Includes tests for Vertex AI batch output transformation to OpenAI format.
"""

import json
import urllib.parse

import httpx
import pytest
from unittest.mock import MagicMock

from litellm.llms.vertex_ai.files.transformation import (
    VertexAIFilesConfig,
    VertexAIJsonlFilesTransformation,
    _sanitize_gcp_label_value,
)
from litellm.types.llms.openai import OpenAIFileObject, HttpxBinaryResponseContent
from openai.types.file_deleted import FileDeleted


@pytest.fixture
def config():
    return VertexAIFilesConfig()


class TestParseGcsUri:
    """Tests for the _parse_gcs_uri helper used by retrieve / content / delete."""

    def test_should_parse_standard_gs_uri(self, config):
        bucket, encoded = config._parse_gcs_uri("gs://my-bucket/path/to/object.jsonl")
        assert bucket == "my-bucket"
        assert encoded == urllib.parse.quote("path/to/object.jsonl", safe="")

    def test_should_parse_uri_with_nested_publisher_path(self, config):
        uri = "gs://litellm-local/litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc-123"
        bucket, encoded = config._parse_gcs_uri(uri)
        assert bucket == "litellm-local"
        expected_path = (
            "litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc-123"
        )
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
        file_id = "gs://my-bucket/path/to/file.jsonl"
        url, params = config.transform_file_content_request(
            file_content_request={"file_id": file_id},
            optional_params={},
            litellm_params={},
        )
        encoded = urllib.parse.quote("path/to/file.jsonl", safe="")
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
        file_id = "gs://my-bucket/path/to/file.jsonl"
        url, params = config.transform_delete_file_request(
            file_id=file_id, optional_params={}, litellm_params={}
        )
        encoded = urllib.parse.quote("path/to/file.jsonl", safe="")
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


class TestVertexBatchOutputTransformation:
    """Test transformation of Vertex AI batch outputs to OpenAI format"""

    def test_transform_successful_vertex_batch_output(self, config):
        """Test transformation of a successful Vertex AI batch output"""
        # Sample Vertex AI batch output (based on actual format)
        vertex_output = {
            "status": "",
            "processed_time": "2024-11-01T18:13:16.826+00:00",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "Hello world!"}]}],
                "labels": {"litellm_custom_id": "request-1"},
            },
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Hello! How can I help you today?"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "modelVersion": "gemini-2.0-flash-001@default",
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 30,
                },
            },
        }

        content = json.dumps(vertex_output).encode("utf-8")
        transformed_content = config._try_transform_vertex_batch_output_to_openai(
            content
        )
        result = json.loads(transformed_content.decode("utf-8"))

        # Verify OpenAI format
        assert "id" in result
        assert "custom_id" in result
        assert "response" in result
        assert "error" in result

        # Verify custom_id was extracted from labels
        assert result["custom_id"] == "request-1"

        # Verify response structure
        assert result["response"]["status_code"] == 200
        assert "body" in result["response"]

        # Verify body has OpenAI format
        body = result["response"]["body"]
        assert "choices" in body
        assert "usage" in body
        assert "model" in body

        # Verify choices
        assert len(body["choices"]) > 0
        choice = body["choices"][0]
        assert "message" in choice
        assert "content" in choice["message"]
        assert "Hello! How can I help you today?" in choice["message"]["content"]

    def test_transform_error_vertex_batch_output(self, config):
        """Test transformation of an error Vertex AI batch output"""
        vertex_output = {
            "status": "Error: Invalid request",
            "processed_time": "2024-11-01T18:13:16.826+00:00",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "Hello world!"}]}],
                "labels": {"litellm_custom_id": "request-error"},
            },
            "response": {},
        }

        content = json.dumps(vertex_output).encode("utf-8")
        transformed_content = config._try_transform_vertex_batch_output_to_openai(
            content
        )
        result = json.loads(transformed_content.decode("utf-8"))

        # Verify error format
        assert result["response"]["status_code"] == 400
        assert result["error"] is not None
        assert "Invalid request" in result["error"]["message"]
        assert result["custom_id"] == "request-error"

    def test_transform_vertex_batch_output_legacy_labels_only_sanitized(self, config):
        """Older LiteLLM batches only stored litellm_custom_id (sanitized); read path still works."""
        vertex_output = {
            "status": "",
            "processed_time": "2024-11-01T18:13:16.826+00:00",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "Hello world!"}]}],
                "labels": {"litellm_custom_id": "myrequest-1"},
            },
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Hello!"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "modelVersion": "gemini-2.0-flash-001@default",
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 30,
                },
            },
        }

        content = json.dumps(vertex_output).encode("utf-8")
        transformed_content = config._try_transform_vertex_batch_output_to_openai(
            content
        )
        result = json.loads(transformed_content.decode("utf-8"))

        assert result["custom_id"] == "myrequest-1"

    def test_transform_multiple_vertex_batch_outputs(self, config):
        """Test transformation of multiple Vertex AI batch outputs (JSONL)"""
        vertex_outputs = [
            {
                "status": "",
                "processed_time": "2024-11-01T18:13:16.826+00:00",
                "request": {
                    "contents": [
                        {"role": "user", "parts": [{"text": "First request"}]}
                    ],
                    "labels": {"litellm_custom_id": "request-1"},
                },
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "First response"}],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "modelVersion": "gemini-2.0-flash-001@default",
                    "usageMetadata": {
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 10,
                        "totalTokenCount": 15,
                    },
                },
            },
            {
                "status": "",
                "processed_time": "2024-11-01T18:13:17.826+00:00",
                "request": {
                    "contents": [
                        {"role": "user", "parts": [{"text": "Second request"}]}
                    ],
                    "labels": {"litellm_custom_id": "request-2"},
                },
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Second response"}],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "modelVersion": "gemini-2.0-flash-001@default",
                    "usageMetadata": {
                        "promptTokenCount": 6,
                        "candidatesTokenCount": 11,
                        "totalTokenCount": 17,
                    },
                },
            },
        ]

        content = "\n".join(json.dumps(output) for output in vertex_outputs).encode(
            "utf-8"
        )
        transformed_content = config._try_transform_vertex_batch_output_to_openai(
            content
        )
        lines = transformed_content.decode("utf-8").strip().split("\n")

        assert len(lines) == 2

        for i, line in enumerate(lines):
            result = json.loads(line)
            assert "id" in result
            assert "response" in result
            assert result["response"]["status_code"] == 200
            assert result["custom_id"] == f"request-{i+1}"
            body = result["response"]["body"]
            assert "choices" in body
            assert len(body["choices"]) > 0

    def test_non_batch_output_passthrough(self, config):
        """Test that non-batch output is returned as-is"""
        regular_content = b"This is just a regular file content"
        transformed_content = config._try_transform_vertex_batch_output_to_openai(
            regular_content
        )
        assert transformed_content == regular_content

    def test_invalid_json_passthrough(self, config):
        """Test that invalid JSON is returned as-is"""
        invalid_content = b'{"invalid": json content}'
        transformed_content = config._try_transform_vertex_batch_output_to_openai(
            invalid_content
        )
        assert transformed_content == invalid_content


class TestVertexBatchCustomIdLabels:
    """Test custom_id handling in batch transformations"""

    def test_custom_id_added_to_labels_in_vertex_request(self):
        """Test that custom_id from OpenAI format is added as a label in Vertex AI format"""
        transformation = VertexAIJsonlFilesTransformation()

        openai_jsonl_content = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gemini-1.5-flash-001",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "max_tokens": 10,
                },
            }
        ]

        vertex_jsonl_content = (
            transformation._transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
                openai_jsonl_content
            )
        )

        assert len(vertex_jsonl_content) == 1
        vertex_request = vertex_jsonl_content[0]

        # Verify labels were added
        assert "labels" in vertex_request["request"]
        assert "litellm_custom_id" in vertex_request["request"]["labels"]
        assert vertex_request["request"]["labels"]["litellm_custom_id"] == "request-1"
        raw_label = vertex_request["request"]["labels"]["litellm_custom_id_raw"]
        assert raw_label != "request-1"
        assert _sanitize_gcp_label_value(raw_label) == raw_label

    def test_multiple_requests_each_get_their_own_label(self):
        """Test that multiple requests each get their own custom_id label"""
        transformation = VertexAIJsonlFilesTransformation()

        openai_jsonl_content = [
            {
                "custom_id": f"request-{i+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gemini-1.5-flash-001",
                    "messages": [{"role": "user", "content": f"Question {i+1}"}],
                },
            }
            for i in range(3)
        ]

        vertex_jsonl_content = (
            transformation._transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
                openai_jsonl_content
            )
        )

        assert len(vertex_jsonl_content) == 3

        for i, vertex_request in enumerate(vertex_jsonl_content):
            expected_custom_id = f"request-{i+1}"
            assert (
                vertex_request["request"]["labels"]["litellm_custom_id"]
                == expected_custom_id
            )
            raw_label = vertex_request["request"]["labels"]["litellm_custom_id_raw"]
            assert raw_label != expected_custom_id
            assert _sanitize_gcp_label_value(raw_label) == raw_label

    def test_request_without_custom_id_has_no_label(self):
        """Test that requests without custom_id don't get a label"""
        transformation = VertexAIJsonlFilesTransformation()

        openai_jsonl_content = [
            {
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gemini-1.5-flash-001",
                    "messages": [{"role": "user", "content": "Question"}],
                },
            }
        ]

        vertex_jsonl_content = (
            transformation._transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
                openai_jsonl_content
            )
        )

        # Should not have labels if no custom_id was provided
        assert "labels" not in vertex_jsonl_content[0]["request"]

    def test_end_to_end_custom_id_round_trip(self):
        """
        Test the full round trip: OpenAI format -> Vertex AI format -> Vertex AI output -> OpenAI output
        Verify that custom_id is preserved through the entire flow.
        """
        transformation = VertexAIJsonlFilesTransformation()
        config = VertexAIFilesConfig()

        # Step 1: Transform OpenAI input to Vertex AI format (mixed case exercises raw label)
        openai_input = [
            {
                "custom_id": "MyRequest-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gemini-1.5-flash-001",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            }
        ]

        vertex_input = (
            transformation._transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
                openai_input
            )
        )

        # Verify both labels are GCP-safe and encoded raw preserves round-trip.
        assert (
            vertex_input[0]["request"]["labels"]["litellm_custom_id"] == "myrequest-1"
        )
        raw_label = vertex_input[0]["request"]["labels"]["litellm_custom_id_raw"]
        assert raw_label != "MyRequest-1"
        assert _sanitize_gcp_label_value(raw_label) == raw_label

        # Step 2: Simulate Vertex AI batch output (with the label echoed back)
        vertex_output = {
            "status": "",
            "processed_time": "2024-11-01T18:13:16.826+00:00",
            "request": vertex_input[0]["request"],
            "response": {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Hi there!"}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ],
                "modelVersion": "gemini-2.0-flash-001@default",
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 15,
                },
            },
        }

        # Step 3: Transform Vertex AI output back to OpenAI format
        content = json.dumps(vertex_output).encode("utf-8")
        transformed_content = config._try_transform_vertex_batch_output_to_openai(
            content
        )
        openai_output = json.loads(transformed_content.decode("utf-8"))

        # Step 4: Verify custom_id was preserved (original casing, not sanitized label)
        assert openai_output["custom_id"] == "MyRequest-1"
        assert openai_output["response"]["status_code"] == 200

    def test_custom_id_label_sanitization(self):
        """Test that custom_id values are sanitized to meet GCP label constraints"""
        transformation = VertexAIJsonlFilesTransformation()

        # Test sanitization function
        assert _sanitize_gcp_label_value("MyRequest-1") == "myrequest-1"
        assert _sanitize_gcp_label_value("Request.With.Dots") == "request_with_dots"
        assert _sanitize_gcp_label_value("Request With Spaces") == "request_with_spaces"
        assert _sanitize_gcp_label_value("Request@#$%Special") == "request____special"

        # Test max length (63 chars)
        long_id = "a" * 100
        assert len(_sanitize_gcp_label_value(long_id)) == 63

        # Test in actual transformation
        openai_input = [
            {
                "custom_id": "MyRequest-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gemini-1.5-flash-001",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            }
        ]

        vertex_input = (
            transformation._transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
                openai_input
            )
        )

        # Verify both labels are safe for GCP labels.
        assert (
            vertex_input[0]["request"]["labels"]["litellm_custom_id"] == "myrequest-1"
        )
        raw_label = vertex_input[0]["request"]["labels"]["litellm_custom_id_raw"]
        assert raw_label != "MyRequest-1"
        assert _sanitize_gcp_label_value(raw_label) == raw_label
