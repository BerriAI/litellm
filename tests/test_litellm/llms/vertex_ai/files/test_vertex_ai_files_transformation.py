"""
Tests for VertexAIFilesConfig transformation methods (Issues 5-7).
Includes tests for Vertex AI batch output transformation to OpenAI format.
"""

import json
import urllib.parse
from types import MappingProxyType
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from unittest.mock import MagicMock

from litellm.llms.vertex_ai.files.transformation import (
    VertexAIFilesConfig,
    VertexAIJsonlFilesTransformation,
    _get_litellm_batch_custom_id_from_labels,
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

    def test_should_reject_request_supplied_legacy_flag(self, config):
        with pytest.raises(ValueError, match="LiteLLM-managed"):
            config._parse_gcs_uri(
                "gs://my-bucket/private/object.txt",
                litellm_params={
                    "bucket_name": "my-bucket",
                    "allow_legacy_cloud_file_ids": True,
                },
            )

    def test_should_allow_legacy_object_path_with_trusted_server_flag(self, config):
        trusted_credentials = MappingProxyType({"allow_legacy_cloud_file_ids": True})
        bucket, encoded = config._parse_gcs_uri(
            "gs://my-bucket/private/object.txt",
            litellm_params={
                "bucket_name": "my-bucket",
                "_litellm_internal_model_credentials": trusted_credentials,
            },
        )

        assert bucket == "my-bucket"
        assert encoded == urllib.parse.quote("private/object.txt", safe="")

    def test_should_reject_user_supplied_legacy_flag_snapshot(self, config):
        with pytest.raises(ValueError, match="LiteLLM-managed"):
            config._parse_gcs_uri(
                "gs://my-bucket/private/object.txt",
                litellm_params={
                    "bucket_name": "my-bucket",
                    "_litellm_internal_model_credentials": {
                        "allow_legacy_cloud_file_ids": True
                    },
                },
            )

    def test_should_keep_configured_prefix_for_legacy_object_path(self, config):
        trusted_credentials = MappingProxyType({"allow_legacy_cloud_file_ids": True})
        bucket, encoded = config._parse_gcs_uri(
            "gs://my-bucket/team-a/private/object.txt",
            litellm_params={
                "bucket_name": "my-bucket/team-a",
                "_litellm_internal_model_credentials": trusted_credentials,
            },
        )

        assert bucket == "my-bucket"
        assert encoded == urllib.parse.quote("team-a/private/object.txt", safe="")

    def test_should_reject_legacy_object_outside_configured_prefix(self, config):
        trusted_credentials = MappingProxyType({"allow_legacy_cloud_file_ids": True})
        with pytest.raises(ValueError, match="configured storage prefix"):
            config._parse_gcs_uri(
                "gs://my-bucket/team-b/private/object.txt",
                litellm_params={
                    "bucket_name": "my-bucket/team-a",
                    "_litellm_internal_model_credentials": trusted_credentials,
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

    def test_should_not_mutate_caller_logging_obj_for_batch_output_transform(
        self, config, monkeypatch
    ):
        original_model = "vertex_ai/original-model"
        original_start_time = 123.456
        original_optional_params = {"temperature": 0.1}
        raw_response = httpx.Response(
            status_code=200,
            content=json.dumps(
                {
                    "status": "",
                    "processed_time": "2024-11-01T18:13:16.826+00:00",
                    "request": {"labels": {"litellm_custom_id": "request-1"}},
                    "response": {
                        "candidates": [
                            {"content": {"parts": [{"text": "ok"}], "role": "model"}}
                        ],
                        "modelVersion": "gemini-2.0-flash-001@default",
                    },
                }
            ).encode("utf-8"),
            headers={"content-type": "application/octet-stream"},
            request=httpx.Request("GET", "https://example.com"),
        )
        logging_obj = MagicMock()
        logging_obj.model = original_model
        logging_obj.start_time = original_start_time
        logging_obj.optional_params = original_optional_params
        captured = {}

        def mock_transform_single(
            vertex_output,
            vertex_gemini_config,
            logging_obj,
            mock_httpx_response,
        ):
            captured["logging_obj"] = logging_obj
            logging_obj.model = "gemini-2.0-flash-001"
            logging_obj.start_time = 789.0
            return {
                "custom_id": vertex_output["request"]["labels"]["litellm_custom_id"]
            }

        monkeypatch.setattr(
            config,
            "_transform_single_vertex_batch_output_to_openai",
            mock_transform_single,
        )

        result = config.transform_file_content_response(
            raw_response=raw_response,
            logging_obj=logging_obj,
            litellm_params={},
        )

        assert captured["logging_obj"] is not logging_obj
        assert logging_obj.model == original_model
        assert logging_obj.start_time == original_start_time
        assert logging_obj.optional_params == original_optional_params
        assert result.response is not raw_response

    def test_should_skip_batch_output_transformation_when_opt_out_flag_set(
        self, config, monkeypatch
    ):
        """When `litellm.disable_vertex_batch_output_transformation` is True the
        Vertex predictions.jsonl content must be returned untouched, so callers
        that parse raw `candidates`/`modelVersion` keep working."""
        import litellm

        raw_jsonl = json.dumps(
            {
                "status": "",
                "processed_time": "2024-11-01T18:13:16.826+00:00",
                "request": {"labels": {"litellm_custom_id": "request-1"}},
                "response": {
                    "candidates": [
                        {"content": {"parts": [{"text": "ok"}], "role": "model"}}
                    ],
                    "modelVersion": "gemini-2.0-flash-001@default",
                },
            }
        ).encode("utf-8")
        raw_response = httpx.Response(
            status_code=200,
            content=raw_jsonl,
            headers={"content-type": "application/octet-stream"},
            request=httpx.Request("GET", "https://example.com"),
        )

        monkeypatch.setattr(
            litellm, "disable_vertex_batch_output_transformation", True, raising=False
        )

        result = config.transform_file_content_response(
            raw_response=raw_response,
            logging_obj=MagicMock(),
            litellm_params={},
        )

        assert isinstance(result, HttpxBinaryResponseContent)
        assert result.response.content == raw_jsonl


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

        # Per OpenAI Batch output spec, error entries set response to null
        # and populate the top-level error object.
        assert result["response"] is None
        assert result["error"] is not None
        assert "Invalid request" in result["error"]["message"]
        assert result["error"]["code"] == "vertex_ai_error"
        assert result["custom_id"] == "request-error"

    def test_transform_exception_path_sets_response_null(self, config):
        """
        The except-Exception branch in _transform_single_vertex_batch_output_to_openai
        must also emit response=null per the OpenAI Batch output spec. The outer
        _try_transform path swallows exceptions and falls back to original content,
        so this test invokes the single-line transformer directly with a vertex_gemini_config
        stub that raises during transformation.
        """
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
            VertexGeminiConfig,
        )

        vertex_output = {
            "status": "",
            "processed_time": "2024-11-01T18:13:16.826+00:00",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "Hello world!"}]}],
                "labels": {"litellm_custom_id": "request-boom"},
            },
            "response": {"modelVersion": "gemini-2.0-flash-001@default"},
        }

        class _RaisingGeminiConfig(VertexGeminiConfig):
            def _transform_google_generate_content_to_openai_model_response(
                self, *args, **kwargs
            ):
                raise ValueError("simulated transform failure")

        mock_response = httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            request=httpx.Request(method="POST", url="https://example.com"),
        )

        result = config._transform_single_vertex_batch_output_to_openai(
            vertex_output=vertex_output,
            vertex_gemini_config=_RaisingGeminiConfig(),
            logging_obj=MagicMock(),
            mock_httpx_response=mock_response,
        )

        assert result["response"] is None
        assert result["error"] is not None
        assert result["error"]["code"] == "transformation_error"
        assert "simulated transform failure" in result["error"]["message"]
        assert result["custom_id"] == "request-boom"

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

    def test_transform_vertex_batch_output_with_first_line_prompt_feedback(
        self, config, monkeypatch
    ):
        """Test that promptFeedback-only first lines are detected as Vertex batch output."""
        vertex_outputs = [
            {
                "status": "",
                "processed_time": "2024-11-01T18:13:16.826+00:00",
                "request": {"labels": {"litellm_custom_id": "blocked-request"}},
                "response": {
                    "promptFeedback": {"blockReason": "SAFETY"},
                    "modelVersion": "gemini-2.0-flash-001@default",
                },
            },
            {
                "status": "",
                "processed_time": "2024-11-01T18:13:17.826+00:00",
                "request": {"labels": {"litellm_custom_id": "request-2"}},
                "response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
            },
        ]

        def mock_transform_single(
            vertex_output,
            vertex_gemini_config,
            logging_obj,
            mock_httpx_response,
        ):
            return {
                "custom_id": vertex_output["request"]["labels"]["litellm_custom_id"]
            }

        monkeypatch.setattr(
            config,
            "_transform_single_vertex_batch_output_to_openai",
            mock_transform_single,
        )

        content = "\n".join(json.dumps(output) for output in vertex_outputs).encode(
            "utf-8"
        )
        transformed_content = config._try_transform_vertex_batch_output_to_openai(
            content
        )
        results = [
            json.loads(line) for line in transformed_content.decode("utf-8").split("\n")
        ]

        assert [result["custom_id"] for result in results] == [
            "blocked-request",
            "request-2",
        ]

    def test_batch_detection_requires_candidates_or_non_empty_status(self, config):
        """Test that JSONL with a blank status but no candidates is returned as-is."""
        non_batch_output = {
            "status": "",
            "processed_time": "2024-11-01T18:13:16.826+00:00",
            "request": {"metadata": "not a Vertex batch request"},
            "response": {"metadata": "not a Gemini response"},
        }

        content = json.dumps(non_batch_output).encode("utf-8")
        transformed_content = config._try_transform_vertex_batch_output_to_openai(
            content
        )

        assert transformed_content == content

    def test_reuses_batch_transform_helpers_per_jsonl_file(self, config, monkeypatch):
        """Test that heavy helper objects are reused while transforming a JSONL file."""
        vertex_outputs = [
            {
                "status": "",
                "processed_time": "2024-11-01T18:13:16.826+00:00",
                "request": {"labels": {"litellm_custom_id": f"request-{i}"}},
                "response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
            }
            for i in range(2)
        ]
        helper_ids = []

        def mock_transform_single(
            vertex_output,
            vertex_gemini_config,
            logging_obj,
            mock_httpx_response,
        ):
            helper_ids.append(
                (
                    id(vertex_gemini_config),
                    id(logging_obj),
                    id(mock_httpx_response),
                )
            )
            return {
                "custom_id": vertex_output["request"]["labels"]["litellm_custom_id"]
            }

        monkeypatch.setattr(
            config,
            "_transform_single_vertex_batch_output_to_openai",
            mock_transform_single,
        )

        content = "\n".join(json.dumps(output) for output in vertex_outputs).encode(
            "utf-8"
        )
        transformed_content = config._try_transform_vertex_batch_output_to_openai(
            content
        )

        assert len(transformed_content.decode("utf-8").strip().split("\n")) == 2
        assert len(set(helper_ids)) == 1

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


class TestTryTransformDoesNotMutateCallerLoggingObj:
    """Regression tests: _try_transform_vertex_batch_output_to_openai must not mutate
    the caller's logging_obj (model, start_time, optional_params)."""

    def _make_vertex_batch_line(self) -> bytes:
        return json.dumps(
            {
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
                                "parts": [{"text": "Hi!"}],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "modelVersion": "gemini-2.0-flash-001@default",
                    "usageMetadata": {
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 3,
                        "totalTokenCount": 8,
                    },
                },
            }
        ).encode("utf-8")

    def test_should_not_overwrite_model_on_caller_logging_obj(self, config):
        sentinel_model = "original-caller-model"
        logging_obj = MagicMock()
        logging_obj.model = sentinel_model
        logging_obj.optional_params = {"temperature": 0.9}

        config._try_transform_vertex_batch_output_to_openai(
            content=self._make_vertex_batch_line(),
            logging_obj=logging_obj,
        )

        assert (
            logging_obj.model == sentinel_model
        ), "logging_obj.model was mutated by _try_transform_vertex_batch_output_to_openai"

    def test_should_not_overwrite_start_time_on_caller_logging_obj(self, config):
        sentinel_start = 1234567890.0
        logging_obj = MagicMock()
        logging_obj.start_time = sentinel_start
        logging_obj.optional_params = {}

        config._try_transform_vertex_batch_output_to_openai(
            content=self._make_vertex_batch_line(),
            logging_obj=logging_obj,
        )

        assert (
            logging_obj.start_time == sentinel_start
        ), "logging_obj.start_time was mutated by _try_transform_vertex_batch_output_to_openai"

    def test_should_not_overwrite_optional_params_on_caller_logging_obj(self, config):
        sentinel_params = {"temperature": 0.5, "top_p": 0.9}
        logging_obj = MagicMock()
        logging_obj.optional_params = sentinel_params

        config._try_transform_vertex_batch_output_to_openai(
            content=self._make_vertex_batch_line(),
            logging_obj=logging_obj,
        )

        assert (
            logging_obj.optional_params is sentinel_params
        ), "logging_obj.optional_params was replaced by _try_transform_vertex_batch_output_to_openai"
        assert logging_obj.optional_params == {
            "temperature": 0.5,
            "top_p": 0.9,
        }, "logging_obj.optional_params contents were mutated"

    def test_should_still_transform_content_correctly(self, config):
        logging_obj = MagicMock()
        logging_obj.model = "original-model"
        logging_obj.start_time = 9999.0
        logging_obj.optional_params = {"max_tokens": 100}

        result = config._try_transform_vertex_batch_output_to_openai(
            content=self._make_vertex_batch_line(),
            logging_obj=logging_obj,
        )

        # Transformation should still succeed
        transformed = json.loads(result.decode("utf-8"))
        assert transformed["custom_id"] == "request-1"
        assert transformed["response"]["status_code"] == 200


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

    def test_long_custom_id_round_trips_across_raw_label_chunks(self):
        """Test that long custom_ids are not truncated in raw labels."""
        transformation = VertexAIJsonlFilesTransformation()
        custom_id_a = "shared-prefix-that-is-longer-than-thirty-six-bytes-A"
        custom_id_b = "shared-prefix-that-is-longer-than-thirty-six-bytes-B"

        openai_jsonl_content = [
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gemini-1.5-flash-001",
                    "messages": [{"role": "user", "content": "Question"}],
                },
            }
            for custom_id in (custom_id_a, custom_id_b)
        ]

        vertex_jsonl_content = (
            transformation._transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
                openai_jsonl_content
            )
        )
        labels_a = vertex_jsonl_content[0]["request"]["labels"]
        labels_b = vertex_jsonl_content[1]["request"]["labels"]

        assert "litellm_custom_id_raw_1" in labels_a
        assert "litellm_custom_id_raw_1" in labels_b
        assert labels_a["litellm_custom_id_raw"] == labels_b["litellm_custom_id_raw"]
        assert (
            labels_a["litellm_custom_id_raw_1"] != labels_b["litellm_custom_id_raw_1"]
        )
        assert _get_litellm_batch_custom_id_from_labels(labels_a) == custom_id_a
        assert _get_litellm_batch_custom_id_from_labels(labels_b) == custom_id_b

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
