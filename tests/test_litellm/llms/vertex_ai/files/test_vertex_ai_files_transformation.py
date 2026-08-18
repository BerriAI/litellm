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
    _get_litellm_batch_custom_id_from_labels,
    _openai_batch_jsonl_entry_to_vertex_rows,
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
        bucket, encoded = config._parse_gcs_uri(file_id, litellm_params={"gcs_bucket_name": "my-bucket"})
        assert bucket == "my-bucket"
        assert encoded == urllib.parse.quote("litellm-vertex-files/path/to/object.jsonl", safe="")

    def test_should_parse_uri_with_nested_publisher_path(self, config):
        uri = "gs://litellm-local/litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc-123"
        bucket, encoded = config._parse_gcs_uri(uri, litellm_params={"gcs_bucket_name": "litellm-local"})
        assert bucket == "litellm-local"
        expected_path = "litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc-123"
        assert encoded == urllib.parse.quote(expected_path, safe="")

    def test_should_handle_url_encoded_input(self, config):
        encoded_uri = urllib.parse.quote("gs://my-bucket/litellm-vertex-files/some/path", safe="")
        bucket, encoded = config._parse_gcs_uri(encoded_uri, litellm_params={"gcs_bucket_name": "my-bucket"})
        assert bucket == "my-bucket"
        assert encoded == urllib.parse.quote("litellm-vertex-files/some/path", safe="")

    def test_should_reject_bucket_only(self, config):
        with pytest.raises(ValueError, match="object name"):
            config._parse_gcs_uri("gs://my-bucket", litellm_params={"gcs_bucket_name": "my-bucket"})

    def test_should_reject_no_gs_prefix(self, config):
        with pytest.raises(ValueError, match="gs://"):
            config._parse_gcs_uri(
                "my-bucket/litellm-vertex-files/object.txt",
                litellm_params={"gcs_bucket_name": "my-bucket"},
            )

    def test_should_reject_unmanaged_object_path(self, config):
        with pytest.raises(ValueError, match="LiteLLM-managed"):
            config._parse_gcs_uri(
                "gs://my-bucket/private/object.txt",
                litellm_params={"gcs_bucket_name": "my-bucket"},
            )

    def test_should_reject_request_supplied_legacy_flag(self, config):
        with pytest.raises(ValueError, match="LiteLLM-managed"):
            config._parse_gcs_uri(
                "gs://my-bucket/private/object.txt",
                litellm_params={
                    "gcs_bucket_name": "my-bucket",
                    "allow_legacy_cloud_file_ids": True,
                },
            )

    def test_should_allow_legacy_object_path_with_trusted_server_flag(self, config):
        trusted_credentials = MappingProxyType({"allow_legacy_cloud_file_ids": True})
        bucket, encoded = config._parse_gcs_uri(
            "gs://my-bucket/private/object.txt",
            litellm_params={
                "gcs_bucket_name": "my-bucket",
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
                    "gcs_bucket_name": "my-bucket",
                    "_litellm_internal_model_credentials": {"allow_legacy_cloud_file_ids": True},
                },
            )

    def test_should_keep_configured_prefix_for_legacy_object_path(self, config):
        trusted_credentials = MappingProxyType({"allow_legacy_cloud_file_ids": True})
        bucket, encoded = config._parse_gcs_uri(
            "gs://my-bucket/team-a/private/object.txt",
            litellm_params={
                "gcs_bucket_name": "my-bucket/team-a",
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
                    "gcs_bucket_name": "my-bucket/team-a",
                    "_litellm_internal_model_credentials": trusted_credentials,
                },
            )

    def test_should_reject_unconfigured_bucket(self, config):
        with pytest.raises(ValueError, match="configured storage bucket"):
            config._parse_gcs_uri(
                "gs://other-bucket/litellm-vertex-files/object.txt",
                litellm_params={"gcs_bucket_name": "my-bucket"},
            )


class TestCreateFileUrl:
    def test_should_ignore_request_metadata_bucket_and_sanitize_filename(self, config):
        url = config.get_complete_file_url(
            api_base=None,
            api_key=None,
            model="",
            optional_params={},
            litellm_params={
                "gcs_bucket_name": "safe-bucket",
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
            litellm_params={"gcs_bucket_name": "my-bucket"},
        )
        expected_encoded = urllib.parse.quote("litellm-vertex-files/path/to/file.jsonl", safe="")
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
        file_id = "gs://my-bucket/litellm-vertex-files/path/to/file.jsonl"
        url, params = config.transform_file_content_request(
            file_content_request={"file_id": file_id},
            optional_params={},
            litellm_params={"gcs_bucket_name": "my-bucket"},
        )
        encoded = urllib.parse.quote("litellm-vertex-files/path/to/file.jsonl", safe="")
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

    def test_should_not_mutate_caller_logging_obj_for_batch_output_transform(self, config, monkeypatch):
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
                        "candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}}],
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
            return {"custom_id": vertex_output["request"]["labels"]["litellm_custom_id"]}

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

    def test_should_skip_batch_output_transformation_when_opt_out_flag_set(self, config, monkeypatch):
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
                    "candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}}],
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

        monkeypatch.setattr(litellm, "disable_vertex_batch_output_transformation", True, raising=False)

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
            litellm_params={"gcs_bucket_name": "my-bucket"},
        )
        encoded = urllib.parse.quote("litellm-vertex-files/path/to/file.jsonl", safe="")
        assert url == f"https://storage.googleapis.com/storage/v1/b/my-bucket/o/{encoded}"
        assert params == {}

    def test_should_return_file_deleted_with_reconstructed_id(self, config):
        raw_response = MagicMock(spec=httpx.Response)
        mock_request = MagicMock()
        encoded_name = urllib.parse.quote(
            "litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc",
            safe="",
        )
        mock_request.url = f"https://storage.googleapis.com/storage/v1/b/my-bucket/o/{encoded_name}"
        raw_response.request = mock_request

        result = config.transform_delete_file_response(
            raw_response=raw_response,
            logging_obj=MagicMock(),
            litellm_params={},
        )

        assert isinstance(result, FileDeleted)
        assert result.deleted is True
        assert result.object == "file"
        assert result.id == "gs://my-bucket/litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc"

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
        mock_request.url = f"https://storage.googleapis.com/storage/v1/b/my-bucket/o/{encoded_object}"
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
            "gs://prod-bucket/litellm-vertex-files/publishers/google/models/gemini-2.0-flash-001/abc-123"
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
        transformed_content = config._try_transform_vertex_batch_output_to_openai(content)
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
        transformed_content = config._try_transform_vertex_batch_output_to_openai(content)
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
            def _transform_google_generate_content_to_openai_model_response(self, *args, **kwargs):
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
        transformed_content = config._try_transform_vertex_batch_output_to_openai(content)
        result = json.loads(transformed_content.decode("utf-8"))

        assert result["custom_id"] == "myrequest-1"

    def test_transform_multiple_vertex_batch_outputs(self, config):
        """Test transformation of multiple Vertex AI batch outputs (JSONL)"""
        vertex_outputs = [
            {
                "status": "",
                "processed_time": "2024-11-01T18:13:16.826+00:00",
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": "First request"}]}],
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
                    "contents": [{"role": "user", "parts": [{"text": "Second request"}]}],
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

        content = "\n".join(json.dumps(output) for output in vertex_outputs).encode("utf-8")
        transformed_content = config._try_transform_vertex_batch_output_to_openai(content)
        lines = transformed_content.decode("utf-8").strip().split("\n")

        assert len(lines) == 2

        for i, line in enumerate(lines):
            result = json.loads(line)
            assert "id" in result
            assert "response" in result
            assert result["response"]["status_code"] == 200
            assert result["custom_id"] == f"request-{i + 1}"
            body = result["response"]["body"]
            assert "choices" in body
            assert len(body["choices"]) > 0

    def test_transform_vertex_batch_output_with_first_line_prompt_feedback(self, config, monkeypatch):
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
            return {"custom_id": vertex_output["request"]["labels"]["litellm_custom_id"]}

        monkeypatch.setattr(
            config,
            "_transform_single_vertex_batch_output_to_openai",
            mock_transform_single,
        )

        content = "\n".join(json.dumps(output) for output in vertex_outputs).encode("utf-8")
        transformed_content = config._try_transform_vertex_batch_output_to_openai(content)
        results = [json.loads(line) for line in transformed_content.decode("utf-8").split("\n")]

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
        transformed_content = config._try_transform_vertex_batch_output_to_openai(content)

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
            return {"custom_id": vertex_output["request"]["labels"]["litellm_custom_id"]}

        monkeypatch.setattr(
            config,
            "_transform_single_vertex_batch_output_to_openai",
            mock_transform_single,
        )

        content = "\n".join(json.dumps(output) for output in vertex_outputs).encode("utf-8")
        transformed_content = config._try_transform_vertex_batch_output_to_openai(content)

        assert len(transformed_content.decode("utf-8").strip().split("\n")) == 2
        assert len(set(helper_ids)) == 1

    def test_non_batch_output_passthrough(self, config):
        """Test that non-batch output is returned as-is"""
        regular_content = b"This is just a regular file content"
        transformed_content = config._try_transform_vertex_batch_output_to_openai(regular_content)
        assert transformed_content == regular_content

    def test_invalid_json_passthrough(self, config):
        """Test that invalid JSON is returned as-is"""
        invalid_content = b'{"invalid": json content}'
        transformed_content = config._try_transform_vertex_batch_output_to_openai(invalid_content)
        assert transformed_content == invalid_content

    def test_binary_content_passthrough(self, config):
        """A binary file (PDF/video) whose first bytes are not valid UTF-8 must be
        returned unchanged. The row-by-row transform only engages for a JSONL
        batch output and must never line-parse or corrupt binary content."""
        binary = b"%PDF-1.4\n%\xc4\xe5\xf2\xe5\xeb\xa7\n" + b"\x00\x01\x02\xff\xfe" * 64
        assert config._try_transform_vertex_batch_output_to_openai(binary) == binary

    def test_streaming_transform_peaks_below_list_pipeline(self, config):
        """The output transform must stream row-by-row, not build a list of every
        parsed row and a second list of transformed rows. This guards against a
        regression to the list pipeline, which peaks at several full copies and
        OOMs on large result files. The relative comparison cancels shared noise
        (per-row transform cost, GC timing) and only the list overhead differs.
        """
        import gc
        import tracemalloc

        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
            VertexGeminiConfig,
        )

        def vertex_row(index: int) -> dict:
            return {
                "status": "",
                "processed_time": "2024-11-01T18:13:16.826+00:00",
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                    "labels": {"litellm_custom_id": f"r-{index}"},
                },
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "hello " * 20}],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "modelVersion": "gemini-2.0-flash-001",
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 20,
                        "totalTokenCount": 30,
                    },
                },
            }

        content = ("\n".join(json.dumps(vertex_row(i)) for i in range(4000))).encode("utf-8")

        def list_pipeline() -> bytes:
            gemini_config = VertexGeminiConfig()
            logging_obj = Logging(
                model="",
                messages=[],
                stream=False,
                call_type="batch_transform",
                start_time=0.1,
                litellm_call_id="",
                function_id="",
            )
            logging_obj.optional_params = {}
            mock_response = httpx.Response(
                status_code=200,
                headers={"content-type": "application/json"},
                request=httpx.Request("POST", "https://example.com"),
            )
            rows = content.decode("utf-8").strip().split("\n")
            transformed = [
                json.dumps(
                    config._transform_single_vertex_batch_output_to_openai(
                        json.loads(row), gemini_config, logging_obj, mock_response
                    )
                )
                for row in rows
            ]
            return "\n".join(transformed).encode("utf-8")

        def peak_of(fn) -> int:
            gc.collect()
            tracemalloc.start()
            try:
                fn()
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

        streaming_peak = peak_of(lambda: config._try_transform_vertex_batch_output_to_openai(content))
        list_peak = peak_of(list_pipeline)

        assert streaming_peak < list_peak * 0.75, (
            f"streaming peak {streaming_peak} is not a clear win over the list "
            f"pipeline {list_peak} (ratio {streaming_peak / list_peak:.2f})"
        )


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

        assert logging_obj.model == sentinel_model, (
            "logging_obj.model was mutated by _try_transform_vertex_batch_output_to_openai"
        )

    def test_should_not_overwrite_start_time_on_caller_logging_obj(self, config):
        sentinel_start = 1234567890.0
        logging_obj = MagicMock()
        logging_obj.start_time = sentinel_start
        logging_obj.optional_params = {}

        config._try_transform_vertex_batch_output_to_openai(
            content=self._make_vertex_batch_line(),
            logging_obj=logging_obj,
        )

        assert logging_obj.start_time == sentinel_start, (
            "logging_obj.start_time was mutated by _try_transform_vertex_batch_output_to_openai"
        )

    def test_should_not_overwrite_optional_params_on_caller_logging_obj(self, config):
        sentinel_params = {"temperature": 0.5, "top_p": 0.9}
        logging_obj = MagicMock()
        logging_obj.optional_params = sentinel_params

        config._try_transform_vertex_batch_output_to_openai(
            content=self._make_vertex_batch_line(),
            logging_obj=logging_obj,
        )

        assert logging_obj.optional_params is sentinel_params, (
            "logging_obj.optional_params was replaced by _try_transform_vertex_batch_output_to_openai"
        )
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


def _wrap_entries(openai_jsonl_content):
    """Vertex rows for a list of OpenAI batch entries, built via the live
    single-entry transform that the streaming upload path uses."""
    cfg = VertexAIFilesConfig()
    return [
        row
        for entry in openai_jsonl_content
        for row in _openai_batch_jsonl_entry_to_vertex_rows(entry, cfg._map_openai_to_vertex_params)
    ]


class TestVertexBatchCustomIdLabels:
    """Test custom_id handling in batch transformations"""

    def test_custom_id_added_to_labels_in_vertex_request(self):
        """Test that custom_id from OpenAI format is added as a label in Vertex AI format"""

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

        vertex_jsonl_content = _wrap_entries(openai_jsonl_content)

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

        vertex_jsonl_content = _wrap_entries(openai_jsonl_content)
        labels_a = vertex_jsonl_content[0]["request"]["labels"]
        labels_b = vertex_jsonl_content[1]["request"]["labels"]

        assert "litellm_custom_id_raw_1" in labels_a
        assert "litellm_custom_id_raw_1" in labels_b
        assert labels_a["litellm_custom_id_raw"] == labels_b["litellm_custom_id_raw"]
        assert labels_a["litellm_custom_id_raw_1"] != labels_b["litellm_custom_id_raw_1"]
        assert _get_litellm_batch_custom_id_from_labels(labels_a) == custom_id_a
        assert _get_litellm_batch_custom_id_from_labels(labels_b) == custom_id_b

    def test_multiple_requests_each_get_their_own_label(self):
        """Test that multiple requests each get their own custom_id label"""

        openai_jsonl_content = [
            {
                "custom_id": f"request-{i + 1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gemini-1.5-flash-001",
                    "messages": [{"role": "user", "content": f"Question {i + 1}"}],
                },
            }
            for i in range(3)
        ]

        vertex_jsonl_content = _wrap_entries(openai_jsonl_content)

        assert len(vertex_jsonl_content) == 3

        for i, vertex_request in enumerate(vertex_jsonl_content):
            expected_custom_id = f"request-{i + 1}"
            assert vertex_request["request"]["labels"]["litellm_custom_id"] == expected_custom_id
            raw_label = vertex_request["request"]["labels"]["litellm_custom_id_raw"]
            assert raw_label != expected_custom_id
            assert _sanitize_gcp_label_value(raw_label) == raw_label

    def test_request_without_custom_id_has_no_label(self):
        """Test that requests without custom_id don't get a label"""

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

        vertex_jsonl_content = _wrap_entries(openai_jsonl_content)

        # Should not have labels if no custom_id was provided
        assert "labels" not in vertex_jsonl_content[0]["request"]

    def test_end_to_end_custom_id_round_trip(self):
        """
        Test the full round trip: OpenAI format -> Vertex AI format -> Vertex AI output -> OpenAI output
        Verify that custom_id is preserved through the entire flow.
        """
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

        vertex_input = _wrap_entries(openai_input)

        # Verify both labels are GCP-safe and encoded raw preserves round-trip.
        assert vertex_input[0]["request"]["labels"]["litellm_custom_id"] == "myrequest-1"
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
        transformed_content = config._try_transform_vertex_batch_output_to_openai(content)
        openai_output = json.loads(transformed_content.decode("utf-8"))

        # Step 4: Verify custom_id was preserved (original casing, not sanitized label)
        assert openai_output["custom_id"] == "MyRequest-1"
        assert openai_output["response"]["status_code"] == 200

    def test_custom_id_label_sanitization(self):
        """Test that custom_id values are sanitized to meet GCP label constraints"""

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

        vertex_input = _wrap_entries(openai_input)

        # Verify both labels are safe for GCP labels.
        assert vertex_input[0]["request"]["labels"]["litellm_custom_id"] == "myrequest-1"
        raw_label = vertex_input[0]["request"]["labels"]["litellm_custom_id_raw"]
        assert raw_label != "MyRequest-1"
        assert _sanitize_gcp_label_value(raw_label) == raw_label


class TestConfiguredBucketNameResolution:
    def test_should_resolve_new_gcs_bucket_name_key(self, config, monkeypatch):
        monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
        assert config._get_configured_bucket_name({"gcs_bucket_name": "my-new-bucket"}) == "my-new-bucket"

    def test_should_resolve_legacy_bucket_name_key(self, config, monkeypatch):
        monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
        assert config._get_configured_bucket_name({"bucket_name": "my-legacy-bucket"}) == "my-legacy-bucket"

    def test_should_prefer_new_key_over_legacy(self, config, monkeypatch):
        monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
        assert config._get_configured_bucket_name({"gcs_bucket_name": "new", "bucket_name": "legacy"}) == "new"

    def test_should_fall_back_to_env(self, config, monkeypatch):
        monkeypatch.setenv("GCS_BUCKET_NAME", "env-bucket")
        assert config._get_configured_bucket_name({}) == "env-bucket"

    def test_should_raise_when_no_bucket_anywhere(self, config, monkeypatch):
        monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
        with pytest.raises(ValueError, match="GCS bucket_name is required"):
            config._get_configured_bucket_name({})

    def test_legacy_kwarg_survives_get_litellm_params(self):
        from litellm.litellm_core_utils.get_litellm_params import (
            OPTIONAL_KWARGS_KEYS,
            get_litellm_params,
        )

        assert "bucket_name" in OPTIONAL_KWARGS_KEYS
        params = get_litellm_params(bucket_name="my-legacy-bucket")
        assert params.get("bucket_name") == "my-legacy-bucket"


def _embeddings_entry(**overrides):
    entry = {
        "custom_id": "request-1",
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {"model": "gemini-embedding-2", "input": "hello world"},
    }
    entry.update(overrides)
    return entry


class TestVertexEmbeddingsBatchInputTranslation:
    """
    /v1/embeddings batch lines must be translated to Vertex's Gemini Embedding batch
    shape, not the generateContent shape.

    Ref: https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/batch-prediction-genai-embeddings
    """

    def test_should_emit_embed_content_request_shape(self):
        (row,) = _wrap_entries([_embeddings_entry()])

        assert row["request"] == {"content": {"parts": [{"text": "hello world"}]}}
        assert "contents" not in row["request"]
        assert "labels" not in row["request"]

    def test_should_round_trip_custom_id_through_top_level_key(self):
        (row,) = _wrap_entries([_embeddings_entry(custom_id="MyRequest-1")])

        assert row["key"] == "MyRequest-1"

    def test_should_omit_key_when_no_custom_id(self):
        entry = _embeddings_entry()
        del entry["custom_id"]

        (row,) = _wrap_entries([entry])

        assert "key" not in row

    def test_should_map_openai_params_into_the_embed_content_request(self):
        """
        The docs put these in an `embed_content_config` sibling of `request`, but Vertex
        rejects that key and fails the whole job, so they belong inside the request.
        """
        (row,) = _wrap_entries(
            [
                _embeddings_entry(
                    body={
                        "model": "gemini-embedding-001",
                        "input": "hello world",
                        "dimensions": 768,
                        "task_type": "RETRIEVAL_DOCUMENT",
                        "title": "some_title",
                    }
                )
            ]
        )

        assert row == {
            "key": "request-1",
            "request": {
                "content": {"parts": [{"text": "hello world"}]},
                "output_dimensionality": 768,
                "task_type": "RETRIEVAL_DOCUMENT",
                "title": "some_title",
            },
        }

    def test_should_omit_config_fields_when_no_params_given(self):
        (row,) = _wrap_entries([_embeddings_entry()])

        assert set(row["request"]) == {"content"}

    def test_should_translate_multimodal_gcs_input(self):
        (row,) = _wrap_entries(
            [
                _embeddings_entry(
                    body={
                        "model": "gemini-embedding-2",
                        "input": "gs://cloud-samples-data/generative-ai/image/benchmark.jpeg",
                    }
                )
            ]
        )

        assert row["request"]["content"]["parts"] == [
            {
                "file_data": {
                    "mime_type": "image/jpeg",
                    "file_uri": "gs://cloud-samples-data/generative-ai/image/benchmark.jpeg",
                }
            }
        ]

    @pytest.mark.parametrize("url", ["/v1/embeddings", "v1/embeddings", "/v1/embeddings/"])
    def test_should_detect_embeddings_route_variants(self, url):
        (row,) = _wrap_entries([_embeddings_entry(url=url)])

        assert "content" in row["request"]

    def test_should_raise_when_input_missing(self):
        with pytest.raises(ValueError, match="`input` is required"):
            _wrap_entries([_embeddings_entry(body={"model": "gemini-embedding-2"})])

    def test_should_raise_when_input_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _wrap_entries([_embeddings_entry(body={"model": "gemini-embedding-2", "input": []})])

    def test_should_fan_an_input_array_out_into_one_row_per_element(self):
        """
        An `EmbedContentRequest` returns exactly one vector, so an OpenAI entry asking
        for several embeddings needs several Vertex rows.
        """
        rows = _wrap_entries(
            [
                _embeddings_entry(
                    body={
                        "model": "gemini-embedding-001",
                        "input": ["first", "second"],
                        "dimensions": 768,
                    }
                )
            ]
        )

        assert rows == [
            {
                "key": "request-1#0/2",
                "request": {
                    "content": {"parts": [{"text": "first"}]},
                    "output_dimensionality": 768,
                },
            },
            {
                "key": "request-1#1/2",
                "request": {
                    "content": {"parts": [{"text": "second"}]},
                    "output_dimensionality": 768,
                },
            },
        ]

    def test_should_keep_the_bare_custom_id_for_single_element_arrays(self):
        (row,) = _wrap_entries([_embeddings_entry(body={"model": "gemini-embedding-2", "input": ["only one"]})])

        assert row["key"] == "request-1"

    def test_should_encode_a_custom_id_that_looks_like_a_fan_out_tag(self):
        """A customer custom_id ending in `#<i>/<n>` must not read back as fan-out metadata."""
        (row,) = _wrap_entries(
            [
                _embeddings_entry(
                    custom_id="request-1#0/2",
                    body={"model": "gemini-embedding-2", "input": "hello world"},
                )
            ]
        )

        assert row["key"] == "request-1%230%2F2"

    def test_should_combine_a_nested_input_into_one_multipart_row(self):
        """Nested arrays are the combined-embedding shape, as on the online path."""
        (row,) = _wrap_entries(
            [
                _embeddings_entry(
                    body={
                        "model": "gemini-embedding-2",
                        "input": [
                            [
                                "a caption",
                                "gs://cloud-samples-data/generative-ai/image/benchmark.jpeg",
                            ]
                        ],
                    }
                )
            ]
        )

        assert row["key"] == "request-1"
        assert row["request"]["content"]["parts"] == [
            {"text": "a caption"},
            {
                "file_data": {
                    "mime_type": "image/jpeg",
                    "file_uri": "gs://cloud-samples-data/generative-ai/image/benchmark.jpeg",
                }
            },
        ]

    def test_should_keep_chat_completions_lines_on_generate_content_path(self):
        (row,) = _wrap_entries(
            [
                {
                    "custom_id": "request-1",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gemini-2.0-flash-001",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                }
            ]
        )

        assert row["request"]["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
        assert row["request"]["labels"]["litellm_custom_id"] == "request-1"
        assert "key" not in row

    def test_should_keep_lines_without_a_url_on_generate_content_path(self):
        """`url` is optional on a batch line, and chat is the shape LiteLLM has always assumed."""
        (row,) = _wrap_entries(
            [
                {
                    "custom_id": "request-1",
                    "body": {
                        "model": "gemini-2.0-flash-001",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                }
            ]
        )

        assert row["request"]["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]

    def test_should_translate_each_line_by_its_own_url(self):
        chat_row, embeddings_row = _wrap_entries(
            [
                {
                    "custom_id": "chat-1",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gemini-2.0-flash-001",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                },
                _embeddings_entry(custom_id="embed-1"),
            ]
        )

        assert "contents" in chat_row["request"]
        assert "content" in embeddings_row["request"]


class TestVertexEmbeddingsBatchOutputTranslation:
    """Vertex Gemini Embedding batch output rows must come back as OpenAI batch rows."""

    def _vertex_embeddings_output_row(self, **overrides):
        row = {
            "key": "request-1",
            "request": {"content": {"parts": [{"text": "hello world"}]}},
            "response": {
                "embedding": {"values": [-0.015, 0.024]},
                "usageMetadata": {"promptTokenCount": 2},
            },
        }
        row.update(overrides)
        return row

    def _transform(self, config, rows, url="https://example.com"):
        content = "\n".join(json.dumps(row) for row in rows).encode("utf-8")
        result = config.transform_file_content_response(
            raw_response=httpx.Response(
                status_code=200,
                content=content,
                headers={"content-type": "application/octet-stream"},
                request=httpx.Request("GET", url),
            ),
            logging_obj=MagicMock(),
            litellm_params={},
        )
        return [json.loads(line) for line in result.response.content.decode("utf-8").split("\n")]

    def test_should_transform_embeddings_output_to_openai_batch_row(self, config):
        (result,) = self._transform(config, [self._vertex_embeddings_output_row()])

        assert result["custom_id"] == "request-1"
        assert result["error"] is None
        assert result["response"]["status_code"] == 200
        body = result["response"]["body"]
        assert body["object"] == "list"
        assert body["data"] == [{"embedding": [-0.015, 0.024], "index": 0, "object": "embedding"}]
        assert body["usage"]["prompt_tokens"] == 2
        assert body["usage"]["total_tokens"] == 2

    def test_should_fall_back_to_documented_token_count_field(self, config):
        (result,) = self._transform(
            config,
            [
                self._vertex_embeddings_output_row(
                    response={
                        "embedding": {"values": [-0.015, 0.024]},
                        "tokenCount": "2",
                    }
                )
            ],
        )

        assert result["response"]["body"]["usage"]["prompt_tokens"] == 2

    def test_should_resolve_model_from_managed_gcs_object_path(self, config):
        object_path = urllib.parse.quote(
            "litellm-vertex-files/publishers/google/models/gemini-embedding-2/"
            "prediction-model-2026-07-29T05:55:52Z/predictions.jsonl",
            safe="",
        )
        url = f"https://storage.googleapis.com/storage/v1/b/my-bucket/o/{object_path}?alt=media"

        (result,) = self._transform(config, [self._vertex_embeddings_output_row()], url=url)

        assert result["response"]["body"]["model"] == "gemini-embedding-2"

    def test_should_surface_failed_embeddings_row_as_error(self, config):
        (result,) = self._transform(
            config,
            [self._vertex_embeddings_output_row(status="Failed to parse JSON into proto", response={})],
        )

        assert result["custom_id"] == "request-1"
        assert result["response"] is None
        assert result["error"]["code"] == "vertex_ai_error"
        assert "Failed to parse JSON into proto" in result["error"]["message"]

    def test_should_transform_every_row_of_a_multi_row_file(self, config):
        results = self._transform(
            config,
            [self._vertex_embeddings_output_row(key=f"request-{index}") for index in range(3)],
        )

        assert [result["custom_id"] for result in results] == [
            "request-0",
            "request-1",
            "request-2",
        ]

    def test_should_reassemble_a_fanned_out_input_array_into_one_row(self, config):
        """Vertex returns the rows of one entry in arbitrary order."""
        (result,) = self._transform(
            config,
            [
                self._vertex_embeddings_output_row(
                    key="request-1#1/2",
                    response={
                        "embedding": {"values": [0.3, 0.4]},
                        "usageMetadata": {"promptTokenCount": 5},
                    },
                ),
                self._vertex_embeddings_output_row(
                    key="request-1#0/2",
                    response={
                        "embedding": {"values": [0.1, 0.2]},
                        "usageMetadata": {"promptTokenCount": 3},
                    },
                ),
            ],
        )

        assert result["custom_id"] == "request-1"
        assert result["response"]["body"]["data"] == [
            {"embedding": [0.1, 0.2], "index": 0, "object": "embedding"},
            {"embedding": [0.3, 0.4], "index": 1, "object": "embedding"},
        ]
        assert result["response"]["body"]["usage"]["prompt_tokens"] == 8

    def test_should_keep_fanned_out_entries_apart_and_in_file_order(self, config):
        results = self._transform(
            config,
            [
                self._vertex_embeddings_output_row(key="request-2#0/2"),
                self._vertex_embeddings_output_row(key="request-1"),
                self._vertex_embeddings_output_row(key="request-2#1/2"),
            ],
        )

        assert [result["custom_id"] for result in results] == ["request-2", "request-1"]
        assert len(results[0]["response"]["body"]["data"]) == 2
        assert len(results[1]["response"]["body"]["data"]) == 1

    def test_should_not_merge_an_entry_whose_custom_id_looks_like_a_fan_out_tag(self, config):
        """`request-1#0/2` is a legal custom_id, and a distinct entry from `request-1`."""
        lookalike_row, plain_row = _wrap_entries(
            [
                _embeddings_entry(
                    custom_id="request-1#0/2",
                    body={"model": "gemini-embedding-2", "input": "lookalike"},
                ),
                _embeddings_entry(
                    custom_id="request-1",
                    body={"model": "gemini-embedding-2", "input": "plain"},
                ),
            ]
        )

        results = self._transform(
            config,
            [
                {**row, "status": "", "response": {"embedding": {"values": values}}}
                for row, values in ((lookalike_row, [0.1]), (plain_row, [0.2]))
            ],
        )

        assert [result["custom_id"] for result in results] == [
            "request-1#0/2",
            "request-1",
        ]
        assert [result["response"]["body"]["data"][0]["embedding"] for result in results] == [[0.1], [0.2]]

    def test_should_round_trip_a_fan_out_of_a_custom_id_holding_the_separator(self, config):
        rows = _wrap_entries(
            [
                _embeddings_entry(
                    custom_id="request#1/1",
                    body={
                        "model": "gemini-embedding-2",
                        "input": ["first", "second"],
                    },
                )
            ]
        )

        assert [row["key"] for row in rows] == [
            "request%231%2F1#0/2",
            "request%231%2F1#1/2",
        ]

        (result,) = self._transform(
            config,
            [
                {**row, "status": "", "response": {"embedding": {"values": values}}}
                for row, values in zip(reversed(rows), ([0.3], [0.1]))
            ],
        )

        assert result["custom_id"] == "request#1/1"
        assert [embedding["embedding"] for embedding in result["response"]["body"]["data"]] == [[0.1], [0.3]]

    def test_should_fail_the_whole_entry_when_one_of_its_rows_failed(self, config):
        """An OpenAI batch row is either a response or an error, never both."""
        (result,) = self._transform(
            config,
            [
                self._vertex_embeddings_output_row(key="request-1#0/2"),
                self._vertex_embeddings_output_row(key="request-1#1/2", status="Quota exceeded", response={}),
            ],
        )

        assert result["custom_id"] == "request-1"
        assert result["response"] is None
        assert result["error"]["message"] == "Quota exceeded"

    def test_should_fail_the_whole_entry_when_a_fanned_out_row_is_missing(self, config):
        """A partial `data` array would shift embeddings onto the wrong input positions."""
        (result,) = self._transform(
            config,
            [self._vertex_embeddings_output_row(key="request-1#1/2")],
        )

        assert result["custom_id"] == "request-1"
        assert result["response"] is None
        assert result["error"]["code"] == "vertex_ai_error"
        assert result["error"]["message"] == ("Vertex returned embeddings for input positions [1] of the 2 requested")

    def test_should_fail_the_whole_entry_when_a_fanned_out_row_is_duplicated(self, config):
        (result,) = self._transform(
            config,
            [
                self._vertex_embeddings_output_row(key="request-1#0/2"),
                self._vertex_embeddings_output_row(key="request-1#0/2"),
            ],
        )

        assert result["custom_id"] == "request-1"
        assert result["response"] is None
        assert result["error"]["code"] == "vertex_ai_error"
        assert result["error"]["message"] == (
            "Vertex returned embeddings for input positions [0, 0] of the 2 requested"
        )

    def test_should_end_to_end_round_trip_a_fanned_out_embeddings_batch(self, config):
        first_row, second_row = _wrap_entries(
            [
                _embeddings_entry(
                    custom_id="MyRequest-1",
                    body={
                        "model": "gemini-embedding-2",
                        "input": ["hello world", "goodbye world"],
                    },
                )
            ]
        )

        (result,) = self._transform(
            config,
            [
                {
                    **row,
                    "status": "",
                    "response": {"embedding": {"values": values}},
                }
                for row, values in ((second_row, [0.3]), (first_row, [0.1]))
            ],
        )

        assert result["custom_id"] == "MyRequest-1"
        assert [embedding["embedding"] for embedding in result["response"]["body"]["data"]] == [[0.1], [0.3]]

    def test_should_end_to_end_round_trip_openai_embeddings_batch(self, config):
        (vertex_row,) = _wrap_entries(
            [
                _embeddings_entry(
                    custom_id="MyRequest-1",
                    body={
                        "model": "gemini-embedding-2",
                        "input": "hello world",
                        "dimensions": 2,
                    },
                )
            ]
        )

        (result,) = self._transform(
            config,
            [
                {
                    **vertex_row,
                    "status": "",
                    "processed_time": "2026-07-29T05:55:52.379528Z",
                    "response": {
                        "embedding": {"values": [-0.015, 0.024]},
                        "usageMetadata": {"promptTokenCount": 2},
                    },
                }
            ],
        )

        assert result["custom_id"] == "MyRequest-1"
        assert result["response"]["body"]["data"][0]["embedding"] == [-0.015, 0.024]

    def test_should_leave_legacy_predict_embeddings_output_untouched(self, config):
        legacy_row = {
            "instance": {"content": "hello world"},
            "predictions": [
                {
                    "embeddings": {
                        "statistics": {"token_count": 2, "truncated": False},
                        "values": [0.2],
                    }
                }
            ],
            "status": "",
        }
        content = json.dumps(legacy_row).encode("utf-8")

        assert config._try_transform_vertex_batch_output_to_openai(content) == content
