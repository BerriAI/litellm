"""
Test Google AI Studio (Gemini) files transformation functionality
"""

import json
import re
from unittest.mock import Mock, patch

import httpx
import pytest

from litellm.batches.batch_utils import batch_cost_and_usage
from litellm.llms.gemini.common_utils import GeminiError
from litellm.llms.gemini.files.transformation import GoogleAIStudioFilesHandler
from litellm.types.llms.openai import OpenAIFileObject


class TestGoogleAIStudioFilesTransformation:
    """Test Google AI Studio files transformation"""

    def setup_method(self):
        """Setup test method"""
        self.handler = GoogleAIStudioFilesHandler()

    def test_transform_retrieve_file_request_with_full_uri(self):
        """
        Test that transform_retrieve_file_request returns empty params dict
        to avoid 'Content-Type' query parameter error

        Regression test for: https://github.com/BerriAI/litellm/issues/XXX
        When retrieving a file, the API was incorrectly trying to pass Content-Type
        as a query parameter, which Gemini API rejected.
        """
        file_id = "https://generativelanguage.googleapis.com/v1beta/files/test123"
        litellm_params = {"api_key": "test-api-key"}

        url, params = self.handler.transform_retrieve_file_request(
            file_id=file_id,
            optional_params={},
            litellm_params=litellm_params,
        )

        # API key is passed via x-goog-api-key header, not in URL
        assert url == "https://generativelanguage.googleapis.com/v1beta/files/test123"
        assert "key=" not in url

        # CRITICAL: params should be empty dict, not contain Content-Type or any other params
        # These would be incorrectly interpreted as query parameters
        assert params == {}, f"Expected empty params dict, got: {params}"
        assert "Content-Type" not in params, "Content-Type should not be in query params"

    def test_transform_retrieve_file_request_with_file_name_only(self):
        """
        Test that transform_retrieve_file_request handles file_id without full URI
        """
        file_id = "files/test123"
        litellm_params = {"api_key": "test-api-key"}

        url, params = self.handler.transform_retrieve_file_request(
            file_id=file_id,
            optional_params={},
            litellm_params=litellm_params,
        )

        # API key is passed via x-goog-api-key header, not in URL
        assert url == "https://generativelanguage.googleapis.com/v1beta/files/test123"
        assert "key=" not in url

        # CRITICAL: params should be empty dict
        assert params == {}, f"Expected empty params dict, got: {params}"
        assert "Content-Type" not in params, "Content-Type should not be in query params"

    def test_transform_retrieve_file_request_with_raw_id_only(self):
        """
        Regression guard: API key must NOT appear in the URL.

        The key is sent via x-goog-api-key header to prevent leaking
        credentials in httpx error tracebacks.
        """
        file_id = "cctqueckiggb"
        litellm_params = {"api_key": "test-api-key"}

        url, params = self.handler.transform_retrieve_file_request(
            file_id=file_id,
            optional_params={},
            litellm_params=litellm_params,
        )

        assert url == "https://generativelanguage.googleapis.com/v1beta/files/cctqueckiggb"
        assert "key=" not in url
        assert params == {}

    def test_transform_retrieve_file_request_encodes_file_id_path_segment(self):
        file_id = "files/../../models/gemini-pro?x=1#frag"
        litellm_params = {"api_key": "test-api-key"}

        url, params = self.handler.transform_retrieve_file_request(
            file_id=file_id,
            optional_params={},
            litellm_params=litellm_params,
        )

        assert (
            url == "https://generativelanguage.googleapis.com/v1beta/files/..%2F..%2Fmodels%2Fgemini-pro%3Fx%3D1%23frag"
        )
        assert params == {}

    def test_transform_retrieve_file_request_rejects_dot_path_segment(self):
        with pytest.raises(ValueError, match="file_id cannot be a dot path segment"):
            self.handler.transform_retrieve_file_request(
                file_id="files/..",
                optional_params={},
                litellm_params={"api_key": "test-api-key"},
            )

    @patch.dict("os.environ", {}, clear=True)
    @patch("litellm.llms.gemini.common_utils.get_secret_str", return_value=None)
    def test_transform_retrieve_file_request_missing_api_key(self, mock_get_secret):
        """Test that transform_retrieve_file_request raises error when API key is missing"""
        file_id = "files/test123"
        litellm_params = {}

        with pytest.raises(ValueError, match="api_key is required"):
            self.handler.transform_retrieve_file_request(
                file_id=file_id,
                optional_params={},
                litellm_params=litellm_params,
            )

    def test_transform_retrieve_file_response_success(self):
        """Test successful transformation of Gemini file retrieval response"""
        # Mock response data from Gemini API
        mock_response_data = {
            "name": "files/test123",
            "displayName": "test_file.pdf",
            "mimeType": "application/pdf",
            "sizeBytes": "1024",
            "createTime": "2024-01-15T10:30:00.123456Z",
            "updateTime": "2024-01-15T10:30:00.123456Z",
            "expirationTime": "2024-01-17T10:30:00.123456Z",
            "sha256Hash": "abcd1234",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/test123",
            "state": "ACTIVE",
        }

        # Create mock httpx response
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data

        # Create mock logging object
        mock_logging_obj = Mock()

        # Transform response
        result = self.handler.transform_retrieve_file_response(
            raw_response=mock_response,
            logging_obj=mock_logging_obj,
            litellm_params={},
        )

        # Verify transformation
        assert isinstance(result, OpenAIFileObject)
        assert result.id == mock_response_data["uri"]
        assert result.filename == mock_response_data["displayName"]
        assert result.bytes == int(mock_response_data["sizeBytes"])
        assert result.object == "file"
        assert result.purpose == "user_data"
        assert result.status == "processed"  # ACTIVE state maps to processed
        assert result.status_details is None

    def test_transform_retrieve_file_response_failed_state(self):
        """Test transformation of Gemini file retrieval response with FAILED state"""
        mock_response_data = {
            "name": "files/test123",
            "displayName": "test_file.pdf",
            "mimeType": "application/pdf",
            "sizeBytes": "1024",
            "createTime": "2024-01-15T10:30:00.123456Z",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/test123",
            "state": "FAILED",
            "error": {"message": "Upload failed", "code": "INTERNAL"},
        }

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_logging_obj = Mock()

        result = self.handler.transform_retrieve_file_response(
            raw_response=mock_response,
            logging_obj=mock_logging_obj,
            litellm_params={},
        )

        # Verify error state handling
        assert result.status == "error"
        assert result.status_details is not None
        assert "message" in result.status_details

    def test_transform_retrieve_file_response_processing_state(self):
        """Test transformation of Gemini file retrieval response with PROCESSING state"""
        mock_response_data = {
            "name": "files/test123",
            "displayName": "test_file.pdf",
            "mimeType": "application/pdf",
            "sizeBytes": "1024",
            "createTime": "2024-01-15T10:30:00.123456Z",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/test123",
            "state": "PROCESSING",
        }

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_logging_obj = Mock()

        result = self.handler.transform_retrieve_file_response(
            raw_response=mock_response,
            logging_obj=mock_logging_obj,
            litellm_params={},
        )

        # PROCESSING state should map to "uploaded" status
        assert result.status == "uploaded"

    def test_transform_retrieve_file_response_missing_createTime(self):
        """
        Test that transform_retrieve_file_response raises proper error when createTime is missing

        This tests the error scenario that occurs when API returns an error response
        without the expected file metadata fields.
        """
        # Mock error response from Gemini API (missing createTime)
        mock_response_data = {
            "error": {
                "code": 400,
                "message": "Invalid request",
                "status": "INVALID_ARGUMENT",
            }
        }

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = mock_response_data
        mock_logging_obj = Mock()

        # Should raise ValueError with helpful message
        with pytest.raises(ValueError, match="Error parsing file retrieve response"):
            self.handler.transform_retrieve_file_response(
                raw_response=mock_response,
                logging_obj=mock_logging_obj,
                litellm_params={},
            )

    def test_validate_environment(self):
        """Test that validate_environment properly adds API key to headers"""
        headers = {}
        api_key = "test-gemini-api-key"

        result_headers = self.handler.validate_environment(
            headers=headers,
            model="gemini-pro",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=api_key,
        )

        # Verify API key is added to headers
        assert "x-goog-api-key" in result_headers
        assert result_headers["x-goog-api-key"] == api_key

    @patch.dict("os.environ", {}, clear=True)
    @patch("litellm.llms.gemini.common_utils.get_secret_str", return_value=None)
    def test_validate_environment_missing_api_key(self, mock_get_secret):
        """Test that validate_environment raises error when API key is missing"""
        headers = {}

        with pytest.raises(
            ValueError,
            match="GEMINI_API_KEY is required for Google AI Studio file operations",
        ):
            self.handler.validate_environment(
                headers=headers,
                model="gemini-pro",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    def test_get_complete_url(self):
        """Test that get_complete_url constructs proper upload URL"""
        api_base = "https://generativelanguage.googleapis.com"
        api_key = "test-api-key"

        url = self.handler.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model="gemini-pro",
            optional_params={},
            litellm_params={},
        )

        # Verify URL structure - API key must NOT be in URL
        assert api_base in url
        assert "upload/v1beta/files" in url
        assert "key=" not in url

    def test_transform_delete_file_request_with_full_uri(self):
        """Test delete file request transformation with full URI"""
        file_id = "https://generativelanguage.googleapis.com/v1beta/files/test123"
        litellm_params = {
            "api_key": "test-api-key",
            "api_base": "https://generativelanguage.googleapis.com",
        }

        url, params = self.handler.transform_delete_file_request(
            file_id=file_id,
            optional_params={},
            litellm_params=litellm_params,
        )

        assert url == "https://generativelanguage.googleapis.com/v1beta/files/test123"

        # Params should be empty (API key goes in header via validate_environment)
        assert params == {}

    def test_transform_delete_file_request_with_file_name_only(self):
        """Test delete file request transformation with file name only"""
        file_id = "files/test123"
        litellm_params = {
            "api_key": "test-api-key",
            "api_base": "https://generativelanguage.googleapis.com",
        }

        url, params = self.handler.transform_delete_file_request(
            file_id=file_id,
            optional_params={},
            litellm_params=litellm_params,
        )

        # Verify URL construction
        assert file_id in url
        assert "generativelanguage.googleapis.com" in url
        assert params == {}

    def test_transform_delete_file_request_encodes_file_id_path_segment(self):
        file_id = "files/../../models/gemini-pro?x=1#frag"
        litellm_params = {
            "api_key": "test-api-key",
            "api_base": "https://generativelanguage.googleapis.com",
        }

        url, params = self.handler.transform_delete_file_request(
            file_id=file_id,
            optional_params={},
            litellm_params=litellm_params,
        )

        assert (
            url == "https://generativelanguage.googleapis.com/v1beta/files/..%2F..%2Fmodels%2Fgemini-pro%3Fx%3D1%23frag"
        )
        assert params == {}


@pytest.mark.parametrize(
    "file_id",
    [
        "batch-la6",
        "files/batch-la6",
        "https://generativelanguage.googleapis.com/v1beta/files/batch-la6",
        "https://example.invalid/batch-la6",
        "http://example.invalid/batch-la6?alt=media#download",
    ],
)
def test_file_content_request_downloads_the_file_as_media(file_id):
    handler = GoogleAIStudioFilesHandler()
    url, params = handler.transform_file_content_request(
        file_content_request={"file_id": file_id},
        optional_params={},
        litellm_params={"api_base": "https://proxy.example"},
    )
    assert (url, params) == ("https://proxy.example/download/v1beta/files/batch-la6:download", {"alt": "media"})


def test_file_content_response_passes_the_jsonl_bytes_through():
    handler = GoogleAIStudioFilesHandler()
    body = b'{"custom_id":"r0","response":{"status_code":200,"body":{"object":"chat.completion"}}}\n'
    raw = httpx.Response(200, content=body, request=httpx.Request("GET", "https://generativelanguage.googleapis.com/x"))
    assert (
        handler.transform_file_content_response(raw_response=raw, logging_obj=Mock(), litellm_params={}).content == body
    )


_CAPITAL_SCHEMA = {
    "type": "object",
    "properties": {"capital": {"type": "string"}},
    "required": ["capital"],
    "additionalProperties": False,
}


def _chat_row(custom_id: str, **body) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Capital of France?"}],
            **body,
        },
    }


def _upload_data(lines: list[bytes], purpose: str = "batch", litellm_params: dict | None = None) -> bytes:
    request = GoogleAIStudioFilesHandler().transform_create_file_request(
        model="",
        create_file_data={"file": ("batch.jsonl", b"\n".join(lines), "application/jsonl"), "purpose": purpose},
        optional_params={},
        litellm_params=litellm_params or {},
    )
    data = request["upload_request"]["data"]
    assert request["initial_request"]["headers"]["X-Goog-Upload-Header-Content-Length"] == str(len(data))
    return data


def _uploaded_rows(rows: list[dict], litellm_params: dict | None = None) -> list[dict]:
    data = _upload_data([json.dumps(row).encode() for row in rows], litellm_params=litellm_params)
    return [json.loads(line) for line in data.splitlines()]


def test_batch_upload_turns_an_openai_chat_row_into_a_gemini_row():
    row = _chat_row(
        "row-1",
        messages=[
            {"role": "developer", "content": "Answer in JSON."},
            {"role": "user", "content": "Capital of France?"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "c", "strict": True, "schema": _CAPITAL_SCHEMA},
        },
        seed=7,
        metadata={"run": "1"},
    )

    [uploaded] = _uploaded_rows([row])

    request = uploaded["request"]
    assert (uploaded["key"], request["system_instruction"], request["contents"]) == (
        "row-1",
        {"parts": [{"text": "Answer in JSON."}]},
        [{"role": "user", "parts": [{"text": "Capital of France?"}]}],
    )
    assert (request["generationConfig"]["response_json_schema"], request["generationConfig"]["seed"]) == (
        _CAPITAL_SCHEMA,
        7,
    )
    assert "metadata" not in json.dumps(request)


@pytest.mark.parametrize("unsupported", [{"logit_bias": {"1": 2}}, {"prompt_cache_key": "k"}, {"store": False}])
def test_batch_upload_rejects_a_param_gemini_chat_rejects(unsupported):
    with pytest.raises(GeminiError, match=f"line 1 .*{next(iter(unsupported))}"):
        _uploaded_rows([_chat_row("row-1", **unsupported)])


@pytest.mark.parametrize(
    "row_drop_params,deployment_params",
    [(True, {}), (None, {"drop_params": True})],
)
def test_batch_upload_drops_unsupported_params_when_asked(row_drop_params, deployment_params):
    flags = {} if row_drop_params is None else {"drop_params": row_drop_params}
    with_unsupported = _chat_row("row-1", logit_bias={"1": 2}, prompt_cache_key="k", **flags)

    assert _uploaded_rows([with_unsupported], litellm_params=deployment_params) == _uploaded_rows([_chat_row("row-1")])


def test_batch_upload_maps_tools_to_function_declarations():
    parameters = {"type": "object", "properties": {"country": {"type": "string"}}, "required": ["country"]}
    tool = {"type": "function", "function": {"name": "lookup", "parameters": parameters}}

    [uploaded] = _uploaded_rows([_chat_row("row-1", tools=[tool], tool_choice="required")])

    assert (uploaded["request"]["tools"], uploaded["request"]["toolConfig"]) == (
        [{"function_declarations": [{"name": "lookup", "parameters": parameters}]}],
        {"functionCallingConfig": {"mode": "ANY"}},
    )


def test_batch_upload_runs_each_row_on_the_deployment_model():
    on_deployment = _uploaded_rows(
        [_chat_row("row-1", model="gemini-2.5-flash", reasoning_effort="low")],
        litellm_params={"model": "gemini/gemini-3-pro-preview"},
    )
    on_body = _uploaded_rows([_chat_row("row-1", model="gemini-3-pro-preview", reasoning_effort="low")])
    on_other_model = _uploaded_rows([_chat_row("row-1", model="gemini-2.5-flash", reasoning_effort="low")])

    assert on_deployment == on_body
    assert on_deployment != on_other_model


def test_batch_upload_keeps_rows_it_does_not_convert():
    native_row = {"key": "row-1", "request": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}}
    embeddings_row = {"custom_id": "row-2", "method": "POST", "url": "/v1/embeddings", "body": {"input": "hi"}}

    assert _uploaded_rows([native_row, embeddings_row]) == [native_row, embeddings_row]


def test_batch_upload_reports_the_line_that_cannot_be_converted():
    lines = [json.dumps(_chat_row("row-1")).encode(), b"", json.dumps(_chat_row("row-2") | {"body": "text"}).encode()]

    with pytest.raises(GeminiError, match="Batch input file line 3 ") as raised:
        _upload_data(lines)
    assert raised.value.status_code == 400


def test_non_batch_upload_is_sent_unchanged():
    content = json.dumps(_chat_row("row-1")).encode()
    assert _upload_data([content], purpose="user_data") == content


_NATIVE_OUTPUT_ROWS = [
    {
        "key": "row-1",
        "response": {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": '{"capital": "Paris"}'}]},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 27,
                "candidatesTokenCount": 6,
                "thoughtsTokenCount": 89,
                "totalTokenCount": 122,
            },
            "modelVersion": "gemini-2.5-flash",
            "responseId": "resp-1",
        },
    },
    {
        "key": "row-2",
        "response": {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Look it up.", "thought": True},
                            {"functionCall": {"name": "lookup", "args": {"country": "France"}}},
                        ],
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 15,
                "totalTokenCount": 40,
            },
            "modelVersion": "gemini-2.5-flash",
            "responseId": "resp-2",
        },
    },
    {"key": "row-3", "error": {"code": 3, "message": "Request contains an invalid argument."}},
]
_NATIVE_OUTPUT = b"\n".join(json.dumps(row).encode() for row in _NATIVE_OUTPUT_ROWS)


def _downloaded_rows(content: bytes) -> list[dict]:
    raw = httpx.Response(
        200, content=content, request=httpx.Request("GET", "https://generativelanguage.googleapis.com/x")
    )
    response = GoogleAIStudioFilesHandler().transform_file_content_response(
        raw_response=raw, logging_obj=Mock(), litellm_params={}
    )
    assert response.response.headers["content-length"] == str(len(response.content))
    return [json.loads(line) for line in response.content.splitlines()]


def test_file_content_turns_native_output_rows_into_openai_rows():
    text_row, tool_row, failed_row = _downloaded_rows(_NATIVE_OUTPUT)

    text_body = text_row["response"]["body"]
    assert (text_row["custom_id"], text_body["choices"][0]["message"]["content"]) == ("row-1", '{"capital": "Paris"}')
    assert text_body["usage"]["completion_tokens"] == 6 + 89

    tool_message = tool_row["response"]["body"]["choices"][0]["message"]
    [tool_call] = tool_message["tool_calls"]
    assert (tool_row["response"]["body"]["choices"][0]["finish_reason"], tool_message["reasoning_content"]) == (
        "tool_calls",
        "Look it up.",
    )
    assert (tool_call["function"]["name"], json.loads(tool_call["function"]["arguments"])) == (
        "lookup",
        {"country": "France"},
    )

    assert (failed_row["custom_id"], failed_row["response"], failed_row["error"]) == (
        "row-3",
        None,
        {"code": "3", "message": "Request contains an invalid argument."},
    )


def test_converted_output_rows_cost_what_the_native_rows_cost():
    native = batch_cost_and_usage(_NATIVE_OUTPUT_ROWS, "gemini", "gemini-2.5-flash")
    converted = batch_cost_and_usage(_downloaded_rows(_NATIVE_OUTPUT), "gemini", "gemini-2.5-flash")

    assert (converted.cost, converted.usage.total_tokens, converted.failed_requests) == (
        native.cost,
        native.usage.total_tokens,
        native.failed_requests,
    )


def _without_generated_values(content: bytes) -> list[bytes]:
    return re.sub(rb'"(batch_req|call)_[0-9a-f-]+"|"created": \d+', b"<generated>", content).splitlines()


async def _chunks(content: bytes, size: int):
    for offset in range(0, len(content), size):
        yield content[offset : offset + size]


@pytest.mark.parametrize("chunk_size", [1, 37, len(_NATIVE_OUTPUT)])
async def test_streamed_file_content_matches_the_buffered_conversion(chunk_size):
    result = await GoogleAIStudioFilesHandler().transform_file_content_stream(
        stream_iterator=_chunks(_NATIVE_OUTPUT, chunk_size),
        headers={"content-type": "application/jsonl", "content-length": str(len(_NATIVE_OUTPUT))},
        request_url="https://generativelanguage.googleapis.com/x",
        logging_obj=Mock(),
        litellm_params={},
    )
    streamed = b"".join([chunk async for chunk in result.stream_iterator])
    buffered = b"\n".join(json.dumps(row).encode() for row in _downloaded_rows(_NATIVE_OUTPUT))

    assert "content-length" not in result.headers
    assert _without_generated_values(streamed) == _without_generated_values(buffered)


def test_retrieve_file_response_accepts_nanosecond_create_time():
    handler = GoogleAIStudioFilesHandler()
    raw = httpx.Response(
        200,
        json={
            "name": "files/batch-la6",
            "displayName": "Results of batch la6",
            "mimeType": "application/jsonl",
            "sizeBytes": "1024",
            "createTime": "2026-11-10T21:35:00.902340270Z",
            "updateTime": "2026-11-10T21:35:00.902340270Z",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/batch-la6",
            "state": "ACTIVE",
        },
        request=httpx.Request("GET", "https://generativelanguage.googleapis.com/v1beta/files/batch-la6"),
    )
    file_object = handler.transform_retrieve_file_response(raw_response=raw, logging_obj=Mock(), litellm_params={})
    assert (file_object.created_at, file_object.bytes, file_object.status) == (1794346500, 1024, "processed")
