import io
import json
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.llms.vertex_ai.files.transformation import VertexAIFilesConfig, is_passthrough_managed_gcs_url

NATIVE_VERTEX_ROW = json.dumps(
    {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": "What is the tallest building in the world?"}]}],
            "tools": [{"googleSearch": {"excludeDomains": ["example.com"]}}],
        }
    }
).encode()
NATIVE_VERTEX_JSONL = NATIVE_VERTEX_ROW + b"\n" + NATIVE_VERTEX_ROW + b"\n"
OPENAI_BATCH_JSONL = (
    b'{"custom_id": "r1", "method": "POST", "url": "/v1/chat/completions",'
    b' "body": {"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]}}\n'
)
PASSTHROUGH_OBJECT = (
    "litellm-vertex-files/passthrough/publishers/google/models/gemini-2.5-flash/uuid-1/predictions.jsonl"
)
TRANSFORMED_OBJECT = "litellm-vertex-files/publishers/google/models/gemini-2.5-flash/uuid-1/predictions.jsonl"
UPLOAD_CHUNK_BYTES = 1024 * 1024


@pytest.fixture
def config() -> VertexAIFilesConfig:
    return VertexAIFilesConfig()


def _gcs_media_url(object_name: str) -> str:
    return (
        f"https://storage.googleapis.com/storage/v1/b/my-bucket/o/{urllib.parse.quote(object_name, safe='')}?alt=media"
    )


def _native_output_jsonl() -> bytes:
    return (
        json.dumps(
            {
                "request": json.loads(NATIVE_VERTEX_ROW)["request"],
                "status": "",
                "response": {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "The Burj Khalifa."}]},
                            "finishReason": "STOP",
                            "groundingMetadata": {"webSearchQueries": ["tallest building in the world"]},
                        }
                    ],
                    "modelVersion": "gemini-2.5-flash",
                    "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 48, "totalTokenCount": 68},
                },
                "processed_time": "2026-09-23T19:02:00.000+00:00",
            }
        ).encode()
        + b"\n"
    )


def _upload_chunks(config: VertexAIFilesConfig, file: object, litellm_params: dict) -> list[bytes]:
    body = config.transform_create_file_request(
        model="",
        create_file_data={"file": file, "purpose": "batch"},
        optional_params={},
        litellm_params=litellm_params,
    )
    return list(body["streaming_media_upload"]["body_stream"].iter_bytes())


def _upload_body_bytes(config: VertexAIFilesConfig, file: object, litellm_params: dict) -> bytes:
    return b"".join(_upload_chunks(config, file, litellm_params))


class TestPassthroughBatchUpload:
    """`passthrough=True` on a batch upload ships the caller's native Vertex JSONL
    to GCS byte for byte, filed under a `passthrough/` object path so the batch
    output that lands beside it is recognized and returned untouched as well."""

    def _upload_url(self, config, litellm_params, file, purpose="batch") -> str:
        return config.get_complete_file_url(
            api_base=None,
            api_key=None,
            model="",
            optional_params={},
            litellm_params=litellm_params,
            data={"file": file, "purpose": purpose},
        )

    def test_passthrough_object_is_filed_under_passthrough_prefix_named_by_deployment_model(self, config):
        url = self._upload_url(
            config,
            {"gcs_bucket_name": "my-bucket", "model": "vertex_ai/gemini-2.5-flash", "passthrough": True},
            ("batch.jsonl", NATIVE_VERTEX_JSONL, "application/jsonl"),
        )
        object_name = parse_qs(urlparse(url).query)["name"][0]
        assert object_name.startswith("litellm-vertex-files/passthrough/publishers/google/models/gemini-2.5-flash/")

    def test_passthrough_upload_without_deployment_model_is_rejected(self, config):
        with pytest.raises(VertexAIError) as exc_info:
            self._upload_url(
                config,
                {"gcs_bucket_name": "my-bucket", "passthrough": True},
                ("batch.jsonl", NATIVE_VERTEX_JSONL, "application/jsonl"),
            )
        assert exc_info.value.status_code == 400
        assert "target_model_names" in exc_info.value.message

    def test_passthrough_flag_does_not_ship_a_non_batch_upload_raw(self, config):
        result = config.transform_create_file_request(
            model="",
            create_file_data={"file": ("notes.txt", b"plain text", "text/plain"), "purpose": "user_data"},
            optional_params={},
            litellm_params={"gcs_bucket_name": "my-bucket", "passthrough": True},
        )
        assert result == b"plain text"

    def test_passthrough_flag_is_ignored_for_non_batch_purposes(self, config):
        url = self._upload_url(
            config,
            {"gcs_bucket_name": "my-bucket", "model": "vertex_ai/gemini-2.5-flash", "passthrough": True},
            ("notes.txt", b"plain text", "text/plain"),
            purpose="user_data",
        )
        object_name = parse_qs(urlparse(url).query)["name"][0]
        assert object_name.startswith("litellm-vertex-files/uploads/")
        assert "passthrough" not in object_name

    @pytest.mark.parametrize(
        "file",
        [
            ("batch.jsonl", NATIVE_VERTEX_JSONL, "application/jsonl"),
            NATIVE_VERTEX_JSONL,
            ("batch.jsonl", io.BytesIO(NATIVE_VERTEX_JSONL), "application/jsonl"),
            ("batch.jsonl", NATIVE_VERTEX_JSONL.decode(), "application/jsonl"),
        ],
        ids=["bytes-tuple", "bare-bytes", "handle-tuple", "text-tuple"],
    )
    def test_passthrough_upload_body_is_the_callers_bytes(self, config, file):
        body = config.transform_create_file_request(
            model="",
            create_file_data={"file": file, "purpose": "batch"},
            optional_params={},
            litellm_params={"passthrough": True},
        )
        stream = body["streaming_media_upload"]["body_stream"]
        assert b"".join(stream.iter_bytes()) == NATIVE_VERTEX_JSONL
        assert b"".join(stream.iter_bytes()) == NATIVE_VERTEX_JSONL
        assert body["streaming_media_upload"]["content_type"] == "application/json"

    def test_passthrough_upload_streams_a_large_handle_in_bounded_chunks(self, config):
        content = NATIVE_VERTEX_ROW * (3 * UPLOAD_CHUNK_BYTES // len(NATIVE_VERTEX_ROW) + 1)
        chunks = _upload_chunks(
            config, ("batch.jsonl", io.BytesIO(content), "application/jsonl"), {"passthrough": True}
        )
        assert len(chunks) >= 3
        assert max(len(chunk) for chunk in chunks) <= UPLOAD_CHUNK_BYTES
        assert b"".join(chunks) == content

    def test_passthrough_upload_streams_a_path_in_bounded_chunks(self, config, tmp_path: Path):
        content = NATIVE_VERTEX_ROW * (2 * UPLOAD_CHUNK_BYTES // len(NATIVE_VERTEX_ROW) + 1)
        batch_path = tmp_path / "batch.jsonl"
        batch_path.write_bytes(content)
        chunks = _upload_chunks(config, ("batch.jsonl", batch_path, "application/jsonl"), {"passthrough": True})
        assert len(chunks) >= 2
        assert max(len(chunk) for chunk in chunks) <= UPLOAD_CHUNK_BYTES
        assert b"".join(chunks) == content

    def test_passthrough_upload_rejects_a_non_seekable_handle(self, config):
        class _Pipe:
            def read(self, size=-1):
                return b""

        with pytest.raises(ValueError, match="seekable"):
            _upload_body_bytes(config, ("batch.jsonl", _Pipe(), "application/jsonl"), {"passthrough": True})

    def test_passthrough_upload_rejects_content_that_is_neither_bytes_path_nor_handle(self, config):
        with pytest.raises(ValueError, match="Unsupported file content type"):
            _upload_body_bytes(config, ("batch.jsonl", 42, "application/jsonl"), {"passthrough": True})

    def test_openai_rows_are_translated_unless_passthrough_is_set(self, config):
        file = ("batch.jsonl", OPENAI_BATCH_JSONL, "application/jsonl")
        translated = _upload_body_bytes(config, file, {})
        untouched = _upload_body_bytes(config, file, {"passthrough": True})
        assert untouched == OPENAI_BATCH_JSONL
        assert translated != OPENAI_BATCH_JSONL
        assert b'"contents"' in translated

    def test_passthrough_output_content_is_returned_untouched(self, config):
        raw_jsonl = _native_output_jsonl()

        def _download(object_name: str) -> bytes:
            raw_response = httpx.Response(
                status_code=200,
                content=raw_jsonl,
                headers={"content-type": "application/octet-stream"},
                request=httpx.Request("GET", _gcs_media_url(object_name)),
            )
            result = config.transform_file_content_response(
                raw_response=raw_response, logging_obj=MagicMock(), litellm_params={}
            )
            return result.response.content

        assert _download(PASSTHROUGH_OBJECT) == raw_jsonl
        assert _download(f"team-a/{PASSTHROUGH_OBJECT}") == raw_jsonl
        transformed = _download(TRANSFORMED_OBJECT)
        assert transformed != raw_jsonl
        assert json.loads(transformed.splitlines()[0])["response"]["body"]["choices"]
        nested = _download(f"litellm-vertex-files/{PASSTHROUGH_OBJECT}")
        assert nested != raw_jsonl
        assert json.loads(nested.splitlines()[0])["response"]["body"]["choices"]

    def test_output_of_an_upload_whose_model_smuggles_the_passthrough_segment_is_still_transformed(self, config):
        smuggled_model = b"litellm-vertex-files/passthrough/publishers/google/models/gemini-2.5-flash"
        upload_url = self._upload_url(
            config,
            {"gcs_bucket_name": "my-bucket"},
            ("batch.jsonl", OPENAI_BATCH_JSONL.replace(b"gemini-2.5-flash", smuggled_model), "application/jsonl"),
        )
        object_name = parse_qs(urlparse(upload_url).query)["name"][0]
        raw_jsonl = _native_output_jsonl()
        raw_response = httpx.Response(
            status_code=200,
            content=raw_jsonl,
            headers={"content-type": "application/octet-stream"},
            request=httpx.Request("GET", _gcs_media_url(f"{object_name}/predictions.jsonl")),
        )

        result = config.transform_file_content_response(
            raw_response=raw_response, logging_obj=MagicMock(), litellm_params={}
        )

        assert object_name.startswith("litellm-vertex-files/litellm-vertex-files/passthrough/")
        assert json.loads(result.response.content.splitlines()[0])["response"]["body"]["choices"]

    @pytest.mark.parametrize(
        "url, expected",
        [
            (f"gs://my-bucket/{PASSTHROUGH_OBJECT}", True),
            (f"gs://my-bucket/team-a/{PASSTHROUGH_OBJECT}", True),
            (f"gs://my-bucket/litellm-vertex-files/{PASSTHROUGH_OBJECT}", False),
            (_gcs_media_url(f"team-a/sub/{PASSTHROUGH_OBJECT}"), True),
            (_gcs_media_url(f"litellm-vertex-files/publishers/google/models/x/{PASSTHROUGH_OBJECT}"), False),
            (_gcs_media_url(TRANSFORMED_OBJECT), False),
        ],
        ids=["gs", "gs-prefixed", "gs-smuggled", "https-prefixed", "https-model-path-smuggled", "https-transformed"],
    )
    def test_passthrough_detection_anchors_on_the_first_managed_segment(self, url, expected):
        assert is_passthrough_managed_gcs_url(url) is expected

    @pytest.mark.asyncio
    async def test_passthrough_output_stream_is_returned_untouched(self, config):
        stream_iterator = object()
        headers = {"content-type": "application/octet-stream"}
        result = await config.transform_file_content_stream(
            stream_iterator=stream_iterator,
            headers=headers,
            request_url=_gcs_media_url(f"team-a/{PASSTHROUGH_OBJECT}"),
            logging_obj=MagicMock(),
            litellm_params={},
        )
        assert result.stream_iterator is stream_iterator
        assert result.headers == headers


class TestEmbeddingOutputTranslation:
    EMBEDDING_OBJECT = (
        "litellm-vertex-files/publishers/google/models/gemini-embedding-2/prediction-model-1/predictions.jsonl"
    )

    def _transform(self, config: VertexAIFilesConfig, rows: list[dict]) -> list[dict]:
        raw_response = httpx.Response(
            status_code=200,
            content="\n".join(json.dumps(row) for row in rows).encode(),
            headers={"content-type": "application/octet-stream"},
            request=httpx.Request("GET", _gcs_media_url(self.EMBEDDING_OBJECT)),
        )
        result = config.transform_file_content_response(
            raw_response=raw_response, logging_obj=MagicMock(), litellm_params={}
        )
        return [json.loads(line) for line in result.response.content.decode().splitlines()]

    def test_embedding_rows_become_openai_batch_rows_billed_by_their_prompt_tokens(self, config):
        live_row = {
            "key": "request-1",
            "request": {"content": {"parts": [{"text": "hello world"}]}},
            "response": {"embedding": {"values": [-0.015, 0.024]}, "usageMetadata": {"promptTokenCount": 2}},
        }
        documented_row = {
            "key": "request-2",
            "request": {"content": {"parts": [{"text": "hello"}]}},
            "response": {"embedding": {"values": [0.5]}, "tokenCount": "3"},
        }

        live, documented = self._transform(config, [live_row, documented_row])

        assert (live["custom_id"], live["error"], live["response"]["status_code"]) == ("request-1", None, 200)
        assert live["response"]["body"]["model"] == "gemini-embedding-2"
        assert live["response"]["body"]["data"] == [{"embedding": [-0.015, 0.024], "index": 0, "object": "embedding"}]
        live_usage, documented_usage = (row["response"]["body"]["usage"] for row in (live, documented))
        assert (live_usage["prompt_tokens"], live_usage["total_tokens"]) == (2, 2)
        assert (documented_usage["prompt_tokens"], documented_usage["total_tokens"]) == (3, 3)
