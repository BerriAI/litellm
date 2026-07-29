"""Vertex Gemini: extensionless gs:// MIME + GCS metadata tests.

Split from test_vertex.py to satisfy CI per-file size limits.
"""
import asyncio
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import pytest

import litellm
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../.."))

from litellm.llms.vertex_ai.gemini.transformation import _process_gemini_media


def test_process_gemini_media_gcs_explicit_format_octet_stream_and_alias():
    """Explicit format bypasses registry; image/jpg alias still applies."""
    from litellm.types.llms.vertex_ai import FileDataType

    r1 = _process_gemini_media(
        "gs://bucket/object-no-ext",
        format="application/octet-stream",
    )
    assert r1["file_data"] == FileDataType(
        mime_type="application/octet-stream",
        file_uri="gs://bucket/object-no-ext",
    )
    r2 = _process_gemini_media("gs://bucket/object-no-ext", format="image/jpg")
    assert r2["file_data"] == FileDataType(
        mime_type="image/jpeg",
        file_uri="gs://bucket/object-no-ext",
    )


def test_process_gemini_media_gcs_without_extension_errors_and_metadata_mock():
    with patch(
        "litellm.llms.vertex_ai.gemini.transformation._get_gcs_object_content_type",
        return_value=None,
    ):
        with pytest.raises(litellm.BadRequestError) as exc:
            _process_gemini_media("gs://bucket/image-without-extension")
    assert "Unable to determine mime type for gs URI" in str(exc.value)

    from litellm.types.llms.vertex_ai import FileDataType

    with patch(
        "litellm.llms.vertex_ai.gemini.transformation._get_gcs_object_content_type",
        return_value="image/jpeg",
    ) as m:
        r = _process_gemini_media("gs://bucket/image-without-extension")
    assert r["file_data"] == FileDataType(
        mime_type="image/jpeg", file_uri="gs://bucket/image-without-extension"
    )
    m.assert_called()

    with patch(
        "litellm.llms.vertex_ai.gemini.transformation._get_gcs_object_content_type",
        return_value="image/jpg",
    ):
        r_alias = _process_gemini_media("gs://bucket/image-without-extension")
    assert r_alias["file_data"]["mime_type"] == "image/jpeg"


def test_process_gemini_media_rejects_gcs_metadata_mime_not_supported_by_gemini():
    """Non-empty GCS contentType that fails _normalize_and_validate_gemini_mime_type."""
    with patch(
        "litellm.llms.vertex_ai.gemini.transformation._get_gcs_object_content_type",
        return_value="application/x-litellm-unit-test-unknown-mime",
    ):
        with pytest.raises(
            litellm.BadRequestError,
            match="File type not supported by gemini",
        ):
            _process_gemini_media("gs://bucket/object-without-extension")


def test_file_block_uses_mime_type_alias_for_extensionless_gcs():
    from litellm.llms.vertex_ai.gemini.transformation import (
        _gemini_convert_messages_with_history,
    )
    from litellm.types.llms.vertex_ai import FileDataType

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "file_id": "gs://bucket/no-extension-object",
                        "mime_type": "application/pdf",
                    },
                }
            ],
        }
    ]
    converted = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-2.5-flash"
    )
    assert converted[0]["parts"][0]["file_data"] == FileDataType(
        mime_type="application/pdf", file_uri="gs://bucket/no-extension-object"
    )


@pytest.mark.parametrize(
    "bucket,expected",
    [
        (("a." * 110) + "aa", True),
        ("ab", False),
        ("a" * 64, False),
        ("ab..cd", False),
        ("1.2.3.4", False),
        ("192.168.0.1", False),
        ("Bucket-Upper", False),
        ("bucket@name", False),
        ("bucket name", False),
        ("-mybucket", False),
        ("mybucket-", False),
        (".mybucket", False),
        ("mybucket.", False),
    ],
)
def test_is_valid_gcs_bucket_name_matrix(bucket, expected):
    from litellm.llms.vertex_ai.gemini.transformation import _is_valid_gcs_bucket_name

    assert _is_valid_gcs_bucket_name(bucket) is expected


def test_get_gcs_object_content_type_explicit_vertex_success_and_token_failure():
    from litellm.llms.vertex_ai.gemini import transformation as gt

    mock_v = MagicMock()
    mock_v.get_access_token.return_value = ("test-token", "test-project")
    resp = MagicMock()
    resp.is_error = False
    resp.status_code = 200
    resp.json.return_value = {"contentType": "image/png"}
    http = MagicMock()
    http.get.return_value = resp

    with (
        patch.object(gt, "_GCS_METADATA_VERTEX_BASE", mock_v),
        patch(
            "litellm.llms.vertex_ai.gemini.transformation._get_gcs_metadata_http_handler",
            return_value=http,
        ),
    ):
        assert (
            gt._get_gcs_object_content_type(
                image_url="gs://my-bucket/path/to/image-without-extension",
                vertex_project="project-123",
                vertex_credentials="credential-json",
            )
            == "image/png"
        )
    mock_v.get_access_token.assert_called_once_with(
        credentials="credential-json",
        project_id="project-123",
    )

    mock_v2 = MagicMock()
    mock_v2.get_access_token.side_effect = Exception("token failure")
    with patch.object(gt, "_GCS_METADATA_VERTEX_BASE", mock_v2):
        with pytest.raises(
            litellm.BadRequestError,
            match="Unable to fetch GCS metadata with provided Vertex credentials/project",
        ):
            gt._get_gcs_object_content_type(
                image_url="gs://my-bucket/path/to/image-without-extension",
                vertex_project="project-123",
                vertex_credentials="credential-json",
            )


def test_get_gcs_object_content_type_http_error_explicit_vs_anonymous():
    from litellm.llms.vertex_ai.gemini import transformation as gt

    mock_v = MagicMock()
    mock_v.get_access_token.return_value = ("t", "p")
    err_resp = MagicMock()
    err_resp.is_error = True
    err_resp.status_code = 403
    err_resp.text = '{"error":{"message":"Permission denied"}}'
    http = MagicMock()
    http.get.return_value = err_resp

    with (
        patch.object(gt, "_GCS_METADATA_VERTEX_BASE", mock_v),
        patch(
            "litellm.llms.vertex_ai.gemini.transformation._get_gcs_metadata_http_handler",
            return_value=http,
        ),
    ):
        with pytest.raises(litellm.BadRequestError, match="HTTP 403") as ei:
            gt._get_gcs_object_content_type(
                image_url="gs://my-bucket/path/to/obj",
                vertex_project="project-123",
                vertex_credentials="credential-json",
            )
    assert "Permission denied" in str(ei.value)

    mock_v2 = MagicMock()
    anon_err = MagicMock()
    anon_err.is_error = True
    anon_err.status_code = 403
    anon_err.text = "Forbidden"
    http2 = MagicMock()
    http2.get.return_value = anon_err
    with (
        patch.object(gt, "_GCS_METADATA_VERTEX_BASE", mock_v2),
        patch(
            "litellm.llms.vertex_ai.gemini.transformation._get_gcs_metadata_http_handler",
            return_value=http2,
        ),
    ):
        assert (
            gt._get_gcs_object_content_type(image_url="gs://public-bucket/public-object")
            is None
        )
    mock_v2.get_access_token.assert_not_called()


def test_get_gcs_object_content_type_anonymous_success_no_auth_header():
    from litellm.llms.vertex_ai.gemini import transformation as gt

    mock_v = MagicMock()
    ok = MagicMock()
    ok.is_error = False
    ok.status_code = 200
    ok.json.return_value = {"contentType": "image/jpeg"}
    http = MagicMock()
    http.get.return_value = ok

    with (
        patch.object(gt, "_GCS_METADATA_VERTEX_BASE", mock_v),
        patch(
            "litellm.llms.vertex_ai.gemini.transformation._get_gcs_metadata_http_handler",
            return_value=http,
        ),
    ):
        assert (
            gt._get_gcs_object_content_type(image_url="gs://public-bucket/public-object")
            == "image/jpeg"
        )
    mock_v.get_access_token.assert_not_called()
    hdrs = http.get.call_args.kwargs.get("headers")
    assert hdrs is None or "Authorization" not in hdrs


def test_async_transform_request_body_offloads_extensionless_gs_not_plain_text():
    from litellm.llms.vertex_ai.gemini import transformation as gemini_transformation

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "gs://bucket/image-without-extension"},
                }
            ],
        }
    ]

    def slow_http_get(*args, **kwargs):
        time.sleep(0.5)
        response = MagicMock()
        response.is_error = False
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {"contentType": "image/png"}
        return response

    async def fake_check_and_create_cache(self, **kwargs):
        return kwargs["messages"], kwargs["optional_params"], None

    mock_v = MagicMock()
    mock_v.get_access_token.return_value = ("token", "project")
    mock_http = MagicMock()
    mock_http.get.side_effect = slow_http_get

    async def run_scenario() -> float:
        async def concurrent_sleep() -> float:
            start = time.monotonic()
            await asyncio.sleep(0.05)
            return time.monotonic() - start

        task = asyncio.create_task(
            gemini_transformation.async_transform_request_body(
                gemini_api_key=None,
                messages=messages,
                api_base=None,
                model="gemini-2.5-flash",
                client=None,
                timeout=None,
                extra_headers=None,
                optional_params={},
                logging_obj=MagicMock(),
                custom_llm_provider="vertex_ai",
                litellm_params={},
                vertex_project=None,
                vertex_location=None,
                vertex_auth_header=None,
            )
        )
        elapsed = await concurrent_sleep()
        await task
        return elapsed

    with (
        patch.object(gemini_transformation, "_GCS_METADATA_VERTEX_BASE", mock_v),
        patch(
            "litellm.llms.vertex_ai.gemini.transformation._get_gcs_metadata_http_handler",
            return_value=mock_http,
        ),
        patch(
            "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching."
            "ContextCachingEndpoints.async_check_and_create_cache",
            new=fake_check_and_create_cache,
        ),
    ):
        sleep_elapsed = asyncio.run(run_scenario())

    assert sleep_elapsed < 0.4, (
        f"Event loop blocked for {sleep_elapsed:.3f}s; "
        "async_transform_request_body did not offload sync GCS metadata"
    )

    async def fake_cache2(self, **kwargs):
        return kwargs["messages"], kwargs["optional_params"], None

    async def run_plain():
        with patch(
            "litellm.llms.vertex_ai.gemini.transformation.asyncify",
            side_effect=AssertionError("asyncify must not run without extensionless gs://"),
        ):
            return await gemini_transformation.async_transform_request_body(
                gemini_api_key=None,
                messages=[{"role": "user", "content": "hello"}],
                api_base=None,
                model="gemini-2.5-flash",
                client=None,
                timeout=None,
                extra_headers=None,
                optional_params={},
                logging_obj=MagicMock(),
                custom_llm_provider="vertex_ai",
                litellm_params={},
                vertex_project=None,
                vertex_location=None,
                vertex_auth_header=None,
            )

    with patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching."
        "ContextCachingEndpoints.async_check_and_create_cache",
        new=fake_cache2,
    ):
        body = asyncio.run(run_plain())
    assert body is not None and "contents" in body


@pytest.mark.parametrize(
    "messages,expected",
    [
        ([{"role": "user", "content": "hello"}], False),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "gs://bucket/image-without-extension"},
                        }
                    ],
                }
            ],
            True,
        ),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "gs://bucket/image.png"},
                        }
                    ],
                }
            ],
            False,
        ),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "gs://bucket/image-without-extension",
                                "mime_type": "image/png",
                            },
                        }
                    ],
                }
            ],
            False,
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": [],
                    "images": [
                        {"image_url": {"url": "gs://bucket/gen-without-extension"}},
                    ],
                }
            ],
            True,
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": [],
                    "images": [{"image_url": {"url": "gs://bucket/gen.png"}}],
                }
            ],
            False,
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": [],
                    "images": [
                        {
                            "image_url": {
                                "url": "gs://bucket/gen-no-ext",
                                "mime_type": "image/png",
                            },
                        }
                    ],
                }
            ],
            False,
        ),
    ],
)
def test_openai_messages_may_need_sync_gcs_metadata_fetch_matrix(messages, expected):
    from litellm.llms.vertex_ai.gemini.transformation import (
        _openai_messages_may_need_sync_gcs_metadata_fetch,
    )

    assert _openai_messages_may_need_sync_gcs_metadata_fetch(messages) is expected
