"""
Unit tests for ``VertexAIBatchPrediction`` (litellm/llms/vertex_ai/batches/handler.py).

The handler is HTTP/auth glue around the (separately-tested) pure
``VertexAIBatchTransformation``. Each public method (create / retrieve / list /
cancel) resolves a Vertex access token + URL, branches on ``_is_async``
(returning the coroutine in the async case, doing the sync HTTP call otherwise),
checks the HTTP status, and parses the JSON into ``LiteLLMBatch`` (or the OpenAI
list shape).

We mock only true I/O / auth seams:
  * ``_ensure_access_token`` - the Vertex credential seam. Returns a fixed
    (token, project) so we can assert the ``Authorization: Bearer <token>``
    header is forwarded.
  * ``_check_custom_proxy`` - returns ``(None, url)``; we let it pass the
    computed default url straight through so we can assert the request URL.
  * the httpx client factories (``_get_httpx_client`` /
    ``get_async_httpx_client``) and the SSRF wrappers (``safe_get`` /
    ``async_safe_get``) - the network calls. We assert which seam fired with
    what URL/headers/body, and that the response is parsed into the litellm
    type. Sibling seams are asserted NOT called where relevant.

The ``_is_async`` branch, status-code error paths, and the cancel
retrieve-after-cancel sequencing run for real.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.vertex_ai.batches.handler import (  # noqa: E402
    VertexAIBatchPrediction,
)
from litellm.types.utils import LiteLLMBatch  # noqa: E402

HMOD = "litellm.llms.vertex_ai.batches.handler"
TOKEN = "ya29.fake-access-token"
PROJECT = "my-project"
LOCATION = "us-central1"
BATCH_ID = "3814889423749775360"

CREATE_DATA = {
    "input_file_id": (
        "gs://bucket/publishers/google/models/gemini-1.5-flash-001/file-uuid"
    )
}


def _vertex_job_response(state: str = "JOB_STATE_SUCCEEDED") -> dict:
    return {
        "name": f"projects/p/locations/{LOCATION}/batchPredictionJobs/{BATCH_ID}",
        "state": state,
        "createTime": "2024-12-04T21:53:12.120184Z",
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {"uris": ["gs://bucket/in.jsonl"]},
        },
        "outputInfo": {"gcsOutputDirectory": "gs://bucket/out"},
    }


def _http_response(status_code: int = 200, json_body: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = "error text"
    resp.json.return_value = json_body if json_body is not None else _vertex_job_response()
    return resp


def _make_handler() -> VertexAIBatchPrediction:
    """Construct the handler with auth + proxy seams patched at the instance level.

    ``_ensure_access_token`` and ``_check_custom_proxy`` are inherited from
    ``VertexLLM``; we patch them on the instance (DI-style) so the URL/auth
    plumbing is deterministic and we can assert what got forwarded downstream.
    """
    h = VertexAIBatchPrediction(gcs_bucket_name="litellm-testing-bucket")
    h._ensure_access_token = MagicMock(return_value=(TOKEN, PROJECT))  # type: ignore[method-assign]
    # pass the computed default url straight through (no custom proxy)
    h._check_custom_proxy = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda **kw: (None, kw["url"])
    )
    return h


def _run(coro):
    return asyncio.run(coro)


# =========================================================================== #
# create_vertex_batch_url
# =========================================================================== #


def test_create_vertex_batch_url():
    h = _make_handler()
    url = h.create_vertex_batch_url(vertex_location=LOCATION, vertex_project=PROJECT)
    assert url == (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}"
        f"/locations/{LOCATION}/batchPredictionJobs"
    )


# =========================================================================== #
# create_batch
# =========================================================================== #


def test_create_batch_sync_posts_and_parses():
    h = _make_handler()
    client = MagicMock()
    client.post.return_value = _http_response()

    with patch(f"{HMOD}._get_httpx_client", return_value=client):
        out = h.create_batch(
            _is_async=False,
            create_batch_data=CREATE_DATA,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )

    assert isinstance(out, LiteLLMBatch)
    assert out.id == BATCH_ID
    assert out.status == "completed"

    # auth seam fired
    h._ensure_access_token.assert_called_once()
    # the POST hit the batchPredictionJobs collection url with bearer auth
    _, kwargs = client.post.call_args
    assert kwargs["url"].endswith(f"/projects/{PROJECT}/locations/{LOCATION}/batchPredictionJobs")
    assert kwargs["headers"]["Authorization"] == f"Bearer {TOKEN}"
    # body is the transformed vertex job (json-serialized)
    sent = json.loads(kwargs["data"])
    assert sent["model"] == "publishers/google/models/gemini-1.5-flash-001"
    assert sent["inputConfig"]["gcsSource"]["uris"] == [CREATE_DATA["input_file_id"]]


def test_create_batch_async_returns_coroutine_and_uses_async_client():
    h = _make_handler()
    async_client = MagicMock()
    async_client.post = AsyncMock(return_value=_http_response())
    sync_client = MagicMock()

    with (
        patch(f"{HMOD}._get_httpx_client", return_value=sync_client),
        patch(f"{HMOD}.get_async_httpx_client", return_value=async_client),
    ):
        coro = h.create_batch(
            _is_async=True,
            create_batch_data=CREATE_DATA,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )
        assert asyncio.iscoroutine(coro)
        out = _run(coro)

    assert isinstance(out, LiteLLMBatch)
    assert out.id == BATCH_ID
    async_client.post.assert_awaited_once()
    # the async branch must NOT use the sync client for the request
    sync_client.post.assert_not_called()


def test_create_batch_sync_non_200_raises():
    h = _make_handler()
    client = MagicMock()
    client.post.return_value = _http_response(status_code=500)

    with patch(f"{HMOD}._get_httpx_client", return_value=client):
        with pytest.raises(Exception, match="Error: 500"):
            h.create_batch(
                _is_async=False,
                create_batch_data=CREATE_DATA,
                api_base=None,
                vertex_credentials=None,
                vertex_project=PROJECT,
                vertex_location=LOCATION,
                timeout=600.0,
                max_retries=None,
            )


def test_create_batch_async_non_200_raises():
    h = _make_handler()
    async_client = MagicMock()
    async_client.post = AsyncMock(return_value=_http_response(status_code=403))

    with (
        patch(f"{HMOD}._get_httpx_client", return_value=MagicMock()),
        patch(f"{HMOD}.get_async_httpx_client", return_value=async_client),
    ):
        coro = h.create_batch(
            _is_async=True,
            create_batch_data=CREATE_DATA,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )
        with pytest.raises(Exception, match="Error: 403"):
            _run(coro)


# =========================================================================== #
# retrieve_batch
# =========================================================================== #


def test_retrieve_batch_sync_uses_safe_get_with_batch_id_url():
    h = _make_handler()
    sync_client = MagicMock()

    with (
        patch(f"{HMOD}._get_httpx_client", return_value=sync_client),
        patch(f"{HMOD}.safe_get", return_value=_http_response()) as safe_get,
    ):
        out = h.retrieve_batch(
            _is_async=False,
            batch_id=BATCH_ID,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )

    assert isinstance(out, LiteLLMBatch)
    assert out.id == BATCH_ID
    # SSRF-wrapped fetch fired with the batch-id-appended url + bearer header
    args, kwargs = safe_get.call_args
    assert args[0] is sync_client
    assert args[1].endswith(f"/batchPredictionJobs/{BATCH_ID}")
    assert kwargs["headers"]["Authorization"] == f"Bearer {TOKEN}"
    # plain client.get must NOT be used (SSRF wrapper is the seam)
    sync_client.get.assert_not_called()


def test_retrieve_batch_async_returns_coroutine_uses_async_safe_get():
    h = _make_handler()
    async_client = MagicMock()

    with (
        patch(f"{HMOD}._get_httpx_client", return_value=MagicMock()),
        patch(f"{HMOD}.get_async_httpx_client", return_value=async_client),
        patch(
            f"{HMOD}.async_safe_get",
            new=AsyncMock(return_value=_http_response()),
        ) as async_safe_get,
    ):
        coro = h.retrieve_batch(
            _is_async=True,
            batch_id=BATCH_ID,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )
        assert asyncio.iscoroutine(coro)
        out = _run(coro)

    assert isinstance(out, LiteLLMBatch)
    async_safe_get.assert_awaited_once()
    args, _ = async_safe_get.await_args
    assert args[1].endswith(f"/batchPredictionJobs/{BATCH_ID}")


def test_retrieve_batch_sync_non_200_raises():
    h = _make_handler()
    with (
        patch(f"{HMOD}._get_httpx_client", return_value=MagicMock()),
        patch(f"{HMOD}.safe_get", return_value=_http_response(status_code=404)),
    ):
        with pytest.raises(Exception, match="Error: 404"):
            h.retrieve_batch(
                _is_async=False,
                batch_id=BATCH_ID,
                api_base=None,
                vertex_credentials=None,
                vertex_project=PROJECT,
                vertex_location=LOCATION,
                timeout=600.0,
                max_retries=None,
            )


def test_retrieve_batch_sync_invokes_logging_pre_call():
    """When a real ``Logging`` obj is passed, ``pre_call`` is invoked with the
    request url + headers (the curl-redaction branch)."""
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

    h = _make_handler()
    logging_obj = MagicMock(spec=LiteLLMLogging)

    with (
        patch(f"{HMOD}._get_httpx_client", return_value=MagicMock()),
        patch(f"{HMOD}.safe_get", return_value=_http_response()),
    ):
        h.retrieve_batch(
            _is_async=False,
            batch_id=BATCH_ID,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
            logging_obj=logging_obj,
        )

    logging_obj.pre_call.assert_called_once()
    _, kwargs = logging_obj.pre_call.call_args
    assert kwargs["additional_args"]["api_base"].endswith(
        f"/batchPredictionJobs/{BATCH_ID}"
    )


# =========================================================================== #
# list_batches
# =========================================================================== #


def _list_response() -> dict:
    return {
        "batchPredictionJobs": [_vertex_job_response()],
        "nextPageToken": "next-tok",
    }


def test_list_batches_sync_passes_pagination_params():
    h = _make_handler()
    client = MagicMock()
    client.get.return_value = _http_response(json_body=_list_response())

    with patch(f"{HMOD}._get_httpx_client", return_value=client):
        out = h.list_batches(
            _is_async=False,
            after="cursor-xyz",
            limit=7,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )

    assert out["object"] == "list"
    assert out["data"][0].id == BATCH_ID
    assert out["has_more"] is True
    assert out["next_page_token"] == "next-tok"

    _, kwargs = client.get.call_args
    # limit -> pageSize (stringified), after -> pageToken
    assert kwargs["params"] == {"pageSize": "7", "pageToken": "cursor-xyz"}
    assert kwargs["headers"]["Authorization"] == f"Bearer {TOKEN}"


def test_list_batches_sync_omits_unset_pagination_params():
    h = _make_handler()
    client = MagicMock()
    client.get.return_value = _http_response(json_body={"batchPredictionJobs": []})

    with patch(f"{HMOD}._get_httpx_client", return_value=client):
        out = h.list_batches(
            _is_async=False,
            after=None,
            limit=None,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )

    _, kwargs = client.get.call_args
    assert kwargs["params"] == {}
    assert out["data"] == []
    assert out["has_more"] is False


def test_list_batches_async_returns_coroutine():
    h = _make_handler()
    async_client = MagicMock()
    async_client.get = AsyncMock(
        return_value=_http_response(json_body=_list_response())
    )
    sync_client = MagicMock()

    with (
        patch(f"{HMOD}._get_httpx_client", return_value=sync_client),
        patch(f"{HMOD}.get_async_httpx_client", return_value=async_client),
    ):
        coro = h.list_batches(
            _is_async=True,
            after=None,
            limit=None,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )
        assert asyncio.iscoroutine(coro)
        out = _run(coro)

    assert out["data"][0].id == BATCH_ID
    async_client.get.assert_awaited_once()
    sync_client.get.assert_not_called()


def test_list_batches_sync_non_200_raises():
    h = _make_handler()
    client = MagicMock()
    client.get.return_value = _http_response(status_code=500)

    with patch(f"{HMOD}._get_httpx_client", return_value=client):
        with pytest.raises(Exception, match="Error: 500"):
            h.list_batches(
                _is_async=False,
                after=None,
                limit=None,
                api_base=None,
                vertex_credentials=None,
                vertex_project=PROJECT,
                vertex_location=LOCATION,
                timeout=600.0,
                max_retries=None,
            )


# =========================================================================== #
# cancel_batch
# =========================================================================== #


def test_cancel_batch_sync_posts_cancel_then_retrieves():
    h = _make_handler()
    client = MagicMock()
    client.post.return_value = _http_response(json_body={})
    client.get.return_value = _http_response(
        json_body=_vertex_job_response(state="JOB_STATE_CANCELLED")
    )

    with patch(f"{HMOD}._get_httpx_client", return_value=client):
        out = h.cancel_batch(
            _is_async=False,
            batch_id=BATCH_ID,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )

    assert isinstance(out, LiteLLMBatch)
    assert out.status == "cancelled"

    # POST hit the :cancel url
    _, post_kwargs = client.post.call_args
    assert post_kwargs["url"].endswith(f"/batchPredictionJobs/{BATCH_ID}:cancel")
    assert post_kwargs["data"] == json.dumps({})
    # then GET hit the plain retrieve url (no :cancel suffix)
    _, get_kwargs = client.get.call_args
    assert get_kwargs["url"].endswith(f"/batchPredictionJobs/{BATCH_ID}")
    assert not get_kwargs["url"].endswith(":cancel")


def test_cancel_batch_async_returns_coroutine_posts_then_retrieves():
    h = _make_handler()
    async_client = MagicMock()
    async_client.post = AsyncMock(return_value=_http_response(json_body={}))
    async_client.get = AsyncMock(
        return_value=_http_response(
            json_body=_vertex_job_response(state="JOB_STATE_CANCELLED")
        )
    )

    with (
        patch(f"{HMOD}._get_httpx_client", return_value=MagicMock()),
        patch(f"{HMOD}.get_async_httpx_client", return_value=async_client),
    ):
        coro = h.cancel_batch(
            _is_async=True,
            batch_id=BATCH_ID,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )
        assert asyncio.iscoroutine(coro)
        out = _run(coro)

    assert out.status == "cancelled"
    async_client.post.assert_awaited_once()
    async_client.get.assert_awaited_once()
    _, post_kwargs = async_client.post.await_args
    assert post_kwargs["url"].endswith(":cancel")


def test_cancel_batch_sync_cancel_post_non_200_raises():
    h = _make_handler()
    client = MagicMock()
    client.post.return_value = _http_response(status_code=500)

    with patch(f"{HMOD}._get_httpx_client", return_value=client):
        with pytest.raises(Exception, match="Error: 500"):
            h.cancel_batch(
                _is_async=False,
                batch_id=BATCH_ID,
                api_base=None,
                vertex_credentials=None,
                vertex_project=PROJECT,
                vertex_location=LOCATION,
                timeout=600.0,
                max_retries=None,
            )
    # cancel POST failed -> retrieve GET must never fire
    client.get.assert_not_called()


def test_cancel_batch_sync_retrieve_non_200_raises():
    h = _make_handler()
    client = MagicMock()
    client.post.return_value = _http_response(json_body={})
    client.get.return_value = _http_response(status_code=404)

    with patch(f"{HMOD}._get_httpx_client", return_value=client):
        with pytest.raises(Exception, match="Error: 404"):
            h.cancel_batch(
                _is_async=False,
                batch_id=BATCH_ID,
                api_base=None,
                vertex_credentials=None,
                vertex_project=PROJECT,
                vertex_location=LOCATION,
                timeout=600.0,
                max_retries=None,
            )


def test_cancel_batch_sync_proxy_url_without_cancel_suffix_uses_rsplit_branch():
    """If ``_check_custom_proxy`` hands back a url that does NOT end in
    ``:cancel`` (e.g. a custom proxy rewrote it), the retrieve url is derived
    via the ``rsplit(':cancel')`` else-branch rather than ``removesuffix``."""
    h = _make_handler()
    # override the proxy seam to return a non-:cancel-suffixed url
    h._check_custom_proxy = MagicMock(  # type: ignore[method-assign]
        return_value=(None, "https://proxy.internal/vertex/batch")
    )
    client = MagicMock()
    client.post.return_value = _http_response(json_body={})
    client.get.return_value = _http_response(
        json_body=_vertex_job_response(state="JOB_STATE_CANCELLED")
    )

    with patch(f"{HMOD}._get_httpx_client", return_value=client):
        out = h.cancel_batch(
            _is_async=False,
            batch_id=BATCH_ID,
            api_base="https://proxy.internal",
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )

    assert out.status == "cancelled"
    _, get_kwargs = client.get.call_args
    # rsplit(":cancel")[0].rstrip("/") of a url with no :cancel -> url unchanged
    assert get_kwargs["url"] == "https://proxy.internal/vertex/batch"


def test_cancel_batch_sync_httpstatuserror_logged_and_reraised():
    """The cancel POST ``httpx.HTTPStatusError`` except-branch logs + re-raises."""
    h = _make_handler()
    client = MagicMock()
    request = httpx.Request("POST", "https://x/batchPredictionJobs/1:cancel")
    err_response = httpx.Response(status_code=502, request=request, text="bad gw")
    client.post.side_effect = httpx.HTTPStatusError(
        "boom", request=request, response=err_response
    )

    with patch(f"{HMOD}._get_httpx_client", return_value=client):
        with pytest.raises(httpx.HTTPStatusError):
            h.cancel_batch(
                _is_async=False,
                batch_id=BATCH_ID,
                api_base=None,
                vertex_credentials=None,
                vertex_project=PROJECT,
                vertex_location=LOCATION,
                timeout=600.0,
                max_retries=None,
            )
    client.get.assert_not_called()


def test_create_batch_async_httpstatuserror_logged_and_reraised():
    h = _make_handler()
    async_client = MagicMock()
    request = httpx.Request("POST", "https://x/batchPredictionJobs")
    err_response = httpx.Response(status_code=500, request=request, text="boom")
    async_client.post = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "boom", request=request, response=err_response
        )
    )

    with (
        patch(f"{HMOD}._get_httpx_client", return_value=MagicMock()),
        patch(f"{HMOD}.get_async_httpx_client", return_value=async_client),
    ):
        coro = h.create_batch(
            _is_async=True,
            create_batch_data=CREATE_DATA,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )
        with pytest.raises(httpx.HTTPStatusError):
            _run(coro)


def test_async_retrieve_batch_non_200_raises():
    h = _make_handler()
    with (
        patch(f"{HMOD}._get_httpx_client", return_value=MagicMock()),
        patch(f"{HMOD}.get_async_httpx_client", return_value=MagicMock()),
        patch(
            f"{HMOD}.async_safe_get",
            new=AsyncMock(return_value=_http_response(status_code=500)),
        ),
    ):
        coro = h.retrieve_batch(
            _is_async=True,
            batch_id=BATCH_ID,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )
        with pytest.raises(Exception, match="Error: 500"):
            _run(coro)


def test_async_retrieve_batch_invokes_logging_pre_call():
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

    h = _make_handler()
    logging_obj = MagicMock(spec=LiteLLMLogging)

    with (
        patch(f"{HMOD}._get_httpx_client", return_value=MagicMock()),
        patch(f"{HMOD}.get_async_httpx_client", return_value=MagicMock()),
        patch(
            f"{HMOD}.async_safe_get",
            new=AsyncMock(return_value=_http_response()),
        ),
    ):
        coro = h.retrieve_batch(
            _is_async=True,
            batch_id=BATCH_ID,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
            logging_obj=logging_obj,
        )
        _run(coro)

    logging_obj.pre_call.assert_called_once()


def test_async_list_batches_non_200_raises():
    h = _make_handler()
    async_client = MagicMock()
    async_client.get = AsyncMock(return_value=_http_response(status_code=500))

    with (
        patch(f"{HMOD}._get_httpx_client", return_value=MagicMock()),
        patch(f"{HMOD}.get_async_httpx_client", return_value=async_client),
    ):
        coro = h.list_batches(
            _is_async=True,
            after=None,
            limit=None,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )
        with pytest.raises(Exception, match="Error: 500"):
            _run(coro)


def test_async_cancel_batch_httpstatuserror_and_retrieve_non_200():
    """Async cancel: POST HTTPStatusError re-raises; and separately the
    retrieve-after-cancel non-200 raises."""
    h = _make_handler()

    # (a) POST raises HTTPStatusError
    async_client = MagicMock()
    request = httpx.Request("POST", "https://x/batchPredictionJobs/1:cancel")
    err_response = httpx.Response(status_code=502, request=request, text="bad")
    async_client.post = AsyncMock(
        side_effect=httpx.HTTPStatusError("boom", request=request, response=err_response)
    )
    async_client.get = AsyncMock()
    with (
        patch(f"{HMOD}._get_httpx_client", return_value=MagicMock()),
        patch(f"{HMOD}.get_async_httpx_client", return_value=async_client),
    ):
        coro = h.cancel_batch(
            _is_async=True,
            batch_id=BATCH_ID,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )
        with pytest.raises(httpx.HTTPStatusError):
            _run(coro)
    async_client.get.assert_not_awaited()

    # (a2) cancel POST returns a plain non-200 (no exception) -> raises
    async_client_post500 = MagicMock()
    async_client_post500.post = AsyncMock(return_value=_http_response(status_code=500))
    async_client_post500.get = AsyncMock()
    with (
        patch(f"{HMOD}._get_httpx_client", return_value=MagicMock()),
        patch(f"{HMOD}.get_async_httpx_client", return_value=async_client_post500),
    ):
        coro = h.cancel_batch(
            _is_async=True,
            batch_id=BATCH_ID,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )
        with pytest.raises(Exception, match="Error: 500"):
            _run(coro)
    async_client_post500.get.assert_not_awaited()

    # (b) retrieve-after-cancel returns non-200
    async_client2 = MagicMock()
    async_client2.post = AsyncMock(return_value=_http_response(json_body={}))
    async_client2.get = AsyncMock(return_value=_http_response(status_code=404))
    with (
        patch(f"{HMOD}._get_httpx_client", return_value=MagicMock()),
        patch(f"{HMOD}.get_async_httpx_client", return_value=async_client2),
    ):
        coro = h.cancel_batch(
            _is_async=True,
            batch_id=BATCH_ID,
            api_base=None,
            vertex_credentials=None,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            timeout=600.0,
            max_retries=None,
        )
        with pytest.raises(Exception, match="Error: 404"):
            _run(coro)
