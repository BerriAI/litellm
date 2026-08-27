import json
from collections.abc import Sequence

import httpx
import pytest

from litellm.integrations.pointfive.upload_client import PointFiveUploadClient
from litellm.types.integrations.pointfive import PointFiveUploadFailure

API_URL = "https://api.pointfive.co/api/v1/ingestion"
UPLOAD_URL = "https://uploads.example.invalid/some/object.ndjson.gz?signature=sig"
OBJECT_KEY = "some/object.ndjson.gz"
BODY = b"gzipped-bytes"


def _presigned(status_code: int = 200) -> httpx.Response:
    return _response(
        status_code, {"uploadUrl": UPLOAD_URL, "objectKey": OBJECT_KEY, "expiresAt": "2026-08-25T14:35:00Z"}
    )


def _response(status_code: int, payload: object) -> httpx.Response:
    return httpx.Response(status_code, text=json.dumps(payload))


def _refused(status_code: int, error: str) -> httpx.Response:
    """The body PointFive sends with every refusal."""
    return _response(status_code, {"success": False, "error": error})


def _accepted() -> httpx.Response:
    return httpx.Response(200, text="")


def _no_content() -> httpx.Response:
    return httpx.Response(204, text="")


class FakeHTTPClient:
    """
    Stands in for AsyncHTTPHandler, including its habit of raising on error statuses.

    Scripted results are consumed in order, and the last one repeats, so a test that
    cares about a single behaviour passes a single result.
    """

    def __init__(
        self,
        presign: Sequence[httpx.Response | Exception] | None = None,
        put: Sequence[httpx.Response | Exception] | None = None,
    ) -> None:
        self.presign = list(presign) if presign else [_presigned()]  # mutable-ok: results are consumed by popping
        self.put_results = list(put) if put else [_accepted()]  # mutable-ok: results are consumed by popping
        self.presign_calls: list[dict] = []
        self.put_calls: list[dict] = []

    async def post(self, url, json=None, headers=None, **_):
        self.presign_calls.append({"url": url, "json": json, "headers": headers or {}})
        return _next_result(self.presign, url)

    async def put(self, url, data=None, headers=None, **_):
        self.put_calls.append({"url": url, "data": data, "headers": headers or {}})
        return _next_result(self.put_results, url)


def _next_result(results: list, url: str) -> httpx.Response:
    result = results.pop(0) if len(results) > 1 else results[0]
    if isinstance(result, Exception):
        raise result
    if result.status_code >= 400:
        request = httpx.Request("POST", url)
        raise httpx.HTTPStatusError(
            "boom", request=request, response=httpx.Response(result.status_code, text=result.text)
        )
    return result


async def _no_backoff(_seconds: float) -> None:
    return None


def _client(http_client: FakeHTTPClient, max_retries: int = 3, api_url: str = API_URL) -> PointFiveUploadClient:
    return PointFiveUploadClient(
        api_key="p5tu_testkey",
        api_url=api_url,
        http_client=http_client,
        max_retries=max_retries,
        sleep=_no_backoff,
    )


@pytest.mark.asyncio
async def test_uploads_the_body_to_the_url_the_api_returned():
    http_client = FakeHTTPClient()

    outcome = await _client(http_client).upload(BODY)

    assert outcome == OBJECT_KEY
    assert http_client.put_calls[0]["url"] == UPLOAD_URL
    assert http_client.put_calls[0]["data"] == BODY


@pytest.mark.asyncio
async def test_presign_request_is_authenticated_and_sized():
    http_client = FakeHTTPClient()

    await _client(http_client).upload(BODY)

    call = http_client.presign_calls[0]
    assert call["url"] == "https://api.pointfive.co/api/v1/ingestion/upload-url"
    assert call["headers"]["Authorization"] == "Bearer p5tu_testkey"
    assert call["json"] == {"kind": "LITELLM", "byteCount": len(BODY)}


@pytest.mark.asyncio
async def test_a_trailing_slash_on_the_api_url_is_tolerated():
    """A pasted URL often ends in a slash; it must not produce a double slash in the path."""
    http_client = FakeHTTPClient()

    await _client(http_client, api_url=API_URL + "/").upload(BODY)

    assert http_client.presign_calls[0]["url"] == "https://api.pointfive.co/api/v1/ingestion/upload-url"


@pytest.mark.asyncio
async def test_no_bearer_token_is_sent_to_the_presigned_url():
    """The URL carries its own authorization, so the api key must not travel with it."""
    http_client = FakeHTTPClient()

    await _client(http_client).upload(BODY)

    assert "Authorization" not in http_client.put_calls[0]["headers"]


@pytest.mark.asyncio
async def test_the_object_is_declared_as_gzipped_ndjson():
    http_client = FakeHTTPClient()

    await _client(http_client).upload(BODY)

    assert http_client.put_calls[0]["headers"]["Content-Encoding"] == "gzip"
    assert http_client.put_calls[0]["headers"]["Content-Type"] == "application/x-ndjson"


@pytest.mark.asyncio
async def test_each_retry_presigns_again():
    """A retry must never reuse a URL that was consumed or has expired."""
    http_client = FakeHTTPClient(put=[httpx.Response(503), _accepted()])

    outcome = await _client(http_client).upload(BODY)

    assert outcome == OBJECT_KEY
    assert len(http_client.presign_calls) == 2
    assert len(http_client.put_calls) == 2


@pytest.mark.asyncio
async def test_retryable_upload_failure_gives_up_after_max_retries():
    http_client = FakeHTTPClient(put=[httpx.Response(503)])

    outcome = await _client(http_client, max_retries=2).upload(BODY)

    assert outcome == PointFiveUploadFailure("presigned upload returned 503, gave up after 2 attempts", retryable=True)
    assert len(http_client.put_calls) == 2


@pytest.mark.asyncio
async def test_rejected_upload_is_not_retried():
    http_client = FakeHTTPClient(put=[httpx.Response(403)])

    outcome = await _client(http_client).upload(BODY)

    assert outcome == PointFiveUploadFailure("presigned upload returned 403", retryable=False)
    assert len(http_client.put_calls) == 1


@pytest.mark.asyncio
async def test_bad_api_key_is_not_retried():
    http_client = FakeHTTPClient(presign=[httpx.Response(401)])

    outcome = await _client(http_client).upload(BODY)

    assert outcome == PointFiveUploadFailure("pointfive api returned 401", retryable=False)
    assert http_client.put_calls == []


@pytest.mark.asyncio
async def test_the_reason_for_a_refusal_is_surfaced():
    """A 403 means the key no longer maps to an integration; the operator needs to read why."""
    http_client = FakeHTTPClient(presign=[_refused(403, "no integration accepts uploads from this api key")])

    outcome = await _client(http_client).upload(BODY)

    assert outcome == PointFiveUploadFailure(
        "pointfive api returned 403, no integration accepts uploads from this api key", retryable=False
    )
    assert http_client.put_calls == []


@pytest.mark.asyncio
async def test_api_server_error_is_retried():
    http_client = FakeHTTPClient(presign=[httpx.Response(503), _presigned()])

    outcome = await _client(http_client).upload(BODY)

    assert outcome == OBJECT_KEY
    assert len(http_client.presign_calls) == 2


@pytest.mark.asyncio
async def test_too_many_requests_is_retried():
    http_client = FakeHTTPClient(presign=[httpx.Response(429), _presigned()])

    outcome = await _client(http_client).upload(BODY)

    assert outcome == OBJECT_KEY
    assert len(http_client.presign_calls) == 2


@pytest.mark.asyncio
async def test_unreachable_api_is_retried_then_reported_as_retryable():
    http_client = FakeHTTPClient(presign=(ConnectionError("down"),))

    outcome = await _client(http_client, max_retries=2).upload(BODY)

    assert isinstance(outcome, PointFiveUploadFailure)
    assert outcome.retryable
    assert "unreachable" in outcome.detail
    assert len(http_client.presign_calls) == 2


@pytest.mark.asyncio
async def test_malformed_api_body_is_not_retried():
    http_client = FakeHTTPClient(presign=[_response(200, {"objectKey": "k"})])

    outcome = await _client(http_client).upload(BODY)

    assert outcome == PointFiveUploadFailure("pointfive api returned an unreadable body", retryable=False)
    assert http_client.put_calls == []


@pytest.mark.asyncio
async def test_a_body_that_is_not_json_is_reported_as_unreadable():
    http_client = FakeHTTPClient(presign=(httpx.Response(200, text="<html>gateway</html>"),))

    outcome = await _client(http_client).upload(BODY)

    assert outcome == PointFiveUploadFailure("pointfive api returned an unreadable body", retryable=False)
    assert http_client.put_calls == []


@pytest.mark.asyncio
async def test_ping_reports_a_live_shipper():
    http_client = FakeHTTPClient(presign=(_no_content(),))

    failure = await _client(http_client).ping()

    assert failure is None
    assert http_client.presign_calls[0]["url"] == "https://api.pointfive.co/api/v1/ingestion/ping"
    assert http_client.presign_calls[0]["json"] == {"kind": "LITELLM"}


@pytest.mark.asyncio
async def test_ping_surfaces_a_revoked_key():
    http_client = FakeHTTPClient(presign=(_refused(403, "no integration accepts uploads from this api key"),))

    failure = await _client(http_client).ping()

    assert failure is not None
    assert not failure.retryable
    assert "no integration accepts uploads from this api key" in failure.detail


@pytest.mark.asyncio
async def test_ping_surfaces_an_unreachable_api():
    http_client = FakeHTTPClient(presign=(ConnectionError("down"),))

    failure = await _client(http_client).ping()

    assert failure is not None
    assert failure.retryable


@pytest.mark.asyncio
async def test_a_transport_fault_on_the_upload_itself_is_retryable():
    http_client = FakeHTTPClient(put=(ConnectionError("reset"),))

    outcome = await _client(http_client, max_retries=1).upload(BODY)

    assert isinstance(outcome, PointFiveUploadFailure)
    assert outcome.retryable
    assert "presigned upload unreachable" in outcome.detail


@pytest.mark.asyncio
async def test_a_client_that_may_not_try_at_all_says_so():
    """max_upload_retries is validated as >= 1, so this guards the loop against a future zero."""
    outcome = await _client(FakeHTTPClient(), max_retries=0).upload(BODY)

    assert outcome == PointFiveUploadFailure("max_upload_retries must be at least 1", retryable=False)
