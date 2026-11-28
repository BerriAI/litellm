import json
from email import policy
from email.parser import BytesParser
from typing import Final

import httpx
import pytest

import litellm


@pytest.mark.parametrize("model", ("prisma-v2.5", "prisma-v3.0"))
def test_sdk_submits_cloud_audio_with_speaker_labels_and_callback(model: str) -> None:
    jobs: Final = "/stt/v3/batch/jobs"
    payload: Final = {
        "config": {"model": model, "language_code": "en-IN", "with_diarization": True, "num_speakers": 2},
        "source": {"type": "cloud_storage", "auth": {"mode": "public"}, "paths": ["https://media.example/call.wav"]},
        "callback_url": "https://app.example/transcripts",
    }
    created: Final = {"job_id": "00000000-0000-4000-8000-000000000001", "status": "CREATED"}

    def gnani(request: httpx.Request) -> httpx.Response:
        assert (request.method, str(request.url)) == ("POST", "https://gnani.example/prefix" + jobs)
        assert request.headers["X-API-Key-ID"] == "provider-key"
        assert json.loads(request.content) == payload
        return httpx.Response(201, json=created)

    with httpx.Client(transport=httpx.MockTransport(gnani)) as client:
        jobs_client: Final = litellm.gnani_batch(
            api_base="https://gnani.example/prefix", api_key="provider-key", client=client
        )
        response: Final = jobs_client.create(
            model=f"gnani/{model}",
            language_code="en-IN",
            urls=["https://media.example/call.wav"],
            with_diarization=True,
            num_speakers=2,
            callback_url="https://app.example/transcripts",
        )
    assert isinstance(response, httpx.Response)
    assert (response.status_code, response.json()) == (201, created)


@pytest.mark.asyncio
async def test_async_sdk_fetches_another_page_of_transcript_links() -> None:
    files: Final = "/stt/v3/batch/jobs/00000000-0000-4000-8000-000000000001/files"
    page: Final = {
        "data": [{"status": "COMPLETED", "transcript_url": "https://storage.example/recording?signature=value"}],
        "pagination": {"has_more": False, "next_cursor": None},
    }

    def gnani(request: httpx.Request) -> httpx.Response:
        assert request.url.path == files
        assert dict(request.url.params) == {"cursor": "a+/=", "limit": "1"}
        return httpx.Response(200, json=page)

    async with httpx.AsyncClient(transport=httpx.MockTransport(gnani)) as client:
        jobs_client: Final = litellm.agnani_batch(api_key="provider-key", client=client)
        response: Final = await jobs_client.files("00000000-0000-4000-8000-000000000001", cursor="a+/=", limit=1)
    assert isinstance(response, httpx.Response)
    assert response.json() == page


def test_local_audio_files_and_callback_are_separate_multipart_fields() -> None:
    def gnani(request: httpx.Request) -> httpx.Response:
        message: Final = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
        )
        parts: Final = tuple(message.iter_parts())
        assert [part.get_param("name", header="content-disposition") for part in parts] == [
            "config",
            "files",
            "files",
            "callback_url",
        ]
        assert parts[0].get_content_type() == "application/json"
        config: Final[object] = parts[0].get_payload(decode=True)
        assert isinstance(config, bytes)
        assert json.loads(config) == {
            "model": "prisma-v3.0",
            "language_code": "indic-auto",
            "keyterms": ["Park Road"],
        }
        assert [(part.get_filename(), part.get_payload(decode=True)) for part in parts[1:3]] == [
            ("first.wav", b"first recording"),
            ("second.wav", b"second recording"),
        ]
        assert parts[3].get_payload(decode=True) == b"https://app.example/hooks"
        return httpx.Response(201)

    with httpx.Client(transport=httpx.MockTransport(gnani)) as client:
        jobs: Final = litellm.gnani_batch(api_key="provider-key", client=client)
        response: Final = jobs.create(
            model="prisma-v3.0",
            language_code="indic-auto",
            files=[("local/first.wav", b"first recording"), ("second.wav", b"second recording")],
            callback_url="https://app.example/hooks",
            options={"keyterms": ["Park Road"]},
        )
    assert response.status_code == 201


def test_presigned_job_returns_storage_urls_and_surfaces_provider_errors() -> None:
    job_id: Final = "00000000-0000-4000-8000-000000000001"
    upload_url: Final = "https://storage.example/large.flac?signature=value"

    def gnani(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "gnani.example"
        if request.url.path.endswith("/v2/jobs"):
            assert json.loads(request.content) == {
                "config": {"model": "prisma-v2.5", "language_code": "en-IN"},
                "source": {"type": "direct_upload_url", "files": [{"filename": "large.flac"}]},
            }
            return httpx.Response(201, json={"job_id": job_id, "uploads": [{"url": upload_url}]})
        assert (request.method, request.url.path) == ("POST", f"/stt/v3/batch/jobs/{job_id}/start")
        return httpx.Response(409, json={"error": "UPLOAD_INCOMPLETE"})

    with httpx.Client(transport=httpx.MockTransport(gnani)) as client:
        jobs: Final = litellm.gnani_batch(api_key="provider-key", api_base="https://gnani.example", client=client)
        created: Final = jobs.create(model="prisma-v2.5", language_code="en-IN", upload_filenames=["large.flac"])
        assert created.json() == {"job_id": job_id, "uploads": [{"url": upload_url}]}
        with pytest.raises(httpx.HTTPStatusError) as error:
            jobs.start(job_id)
    assert (error.value.response.status_code, error.value.response.json()) == (409, {"error": "UPLOAD_INCOMPLETE"})


@pytest.mark.parametrize(("urls", "filenames"), (((), ()), (("https://media.example/one.wav",), ("one.wav",))))
def test_create_requires_exactly_one_audio_source(urls: tuple[str, ...], filenames: tuple[str, ...]) -> None:
    jobs: Final = litellm.gnani_batch(api_key="provider-key")
    with pytest.raises(ValueError, match="exactly one"):
        jobs.create(model="prisma-v2.5", language_code="en-IN", urls=urls, upload_filenames=filenames)
