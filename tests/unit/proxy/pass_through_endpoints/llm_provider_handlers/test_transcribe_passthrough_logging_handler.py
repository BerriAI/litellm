import asyncio
import io
import json
import wave
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.transcribe_passthrough_logging_handler import (
    TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS,
    TRANSCRIBE_OWNER_TAG,
    TranscribePassthroughLoggingHandler,
    TranscribeRefusal,
    TranscriptionJobRecord,
    media_file_seconds,
    media_predates_job,
    price_transcription_job,
    requested_media_format,
    s3_media_url,
    started_transcription_job,
    transcribe_admin_only_refusal,
    transcribe_cost_per_second,
    transcribe_job_access_refusal,
    transcribe_media_buckets,
    transcribe_owned_start_request,
    transcribe_storage_refusal,
    transcribe_supported_operations,
    transcribe_unpriceable_request_reason,
    write_media_within_limit,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)

COST_PER_SECOND = 0.0001


def _make_response(operation: str) -> httpx.Response:
    request = httpx.Request(
        "POST",
        "https://transcribe.us-west-2.amazonaws.com/",
        headers={"X-Amz-Target": f"Transcribe.{operation}"},
    )
    return httpx.Response(200, request=request, text='{"TranscriptionJob": {}}')


async def _relayed_response(operation: str, body: bytes) -> httpx.Response:
    response = httpx.Response(
        200,
        request=_make_response(operation).request,
        headers={"content-type": "application/x-amz-json-1.1"},
        stream=httpx.ByteStream(body),
    )
    async for _ in response.aiter_bytes():
        pass
    await response.aclose()
    return response


def _make_logging_obj() -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.litellm_call_id = "test-call-id"
    logging_obj.model_call_details = {}
    return logging_obj


async def _no_sleep(_: float) -> None:
    return None


MEDIA_URI = "s3://b/a.wav"
CREATED_AT = 1_789_682_363.696


def _job(
    status: str, media_uri: str | None = MEDIA_URI, created_at: float | None = CREATED_AT, **members: object
) -> dict[str, object]:
    media = {"Media": {"MediaFileUri": media_uri}} if media_uri else {}
    created = {"CreationTime": created_at} if created_at is not None else {}
    return {"TranscriptionJob": {"TranscriptionJobStatus": status, **media, **created, **members}}


async def _no_media(uri: str, created_at: float) -> float | None:
    raise AssertionError("the media must not be measured on this path")


def _media_probe(*durations: float | None | Exception):
    remaining = list(durations)
    measured: list[tuple[str, float]] = []

    async def media_seconds(uri: str, created_at: float) -> float | None:
        measured.append((uri, created_at))
        outcome = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return media_seconds, measured


def _sequence(*jobs: dict[str, object]):
    remaining = list(jobs)
    seen: list[str] = []

    async def get_job(job_name: str) -> dict[str, object]:
        seen.append(job_name)
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return get_job, seen


def _aws_error(error_type: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://transcribe.us-west-2.amazonaws.com/")
    response = httpx.Response(400, request=request, json={"__type": error_type, "message": "nope"})
    return httpx.HTTPStatusError("400", request=request, response=response)


def _missing_job(error_type: str):
    seen: list[str] = []

    async def get_job(job_name: str) -> dict[str, object]:
        seen.append(job_name)
        raise _aws_error(error_type)

    return get_job, seen


class TestTranscribeSupportedOperations:
    def test_matches_the_installed_botocore_service_model(self):
        from botocore.session import get_session

        assert transcribe_supported_operations() == frozenset(
            get_session().get_service_model("transcribe").operation_names
        )


class TestTranscribeCostMap:
    def test_start_transcription_job_is_priced_per_second_of_audio(self):
        entry = litellm.model_cost["transcribe/StartTranscriptionJob"]

        assert entry["litellm_provider"] == "transcribe"
        assert entry["mode"] == "audio_transcription"
        assert transcribe_cost_per_second() == entry["input_cost_per_second"] > 0

    def test_missing_or_malformed_entry_yields_no_rate(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(litellm.model_cost, "transcribe/StartTranscriptionJob", {"input_cost_per_second": "x"})
        assert transcribe_cost_per_second() is None
        monkeypatch.delitem(litellm.model_cost, "transcribe/StartTranscriptionJob")
        assert transcribe_cost_per_second() is None


class TestTranscribeUnpriceableRequestReason:
    def test_plain_start_transcription_job_is_allowed(self):
        body = {"TranscriptionJobName": "j", "Media": {"MediaFileUri": MEDIA_URI}}
        assert transcribe_unpriceable_request_reason("StartTranscriptionJob", body, COST_PER_SECOND) is None

    @pytest.mark.parametrize(
        "body",
        [
            {"Media": {"MediaFileUri": "s3://b/a.mp4"}},
            {"Media": {"MediaFileUri": "s3://b/a.wav"}, "MediaFormat": "webm"},
            {"Media": {"MediaFileUri": "s3://b/recording"}},
            {"TranscriptionJobName": "j"},
        ],
    )
    def test_media_whose_length_cannot_be_read_is_rejected(self, body: dict[str, object]):
        reason = transcribe_unpriceable_request_reason("StartTranscriptionJob", body, COST_PER_SECOND)
        assert reason is not None and "MediaFormat" in reason

    @pytest.mark.parametrize(
        "body",
        [
            {"Media": {"MediaFileUri": "s3://b/a.mp4"}, "MediaFormat": "mp3"},
            {"Media": {"MediaFileUri": "https://s3.us-west-2.amazonaws.com/b/a.FLAC?x=1"}},
            {"Media": {"MediaFileUri": "s3://b/dir.v2/a.ogg"}},
        ],
    )
    def test_measurable_media_is_allowed(self, body: dict[str, object]):
        assert transcribe_unpriceable_request_reason("StartTranscriptionJob", body, COST_PER_SECOND) is None

    def test_read_only_operations_are_allowed_without_a_rate(self):
        assert transcribe_unpriceable_request_reason("GetTranscriptionJob", {}, None) is None
        assert transcribe_unpriceable_request_reason("ListTranscriptionJobs", {}, None) is None

    def test_start_transcription_job_needs_a_rate(self):
        reason = transcribe_unpriceable_request_reason("StartTranscriptionJob", {"TranscriptionJobName": "j"}, None)
        assert reason is not None and "model cost map" in reason

    @pytest.mark.parametrize(
        "operation", ["StartCallAnalyticsJob", "StartMedicalScribeJob", "StartMedicalTranscriptionJob"]
    )
    def test_unpriced_job_classes_are_rejected(self, operation: str):
        reason = transcribe_unpriceable_request_reason(operation, {}, COST_PER_SECOND)
        assert reason is not None and operation in reason

    @pytest.mark.parametrize(
        ("body", "member"),
        [
            ({"ContentRedaction": {"RedactionType": "PII", "RedactionOutput": "redacted"}}, "ContentRedaction"),
            ({"ToxicityDetection": [{"ToxicityCategories": ["ALL"]}]}, "ToxicityDetection"),
            ({"ModelSettings": {"LanguageModelName": "clm"}}, "ModelSettings.LanguageModelName"),
            (
                {
                    "IdentifyLanguage": True,
                    "LanguageIdSettings": {"en-US": {"VocabularyName": "v"}, "fr-FR": {"LanguageModelName": "clm"}},
                },
                "LanguageIdSettings.fr-FR.LanguageModelName",
            ),
        ],
    )
    def test_surcharged_features_are_rejected(self, body: dict[str, object], member: str):
        reason = transcribe_unpriceable_request_reason(
            "StartTranscriptionJob", {**body, "Media": {"MediaFileUri": MEDIA_URI}}, COST_PER_SECOND
        )
        assert reason is not None and member in reason

    def test_settings_without_a_custom_model_are_allowed(self):
        body = {
            "ModelSettings": {},
            "LanguageIdSettings": {"en-US": {"VocabularyName": "v"}},
            "Media": {"MediaFileUri": MEDIA_URI},
        }
        assert transcribe_unpriceable_request_reason("StartTranscriptionJob", body, COST_PER_SECOND) is None


class TestRequestedMediaFormat:
    def test_explicit_media_format_wins_over_the_extension(self):
        assert requested_media_format({"MediaFormat": "MP3", "Media": {"MediaFileUri": "s3://b/a.wav"}}) == "mp3"

    def test_extension_is_read_from_the_uri_path_only(self):
        assert requested_media_format({"Media": {"MediaFileUri": "https://h/b/a.wav?sig=x.y"}}) == "wav"
        assert requested_media_format({"Media": {"MediaFileUri": "s3://b.name/a"}}) is None
        assert requested_media_format({"Media": {"MediaFileUri": 7}}) is None


class TestS3MediaUrl:
    def test_s3_uri_maps_to_the_regional_virtual_hosted_endpoint(self):
        assert (
            s3_media_url("s3://my-bucket/dir/a b.wav", "us-west-2")
            == "https://my-bucket.s3.us-west-2.amazonaws.com/dir/a%20b.wav"
        )

    def test_dotted_bucket_maps_to_the_regional_path_style_endpoint(self):
        assert (
            s3_media_url("s3://media.example.com/dir/a b.wav", "us-west-2")
            == "https://s3.us-west-2.amazonaws.com/media.example.com/dir/a%20b.wav"
        )

    @pytest.mark.parametrize(
        "media_uri",
        [
            "https://evil.example.com/a.wav",
            "https://my-bucket.s3.us-west-2.amazonaws.com@evil.example.com/a.wav",
            "https://amazonaws.com/a.wav",
            "http://my-bucket.s3.us-west-2.amazonaws.com/a.wav",
        ],
    )
    def test_hosts_outside_the_aws_partition_or_off_https_are_never_signed_for(self, media_uri: str):
        assert s3_media_url(media_uri, "us-west-2") is None

    def test_https_uri_is_used_as_given(self):
        assert (
            s3_media_url("https://my-bucket.s3.eu-west-1.amazonaws.com/a.wav", "us-west-2")
            == "https://my-bucket.s3.eu-west-1.amazonaws.com/a.wav"
        )


class _ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


def _media_response(*chunks: bytes, content_length: int | None) -> httpx.Response:
    headers = {"content-length": str(content_length)} if content_length is not None else {}
    return httpx.Response(200, headers=headers, stream=_ChunkedStream(*chunks))


class TestWriteMediaWithinLimit:
    @pytest.mark.asyncio
    async def test_media_within_the_cap_is_written_whole(self):
        media_file = io.BytesIO()
        assert await write_media_within_limit(_media_response(b"abc", b"def", content_length=6), media_file, 6) is True
        assert media_file.getvalue() == b"abcdef"

    @pytest.mark.asyncio
    async def test_advertised_size_over_the_cap_is_refused_before_downloading(self):
        media_file = io.BytesIO()
        assert await write_media_within_limit(_media_response(b"abcdef", content_length=7), media_file, 6) is False
        assert media_file.getvalue() == b""

    @pytest.mark.asyncio
    async def test_stream_growing_past_the_cap_is_cut_off(self):
        media_file = io.BytesIO()
        response = _media_response(b"abc", b"def", b"ghi", content_length=None)
        assert await write_media_within_limit(response, media_file, 5) is False
        assert media_file.getvalue() == b"abcdef"


class TestPriceTranscriptionJob:
    @pytest.mark.asyncio
    async def test_polls_until_completed_then_charges_the_media_length_rounded_up(self):
        get_job, seen = _sequence(_job("IN_PROGRESS"), _job("IN_PROGRESS"), _job("COMPLETED"))
        media_seconds, measured = _media_probe(17.577)

        cost = await price_transcription_job("job-1", COST_PER_SECOND, get_job, media_seconds, sleep=_no_sleep)

        assert cost == pytest.approx(18 * COST_PER_SECOND)
        assert seen == ["job-1", "job-1", "job-1"]
        assert measured == [(MEDIA_URI, CREATED_AT)]

    @pytest.mark.asyncio
    async def test_a_failed_poll_is_retried_instead_of_ending_pricing(self):
        remaining = [httpx.ConnectError("aws blip"), None]

        async def get_job(job_name: str) -> dict[str, object]:
            outcome = remaining.pop(0)
            if outcome is not None:
                raise outcome
            return _job("COMPLETED")

        media_seconds, _ = _media_probe(3.0)

        cost = await price_transcription_job("job-1", COST_PER_SECOND, get_job, media_seconds, sleep=_no_sleep)

        assert cost == pytest.approx(3 * COST_PER_SECOND)
        assert remaining == []

    @pytest.mark.asyncio
    async def test_failed_job_costs_nothing(self):
        get_job, _ = _sequence(_job("FAILED"))

        assert await price_transcription_job("job-1", COST_PER_SECOND, get_job, _no_media, sleep=_no_sleep) == 0.0

    @pytest.mark.asyncio
    async def test_job_deleted_before_it_is_polled_is_charged_for_the_media_it_was_started_with(self):
        get_job, seen = _missing_job("BadRequestException")
        media_seconds, measured = _media_probe(17.577)
        started = started_transcription_job(
            {"TranscriptionJob": {"Media": {"MediaFileUri": "s3://b/started.wav"}, "CreationTime": 5.0}}
        )

        cost = await price_transcription_job(
            "job-1", COST_PER_SECOND, get_job, media_seconds, sleep=_no_sleep, started_job=started
        )

        assert cost == pytest.approx(18 * COST_PER_SECOND)
        assert seen == ["job-1"]
        assert measured == [("s3://b/started.wav", 5.0)]

    @pytest.mark.asyncio
    async def test_job_not_found_by_transcribe_is_charged_the_maximum_without_a_start_record(self):
        get_job, seen = _missing_job("com.amazonaws.transcribe#NotFoundException")

        cost = await price_transcription_job("job-1", COST_PER_SECOND, get_job, _no_media, sleep=_no_sleep)

        assert cost == pytest.approx(TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS * COST_PER_SECOND)
        assert seen == ["job-1"]

    @pytest.mark.asyncio
    async def test_throttled_poll_is_retried_rather_than_treated_as_a_missing_job(self):
        remaining = ["LimitExceededException", None]

        async def get_job(job_name: str) -> dict[str, object]:
            error_type = remaining.pop(0)
            if error_type is not None:
                raise _aws_error(error_type)
            return _job("COMPLETED")

        media_seconds, _ = _media_probe(3.0)

        cost = await price_transcription_job("job-1", COST_PER_SECOND, get_job, media_seconds, sleep=_no_sleep)

        assert cost == pytest.approx(3 * COST_PER_SECOND)
        assert remaining == []

    @pytest.mark.asyncio
    async def test_job_that_never_finishes_is_charged_the_maximum(self):
        get_job, seen = _sequence(_job("IN_PROGRESS"))

        cost = await price_transcription_job(
            "job-1", COST_PER_SECOND, get_job, _no_media, sleep=_no_sleep, max_attempts=3
        )

        assert cost == pytest.approx(TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS * COST_PER_SECOND)
        assert len(seen) == 3

    @pytest.mark.asyncio
    async def test_media_that_cannot_be_read_is_charged_the_maximum(self):
        get_job, _ = _sequence(_job("COMPLETED"))
        media_seconds, measured = _media_probe(None)

        cost = await price_transcription_job("job-1", COST_PER_SECOND, get_job, media_seconds, sleep=_no_sleep)

        assert cost == pytest.approx(TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS * COST_PER_SECOND)
        assert measured == [(MEDIA_URI, CREATED_AT)]

    @pytest.mark.asyncio
    async def test_media_fetch_is_retried_then_charged_the_maximum(self):
        get_job, _ = _sequence(_job("COMPLETED"))
        media_seconds, measured = _media_probe(httpx.ReadTimeout("s3 slow"))

        cost = await price_transcription_job("job-1", COST_PER_SECOND, get_job, media_seconds, sleep=_no_sleep)

        assert cost == pytest.approx(TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS * COST_PER_SECOND)
        assert len(measured) == 3

    @pytest.mark.asyncio
    async def test_media_fetch_recovers_after_a_transient_failure(self):
        get_job, _ = _sequence(_job("COMPLETED"))
        media_seconds, measured = _media_probe(httpx.ReadTimeout("s3 slow"), 60.0)

        cost = await price_transcription_job("job-1", COST_PER_SECOND, get_job, media_seconds, sleep=_no_sleep)

        assert cost == pytest.approx(60 * COST_PER_SECOND)
        assert len(measured) == 2

    @pytest.mark.asyncio
    async def test_completed_job_without_media_uri_is_charged_the_maximum(self):
        get_job, _ = _sequence(_job("COMPLETED", media_uri=None))

        cost = await price_transcription_job("job-1", COST_PER_SECOND, get_job, _no_media, sleep=_no_sleep)

        assert cost == pytest.approx(TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS * COST_PER_SECOND)

    @pytest.mark.asyncio
    async def test_completed_job_without_creation_time_is_charged_the_maximum_unmeasured(self):
        get_job, _ = _sequence(_job("COMPLETED", created_at=None))
        media_seconds, measured = _media_probe(60.0)

        cost = await price_transcription_job("job-1", COST_PER_SECOND, get_job, media_seconds, sleep=_no_sleep)

        assert cost == pytest.approx(TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS * COST_PER_SECOND)
        assert measured == []


class TestMediaFileSeconds:
    def test_reads_the_duration_from_the_file_on_disk(self, tmp_path: Path):
        media = tmp_path / "a.wav"
        with wave.open(str(media), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(8000)
            out.writeframes(bytes(2 * 12_000))

        assert media_file_seconds(media) == pytest.approx(1.5)

    def test_undecodable_media_yields_no_duration(self, tmp_path: Path):
        media = tmp_path / "a.wav"
        _ = media.write_bytes(b"not audio at all")

        assert media_file_seconds(media) is None


class TestStartedTranscriptionJob:
    def test_reads_the_media_and_creation_time_from_the_start_response(self):
        started = started_transcription_job(
            {
                "TranscriptionJob": {
                    "TranscriptionJobName": "j",
                    "Media": {"MediaFileUri": "s3://b/a.wav"},
                    "CreationTime": 1.5,
                    "TranscriptionJobStatus": "IN_PROGRESS",
                }
            }
        )

        assert started == TranscriptionJobRecord(
            TranscriptionJobStatus="IN_PROGRESS", CreationTime=1.5, Media={"MediaFileUri": "s3://b/a.wav"}
        )

    @pytest.mark.parametrize("body", [None, {"Message": "throttled"}, {"TranscriptionJob": {"CreationTime": "soon"}}])
    def test_unreadable_start_response_yields_no_record(self, body: dict[str, object] | None):
        assert started_transcription_job(body) is None


class TestMediaPredatesJob:
    LAST_MODIFIED = "Thu, 17 Sep 2026 17:45:00 GMT"
    LAST_MODIFIED_EPOCH = 1_789_667_100.0

    def test_object_written_before_the_job_counts(self):
        assert media_predates_job(httpx.Headers({"Last-Modified": self.LAST_MODIFIED}), self.LAST_MODIFIED_EPOCH + 30)

    def test_object_written_in_the_same_second_as_the_job_counts(self):
        assert media_predates_job(httpx.Headers({"Last-Modified": self.LAST_MODIFIED}), self.LAST_MODIFIED_EPOCH - 0.4)

    def test_object_rewritten_after_the_job_does_not_count(self):
        assert not media_predates_job(
            httpx.Headers({"Last-Modified": self.LAST_MODIFIED}), self.LAST_MODIFIED_EPOCH - 30
        )

    @pytest.mark.parametrize("headers", [{}, {"Last-Modified": "yesterday"}])
    def test_unknown_modification_time_does_not_count(self, headers: dict[str, str]):
        assert not media_predates_job(httpx.Headers(headers), self.LAST_MODIFIED_EPOCH + 30)


VIRTUAL_KEY = UserAPIKeyAuth(api_key="hashed-key-a", user_id="user-a", team_id="team-a")
OTHER_VIRTUAL_KEY = UserAPIKeyAuth(api_key="hashed-key-b", user_id="user-b", team_id="team-b")
ADMIN_KEY = UserAPIKeyAuth(api_key="hashed-admin", user_role=LitellmUserRoles.PROXY_ADMIN)


class TestTranscribeAdminOnlyRefusal:
    @pytest.mark.parametrize("operation", ["StartTranscriptionJob", "GetTranscriptionJob", "DeleteTranscriptionJob"])
    def test_job_scoped_operations_are_open_to_virtual_keys(self, operation: str):
        assert transcribe_admin_only_refusal(operation, VIRTUAL_KEY) is None

    @pytest.mark.parametrize("operation", ["ListTranscriptionJobs", "ListVocabularies", "DeleteVocabulary"])
    def test_account_wide_operations_are_refused_for_virtual_keys(self, operation: str):
        refusal = transcribe_admin_only_refusal(operation, VIRTUAL_KEY)

        assert refusal is not None
        assert refusal.status_code == 403
        assert operation in refusal.detail

    @pytest.mark.parametrize("operation", ["ListTranscriptionJobs", "DeleteVocabulary"])
    def test_account_wide_operations_are_open_to_proxy_admins(self, operation: str):
        assert transcribe_admin_only_refusal(operation, ADMIN_KEY) is None


ALLOWED_BUCKETS = frozenset({"tenant-media", "tenant-transcripts"})


def _start_body(media_uri: str = "s3://tenant-media/call.wav", **members: object) -> dict[str, object]:
    return {"TranscriptionJobName": "j", "Media": {"MediaFileUri": media_uri}, **members}


class TestTranscribeMediaBuckets:
    def test_a_list_of_bucket_names_is_read_from_general_settings(self):
        assert transcribe_media_buckets({"transcribe_media_buckets": ["a", "b"]}) == frozenset({"a", "b"})

    @pytest.mark.parametrize("settings", [{}, {"transcribe_media_buckets": "a"}, {"transcribe_media_buckets": [1]}])
    def test_a_missing_or_malformed_setting_reads_as_unset(self, settings: dict[str, object]):
        assert transcribe_media_buckets(settings) is None


class TestTranscribeStorageRefusal:
    def test_media_and_output_in_listed_buckets_are_allowed(self):
        body = _start_body(OutputBucketName="tenant-transcripts", OutputKey="out/")

        assert transcribe_storage_refusal(body, ALLOWED_BUCKETS, VIRTUAL_KEY) is None

    @pytest.mark.parametrize(
        "media_uri",
        [
            "s3://other-tenant/call.wav",
            "https://tenant-media.s3.us-west-2.amazonaws.com/call.wav",
            "s3://",
        ],
    )
    def test_media_outside_the_listed_buckets_is_refused(self, media_uri: str):
        refusal = transcribe_storage_refusal(_start_body(media_uri), ALLOWED_BUCKETS, VIRTUAL_KEY)

        assert refusal is not None
        assert refusal.status_code == 403
        assert "Media.MediaFileUri" in refusal.detail

    def test_redacted_media_outside_the_listed_buckets_is_refused(self):
        body = {
            "TranscriptionJobName": "j",
            "Media": {"MediaFileUri": "s3://tenant-media/call.wav", "RedactedMediaFileUri": "s3://other-tenant/c.wav"},
        }

        refusal = transcribe_storage_refusal(body, ALLOWED_BUCKETS, VIRTUAL_KEY)

        assert refusal is not None
        assert "Media.RedactedMediaFileUri" in refusal.detail

    @pytest.mark.parametrize("output", ["other-tenant", 7])
    def test_an_output_bucket_outside_the_listed_buckets_is_refused(self, output: object):
        refusal = transcribe_storage_refusal(_start_body(OutputBucketName=output), ALLOWED_BUCKETS, VIRTUAL_KEY)

        assert refusal is not None
        assert refusal.status_code == 403
        assert "OutputBucketName" in refusal.detail

    @pytest.mark.parametrize("member", ["DataAccessRoleArn", "JobExecutionSettings"])
    def test_a_caller_chosen_role_is_refused(self, member: str):
        refusal = transcribe_storage_refusal(_start_body(**{member: "x"}), ALLOWED_BUCKETS, VIRTUAL_KEY)

        assert refusal is not None
        assert refusal.status_code == 403
        assert member in refusal.detail

    def test_an_unset_bucket_list_refuses_virtual_keys(self):
        refusal = transcribe_storage_refusal(_start_body(), None, VIRTUAL_KEY)

        assert refusal is not None
        assert refusal.status_code == 403
        assert "transcribe_media_buckets" in refusal.detail

    @pytest.mark.parametrize("allowed", [None, ALLOWED_BUCKETS])
    def test_proxy_admins_are_not_restricted(self, allowed: frozenset[str] | None):
        body = _start_body("s3://other-tenant/call.wav", DataAccessRoleArn="arn:aws:iam::1:role/r")

        assert transcribe_storage_refusal(body, allowed, ADMIN_KEY) is None


class TestTranscribeOwnedStartRequest:
    def test_the_caller_identity_is_appended_to_the_job_tags(self):
        body = {"TranscriptionJobName": "j", "Tags": [{"Key": "env", "Value": "qa"}]}

        owned = transcribe_owned_start_request(body, VIRTUAL_KEY)

        assert owned == {
            "TranscriptionJobName": "j",
            "Tags": ({"Key": "env", "Value": "qa"}, {"Key": TRANSCRIBE_OWNER_TAG, "Value": "user-a"}),
        }
        assert body == {"TranscriptionJobName": "j", "Tags": [{"Key": "env", "Value": "qa"}]}

    def test_a_request_without_tags_gets_the_owner_tag(self):
        owned = transcribe_owned_start_request({"TranscriptionJobName": "j"}, VIRTUAL_KEY)

        assert owned == {"TranscriptionJobName": "j", "Tags": ({"Key": TRANSCRIBE_OWNER_TAG, "Value": "user-a"},)}

    def test_the_caller_cannot_supply_the_owner_tag(self):
        owned = transcribe_owned_start_request(
            {"TranscriptionJobName": "j", "Tags": [{"Key": TRANSCRIBE_OWNER_TAG, "Value": "user-b"}]}, VIRTUAL_KEY
        )

        assert isinstance(owned, TranscribeRefusal)
        assert owned.status_code == 400

    @pytest.mark.parametrize("tags", ["env=qa", ["env"], {"Key": "env"}])
    def test_malformed_tags_are_refused(self, tags: object):
        owned = transcribe_owned_start_request({"TranscriptionJobName": "j", "Tags": tags}, VIRTUAL_KEY)

        assert isinstance(owned, TranscribeRefusal)
        assert owned.status_code == 400

    def test_a_key_without_any_identity_is_refused(self):
        owned = transcribe_owned_start_request({"TranscriptionJobName": "j"}, UserAPIKeyAuth())

        assert isinstance(owned, TranscribeRefusal)
        assert owned.status_code == 400


def _tagged(owner: str | None) -> dict[str, object]:
    tags = {"Tags": [{"Key": TRANSCRIBE_OWNER_TAG, "Value": owner}]} if owner is not None else {}
    return _job("COMPLETED", **tags)


class TestTranscribeJobAccessRefusal:
    @pytest.mark.asyncio
    async def test_the_key_that_started_the_job_may_read_it(self):
        get_job, seen = _sequence(_tagged("user-a"))

        assert await transcribe_job_access_refusal("job-1", VIRTUAL_KEY, get_job) is None
        assert seen == ["job-1"]

    @pytest.mark.asyncio
    async def test_a_job_started_by_another_key_is_reported_missing(self):
        get_job, _ = _sequence(_tagged("user-b"))

        refusal = await transcribe_job_access_refusal("job-1", VIRTUAL_KEY, get_job)

        assert refusal is not None
        assert refusal.status_code == 404

    @pytest.mark.asyncio
    async def test_a_job_started_outside_the_proxy_is_reported_missing(self):
        get_job, _ = _sequence(_tagged(None))

        refusal = await transcribe_job_access_refusal("job-1", OTHER_VIRTUAL_KEY, get_job)

        assert refusal is not None
        assert refusal.status_code == 404

    @pytest.mark.asyncio
    async def test_a_job_that_cannot_be_looked_up_is_reported_missing(self):
        async def get_job(job_name: str) -> dict[str, object]:
            raise httpx.HTTPStatusError("boom", request=MagicMock(), response=MagicMock())

        refusal = await transcribe_job_access_refusal("job-1", VIRTUAL_KEY, get_job)

        assert refusal is not None
        assert refusal.status_code == 404

    @pytest.mark.asyncio
    async def test_a_non_string_job_name_is_refused_before_any_lookup(self):
        get_job, seen = _sequence(_tagged("user-a"))

        refusal = await transcribe_job_access_refusal(["job-1"], VIRTUAL_KEY, get_job)

        assert refusal is not None
        assert refusal.status_code == 400
        assert seen == []

    @pytest.mark.asyncio
    async def test_a_proxy_admin_reads_any_job_without_a_lookup(self):
        get_job, seen = _sequence(_tagged("user-b"))

        assert await transcribe_job_access_refusal("job-1", ADMIN_KEY, get_job) is None
        assert seen == []


class TestTranscribePassthroughHandler:
    def test_records_model_provider_and_the_given_cost(self):
        logging_obj = _make_logging_obj()
        request_body = {"TranscriptionJobName": "litellm-job-1"}

        handler_result = TranscribePassthroughLoggingHandler.transcribe_passthrough_handler(
            httpx_response=_make_response("StartTranscriptionJob"),
            logging_obj=logging_obj,
            url_route="https://transcribe.us-west-2.amazonaws.com/",
            result='{"TranscriptionJob": {}}',
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body=request_body,
            response_cost=0.0018,
        )

        assert handler_result["result"] == {"response": '{"TranscriptionJob": {}}'}
        assert handler_result["kwargs"]["model"] == "transcribe/StartTranscriptionJob"
        assert handler_result["kwargs"]["custom_llm_provider"] == "transcribe"
        assert handler_result["kwargs"]["response_cost"] == 0.0018
        assert handler_result["kwargs"]["standard_logging_object"]["response_cost"] == 0.0018
        assert logging_obj.model_call_details["model"] == "transcribe/StartTranscriptionJob"
        assert logging_obj.model_call_details["custom_llm_provider"] == "transcribe"
        assert logging_obj.model_call_details["response_cost"] == 0.0018
        assert request_body == {"TranscriptionJobName": "litellm-job-1"}

    def test_read_only_operations_default_to_zero_cost(self):
        handler_result = TranscribePassthroughLoggingHandler.transcribe_passthrough_handler(
            httpx_response=_make_response("GetTranscriptionJob"),
            logging_obj=_make_logging_obj(),
            url_route="https://transcribe.us-west-2.amazonaws.com/",
            result='{"TranscriptionJob": {}}',
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"TranscriptionJobName": "litellm-job-1"},
        )

        assert handler_result["kwargs"]["response_cost"] == 0.0


class TestStartTranscriptionJobIsLoggedAtJobCost:
    @pytest.mark.asyncio
    async def test_success_handler_defers_logging_until_the_job_is_priced(self):
        priced: list[tuple[str, str, float, TranscriptionJobRecord | None]] = []

        async def job_pricer(
            job_name: str, aws_region_name: str, cost_per_second: float, started_job: TranscriptionJobRecord | None
        ) -> float:
            priced.append((job_name, aws_region_name, cost_per_second, started_job))
            return 0.0018

        logged: list[dict[str, object]] = []

        async def log(**kwargs: object) -> None:
            logged.append(kwargs)

        handler = TranscribePassthroughLoggingHandler(job_pricer=job_pricer)
        logging_obj = _make_logging_obj()
        task = handler.schedule_priced_job_logging(
            httpx_response=_make_response("StartTranscriptionJob"),
            response_body={"TranscriptionJob": {}},
            logging_obj=logging_obj,
            url_route="https://transcribe.us-west-2.amazonaws.com/",
            result='{"TranscriptionJob": {}}',
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"TranscriptionJobName": "litellm-job-1"},
            log=log,
            standard_pass_through_logging_payload={"cost_per_request": None},
        )
        await task

        assert priced == [("litellm-job-1", "us-west-2", transcribe_cost_per_second(), TranscriptionJobRecord())]
        assert len(logged) == 1
        assert logged[0]["response_cost"] == 0.0018
        assert logged[0]["model"] == "transcribe/StartTranscriptionJob"
        assert logged[0]["standard_pass_through_logging_payload"] == {"cost_per_request": None}
        assert logging_obj.model_call_details["response_cost"] == 0.0018

    @pytest.mark.asyncio
    async def test_job_is_not_logged_for_free_when_the_rate_leaves_the_cost_map(self, monkeypatch: pytest.MonkeyPatch):
        async def job_pricer(
            job_name: str, aws_region_name: str, cost_per_second: float, started_job: TranscriptionJobRecord | None
        ) -> float:
            raise AssertionError("pricer must not run without a rate")

        logged: list[dict[str, object]] = []

        async def log(**kwargs: object) -> None:
            logged.append(kwargs)

        monkeypatch.delitem(litellm.model_cost, "transcribe/StartTranscriptionJob")
        await TranscribePassthroughLoggingHandler(job_pricer=job_pricer).schedule_priced_job_logging(
            httpx_response=_make_response("StartTranscriptionJob"),
            response_body={"TranscriptionJob": {}},
            logging_obj=_make_logging_obj(),
            url_route="https://transcribe.us-west-2.amazonaws.com/",
            result='{"TranscriptionJob": {}}',
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"TranscriptionJobName": "litellm-job-1"},
            log=log,
        )

        assert logged == []

    @pytest.mark.asyncio
    async def test_pass_through_success_handler_routes_job_starts_to_the_pricer(self):
        scheduled: list[str] = []

        async def job_pricer(
            job_name: str, aws_region_name: str, cost_per_second: float, started_job: TranscriptionJobRecord | None
        ) -> float:
            scheduled.append(job_name)
            return 0.0

        immediate: list[dict[str, object]] = []

        async def log_dispatch(**kwargs: object) -> None:
            immediate.append(kwargs)

        logging = PassThroughEndpointLogging(
            TranscribePassthroughLoggingHandler(job_pricer=job_pricer), log_dispatch=log_dispatch
        )

        await logging.pass_through_async_success_handler(
            httpx_response=_make_response("StartTranscriptionJob"),
            response_body={"TranscriptionJob": {}},
            logging_obj=_make_logging_obj(),
            url_route="https://transcribe.us-west-2.amazonaws.com/",
            result='{"TranscriptionJob": {}}',
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"TranscriptionJobName": "litellm-job-1"},
            passthrough_logging_payload={"url": "https://transcribe.us-west-2.amazonaws.com/"},
            custom_llm_provider="transcribe",
        )
        await asyncio.gather(*logging.transcribe_passthrough_logging_handler._pricing_tasks)

        assert scheduled == ["litellm-job-1"]
        assert [entry["response_cost"] for entry in immediate] == [0.0]

    @pytest.mark.asyncio
    async def test_pass_through_success_handler_prices_a_relayed_start_response_from_its_parsed_body(self):
        started_jobs: list[TranscriptionJobRecord | None] = []
        logged_costs: list[object] = []

        async def job_pricer(
            job_name: str, aws_region_name: str, cost_per_second: float, started_job: TranscriptionJobRecord | None
        ) -> float:
            started_jobs.append(started_job)
            return 18 * COST_PER_SECOND

        async def log_dispatch(**kwargs: object) -> None:
            logged_costs.append(kwargs["response_cost"])

        start_response = {
            "TranscriptionJob": {
                "TranscriptionJobName": "litellm-job-1",
                "TranscriptionJobStatus": "IN_PROGRESS",
                "Media": {"MediaFileUri": "s3://b/started.wav"},
                "CreationTime": 5.0,
            }
        }
        logging = PassThroughEndpointLogging(
            TranscribePassthroughLoggingHandler(job_pricer=job_pricer), log_dispatch=log_dispatch
        )

        await logging.pass_through_async_success_handler(
            httpx_response=await _relayed_response("StartTranscriptionJob", json.dumps(start_response).encode()),
            response_body=start_response,
            logging_obj=_make_logging_obj(),
            url_route="https://transcribe.us-west-2.amazonaws.com/",
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"TranscriptionJobName": "litellm-job-1"},
            passthrough_logging_payload={"url": "https://transcribe.us-west-2.amazonaws.com/"},
            custom_llm_provider="transcribe",
        )
        await asyncio.gather(*logging.transcribe_passthrough_logging_handler._pricing_tasks)

        assert started_jobs == [
            TranscriptionJobRecord(
                TranscriptionJobStatus="IN_PROGRESS", CreationTime=5.0, Media={"MediaFileUri": "s3://b/started.wav"}
            )
        ]
        assert logged_costs == [pytest.approx(18 * COST_PER_SECOND)]


class TestIsTranscribeRoute:
    def test_matches_by_provider_tag(self):
        assert PassThroughEndpointLogging().is_transcribe_route("transcribe")

    def test_does_not_match_other_providers(self):
        assert not PassThroughEndpointLogging().is_transcribe_route("comprehendmedical")

    def test_dispatch_reaches_transcribe_handler(self):
        logging_obj = _make_logging_obj()

        normalized = PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
            httpx_response=_make_response("GetTranscriptionJob"),
            response_body={"TranscriptionJob": {}},
            request_body={"TranscriptionJobName": "litellm-job-1"},
            logging_obj=logging_obj,
            url_route="https://transcribe.us-west-2.amazonaws.com/",
            result='{"TranscriptionJob": {}}',
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            custom_llm_provider="transcribe",
        )

        assert normalized["kwargs"]["model"] == "transcribe/GetTranscriptionJob"
        assert normalized["kwargs"]["response_cost"] == 0.0

    def test_config_driven_passthrough_to_transcribe_host_is_not_claimed(self):
        logging_obj = _make_logging_obj()

        normalized = PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
            httpx_response=_make_response("GetTranscriptionJob"),
            response_body={"TranscriptionJob": {}},
            request_body={"TranscriptionJobName": "litellm-job-1"},
            logging_obj=logging_obj,
            url_route="https://transcribe.us-west-2.amazonaws.com/",
            result='{"TranscriptionJob": {}}',
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            custom_llm_provider=None,
        )

        assert normalized["kwargs"].get("model") != "transcribe/GetTranscriptionJob"
