import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.transcribe_passthrough_logging_handler import (
    TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS,
    TranscribePassthroughLoggingHandler,
    price_transcription_job,
    transcribe_cost_per_second,
    transcribe_supported_operations,
    transcribe_unpriceable_request_reason,
    transcript_audio_seconds,
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


def _make_logging_obj() -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.litellm_call_id = "test-call-id"
    logging_obj.model_call_details = {}
    return logging_obj


async def _no_sleep(_: float) -> None:
    return None


def _job(status: str, transcript_uri: str | None = "https://s3.us-west-2.amazonaws.com/b/t.json") -> dict[str, object]:
    transcript = {"Transcript": {"TranscriptFileUri": transcript_uri}} if transcript_uri else {}
    return {"TranscriptionJob": {"TranscriptionJobStatus": status, **transcript}}


def _sequence(*jobs: dict[str, object]):
    remaining = list(jobs)
    seen: list[str] = []

    async def get_job(job_name: str) -> dict[str, object]:
        seen.append(job_name)
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

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
        body = {"TranscriptionJobName": "j", "Media": {"MediaFileUri": "s3://b/a.wav"}}
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
        ],
    )
    def test_surcharged_features_are_rejected(self, body: dict[str, object], member: str):
        reason = transcribe_unpriceable_request_reason("StartTranscriptionJob", body, COST_PER_SECOND)
        assert reason is not None and member in reason

    def test_model_settings_without_a_custom_model_is_allowed(self):
        body = {"ModelSettings": {}}
        assert transcribe_unpriceable_request_reason("StartTranscriptionJob", body, COST_PER_SECOND) is None


class TestTranscriptAudioSeconds:
    def test_reads_the_last_segment_end_time(self):
        transcript = {
            "results": {
                "audio_segments": [{"end_time": "9.5"}, {"end_time": "17.36"}],
                "items": [{"end_time": "17.23"}, {"type": "punctuation"}],
            }
        }
        assert transcript_audio_seconds(transcript) == 17.36

    def test_falls_back_to_items_when_segments_are_absent(self):
        assert transcript_audio_seconds({"results": {"items": [{"end_time": "3.1"}]}}) == 3.1

    def test_without_timings_is_unknown(self):
        assert transcript_audio_seconds({"results": {"items": []}}) is None
        assert transcript_audio_seconds({"jobName": "j"}) is None


class TestPriceTranscriptionJob:
    @pytest.mark.asyncio
    async def test_polls_until_completed_then_charges_rounded_up_audio_seconds(self):
        get_job, seen = _sequence(_job("IN_PROGRESS"), _job("IN_PROGRESS"), _job("COMPLETED"))
        fetched: list[str] = []

        async def fetch_transcript(uri: str) -> dict[str, object]:
            fetched.append(uri)
            return {"results": {"audio_segments": [{"end_time": "17.36"}]}}

        cost = await price_transcription_job("job-1", COST_PER_SECOND, get_job, fetch_transcript, sleep=_no_sleep)

        assert cost == pytest.approx(18 * COST_PER_SECOND)
        assert seen == ["job-1", "job-1", "job-1"]
        assert fetched == ["https://s3.us-west-2.amazonaws.com/b/t.json"]

    @pytest.mark.asyncio
    async def test_failed_job_costs_nothing(self):
        get_job, _ = _sequence(_job("FAILED", transcript_uri=None))

        async def fetch_transcript(uri: str) -> dict[str, object]:
            raise AssertionError("failed jobs have no transcript to fetch")

        assert (
            await price_transcription_job("job-1", COST_PER_SECOND, get_job, fetch_transcript, sleep=_no_sleep) == 0.0
        )

    @pytest.mark.asyncio
    async def test_job_that_never_finishes_is_charged_the_maximum(self):
        get_job, seen = _sequence(_job("IN_PROGRESS"))

        async def fetch_transcript(uri: str) -> dict[str, object]:
            raise AssertionError("unfinished jobs have no transcript to fetch")

        cost = await price_transcription_job(
            "job-1", COST_PER_SECOND, get_job, fetch_transcript, sleep=_no_sleep, max_attempts=3
        )

        assert cost == pytest.approx(TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS * COST_PER_SECOND)
        assert len(seen) == 3

    @pytest.mark.asyncio
    async def test_unreadable_transcript_is_charged_the_maximum(self):
        get_job, _ = _sequence(_job("COMPLETED"))

        async def fetch_transcript(uri: str) -> dict[str, object]:
            return {"results": {}}

        cost = await price_transcription_job("job-1", COST_PER_SECOND, get_job, fetch_transcript, sleep=_no_sleep)

        assert cost == pytest.approx(TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS * COST_PER_SECOND)

    @pytest.mark.asyncio
    async def test_completed_job_without_transcript_uri_is_charged_the_maximum(self):
        get_job, _ = _sequence(_job("COMPLETED", transcript_uri=None))

        async def fetch_transcript(uri: str) -> dict[str, object]:
            raise AssertionError("no URI to fetch")

        cost = await price_transcription_job("job-1", COST_PER_SECOND, get_job, fetch_transcript, sleep=_no_sleep)

        assert cost == pytest.approx(TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS * COST_PER_SECOND)


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
        priced: list[tuple[str, str, float]] = []

        async def job_pricer(job_name: str, aws_region_name: str, cost_per_second: float) -> float:
            priced.append((job_name, aws_region_name, cost_per_second))
            return 0.0018

        logged: list[dict[str, object]] = []

        async def log(**kwargs: object) -> None:
            logged.append(kwargs)

        handler = TranscribePassthroughLoggingHandler(job_pricer=job_pricer)
        logging_obj = _make_logging_obj()
        task = handler.schedule_priced_job_logging(
            httpx_response=_make_response("StartTranscriptionJob"),
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

        assert priced == [("litellm-job-1", "us-west-2", transcribe_cost_per_second())]
        assert len(logged) == 1
        assert logged[0]["response_cost"] == 0.0018
        assert logged[0]["model"] == "transcribe/StartTranscriptionJob"
        assert logged[0]["standard_pass_through_logging_payload"] == {"cost_per_request": None}
        assert logging_obj.model_call_details["response_cost"] == 0.0018

    @pytest.mark.asyncio
    async def test_job_is_not_logged_for_free_when_the_rate_leaves_the_cost_map(self, monkeypatch: pytest.MonkeyPatch):
        async def job_pricer(job_name: str, aws_region_name: str, cost_per_second: float) -> float:
            raise AssertionError("pricer must not run without a rate")

        logged: list[dict[str, object]] = []

        async def log(**kwargs: object) -> None:
            logged.append(kwargs)

        monkeypatch.delitem(litellm.model_cost, "transcribe/StartTranscriptionJob")
        await TranscribePassthroughLoggingHandler(job_pricer=job_pricer).schedule_priced_job_logging(
            httpx_response=_make_response("StartTranscriptionJob"),
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

        async def job_pricer(job_name: str, aws_region_name: str, cost_per_second: float) -> float:
            scheduled.append(job_name)
            return 0.0

        logging = PassThroughEndpointLogging(TranscribePassthroughLoggingHandler(job_pricer=job_pricer))
        immediate: list[dict[str, object]] = []

        async def handle_logging(**kwargs: object) -> None:
            immediate.append(kwargs)

        logging._handle_logging = handle_logging  # rebind-ok: the shared dispatch is the observable under test

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
