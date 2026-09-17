import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from functools import lru_cache, partial
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    TRANSCRIBE_JOB_MAX_POLLING_ATTEMPTS,
    TRANSCRIBE_JOB_POLLING_INTERVAL_SECONDS,
    TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS,
)
from litellm.litellm_core_utils.aws_partition import get_aws_dns_suffix
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._types import PassThroughEndpointLoggingResultValues, PassThroughEndpointLoggingTypedDict
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.utils import StandardPassThroughResponseObject

TRANSCRIBE_TARGET_PREFIX: Final = "Transcribe"
TRANSCRIBE_CUSTOM_LLM_PROVIDER: Final = "transcribe"
TRANSCRIBE_PRICED_OPERATION: Final = "StartTranscriptionJob"
TRANSCRIBE_PRICED_MODEL: Final = f"{TRANSCRIBE_CUSTOM_LLM_PROVIDER}/{TRANSCRIBE_PRICED_OPERATION}"
TRANSCRIBE_UNPRICED_OPERATIONS: Final = frozenset(
    {"StartCallAnalyticsJob", "StartMedicalScribeJob", "StartMedicalTranscriptionJob"}
)
TRANSCRIBE_SURCHARGE_MEMBERS: Final = ("ContentRedaction", "ToxicityDetection")
TRANSCRIBE_TERMINAL_JOB_STATUSES: Final = frozenset({"COMPLETED", "FAILED"})

JobLookup: TypeAlias = Callable[[str], Awaitable[Mapping[str, object]]]  # mutable-ok: Callable parameter syntax
TranscriptFetch: TypeAlias = Callable[[str], Awaitable[Mapping[str, object]]]  # mutable-ok: Callable parameter syntax
JobPricer: TypeAlias = Callable[[str, str, float], Awaitable[float]]  # mutable-ok: Callable parameter syntax


class GetTranscriptionJobRequest(TypedDict):
    TranscriptionJobName: ReadOnly[str]


class _TranscriptRef(BaseModel):
    model_config = ConfigDict(frozen=True)
    TranscriptFileUri: str | None = None


class _TranscriptionJob(BaseModel):
    model_config = ConfigDict(frozen=True)
    TranscriptionJobStatus: str | None = None
    Transcript: _TranscriptRef | None = None


class _GetTranscriptionJobResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    TranscriptionJob: _TranscriptionJob | None = None


class _TranscriptItem(BaseModel):
    model_config = ConfigDict(frozen=True)
    end_time: float | None = None


class _TranscriptResults(BaseModel):
    model_config = ConfigDict(frozen=True)
    audio_segments: tuple[_TranscriptItem, ...] = ()
    items: tuple[_TranscriptItem, ...] = ()


class _Transcript(BaseModel):
    model_config = ConfigDict(frozen=True)
    results: _TranscriptResults | None = None


class _PricedCostMapEntry(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    input_cost_per_second: float


_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])


class PassThroughLogDispatch(Protocol):
    def __call__(
        self,
        *,
        logging_obj: LiteLLMLoggingObj,
        standard_logging_response_object: PassThroughEndpointLoggingResultValues | None,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs: object,  # kwargs-ok: mirrors the shared pass-through logging dispatch signature
    ) -> Awaitable[None]: ...


@lru_cache(maxsize=1)
def transcribe_supported_operations() -> frozenset[str]:
    """
    Operation names of the Amazon Transcribe JSON 1.1 API, read from the botocore
    service model so the allowlist tracks the installed SDK instead of a hand-typed copy.
    """
    from botocore.session import get_session

    return frozenset(get_session().get_service_model("transcribe").operation_names)


def transcribe_cost_per_second() -> float | None:
    try:
        return _PricedCostMapEntry.model_validate(litellm.model_cost.get(TRANSCRIBE_PRICED_MODEL)).input_cost_per_second
    except ValidationError:
        return None


def transcribe_unpriceable_request_reason(
    operation: str,
    request_body: Mapping[str, object],
    cost_per_second: float | None,
) -> str | None:
    if operation in TRANSCRIBE_UNPRICED_OPERATIONS:
        return (
            f"{operation} is billed per second of audio at a rate LiteLLM does not price yet, so it cannot be"
            f" submitted through this route; only {TRANSCRIBE_PRICED_OPERATION} is priced and budgeted"
        )
    if operation != TRANSCRIBE_PRICED_OPERATION:
        return None
    if cost_per_second is None:
        return (
            f"{TRANSCRIBE_PRICED_MODEL} has no input_cost_per_second in the LiteLLM model cost map, so billable"
            " transcription jobs cannot be submitted through this route"
        )
    model_settings: Final = request_body.get("ModelSettings")
    custom_language_model: Final = (
        ("ModelSettings.LanguageModelName",)
        if isinstance(model_settings, Mapping) and "LanguageModelName" in model_settings
        else ()
    )
    surcharges: Final = tuple(m for m in TRANSCRIBE_SURCHARGE_MEMBERS if m in request_body) + custom_language_model
    if not surcharges:
        return None
    return (
        f"{TRANSCRIBE_PRICED_OPERATION} with {', '.join(surcharges)} adds a per-second surcharge LiteLLM does not"
        " price yet; remove it to submit the job through this route"
    )


def transcription_job_cost(audio_seconds: float, cost_per_second: float) -> float:
    return math.ceil(audio_seconds) * cost_per_second


def transcribe_max_job_cost(cost_per_second: float) -> float:
    return transcription_job_cost(TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS, cost_per_second)


def transcript_audio_seconds(transcript: Mapping[str, object]) -> float | None:
    results: Final = _Transcript.model_validate(transcript).results
    if results is None:
        return None
    end_times: Final = tuple(
        item.end_time for item in results.audio_segments + results.items if item.end_time is not None
    )
    return max(end_times, default=None)


async def await_transcription_job(
    job_name: str,
    get_job: JobLookup,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    max_attempts: int = TRANSCRIBE_JOB_MAX_POLLING_ATTEMPTS,
) -> _TranscriptionJob | None:
    for _ in range(max_attempts):
        job = _GetTranscriptionJobResponse.model_validate(await get_job(job_name)).TranscriptionJob
        if job is not None and job.TranscriptionJobStatus in TRANSCRIBE_TERMINAL_JOB_STATUSES:
            return job
        await sleep(TRANSCRIBE_JOB_POLLING_INTERVAL_SECONDS)
    return None


async def price_transcription_job(
    job_name: str,
    cost_per_second: float,
    get_job: JobLookup,
    fetch_transcript: TranscriptFetch,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    max_attempts: int = TRANSCRIBE_JOB_MAX_POLLING_ATTEMPTS,
) -> float:
    """
    Amazon Transcribe bills per second of audio and reports the duration only inside the
    transcript artifact, so the job is polled to completion and priced from the last end_time.
    Anything that stops the duration from being read is charged as the longest media AWS accepts.
    """
    job: Final = await await_transcription_job(job_name, get_job, sleep=sleep, max_attempts=max_attempts)
    if job is None:
        verbose_proxy_logger.warning("Transcribe job %s did not finish while polling, charging maximum", job_name)
        return transcribe_max_job_cost(cost_per_second)
    if job.TranscriptionJobStatus == "FAILED":
        return 0.0
    transcript_uri: Final = job.Transcript.TranscriptFileUri if job.Transcript is not None else None
    if transcript_uri is None:
        return transcribe_max_job_cost(cost_per_second)
    audio_seconds: Final = transcript_audio_seconds(await fetch_transcript(transcript_uri))
    if audio_seconds is None:
        return transcribe_max_job_cost(cost_per_second)
    return transcription_job_cost(audio_seconds, cost_per_second)


def _as_json_object(response: httpx.Response) -> Mapping[str, object]:
    return _JSON_OBJECT.validate_python(response.raise_for_status().json())


def transcribe_job_lookup(aws_region_name: str) -> JobLookup:
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, run_aws_signing, sign_aws_json_post

    url: Final = f"https://transcribe.{aws_region_name}.{get_aws_dns_suffix(aws_region_name)}/"
    headers: Final = MappingProxyType(
        {
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": f"{TRANSCRIBE_TARGET_PREFIX}.GetTranscriptionJob",
        }
    )

    async def get_job(job_name: str) -> Mapping[str, object]:
        body: Final[GetTranscriptionJobRequest] = {"TranscriptionJobName": job_name}
        payload: Final = json.dumps(body)
        prepped: Final = await run_aws_signing(
            sign_aws_json_post,
            get_credentials=partial(BaseAWSLLM().get_credentials, aws_region_name=aws_region_name),
            service_name="transcribe",
            aws_region_name=aws_region_name,
            url=url,
            body=payload,
            headers=headers,
        )
        client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.PassThroughEndpoint)
        signed_headers: Final = dict(prepped.headers.items())  # mutable-ok: AsyncHTTPHandler.post takes a dict
        return _as_json_object(await client.post(str(prepped.url), data=payload, headers=signed_headers))

    return get_job


def transcribe_transcript_fetch(aws_region_name: str) -> TranscriptFetch:
    from botocore.auth import S3SigV4Auth
    from botocore.awsrequest import AWSRequest

    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, run_aws_signing

    def sign_s3_get(transcript_uri: str) -> dict[str, str]:  # mutable-ok: AsyncHTTPHandler.get takes a dict
        aws_request: Final = AWSRequest(method="GET", url=transcript_uri)
        credentials: Final = BaseAWSLLM().get_credentials(aws_region_name=aws_region_name)
        S3SigV4Auth(credentials, "s3", aws_region_name).add_auth(aws_request)
        return dict(aws_request.prepare().headers.items())  # mutable-ok: AsyncHTTPHandler.get takes a dict

    async def fetch_transcript(transcript_uri: str) -> Mapping[str, object]:
        presigned: Final = "X-Amz-Signature" in httpx.URL(transcript_uri).params
        headers: Final = None if presigned else await run_aws_signing(sign_s3_get, transcript_uri)
        client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.PassThroughEndpoint)
        return _as_json_object(await client.get(transcript_uri, headers=headers))

    return fetch_transcript


async def price_transcription_job_live(job_name: str, aws_region_name: str, cost_per_second: float) -> float:
    try:
        return await price_transcription_job(
            job_name,
            cost_per_second,
            get_job=transcribe_job_lookup(aws_region_name),
            fetch_transcript=transcribe_transcript_fetch(aws_region_name),
        )
    except Exception as e:  # noqa: BLE001  # an unreadable job must still be charged, so fail closed at the maximum
        verbose_proxy_logger.exception("Pricing Transcribe job %s failed, charging maximum: %s", job_name, e)
        return transcribe_max_job_cost(cost_per_second)


class TranscribePassthroughLoggingHandler:
    def __init__(self, job_pricer: JobPricer = price_transcription_job_live) -> None:
        self._job_pricer: Final = job_pricer
        self._pricing_tasks: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: asyncio holds tasks weakly

    @staticmethod
    def _operation_from_response(httpx_response: httpx.Response) -> str:
        headers: Final[Mapping[str, str]] = httpx_response.request.headers
        target: Final = headers.get("x-amz-target", "")
        return target.split(".")[-1]

    @staticmethod
    def is_priced_job_start(httpx_response: httpx.Response) -> bool:
        return (
            TranscribePassthroughLoggingHandler._operation_from_response(httpx_response) == TRANSCRIBE_PRICED_OPERATION
        )

    def schedule_priced_job_logging(
        self,
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: Mapping[str, object],
        log: PassThroughLogDispatch,
        **kwargs: object,  # kwargs-ok: the passthrough logging dispatch forwards shared logging kwargs to every handler
    ) -> asyncio.Task[None]:
        task: Final = asyncio.create_task(
            self._price_then_log(
                httpx_response=httpx_response,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                log=log,
                **kwargs,
            )
        )
        self._pricing_tasks.add(task)
        task.add_done_callback(self._pricing_tasks.discard)
        return task

    async def _price_then_log(
        self,
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: Mapping[str, object],
        log: PassThroughLogDispatch,
        **kwargs: object,  # kwargs-ok: the passthrough logging dispatch forwards shared logging kwargs to every handler
    ) -> None:
        cost_per_second: Final = transcribe_cost_per_second()
        if cost_per_second is None:
            verbose_proxy_logger.error("%s left the model cost map, spend not recorded", TRANSCRIBE_PRICED_MODEL)
            return
        job_name: Final = request_body.get("TranscriptionJobName")
        aws_region_name: Final = httpx_response.request.url.host.split(".")[1]
        response_cost: Final = await self._job_pricer(
            job_name if isinstance(job_name, str) else "", aws_region_name, cost_per_second
        )
        payload: Final = self.transcribe_passthrough_handler(
            httpx_response=httpx_response,
            logging_obj=logging_obj,
            url_route=url_route,
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=cache_hit,
            request_body=request_body,
            response_cost=response_cost,
            **kwargs,
        )
        await log(
            logging_obj=logging_obj,
            standard_logging_response_object=payload["result"],
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=cache_hit,
            **payload["kwargs"],
        )

    @staticmethod
    def transcribe_passthrough_handler(
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: Mapping[str, object],
        response_cost: float = 0.0,
        **kwargs: object,  # kwargs-ok: the passthrough logging dispatch forwards shared logging kwargs to every handler
    ) -> PassThroughEndpointLoggingTypedDict:
        try:
            operation: Final = TranscribePassthroughLoggingHandler._operation_from_response(httpx_response)
            model_name: Final = f"{TRANSCRIBE_CUSTOM_LLM_PROVIDER}/{operation}"

            updated_kwargs: Final = {  # mutable-ok: the logging pipeline requires a plain kwargs dict
                **kwargs,
                "model": model_name,
                "custom_llm_provider": TRANSCRIBE_CUSTOM_LLM_PROVIDER,
                "response_cost": response_cost,
            }
            logging_obj.model_call_details.update(
                model=model_name,
                custom_llm_provider=TRANSCRIBE_CUSTOM_LLM_PROVIDER,
                response_cost=response_cost,
            )

            standard_logging_object: Final = get_standard_logging_object_payload(
                kwargs=updated_kwargs,
                init_response_obj=StandardPassThroughResponseObject(response=result),
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                status="success",
            )

            handler_payload: Final[PassThroughEndpointLoggingTypedDict] = {
                "result": StandardPassThroughResponseObject(response=result),
                "kwargs": {**updated_kwargs, "standard_logging_object": standard_logging_object},
            }
        except Exception as e:  # noqa: BLE001  # logging must never fail the forwarded request
            verbose_proxy_logger.exception("Error in Amazon Transcribe passthrough logging handler: %s", e)
            fallback_payload: Final[PassThroughEndpointLoggingTypedDict] = {
                "result": StandardPassThroughResponseObject(response=result),
                "kwargs": kwargs,
            }
            return fallback_payload
        return handler_payload
