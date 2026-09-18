import asyncio
import json
import math
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from functools import lru_cache, partial
from pathlib import Path
from types import MappingProxyType
from typing import IO, Final, Protocol, TypeAlias
from urllib.parse import quote

import httpx
import soundfile
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    TRANSCRIBE_JOB_MAX_POLLING_ATTEMPTS,
    TRANSCRIBE_JOB_POLLING_INTERVAL_SECONDS,
    TRANSCRIBE_MAX_MEDIA_BYTES,
    TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS,
    TRANSCRIBE_MEASURABLE_MEDIA_FORMATS,
    TRANSCRIBE_MEDIA_DOWNLOAD_CONCURRENCY,
    TRANSCRIBE_MEDIA_FETCH_ATTEMPTS,
    TRANSCRIBE_MEDIA_LAST_MODIFIED_TOLERANCE_SECONDS,
)
from litellm.litellm_core_utils.aws_partition import get_aws_dns_suffix
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._types import (
    PassThroughEndpointLoggingResultValues,
    PassThroughEndpointLoggingTypedDict,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.resource_ownership import (
    get_primary_resource_owner_scope,
    is_proxy_admin,
    user_can_access_resource_owner,
)
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
TRANSCRIBE_MISSING_JOB_ERRORS: Final = frozenset({"BadRequestException", "NotFoundException"})
TRANSCRIBE_OWNER_TAG: Final = "litellm-owner"
TRANSCRIBE_OWNED_JOB_OPERATIONS: Final = frozenset({"GetTranscriptionJob", "DeleteTranscriptionJob"})
TRANSCRIBE_MEDIA_BUCKETS_SETTING: Final = "transcribe_media_buckets"
TRANSCRIBE_ROLE_MEMBERS: Final = ("DataAccessRoleArn", "JobExecutionSettings")
TRANSCRIBE_MEDIA_URI_MEMBERS: Final = ("MediaFileUri", "RedactedMediaFileUri")

JobLookup: TypeAlias = Callable[[str], Awaitable[Mapping[str, object]]]  # mutable-ok: Callable parameter syntax
MediaDurationProbe: TypeAlias = Callable[[str, float], Awaitable[float | None]]  # mutable-ok: Callable parameter syntax


class GetTranscriptionJobRequest(TypedDict):
    TranscriptionJobName: ReadOnly[str]


class _MediaRef(BaseModel):
    model_config = ConfigDict(frozen=True)
    MediaFileUri: str | None = None


class _JobTag(BaseModel):
    model_config = ConfigDict(frozen=True)
    Key: str | None = None
    Value: str | None = None


class TranscriptionJobRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    TranscriptionJobStatus: str | None = None
    CreationTime: float | None = None
    Media: _MediaRef | None = None
    Tags: tuple[_JobTag, ...] = ()


class _TranscriptionJobResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    TranscriptionJob: TranscriptionJobRecord | None = None


@dataclass(frozen=True, slots=True)
class MissingJob:
    """Transcribe no longer knows the job, so polling it again can never reach a terminal status."""


StartedJob: TypeAlias = TranscriptionJobRecord | None
JobPricer: TypeAlias = Callable[[str, str, float, StartedJob], Awaitable[float]]  # mutable-ok: Callable params


class _PricedCostMapEntry(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    input_cost_per_second: float


_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])
_JSON_OBJECTS: Final = TypeAdapter(tuple[Mapping[str, object], ...])
_BUCKET_NAMES: Final = TypeAdapter(frozenset[str])


@dataclass(frozen=True, slots=True)
class TranscribeRefusal:
    status_code: int
    detail: str


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
    surcharges: Final = tuple(m for m in TRANSCRIBE_SURCHARGE_MEMBERS if m in request_body) + tuple(
        _custom_language_model_members(request_body)
    )
    if surcharges:
        return (
            f"{TRANSCRIBE_PRICED_OPERATION} with {', '.join(surcharges)} adds a per-second surcharge LiteLLM does not"
            " price yet; remove it to submit the job through this route"
        )
    if requested_media_format(request_body) not in TRANSCRIBE_MEASURABLE_MEDIA_FORMATS:
        return (
            "LiteLLM bills a transcription job by reading the length of the media file, which it can only do for"
            f" {', '.join(sorted(TRANSCRIBE_MEASURABLE_MEDIA_FORMATS))}; set MediaFormat to one of those or point"
            " Media.MediaFileUri at a file with that extension"
        )
    return None


def _custom_language_model_members(request_body: Mapping[str, object]) -> tuple[str, ...]:
    model_settings: Final = request_body.get("ModelSettings")
    language_id_settings: Final = request_body.get("LanguageIdSettings")
    from_model_settings: Final = (
        ("ModelSettings.LanguageModelName",)
        if isinstance(model_settings, Mapping) and "LanguageModelName" in model_settings
        else ()
    )
    from_language_id: Final = (
        tuple(
            f"LanguageIdSettings.{language}.LanguageModelName"
            for language, settings in _JSON_OBJECT.validate_python(language_id_settings).items()
            if isinstance(settings, Mapping) and "LanguageModelName" in settings
        )
        if isinstance(language_id_settings, Mapping)
        else ()
    )
    return from_model_settings + from_language_id


def requested_media_format(request_body: Mapping[str, object]) -> str | None:
    media_format: Final = request_body.get("MediaFormat")
    if isinstance(media_format, str):
        return media_format.lower()
    media: Final = request_body.get("Media")
    media_uri: Final = _JSON_OBJECT.validate_python(media).get("MediaFileUri") if isinstance(media, Mapping) else None
    if not isinstance(media_uri, str):
        return None
    path: Final = httpx.URL(media_uri).path if "://" in media_uri else media_uri
    _, dot, suffix = path.rpartition(".")
    return suffix.lower() if dot else None


def transcribe_admin_only_refusal(operation: str, user_api_key_dict: UserAPIKeyAuth) -> TranscribeRefusal | None:
    if (
        operation == TRANSCRIBE_PRICED_OPERATION
        or operation in TRANSCRIBE_OWNED_JOB_OPERATIONS
        or is_proxy_admin(user_api_key_dict)
    ):
        return None
    return TranscribeRefusal(
        403,
        f"{operation} reaches every Amazon Transcribe resource in the AWS account, so only a proxy admin may call it;"
        f" other keys may {TRANSCRIBE_PRICED_OPERATION} and {' or '.join(sorted(TRANSCRIBE_OWNED_JOB_OPERATIONS))}"
        " for the jobs they started",
    )


def transcribe_media_buckets(general_settings: Mapping[str, object]) -> frozenset[str] | None:
    try:
        return _BUCKET_NAMES.validate_python(general_settings.get(TRANSCRIBE_MEDIA_BUCKETS_SETTING))
    except ValidationError:
        return None


def s3_bucket_name(uri: object) -> str | None:
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        return None
    bucket, _, _ = uri.removeprefix("s3://").partition("/")
    return bucket or None


def transcribe_storage_refusal(
    request_body: Mapping[str, object],
    allowed_buckets: frozenset[str] | None,
    user_api_key_dict: UserAPIKeyAuth,
) -> TranscribeRefusal | None:
    """
    Transcribe reads the media and writes the transcript with the proxy's own AWS credentials, so a
    non-admin key may only point a job at buckets the operator listed; otherwise any object those
    credentials can reach could be transcribed and read back through the caller's own job.
    """
    if is_proxy_admin(user_api_key_dict):
        return None
    if allowed_buckets is None:
        return TranscribeRefusal(
            403,
            f"general_settings.{TRANSCRIBE_MEDIA_BUCKETS_SETTING} is not a list of S3 bucket names, so only a proxy"
            f" admin may {TRANSCRIBE_PRICED_OPERATION}; list the buckets other keys may read media from and write"
            " transcripts to",
        )
    roles: Final = tuple(m for m in TRANSCRIBE_ROLE_MEMBERS if m in request_body)
    if roles:
        return TranscribeRefusal(
            403,
            f"{', '.join(roles)} would run the job under a role other than the proxy's own AWS credentials, so"
            " only a proxy admin may set it",
        )
    media: Final = request_body.get("Media")
    media_uris: Final = (
        tuple((f"Media.{m}", s3_bucket_name(media.get(m))) for m in TRANSCRIBE_MEDIA_URI_MEMBERS if m in media)
        if isinstance(media, Mapping)
        else ()
    )
    output: Final = request_body.get("OutputBucketName")
    locations: Final = media_uris + (
        (("OutputBucketName", output if isinstance(output, str) else None),)
        if "OutputBucketName" in request_body
        else ()
    )
    offending: Final = tuple(member for member, bucket in locations if bucket not in allowed_buckets)
    if offending:
        return TranscribeRefusal(
            403,
            f"{', '.join(offending)} must name one of the S3 buckets in general_settings."
            f"{TRANSCRIBE_MEDIA_BUCKETS_SETTING} ({', '.join(sorted(allowed_buckets))}), as s3://bucket/key for media",
        )
    return None


def transcribe_owned_start_request(
    request_body: Mapping[str, object], user_api_key_dict: UserAPIKeyAuth
) -> dict[str, object] | TranscribeRefusal:
    owner: Final = get_primary_resource_owner_scope(user_api_key_dict)
    if owner is None:
        return TranscribeRefusal(400, "The calling key has no identity to record as the owner of the transcription job")
    try:
        tags: Final = _JSON_OBJECTS.validate_python(request_body.get("Tags", ()))
    except ValidationError:
        return TranscribeRefusal(400, "Tags must be a list of objects with Key and Value members")
    if any(tag.get("Key") == TRANSCRIBE_OWNER_TAG for tag in tags):
        return TranscribeRefusal(
            400, f"The {TRANSCRIBE_OWNER_TAG} tag is assigned by LiteLLM and cannot be supplied by the caller"
        )
    owner_tag: Final = _JobTag(Key=TRANSCRIBE_OWNER_TAG, Value=owner).model_dump()
    return {**request_body, "Tags": (*tags, owner_tag)}  # mutable-ok: json.dumps and the body state key take a dict


async def transcribe_job_access_refusal(
    job_name: object, user_api_key_dict: UserAPIKeyAuth, get_job: JobLookup
) -> TranscribeRefusal | None:
    if is_proxy_admin(user_api_key_dict):
        return None
    if not isinstance(job_name, str):
        return TranscribeRefusal(400, "TranscriptionJobName must be a string")
    not_found: Final = TranscribeRefusal(
        404, f"No transcription job named {job_name} was started through this proxy by the calling key"
    )
    try:
        job: Final = _TranscriptionJobResponse.model_validate(await get_job(job_name)).TranscriptionJob
    except Exception as e:  # noqa: BLE001  # a job that cannot be read cannot be shown to belong to the caller
        verbose_proxy_logger.warning("Looking up Transcribe job %s for an ownership check failed: %s", job_name, e)
        return not_found
    owner: Final = (
        next((tag.Value for tag in job.Tags if tag.Key == TRANSCRIBE_OWNER_TAG), None) if job is not None else None
    )
    return None if user_can_access_resource_owner(owner, user_api_key_dict) else not_found


def transcription_job_cost(audio_seconds: float, cost_per_second: float) -> float:
    return math.ceil(audio_seconds) * cost_per_second


def transcribe_max_job_cost(cost_per_second: float) -> float:
    return transcription_job_cost(TRANSCRIBE_MAX_MEDIA_DURATION_SECONDS, cost_per_second)


def started_transcription_job(response_body: str) -> TranscriptionJobRecord | None:
    try:
        return _TranscriptionJobResponse.model_validate_json(response_body).TranscriptionJob
    except ValidationError:
        return None


def aws_error_type(response: httpx.Response) -> str | None:
    try:
        error_type: Final = _JSON_OBJECT.validate_python(response.json()).get("__type")
    except (ValueError, ValidationError):
        return None
    return error_type.rsplit("#", 1)[-1] if isinstance(error_type, str) else None


async def _poll_transcription_job(job_name: str, get_job: JobLookup) -> TranscriptionJobRecord | MissingJob | None:
    try:
        job: Final = _TranscriptionJobResponse.model_validate(await get_job(job_name)).TranscriptionJob
    except httpx.HTTPStatusError as e:
        if aws_error_type(e.response) in TRANSCRIBE_MISSING_JOB_ERRORS:
            verbose_proxy_logger.warning(
                "Transcribe job %s no longer exists, pricing the media it was started with", job_name
            )
            return MissingJob()
        verbose_proxy_logger.warning("Polling Transcribe job %s failed, retrying: %s", job_name, e)
        return None
    except Exception as e:  # noqa: BLE001  # a failed poll is retried on the next tick instead of ending pricing
        verbose_proxy_logger.warning("Polling Transcribe job %s failed, retrying: %s", job_name, e)
        return None
    return job if job is not None and job.TranscriptionJobStatus in TRANSCRIBE_TERMINAL_JOB_STATUSES else None


async def await_transcription_job(
    job_name: str,
    get_job: JobLookup,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    max_attempts: int = TRANSCRIBE_JOB_MAX_POLLING_ATTEMPTS,
) -> TranscriptionJobRecord | MissingJob | None:
    for _ in range(max_attempts):
        job = await _poll_transcription_job(job_name, get_job)
        if job is not None:
            return job
        await sleep(TRANSCRIBE_JOB_POLLING_INTERVAL_SECONDS)
    return None


async def measure_media_seconds(
    media_uri: str,
    job_created_at: float,
    media_seconds: MediaDurationProbe,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    attempts: int = TRANSCRIBE_MEDIA_FETCH_ATTEMPTS,
) -> float | None:
    for attempt in range(1, attempts + 1):
        try:
            return await media_seconds(media_uri, job_created_at)
        except Exception as e:  # noqa: BLE001  # the media is retried, then charged at the maximum if still unreadable
            verbose_proxy_logger.warning("Measuring Transcribe media %s failed (attempt %d): %s", media_uri, attempt, e)
            if attempt < attempts:
                await sleep(TRANSCRIBE_JOB_POLLING_INTERVAL_SECONDS)
    return None


async def price_transcription_job(
    job_name: str,
    cost_per_second: float,
    get_job: JobLookup,
    media_seconds: MediaDurationProbe,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    max_attempts: int = TRANSCRIBE_JOB_MAX_POLLING_ATTEMPTS,
    started_job: TranscriptionJobRecord | None = None,
) -> float:
    """
    Amazon Transcribe bills every second of the media file, silence included, and reports no
    duration itself, so the job is polled to completion and the media it transcribed is measured.
    The measurement only counts when the object has not been rewritten since the job was created,
    which is what ties it to the bytes Transcribe read. A job deleted before it is polled is
    measured from the media named in its StartTranscriptionJob response. Anything that stops the
    duration from being read is charged as the longest media AWS accepts.
    """
    outcome: Final = await await_transcription_job(job_name, get_job, sleep=sleep, max_attempts=max_attempts)
    if outcome is None:
        verbose_proxy_logger.warning("Transcribe job %s did not finish while polling, charging maximum", job_name)
        return transcribe_max_job_cost(cost_per_second)
    if isinstance(outcome, TranscriptionJobRecord) and outcome.TranscriptionJobStatus == "FAILED":
        return 0.0
    job: Final = outcome if isinstance(outcome, TranscriptionJobRecord) else started_job
    media_uri: Final = job.Media.MediaFileUri if job is not None and job.Media is not None else None
    if job is None or media_uri is None or job.CreationTime is None:
        return transcribe_max_job_cost(cost_per_second)
    audio_seconds: Final = await measure_media_seconds(media_uri, job.CreationTime, media_seconds, sleep=sleep)
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


def s3_media_url(media_uri: str, aws_region_name: str) -> str | None:
    """
    Transcribe accepts media as s3://bucket/key or as an https S3 URL; the bucket is required to
    live in the job's region, so the s3 form maps onto that region's endpoint. Buckets with dots in
    their name use the path-style form because they cannot match the virtual-hosted wildcard
    certificate. The proxy's AWS signature is only ever sent to that partition's own hosts.
    """
    dns_suffix: Final = get_aws_dns_suffix(aws_region_name)
    if not media_uri.startswith("s3://"):
        url: Final = httpx.URL(media_uri)
        return media_uri if url.scheme == "https" and url.host.endswith(f".{dns_suffix}") else None
    bucket, _, key = media_uri.removeprefix("s3://").partition("/")
    if "." in bucket:
        return f"https://s3.{aws_region_name}.{dns_suffix}/{bucket}/{quote(key)}"
    return f"https://{bucket}.s3.{aws_region_name}.{dns_suffix}/{quote(key)}"


def media_predates_job(headers: Mapping[str, str], job_created_at: float) -> bool:
    try:
        modified_at: Final = parsedate_to_datetime(headers["last-modified"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return False
    return modified_at <= job_created_at + TRANSCRIBE_MEDIA_LAST_MODIFIED_TOLERANCE_SECONDS


async def write_media_within_limit(response: httpx.Response, media_file: IO[bytes], max_bytes: int) -> bool:
    if int(response.headers.get("content-length", "0")) > max_bytes:
        return False
    async for chunk in response.aiter_bytes():
        _ = media_file.write(chunk)
        if media_file.tell() > max_bytes:
            return False
    return True


def media_file_seconds(path: Path) -> float | None:
    try:
        with soundfile.SoundFile(str(path)) as audio:
            return len(audio) / audio.samplerate
    except (RuntimeError, ValueError, OSError) as e:
        verbose_proxy_logger.warning("Transcribe media could not be decoded for its duration: %s", e)
        return None


def transcribe_media_duration_probe(aws_region_name: str, download_slots: asyncio.Semaphore) -> MediaDurationProbe:
    from botocore.auth import S3SigV4Auth
    from botocore.awsrequest import AWSRequest

    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, run_aws_signing

    def sign_s3_get(url: str) -> dict[str, str]:  # mutable-ok: httpx request headers take a dict
        aws_request: Final = AWSRequest(method="GET", url=url)
        credentials: Final = BaseAWSLLM().get_credentials(aws_region_name=aws_region_name)
        S3SigV4Auth(credentials, "s3", aws_region_name).add_auth(aws_request)
        return dict(aws_request.prepare().headers.items())  # mutable-ok: httpx request headers take a dict

    async def media_seconds(media_uri: str, job_created_at: float) -> float | None:
        url: Final = s3_media_url(media_uri, aws_region_name)
        if url is None:
            return None
        headers: Final = await run_aws_signing(sign_s3_get, url)
        client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.PassThroughEndpoint).client
        async with download_slots:
            with tempfile.NamedTemporaryFile() as media_file:
                async with client.stream("GET", url, headers=headers) as response:
                    _ = response.raise_for_status()
                    if not media_predates_job(response.headers, job_created_at):
                        verbose_proxy_logger.warning(
                            "Transcribe media %s was rewritten after the job was created, charging maximum", media_uri
                        )
                        return None
                    if not await write_media_within_limit(response, media_file, TRANSCRIBE_MAX_MEDIA_BYTES):
                        verbose_proxy_logger.warning(
                            "Transcribe media %s exceeds the size cap, charging maximum", media_uri
                        )
                        return None
                media_file.flush()
                return await asyncio.to_thread(media_file_seconds, Path(media_file.name))

    return media_seconds


async def price_transcription_job_live(
    job_name: str,
    aws_region_name: str,
    cost_per_second: float,
    started_job: TranscriptionJobRecord | None,
    download_slots: asyncio.Semaphore,
) -> float:
    try:
        return await price_transcription_job(
            job_name,
            cost_per_second,
            get_job=transcribe_job_lookup(aws_region_name),
            media_seconds=transcribe_media_duration_probe(aws_region_name, download_slots),
            started_job=started_job,
        )
    except Exception as e:  # noqa: BLE001  # an unreadable job must still be charged, so fail closed at the maximum
        verbose_proxy_logger.exception("Pricing Transcribe job %s failed, charging maximum: %s", job_name, e)
        return transcribe_max_job_cost(cost_per_second)


class TranscribePassthroughLoggingHandler:
    def __init__(self, job_pricer: JobPricer | None = None) -> None:
        self._job_pricer: Final = (
            job_pricer
            if job_pricer is not None
            else partial(
                price_transcription_job_live,
                download_slots=asyncio.Semaphore(TRANSCRIBE_MEDIA_DOWNLOAD_CONCURRENCY),
            )
        )
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
            job_name if isinstance(job_name, str) else "",
            aws_region_name,
            cost_per_second,
            started_transcription_job(httpx_response.text),
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
