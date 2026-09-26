"""
s3 Bucket Logging Integration

async_log_success_event: Processes the event, stores it in memory for DEFAULT_S3_FLUSH_INTERVAL_SECONDS seconds or until DEFAULT_S3_BATCH_SIZE and then flushes to s3
async_log_failure_event: Processes the event, stores it in memory for DEFAULT_S3_FLUSH_INTERVAL_SECONDS seconds or until DEFAULT_S3_BATCH_SIZE and then flushes to s3
NOTE 1: S3 does not provide a BATCH PUT API endpoint; by default each element is uploaded concurrently with the fixed s3_max_concurrent_uploads bound (or an adaptive bound when s3_adaptive_concurrency is on, backing off only on throttling), or with s3_batch_file_upload the whole flush is written as one .jsonl file
"""

import asyncio
import contextvars
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, cast
from urllib.parse import quote
from uuid import uuid4

import httpx

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.constants import (
    DEFAULT_S3_BATCH_SIZE,
    DEFAULT_S3_FLUSH_INTERVAL_SECONDS,
    DEFAULT_S3_MAX_ADAPTIVE_CONCURRENCY,
    DEFAULT_S3_MAX_CONCURRENT_UPLOADS,
)
from litellm.integrations.adaptive_concurrency import AdaptiveConcurrencyLimiter, PutSample
from litellm.integrations.s3 import (
    get_s3_object_download_filename,
    get_s3_object_key,
    prompts_only_payload,
    resolve_s3_adaptive_concurrency,
    resolve_s3_batch_file_upload,
    resolve_s3_drop_on_terminal_error,
    resolve_s3_log_prompts_only,
    resolve_s3_max_adaptive_concurrency,
    resolve_s3_max_concurrent_uploads,
    resolve_s3_max_queue_size,
    resolve_s3_max_retry_age_seconds,
    resolve_sse_params,
)
from litellm.litellm_core_utils.aws_partition import get_aws_dns_suffix
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, run_aws_signing
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.s3_v2 import s3BatchLoggingElement
from litellm.types.utils import StandardAuditLogPayload, StandardLoggingPayload

from .custom_batch_logger import CustomBatchLogger

if TYPE_CHECKING:
    from botocore.credentials import Credentials


UploadOutcome = Literal["delivered", "retry", "dropped"]

_TERMINAL_ERROR_CODES: Final = frozenset(
    {
        "EntityTooLarge",
        "InvalidArgument",
        "MalformedXML",
        "InvalidDigest",
        "KeyTooLongError",
        "BadDigest",
        "InvalidRequest",
    }
)
_BODY_CODED_STATUSES: Final = frozenset({400, 403})
_RETRYABLE_STATUSES: Final = frozenset({403, 500, 503})
_S3_ERROR_CODE: Final = re.compile(r"<Code>([^<]+)</Code>")


@dataclass(frozen=True, slots=True)
class _PreparedPut:
    json_string: str
    headers: Mapping[str, str]


def _s3_error_code(response: httpx.Response) -> str | None:
    text: Final = response.text
    match: Final = _S3_ERROR_CODE.search(text) if isinstance(text, str) else None
    return match.group(1) if match else None


def _is_terminal(response: httpx.Response) -> bool:
    """True only for object-specific, unrecoverable rejections (400/403 with a terminal XML code).
    Unknown codes, empty or non-XML bodies, and every other status fail safe toward retry."""
    return response.status_code in _BODY_CODED_STATUSES and _s3_error_code(response) in _TERMINAL_ERROR_CODES


def _s3_key_parent(s3_object_key: str) -> str:
    return s3_object_key.rsplit("/", 1)[0] if "/" in s3_object_key else ""


class S3BatchUploadError(Exception):
    def __init__(self, failed: int, total: int) -> None:
        self.failed = failed
        self.total = total
        super().__init__(f"{failed} of {total} S3 uploads failed; transient failures kept in queue for the next flush")


_in_flush: Final[contextvars.ContextVar[bool]] = contextvars.ContextVar("s3_v2_in_flush", default=False)


class S3Logger(CustomBatchLogger, BaseAWSLLM):
    preserve_events_added_during_flush = True
    _flush_retries: int = 0
    _requeued_count: int = 0
    s3_drop_on_terminal_error: bool = False
    s3_max_retry_age_seconds: int | None = None

    def __init__(
        self,
        s3_bucket_name: str | None = None,
        s3_path: str | None = None,
        s3_region_name: str | None = None,
        s3_api_version: str | None = None,
        s3_use_ssl: bool = True,
        s3_verify: bool | None = None,
        s3_endpoint_url: str | None = None,
        s3_aws_access_key_id: str | None = None,
        s3_aws_secret_access_key: str | None = None,
        s3_aws_session_token: str | None = None,
        s3_aws_session_name: str | None = None,
        s3_aws_profile_name: str | None = None,
        s3_aws_role_name: str | None = None,
        s3_aws_web_identity_token: str | None = None,
        s3_aws_sts_endpoint: str | None = None,
        s3_flush_interval: int | None = DEFAULT_S3_FLUSH_INTERVAL_SECONDS,
        s3_batch_size: int | None = DEFAULT_S3_BATCH_SIZE,
        s3_config=None,
        s3_use_team_prefix: bool = False,
        s3_strip_base64_files: bool = False,
        s3_use_key_prefix: bool = False,
        s3_use_virtual_hosted_style: bool = False,
        s3_server_side_encryption: str | None = None,
        s3_sse_kms_key_id: str | None = None,
        s3_log_prompts_only: bool | None = None,
        s3_max_concurrent_uploads: int = DEFAULT_S3_MAX_CONCURRENT_UPLOADS,
        s3_max_queue_size: int | None = None,
        s3_max_retry_age_seconds: int | None = None,
        s3_drop_on_terminal_error: bool = False,
        s3_adaptive_concurrency: bool = False,
        s3_max_adaptive_concurrency: int | None = None,
        s3_batch_file_upload: bool = False,
        s3_callback_params_override: dict | None = None,
        **kwargs,
    ):
        try:
            _masker: Final = SensitiveDataMasker()
            if s3_callback_params_override is not None:
                verbose_logger.debug(
                    "in init s3 logger (audit override) - %s", _masker.mask_dict(dict(s3_callback_params_override))
                )
            else:
                verbose_logger.debug(
                    "in init s3 logger - s3_callback_params %s",
                    _masker.mask_dict(dict(litellm.s3_callback_params or {})),
                )

            # Initialize S3 params first to get the correct s3_verify value
            self._init_s3_params(
                params_source=s3_callback_params_override,
                s3_bucket_name=s3_bucket_name,
                s3_region_name=s3_region_name,
                s3_api_version=s3_api_version,
                s3_use_ssl=s3_use_ssl,
                s3_verify=s3_verify,
                s3_endpoint_url=s3_endpoint_url,
                s3_aws_access_key_id=s3_aws_access_key_id,
                s3_aws_secret_access_key=s3_aws_secret_access_key,
                s3_aws_session_token=s3_aws_session_token,
                s3_aws_session_name=s3_aws_session_name,
                s3_aws_profile_name=s3_aws_profile_name,
                s3_aws_role_name=s3_aws_role_name,
                s3_aws_web_identity_token=s3_aws_web_identity_token,
                s3_aws_sts_endpoint=s3_aws_sts_endpoint,
                s3_config=s3_config,
                s3_path=s3_path,
                s3_use_team_prefix=s3_use_team_prefix,
                s3_strip_base64_files=s3_strip_base64_files,
                s3_use_key_prefix=s3_use_key_prefix,
                s3_use_virtual_hosted_style=s3_use_virtual_hosted_style,
                s3_server_side_encryption=s3_server_side_encryption,
                s3_sse_kms_key_id=s3_sse_kms_key_id,
                s3_log_prompts_only=s3_log_prompts_only,
                s3_max_concurrent_uploads=s3_max_concurrent_uploads,
                s3_max_queue_size=s3_max_queue_size,
                s3_max_retry_age_seconds=s3_max_retry_age_seconds,
                s3_drop_on_terminal_error=s3_drop_on_terminal_error,
                s3_adaptive_concurrency=s3_adaptive_concurrency,
                s3_max_adaptive_concurrency=s3_max_adaptive_concurrency,
                s3_batch_file_upload=s3_batch_file_upload,
            )
            self._upload_limiter: asyncio.Semaphore | AdaptiveConcurrencyLimiter = (
                AdaptiveConcurrencyLimiter(
                    initial=self.s3_max_concurrent_uploads,
                    floor=self.s3_max_concurrent_uploads,
                    ceiling=max(self.s3_max_concurrent_uploads, self.s3_max_adaptive_concurrency),
                )
                if self.s3_adaptive_concurrency
                else asyncio.Semaphore(self.s3_max_concurrent_uploads)
            )
            verbose_logger.debug("s3 logger using endpoint url %s", s3_endpoint_url)

            # IMPORTANT
            # Create httpx client AFTER _init_s3_params so we have the correct s3_verify value
            verbose_logger.debug("s3_v2 logger creating async httpx client with s3_verify=%s", self.s3_verify)
            self.async_httpx_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback,
                params={"ssl_verify": self.s3_verify},
            )

            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()

            verbose_logger.debug("s3 flush interval: %s, s3 batch size: %s", s3_flush_interval, s3_batch_size)
            # Call CustomLogger's __init__
            CustomBatchLogger.__init__(
                self,
                flush_lock=self.flush_lock,
                flush_interval=s3_flush_interval,
                batch_size=s3_batch_size,
                max_queue_size=self.s3_max_queue_size,
            )
            self.log_queue: list[s3BatchLoggingElement] = []
            self._requeued_count: int = 0
            self._flush_retries: int = 0
            self._flush_dropped: dict[int, s3BatchLoggingElement] = {}

            # Call BaseAWSLLM's __init__
            BaseAWSLLM.__init__(self)

        except Exception as e:
            print_verbose(f"Got exception on init s3 client {e}")
            raise e

    def _init_s3_params(
        self,
        s3_bucket_name: str | None = None,
        s3_region_name: str | None = None,
        s3_api_version: str | None = None,
        s3_use_ssl: bool = True,
        s3_verify: bool | None = None,
        s3_endpoint_url: str | None = None,
        s3_aws_access_key_id: str | None = None,
        s3_aws_secret_access_key: str | None = None,
        s3_aws_session_token: str | None = None,
        s3_aws_session_name: str | None = None,
        s3_aws_profile_name: str | None = None,
        s3_aws_role_name: str | None = None,
        s3_aws_web_identity_token: str | None = None,
        s3_aws_sts_endpoint: str | None = None,
        s3_config=None,
        s3_path: str | None = None,
        s3_use_team_prefix: bool = False,
        s3_strip_base64_files: bool = False,
        s3_use_key_prefix: bool = False,
        s3_use_virtual_hosted_style: bool = False,
        s3_server_side_encryption: str | None = None,
        s3_sse_kms_key_id: str | None = None,
        s3_log_prompts_only: bool | None = None,
        s3_max_concurrent_uploads: int = DEFAULT_S3_MAX_CONCURRENT_UPLOADS,
        s3_max_queue_size: int | None = None,
        s3_max_retry_age_seconds: int | None = None,
        s3_drop_on_terminal_error: bool = False,
        s3_adaptive_concurrency: bool = False,
        s3_max_adaptive_concurrency: int | None = None,
        s3_batch_file_upload: bool = False,
        params_source: dict | None = None,
    ):
        """
        Initialize the s3 params for this logging callback. Reads from
        `params_source` if given (e.g. `s3_audit_callback_params` for the
        audit-log instance), otherwise falls back to `litellm.s3_callback_params`.
        Resolves `os.environ/X` markers into a local dict; never mutates the source.
        """
        if params_source is None:
            params_source = litellm.s3_callback_params or {}
        params: Final[dict] = {
            key: (litellm.get_secret(value) if isinstance(value, str) and value.startswith("os.environ/") else value)
            for key, value in params_source.items()
        }

        self.s3_bucket_name = params.get("s3_bucket_name") or s3_bucket_name
        self.s3_region_name = params.get("s3_region_name") or s3_region_name
        self.s3_api_version = params.get("s3_api_version") or s3_api_version
        self.s3_use_ssl = params.get("s3_use_ssl", True) if params.get("s3_use_ssl") is not None else s3_use_ssl
        self.s3_verify = params.get("s3_verify") if params.get("s3_verify") is not None else s3_verify
        self.s3_endpoint_url = params.get("s3_endpoint_url") or s3_endpoint_url
        self.s3_aws_access_key_id = params.get("s3_aws_access_key_id") or s3_aws_access_key_id

        self.s3_aws_secret_access_key = params.get("s3_aws_secret_access_key") or s3_aws_secret_access_key

        self.s3_aws_session_token = params.get("s3_aws_session_token") or s3_aws_session_token

        self.s3_aws_session_name = params.get("s3_aws_session_name") or s3_aws_session_name

        self.s3_aws_profile_name = params.get("s3_aws_profile_name") or s3_aws_profile_name

        self.s3_aws_role_name = params.get("s3_aws_role_name") or s3_aws_role_name

        self.s3_aws_web_identity_token = params.get("s3_aws_web_identity_token") or s3_aws_web_identity_token

        self.s3_aws_sts_endpoint = params.get("s3_aws_sts_endpoint") or s3_aws_sts_endpoint

        self.s3_config = params.get("s3_config") or s3_config
        self.s3_path = params.get("s3_path") or s3_path
        self.s3_use_team_prefix = bool(params.get("s3_use_team_prefix", False)) or s3_use_team_prefix

        self.s3_use_key_prefix = bool(params.get("s3_use_key_prefix", False)) or s3_use_key_prefix

        self.s3_strip_base64_files = bool(params.get("s3_strip_base64_files", False)) or s3_strip_base64_files

        self.s3_use_virtual_hosted_style = (
            bool(params.get("s3_use_virtual_hosted_style", False)) or s3_use_virtual_hosted_style
        )

        self.s3_log_prompts_only: object = (
            params.get("s3_log_prompts_only") if s3_log_prompts_only is None else s3_log_prompts_only
        )

        self.s3_server_side_encryption, self.s3_sse_kms_key_id = resolve_sse_params(
            params.get("s3_server_side_encryption") or s3_server_side_encryption,
            params.get("s3_sse_kms_key_id") or s3_sse_kms_key_id,
        )

        configured_bound: Final = params.get("s3_max_concurrent_uploads")
        self.s3_max_concurrent_uploads = resolve_s3_max_concurrent_uploads(
            s3_max_concurrent_uploads if configured_bound is None or configured_bound == "" else configured_bound,
            DEFAULT_S3_MAX_CONCURRENT_UPLOADS,
        )

        configured_queue_size: Final = params.get("s3_max_queue_size")
        constructor_queue_size: Final = resolve_s3_max_queue_size(
            s3_max_queue_size, CustomBatchLogger.DEFAULT_MAX_QUEUE_SIZE
        )
        self.s3_max_queue_size = resolve_s3_max_queue_size(configured_queue_size, constructor_queue_size)

        configured_retry_age: Final = params.get("s3_max_retry_age_seconds")
        constructor_retry_age: Final = resolve_s3_max_retry_age_seconds(s3_max_retry_age_seconds)
        self.s3_max_retry_age_seconds = (
            constructor_retry_age
            if configured_retry_age is None or configured_retry_age == ""
            else resolve_s3_max_retry_age_seconds(configured_retry_age)
        )

        self.s3_drop_on_terminal_error = s3_drop_on_terminal_error or resolve_s3_drop_on_terminal_error(
            params.get("s3_drop_on_terminal_error")
        )

        self.s3_adaptive_concurrency = s3_adaptive_concurrency or resolve_s3_adaptive_concurrency(
            params.get("s3_adaptive_concurrency")
        )

        configured_adaptive_ceiling: Final = params.get("s3_max_adaptive_concurrency")
        self.s3_max_adaptive_concurrency = resolve_s3_max_adaptive_concurrency(
            s3_max_adaptive_concurrency
            if configured_adaptive_ceiling is None or configured_adaptive_ceiling == ""
            else configured_adaptive_ceiling,
            DEFAULT_S3_MAX_ADAPTIVE_CONCURRENCY,
        )

        self.s3_batch_file_upload = s3_batch_file_upload or resolve_s3_batch_file_upload(
            params.get("s3_batch_file_upload")
        )

    def _build_object_url(self, s3_object_key: str) -> str:
        """
        Build the exact URL that is both signed and sent, with the key percent-encoded once.

        S3SigV4Auth signs the path verbatim while S3 canonicalizes the received path with reserved
        characters encoded, so an unencoded `=`, `+`, `&`, `#`, `?`, `%` or space in the key makes
        the two signatures disagree (403 SignatureDoesNotMatch).
        """
        encoded_key: Final = quote(s3_object_key, safe="/")
        if self.s3_endpoint_url and self.s3_bucket_name:
            if self.s3_use_virtual_hosted_style:
                endpoint_host: Final = self.s3_endpoint_url.replace("https://", "").replace("http://", "")
                protocol: Final = "https://" if self.s3_endpoint_url.startswith("https://") else "http://"
                return f"{protocol}{self.s3_bucket_name}.{endpoint_host}/{encoded_key}"
            return f"{self.s3_endpoint_url}/{self.s3_bucket_name}/{encoded_key}"
        return (
            f"https://{self.s3_bucket_name}.s3.{self.s3_region_name}."
            f"{get_aws_dns_suffix(self.s3_region_name)}/{encoded_key}"
        )

    def _sign_put(
        self, credentials: "Credentials", url: str, json_string: str, headers: Mapping[str, str]
    ) -> dict[str, str]:  # mutable-ok: [LIT001] AsyncHTTPHandler.put/HTTPHandler.put only accept dict headers
        """
        ``RefreshableCredentials`` (IMDS roles) may refresh between the access key, secret and token
        reads SigV4 performs, producing a mixed-generation signature that S3 rejects with 403.
        Freezing first makes the three values one atomic snapshot.
        """
        from botocore.auth import S3SigV4Auth
        from botocore.awsrequest import AWSRequest
        from botocore.credentials import RefreshableCredentials

        frozen: Final = (
            credentials.get_frozen_credentials() if isinstance(credentials, RefreshableCredentials) else credentials
        )
        aws_request: Final = AWSRequest(method="PUT", url=url, data=json_string, headers=dict(headers))
        aws_region_name: Final = self.get_aws_region_name_for_non_llm_api_calls(aws_region_name=self.s3_region_name)
        S3SigV4Auth(frozen, "s3", aws_region_name).add_auth(aws_request)
        return dict(aws_request.headers.items())

    def _sse_headers(self) -> Mapping[str, str]:
        candidates: Final = {
            "x-amz-server-side-encryption": self.s3_server_side_encryption,
            "x-amz-server-side-encryption-aws-kms-key-id": self.s3_sse_kms_key_id,
        }
        return {key: value for key, value in candidates.items() if value}

    def _prepare_put(self, batch_logging_element: s3BatchLoggingElement) -> _PreparedPut:
        try:
            import base64
            import hashlib
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        json_string: Final = (
            batch_logging_element.body
            if batch_logging_element.body is not None
            else safe_dumps(batch_logging_element.payload)
        )
        content_hash: Final = hashlib.sha256(json_string.encode("utf-8")).hexdigest()
        content_md5: Final = base64.b64encode(
            hashlib.md5(json_string.encode("utf-8"), usedforsecurity=False).digest()
        ).decode()
        return _PreparedPut(
            json_string=json_string,
            headers={
                "Content-Type": batch_logging_element.content_type,
                "Content-MD5": content_md5,
                "x-amz-content-sha256": content_hash,
                "Content-Language": "en",
                "Content-Disposition": f'inline; filename="{batch_logging_element.s3_object_download_filename}"',
                "Cache-Control": "private, immutable, max-age=31536000, s-maxage=0",
                **self._sse_headers(),
            },
        )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        await self._async_log_event_base(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        await self._async_log_event_base(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
        )

    async def async_log_audit_log_event(self, audit_log: StandardAuditLogPayload) -> None:
        """Batch audit logs and upload to S3 under audit_logs/ prefix."""
        try:
            from datetime import timezone

            now: Final = datetime.now(timezone.utc)
            audit_log_id: Final = audit_log.get("id", "unknown")

            s3_object_key: Final = get_s3_object_key(
                cast(str | None, self.s3_path) or "",
                "audit_logs/",
                now,
                f"{now.strftime('%H-%M-%S')}_{audit_log_id}",
            )

            element: Final = s3BatchLoggingElement(
                payload=dict(audit_log),
                s3_object_key=s3_object_key,
                s3_object_download_filename=f"audit-{audit_log_id}.json",
            )

            self.log_queue.append(element)

            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()
        except Exception as e:
            verbose_logger.exception("S3 audit log error: %s", e)

    async def _async_log_event_base(self, kwargs, response_obj, start_time, end_time):
        try:
            verbose_logger.debug("s3 Logging - Enters logging function for model %s", kwargs)

            s3_batch_logging_element: Final = self.create_s3_batch_logging_element(
                start_time=start_time,
                standard_logging_payload=kwargs.get("standard_logging_object", None),
            )

            # afile_delete and other non-model call types never produce a standard_logging_object,
            # so s3_batch_logging_element is None. Skip gracefully instead of raising ValueError.
            if s3_batch_logging_element is None:
                verbose_logger.debug(
                    "s3 Logging - skipping event, no standard_logging_object for call_type=%s",
                    kwargs.get("call_type", "unknown"),
                )
                return

            verbose_logger.debug("\ns3 Logger - Logging payload = %s", s3_batch_logging_element)

            self.log_queue.append(s3_batch_logging_element)
            verbose_logger.debug(
                "s3 logging: queue length %s, batch size %s",
                len(self.log_queue),
                self.batch_size,
            )
        except Exception as e:
            verbose_logger.exception("s3 Layer Error - %s", e)
            self.handle_callback_failure(callback_name="S3Logger")

    @property
    def _upload_semaphore(self) -> asyncio.Semaphore | AdaptiveConcurrencyLimiter:
        return self._upload_limiter

    @_upload_semaphore.setter
    def _upload_semaphore(self, value: asyncio.Semaphore | AdaptiveConcurrencyLimiter) -> None:
        self._upload_limiter = value

    async def async_upload_data_to_s3(
        self,
        batch_logging_element: s3BatchLoggingElement,
    ) -> bool:
        try:
            from litellm.litellm_core_utils.asyncify import asyncify

            asyncified_get_credentials: Final = asyncify(self.get_credentials)

            verbose_logger.debug("s3_v2 logger - uploading data to s3 - %s", batch_logging_element.s3_object_key)
            verbose_logger.debug("s3_v2 logger - s3_verify setting: %s", self.s3_verify)

            url: Final = self._build_object_url(batch_logging_element.s3_object_key)

            async def signed_put(prepared: _PreparedPut) -> httpx.Response:
                credentials: Final = await asyncified_get_credentials(
                    aws_access_key_id=self.s3_aws_access_key_id,
                    aws_secret_access_key=self.s3_aws_secret_access_key,
                    aws_session_token=self.s3_aws_session_token,
                    aws_region_name=self.s3_region_name,
                    aws_session_name=self.s3_aws_session_name,
                    aws_profile_name=self.s3_aws_profile_name,
                    aws_role_name=self.s3_aws_role_name,
                    aws_web_identity_token=self.s3_aws_web_identity_token,
                    aws_sts_endpoint=self.s3_aws_sts_endpoint,
                )
                signed_headers: Final = await run_aws_signing(
                    self._sign_put, credentials, url, prepared.json_string, prepared.headers
                )
                try:
                    return await self.async_httpx_client.put(url, data=prepared.json_string, headers=signed_headers)
                except httpx.HTTPStatusError as error:
                    return error.response

            max_retries: Final = 3
            prepared: Final = self._prepare_put(batch_logging_element)
            for attempt in range(max_retries):
                response = await self._recorded_put(partial(signed_put, prepared))
                if (
                    response.status_code in _RETRYABLE_STATUSES
                    and not (self.s3_drop_on_terminal_error and _is_terminal(response))
                    and attempt < max_retries - 1
                ):
                    wait_time = 2**attempt  # 1s, 2s
                    verbose_logger.log(
                        logging.DEBUG if _in_flush.get() else logging.WARNING,
                        "S3 upload returned %s, retrying in %ss (attempt %s/%s) key=%s",
                        response.status_code,
                        wait_time,
                        attempt + 1,
                        max_retries,
                        batch_logging_element.s3_object_key,
                    )
                    self._flush_retries += 1
                    await asyncio.sleep(wait_time)
                    continue
                response.raise_for_status()
                break
        except Exception as e:
            verbose_logger.exception("Error uploading to s3: %s", e)
            self.handle_callback_failure(callback_name="S3Logger")
            if isinstance(e, httpx.HTTPStatusError) and self.s3_drop_on_terminal_error and _is_terminal(e.response):
                verbose_logger.warning(
                    "s3 logging: dropping object %s after terminal status %s",
                    batch_logging_element.s3_object_key,
                    e.response.status_code,
                )
                self._flush_dropped[id(batch_logging_element)] = batch_logging_element
            return False
        return True

    async def async_send_batch(self) -> None:
        """
        Sends runs from self.log_queue.

        Raises S3BatchUploadError when any upload failed; CustomBatchLogger.flush_queue
        keeps the surviving queue entries for the next flush.
        """
        batch: Final = tuple(self.log_queue)
        if not batch:
            return
        verbose_logger.debug("s3_v2 logger - sending batch of %s", len(batch))

        #########################################################
        #  Flush the log queue to s3
        #  the log queue can be bounded by DEFAULT_S3_BATCH_SIZE
        #  see custom_batch_logger.py which triggers the flush
        #########################################################
        uploads: Final = self._batch_file_elements(batch) if self._batch_file_mode_active() else batch
        self._flush_retries = 0
        self._flush_dropped = {}  # mutable-ok: per-flush drop marks read back by _upload_bounded
        stale: Final = min(self._requeued_count, len(uploads)) if len(uploads) == len(batch) else 0
        order: Final = (*range(stale, len(uploads)), *range(stale))
        ordered: Final = await asyncio.gather(*(self._upload_outcome(uploads[i]) for i in order))
        outcomes: Final = dict(zip(order, ordered, strict=True))
        results: Final = tuple(outcomes[i] for i in range(len(uploads)))
        if self._flush_retries:
            verbose_logger.warning(
                "s3 logging: %s in-call retries across %s uploads this flush",
                self._flush_retries,
                len(uploads),
            )
        delivered: Final = sum(1 for outcome in results if outcome == "delivered")
        bucket_wide: Final = delivered == 0
        failed: Final = tuple(
            (element, outcome) for element, outcome in zip(uploads, results, strict=True) if outcome != "delivered"
        )
        now: Final = time.monotonic()
        requeued: Final = (
            tuple(element for element, _ in failed)
            if bucket_wide
            else tuple(
                element
                if element.retrying_since is not None or self.s3_max_retry_age_seconds is None
                else element.model_copy(update={"retrying_since": now})
                for element, outcome in failed
                if outcome != "dropped"
                and not (
                    self.s3_max_retry_age_seconds is not None
                    and element.retrying_since is not None
                    and now - element.retrying_since > self.s3_max_retry_age_seconds
                )
            )
        )
        dropped: Final = len(failed) - len(requeued)
        if dropped:
            verbose_logger.warning(
                "s3 logging: %s uploads dropped (terminal or retrying longer than s3_max_retry_age_seconds=%s)",
                dropped,
                self.s3_max_retry_age_seconds,
            )
        if not requeued:
            self._requeued_count = 0
            return
        arrivals: Final = self.log_queue[len(batch) :]
        overflow: Final = max(0, len(requeued) + len(arrivals) - self.max_queue_size)
        if overflow:
            verbose_logger.warning(
                "s3 logging: queue exceeded max_queue_size=%s after a failed flush, dropped %s oldest events",
                self.max_queue_size,
                overflow,
            )
        self.log_queue = [  # mutable-ok: log_queue is the flush buffer shared with custom_batch_logger
            *requeued,
            *arrivals,
        ][overflow:]
        self._requeued_count = max(0, len(requeued) - overflow)
        raise S3BatchUploadError(failed=len(failed), total=len(uploads))

    def _batch_file_mode_active(self) -> bool:
        if not self.s3_batch_file_upload:
            return False
        if litellm.cold_storage_custom_logger == "s3_v2":
            verbose_logger.warning(
                "s3 logging: s3_batch_file_upload is ignored because s3_v2 is the cold storage logger; "
                "per-request objects are required for spend log lookups"
            )
            return False
        return True

    async def _upload_bounded(self, element: s3BatchLoggingElement) -> bool:
        token: Final = _in_flush.set(True)
        try:
            async with self._upload_semaphore:
                return await self.async_upload_data_to_s3(element)
        finally:
            _in_flush.reset(token)

    async def _upload_outcome(self, element: s3BatchLoggingElement) -> UploadOutcome:
        delivered: Final = await self._upload_bounded(element)
        if delivered:
            return "delivered"
        if id(element) in self._flush_dropped:
            return "dropped"
        return "retry"

    async def _recorded_put(self, signed_put: Callable[[], Awaitable[httpx.Response]]) -> httpx.Response:
        limiter: Final = getattr(self, "_upload_limiter", None)
        adaptive: Final = limiter if isinstance(limiter, AdaptiveConcurrencyLimiter) else None
        try:
            response: Final = await signed_put()
        except Exception:
            if adaptive is not None:
                adaptive.record(PutSample(throttled=True))
            raise
        if adaptive is not None:
            adaptive.record(
                PutSample(
                    throttled=response.status_code in (429, 503) or _s3_error_code(response) == "SlowDown",
                )
            )
        return response

    def _batch_file_elements(self, batch: tuple[s3BatchLoggingElement, ...]) -> tuple[s3BatchLoggingElement, ...]:
        now: Final = datetime.now(timezone.utc)
        groups: Final = {
            parent: tuple(
                element for element in batch if element.body is None and _s3_key_parent(element.s3_object_key) == parent
            )
            for parent in sorted({_s3_key_parent(element.s3_object_key) for element in batch if element.body is None})
        }
        return tuple(element for element in batch if element.body is not None) + tuple(
            self._build_batch_file_element(elements, parent, now) for parent, elements in groups.items()
        )

    def _build_batch_file_element(
        self, elements: tuple[s3BatchLoggingElement, ...], parent: str, now: datetime
    ) -> s3BatchLoggingElement:
        batch_name: Final = f"batch_{now.strftime('%H-%M-%S')}_{uuid4().hex}"
        return s3BatchLoggingElement(
            payload={},
            body="\n".join(safe_dumps(element.payload) for element in elements),
            content_type="application/x-ndjson",
            s3_object_key=f"{parent}/{batch_name}.jsonl" if parent else f"{batch_name}.jsonl",
            s3_object_download_filename=f"{batch_name}.jsonl",
            retrying_since=min(
                (element.retrying_since for element in elements if element.retrying_since is not None), default=None
            ),
        )

    def create_s3_batch_logging_element(
        self,
        start_time: datetime,
        standard_logging_payload: StandardLoggingPayload | None,
    ) -> s3BatchLoggingElement | None:
        """
        Helper function to create an s3BatchLoggingElement.

        Args:
            start_time (datetime): The start time of the logging event.
            standard_logging_payload (Optional[StandardLoggingPayload]): The payload to be logged.
            s3_path (Optional[str]): The S3 path prefix.

        Returns:
            Optional[s3BatchLoggingElement]: The created s3BatchLoggingElement, or None if payload is None.
        """
        if standard_logging_payload is None:
            return None

        if self.s3_strip_base64_files:
            standard_logging_payload = self._strip_base64_from_messages_sync(standard_logging_payload)

        # Base prefix (default empty)
        prefix_components: Final = []
        if self.s3_use_team_prefix:
            team_alias: Final = standard_logging_payload.get("metadata", {}).get("user_api_key_team_alias", None)
            if team_alias:
                prefix_components.append(team_alias)
        if self.s3_use_key_prefix:
            user_api_key_alias: Final = standard_logging_payload.get("metadata", {}).get("user_api_key_alias", None)
            if user_api_key_alias:
                prefix_components.append(user_api_key_alias)

        # Construct full prefix path
        prefix_path = "/".join(prefix_components)
        if prefix_path:
            prefix_path += "/"

        s3_file_name: Final = litellm.utils.get_logging_id(start_time, standard_logging_payload) or ""
        verbose_logger.debug(
            "Creating s3 file with prefix_components=%s,prefix_path=%s and %s",
            prefix_components,
            prefix_path,
            s3_file_name,
        )
        s3_object_key: Final = get_s3_object_key(
            s3_path=cast(str | None, self.s3_path) or "",
            prefix=prefix_path,
            start_time=start_time,
            s3_file_name=s3_file_name,
        )
        verbose_logger.debug("s3_object_key=%s", s3_object_key)

        s3_object_download_filename: Final = get_s3_object_download_filename(start_time, standard_logging_payload["id"])

        payload: Final = (
            prompts_only_payload(standard_logging_payload)
            if resolve_s3_log_prompts_only(self.s3_log_prompts_only)
            else standard_logging_payload
        )
        return s3BatchLoggingElement(
            payload=dict(payload),
            s3_object_key=s3_object_key,
            s3_object_download_filename=s3_object_download_filename,
        )

    def upload_data_to_s3(self, batch_logging_element: s3BatchLoggingElement):
        try:
            verbose_logger.debug("s3_v2 logger - uploading data to s3 - %s", batch_logging_element.s3_object_key)

            url: Final = self._build_object_url(batch_logging_element.s3_object_key)

            prepared: Final = self._prepare_put(batch_logging_element)

            httpx_client: Final = _get_httpx_client(
                params=({"ssl_verify": self.s3_verify} if self.s3_verify is not None else None)
            )

            def signed_put(prepared_put: _PreparedPut) -> httpx.Response:
                credentials: Final = self.get_credentials(
                    aws_access_key_id=self.s3_aws_access_key_id,
                    aws_secret_access_key=self.s3_aws_secret_access_key,
                    aws_session_token=self.s3_aws_session_token,
                    aws_region_name=self.s3_region_name,
                )
                signed_headers: Final = self._sign_put(credentials, url, prepared_put.json_string, prepared_put.headers)
                return httpx_client.put(url, data=prepared_put.json_string, headers=signed_headers)

            max_retries: Final = 3
            for attempt in range(max_retries):
                response = signed_put(prepared)
                if (
                    response.status_code in _RETRYABLE_STATUSES
                    and not (self.s3_drop_on_terminal_error and _is_terminal(response))
                    and attempt < max_retries - 1
                ):
                    wait_time = 2**attempt  # 1s, 2s
                    verbose_logger.warning(
                        "S3 upload returned %s, retrying in %ss (attempt %s/%s) key=%s",
                        response.status_code,
                        wait_time,
                        attempt + 1,
                        max_retries,
                        batch_logging_element.s3_object_key,
                    )
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                break
        except Exception as e:
            verbose_logger.exception("Error uploading to s3: %s", e)
            self.handle_callback_failure(callback_name="S3Logger")
            if isinstance(e, httpx.HTTPStatusError) and self.s3_drop_on_terminal_error and _is_terminal(e.response):
                verbose_logger.warning(
                    "s3 logging: dropping object %s after terminal status %s",
                    batch_logging_element.s3_object_key,
                    e.response.status_code,
                )

    async def _download_object_from_s3(self, s3_object_key: str) -> dict | None:
        """
        Download and parse JSON object from S3.

        Args:
            s3_object_key: The S3 object key to download

        Returns:
            Optional[dict]: The parsed JSON object or None if not found/error
        """
        try:
            import hashlib

            from botocore.auth import S3SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError("Missing boto3 to call S3. Run 'pip install boto3'.")

        try:
            from litellm.litellm_core_utils.asyncify import asyncify

            # Get AWS credentials
            asyncified_get_credentials: Final = asyncify(self.get_credentials)
            credentials: Final = await asyncified_get_credentials(
                aws_access_key_id=self.s3_aws_access_key_id,
                aws_secret_access_key=self.s3_aws_secret_access_key,
                aws_session_token=self.s3_aws_session_token,
                aws_region_name=self.s3_region_name,
                aws_session_name=self.s3_aws_session_name,
                aws_profile_name=self.s3_aws_profile_name,
                aws_role_name=self.s3_aws_role_name,
                aws_web_identity_token=self.s3_aws_web_identity_token,
                aws_sts_endpoint=self.s3_aws_sts_endpoint,
            )

            verbose_logger.debug("s3_v2 logger - downloading data from s3 - %s", s3_object_key)

            url: Final = self._build_object_url(s3_object_key)

            # Prepare the request for GET operation
            # For GET requests, we need x-amz-content-sha256 with hash of empty string
            empty_string_hash: Final = hashlib.sha256(b"").hexdigest()
            headers: Final = {
                "x-amz-content-sha256": empty_string_hash,
            }

            # Sign the request
            aws_request: Final = AWSRequest(method="GET", url=url, headers=headers)
            await run_aws_signing(S3SigV4Auth(credentials, "s3", self.s3_region_name).add_auth, aws_request)

            # Prepare the signed headers
            signed_headers: Final = dict(aws_request.headers.items())

            response: Final = await self.async_httpx_client.get(url, headers=signed_headers)

            if response.status_code != 200:
                verbose_logger.exception("S3 object not found, saw response=", response.text)
                return None

            # Parse JSON response
            return response.json()

        except Exception as e:
            verbose_logger.exception("Error downloading from S3: %s", e)
            return None

    async def get_proxy_server_request_from_cold_storage_with_object_key(
        self,
        object_key: str,
    ) -> dict | None:
        """
        Get the proxy server request from cold storage

        Allows fetching a dict of the proxy server request from s3 or GCS bucket.

        Args:
            request_id: The unique request ID to search for
            start_time: The start time of the request (datetime or ISO string)

        Returns:
            Optional[dict]: The request data dictionary or None if not found
        """
        try:
            # Download and return the object from S3
            downloaded_object: Final = await self._download_object_from_s3(object_key)
            return downloaded_object
        except Exception as e:
            verbose_logger.exception("Error retrieving object %s from cold storage: %s", object_key, e)
            return None
