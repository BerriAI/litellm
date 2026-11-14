import asyncio
import json
import os
from collections.abc import Callable, Collection, Mapping
from typing import TYPE_CHECKING, Final

from typing_extensions import Unpack

import litellm
from litellm._logging import verbose_logger
from litellm.constants import (
    COMPLETION_HTTP_FALLBACK_SECONDS,
    DEFAULT_S3_BATCH_SIZE,
    DEFAULT_S3_FLUSH_INTERVAL_SECONDS,
    DEFAULT_S3_MAX_ADAPTIVE_CONCURRENCY,
    DEFAULT_S3_MAX_CONCURRENT_UPLOADS,
    HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS,
    S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS,
)
from litellm.integrations.batch_line_item_objects import batch_line_items_ndjson, batch_line_items_object_name
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.derivations import (
    cold_storage_object_key,
    log_id,
    logged_key_identity,
    mapping,
    payload_copy,
)
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs
from litellm.integrations.custom_logger_v2.logger import on
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch
from litellm.integrations.custom_logger_v2.rust import RustLogger
from litellm.integrations.custom_logger_v2.selection import Selection
from litellm.integrations.s3 import resolve_s3_max_queue_size, resolve_s3_max_retry_age_seconds
from litellm.llms.bedrock.web_identity_session_policy import S3_LOGGING_SESSION_POLICY, session_policy_document
from litellm.secret_managers.main import get_secret, get_secret_str
from litellm.types.integrations.s3_v2 import S3LoggerSettings, S3PartitionGranularity
from litellm.types.utils import StandardAuditLogPayload, StandardBatchLineItemsPayload, StandardLoggingPayload

if TYPE_CHECKING:
    from litellm.rust_bridge._native import NativeS3Log

CALLBACK_NAME: Final = "s3_v2"
_WEB_IDENTITY_TOKEN_MIN_LENGTH: Final = 4
_DEFAULT_MAX_RETRY_AGE_SECONDS: Final = 3600


class S3BatchUploadError(Exception):
    def __init__(self, failed: int, total: int) -> None:
        self.failed: Final = failed
        self.total: Final = total
        super().__init__(f"{failed} of {total} S3 uploads failed; transient failures kept in queue for the next flush")


def _raise_batch_upload_error(failed: int, total: int) -> None:
    raise S3BatchUploadError(failed=failed, total=total)


class _Host:
    def __init__(self) -> None:
        self.failures: Final = CustomLogger()

    def callback_failed(self) -> None:
        self.failures.handle_callback_failure(callback_name="S3Logger")

    def batch_upload_failed(self, queued: int, failed: int, total: int) -> None:
        try:
            _raise_batch_upload_error(failed, total)
        except S3BatchUploadError:
            verbose_logger.exception(
                "CustomLogger: async_send_batch raised; preserving %s events in queue for retry", queued
            )

    def cold_storage_is_s3_v2(self) -> bool:
        return litellm.cold_storage_custom_logger == "s3_v2"

    def region_from_environment(self) -> str | None:
        region_name: Final = get_secret_str("AWS_REGION_NAME", None)
        region: Final = get_secret_str("AWS_REGION", None)
        return region if region is not None else region_name

    def web_identity_token(self, reference: str) -> str:
        if reference.startswith("os.environ/") or reference in os.environ:
            raise ValueError("Invalid web identity token reference.")
        match _read_secret(get_secret, reference):
            case None:
                raise ValueError("OIDC token could not be retrieved from secret manager.")
            case str() as token if len(token) < _WEB_IDENTITY_TOKEN_MIN_LENGTH:
                raise ValueError(
                    "Parameter validation failed:\nInvalid length for parameter WebIdentityToken, "
                    f"value: {len(token)}, valid min length: {_WEB_IDENTITY_TOKEN_MIN_LENGTH}"
                )
            case str() as token:
                return token
            case other:
                raise ValueError(
                    "Parameter validation failed:\nInvalid type for parameter WebIdentityToken, "
                    f"value: {other}, type: {type(other)}, valid types: {str}"
                )


def _read_secret(read: Callable[[str], object], reference: str) -> object:
    return read(reference)


def _resolved_params(params_source: Mapping[str, object] | None) -> dict[str, object]:  # mutable-ok: handed to Rust
    configured: Final[Mapping[object, object] | None] = getattr(litellm, "s3_callback_params", None)
    source: Final[Mapping[str, object] | Mapping[object, object]] = (
        params_source if params_source is not None else (configured or {})
    )
    return {
        key: (litellm.get_secret(value) if isinstance(value, str) and value.startswith("os.environ/") else value)
        for key, value in source.items()
        if isinstance(key, str)
    }


def _constants(batch_size: int | None = DEFAULT_S3_BATCH_SIZE) -> Mapping[str, object]:
    return {
        "batch_size": batch_size or litellm.DEFAULT_BATCH_SIZE,
        "max_concurrent_uploads": DEFAULT_S3_MAX_CONCURRENT_UPLOADS,
        "max_adaptive_concurrency": DEFAULT_S3_MAX_ADAPTIVE_CONCURRENCY,
        "upload_attempt_timeout_seconds": S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS,
        "download_read_timeout_seconds": COMPLETION_HTTP_FALLBACK_SECONDS,
        "download_connect_timeout_seconds": HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS,
        "web_identity_session_policy": session_policy_document(S3_LOGGING_SESSION_POLICY),
    }


def _flush_interval(flush_interval: float | None = DEFAULT_S3_FLUSH_INTERVAL_SECONDS) -> float:
    return flush_interval or litellm.DEFAULT_FLUSH_INTERVAL_SECONDS


_CONFIGURED_FIRST: Final = frozenset(
    {
        "s3_bucket_name",
        "s3_path",
        "s3_region_name",
        "s3_api_version",
        "s3_endpoint_url",
        "s3_aws_access_key_id",
        "s3_aws_secret_access_key",
        "s3_aws_session_token",
        "s3_aws_session_name",
        "s3_aws_profile_name",
        "s3_aws_role_name",
        "s3_aws_web_identity_token",
        "s3_aws_sts_endpoint",
        "s3_config",
        "s3_server_side_encryption",
        "s3_sse_kms_key_id",
    }
)
_CONSTRUCTOR_ENABLES: Final = frozenset(
    {
        "s3_use_team_prefix",
        "s3_strip_base64_files",
        "s3_use_key_prefix",
        "s3_use_virtual_hosted_style",
        "s3_adaptive_concurrency",
        "s3_batch_file_upload",
    }
)
_CONFIGURED_UNLESS_NONE: Final = frozenset({"s3_use_ssl", "s3_verify", "s3_drop_on_terminal_error"})
_CONSTRUCTOR_UNLESS_NONE: Final = frozenset({"s3_log_prompts_only", "s3_partition_granularity"})
_CONFIGURED_UNLESS_BLANK: Final = frozenset({"s3_max_concurrent_uploads", "s3_max_adaptive_concurrency"})
_CONSTRUCTOR_DEFAULTS: Final[Mapping[str, object]] = {
    "s3_use_ssl": True,
    "s3_drop_on_terminal_error": True,
    "s3_max_concurrent_uploads": DEFAULT_S3_MAX_CONCURRENT_UPLOADS,
    "s3_max_retry_age_seconds": _DEFAULT_MAX_RETRY_AGE_SECONDS,
    "s3_flush_interval": DEFAULT_S3_FLUSH_INTERVAL_SECONDS,
    "s3_batch_size": DEFAULT_S3_BATCH_SIZE,
}


def _blank(value: object) -> bool:
    return value is None or value == ""


def _constructor_wins(key: str, configured: object, value: object) -> bool:
    return (
        (key in _CONFIGURED_FIRST and not configured)
        or (key in _CONFIGURED_UNLESS_NONE and configured is None)
        or (key in _CONFIGURED_UNLESS_BLANK and _blank(configured))
        or (key in _CONSTRUCTOR_UNLESS_NONE and value is not None)
    )


_KEEP_CONFIGURED: Final = object()


def _constructor_value(key: str, configured: object, value: object) -> object:
    if _constructor_wins(key, configured, value):
        return value
    if key in _CONSTRUCTOR_ENABLES and value:
        return True
    if key == "s3_max_queue_size" and value is not None:
        return resolve_s3_max_queue_size(
            configured, resolve_s3_max_queue_size(value, CustomBatchLogger.DEFAULT_MAX_QUEUE_SIZE)
        )
    if key == "s3_max_retry_age_seconds":
        fallback: Final = resolve_s3_max_retry_age_seconds(value, _DEFAULT_MAX_RETRY_AGE_SECONDS)
        resolved: Final = fallback if _blank(configured) else resolve_s3_max_retry_age_seconds(configured, fallback)
        return 0 if resolved is None else resolved
    return _KEEP_CONFIGURED


def _constructed_params(
    params_source: Mapping[str, object] | None, settings: S3LoggerSettings
) -> dict[str, object]:  # mutable-ok: handed to Rust
    params: Final = _resolved_params(params_source)
    overrides: Final = {
        key: replaced
        for key, value in settings.items()
        if value != _CONSTRUCTOR_DEFAULTS.get(key, False if key in _CONSTRUCTOR_ENABLES else None)
        and (replaced := _constructor_value(key, params.get(key), value)) is not _KEEP_CONFIGURED
    }
    return {**params, **overrides}


class _Base64Stripper(CustomLogger):
    def strip(self, payload: StandardLoggingPayload) -> None:
        _ = self._strip_base64_from_messages_sync(payload)


_STRIPPER: Final = _Base64Stripper()


def legacy_should_log(kwargs: Mapping[str, object]) -> bool:
    from litellm.litellm_core_utils import litellm_logging

    if litellm.global_disable_no_log_param:
        return True
    litellm_params: Final = kwargs.get("litellm_params", {})
    lookup: Final = getattr(litellm_params, "get", None)
    if callable(lookup) and lookup("no-log", False) is True:
        return False
    return not litellm_logging.callback_disabled_dynamically(
        callback=CALLBACK_NAME,
        litellm_params=litellm_params,
        standard_callback_dynamic_params=kwargs.get("standard_callback_dynamic_params") or {},  # pyright: ignore[reportArgumentType]  # the dynamic params dict the legacy path hands the same check
    )


async def _periodic_flush(sink: "NativeS3Log", interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            await sink.flush()
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:  # noqa: BLE001  # a failed flush keeps the queue and the next interval tries again, as flush_queue does
            verbose_logger.exception("CustomLogger: periodic flush failed: %s", e)


def _excluding_fields(kwargs: object) -> Mapping[str, object] | None:
    excluded: Final[object] = getattr(litellm, "standard_logging_payload_excluded_fields", None)
    details: Final = mapping(kwargs)
    logged: Final = None if details is None else mapping(details.get("standard_logging_object"))
    if details is None or logged is None or not isinstance(excluded, Collection) or not excluded:
        return None
    if logged.get("status") == "failure":
        return None
    return {**details, "standard_logging_object": {key: value for key, value in logged.items() if key not in excluded}}


def _logged_payload(kwargs: object, details: Mapping[str, object] | None) -> object:
    source: Final = mapping(kwargs) if details is None else details
    return None if source is None else source.get("standard_logging_object")


def _legacy_kwargs(view: object) -> object:
    return getattr(view, Field.deprecated_kwargs.value, None)


_read_attribute: Final[Callable[[object, str], object]] = getattr


class _ExcludedView:
    def __init__(self, view: object, details: Mapping[str, object]) -> None:
        args: Final = HookArgs(request=details)
        self.view: Final = view
        self.fields: Final[Mapping[str, object]] = {
            Field.deprecated_kwargs.value: details,
            Field.standard_logging_payload_ref.value: payload_copy(args),
            Field.log_id.value: log_id(args),
            Field.cold_storage_object_key.value: cold_storage_object_key(args),
            Field.logged_key_identity.value: logged_key_identity(args),
        }

    def __getattr__(self, name: str) -> object:
        if name not in self.fields:
            return _read_attribute(self.view, name)
        value: Final = self.fields[name]
        if value is None:
            raise AttributeError(name)
        return value


_ATTEMPT_ENDED_READS: Final = frozenset(
    {
        Field.deprecated_kwargs,
        Field.attempt_timing,
        Field.log_id,
        Field.cold_storage_object_key,
        Field.standard_logging_payload_ref,
        Field.logged_key_identity,
    }
)


class S3Logger(RustLogger):
    selection = Selection(name=CALLBACK_NAME, honors_no_log=True)

    def __init__(
        self,
        s3_callback_params_override: Mapping[str, object] | None = None,
        **settings: Unpack[S3LoggerSettings],  # kwargs-ok: the S3Logger constructor keywords
    ) -> None:
        from litellm.rust_bridge._native import NativeS3Log

        host: Final = _Host()
        params: Final = _constructed_params(s3_callback_params_override, settings)
        sink: Final = NativeS3Log(params, _constants(settings.get("s3_batch_size", DEFAULT_S3_BATCH_SIZE)), host)
        self.s3_region_name: Final[object] = params.get("s3_region_name") or settings.get("s3_region_name")
        self.sink: Final = sink
        self.s3_host: Final = host
        self.periodic_flush_task: Final = asyncio.create_task(
            _periodic_flush(sink, _flush_interval(settings.get("s3_flush_interval", DEFAULT_S3_FLUSH_INTERVAL_SECONDS)))
        )
        super().__init__(sink.callback(), {Field.deprecated_kwargs: lambda _kwargs: None})

    @on(Hook.attempt_ended, reads=_ATTEMPT_ENDED_READS, awaited=True)
    async def on_attempt_ended(self, view: object, ctx: CallContext[object]) -> Patch | Abort | None:
        return await self.run_rust(Hook.attempt_ended, _ATTEMPT_ENDED_READS, view, ctx)

    async def run_rust(
        self, hook: Hook, reads: frozenset[Field], view: object, ctx: CallContext[object]
    ) -> Patch | Abort | None:
        from litellm.litellm_core_utils.litellm_logging import V2_ASYNC_LOGGING_LINE

        kwargs: Final = getattr(view, Field.deprecated_kwargs.value, None)
        if not V2_ASYNC_LOGGING_LINE.get() or not isinstance(kwargs, Mapping) or not legacy_should_log(kwargs):  # pyright: ignore[reportUnknownArgumentType]  # the legacy kwargs dict
            return None
        payload: Final[object] = kwargs.get("standard_logging_object")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the legacy kwargs dict
        if payload is not None and not isinstance(payload, Mapping):
            return self._failed(AttributeError(f"'{type(payload).__name__}' object has no attribute 'get'"))  # pyright: ignore[reportUnknownArgumentType]  # any non-mapping payload
        try:
            raw: Final = _legacy_kwargs(view)
            details: Final = _excluding_fields(raw)
            logged: Final = _logged_payload(raw, details)
            if self.sink.strip_base64_files and isinstance(logged, Mapping):
                _STRIPPER.strip(logged)  # pyright: ignore[reportArgumentType]  # the shared StandardLoggingPayload, stripped in place as S3Logger does
            return await super().run_rust(hook, reads, view if details is None else _ExcludedView(view, details), ctx)
        except Exception as e:  # noqa: BLE001  # S3Logger counts a failure to build the element and moves on
            return self._failed(e)

    def _failed(self, error: Exception) -> None:
        verbose_logger.error("s3 Layer Error - %s", error, exc_info=error)
        self.s3_host.callback_failed()

    @property
    def s3_path(self) -> str | None:
        return self.sink.s3_path

    @property
    def s3_bucket_name(self) -> str | None:
        return self.sink.s3_bucket_name

    def resolve_partition_granularity(self) -> S3PartitionGranularity:
        return "hour" if self.sink.partition_granularity() == "hour" else "day"

    async def flush_queue(self) -> None:
        await self.sink.flush()

    async def async_log_audit_log_event(self, audit_log: StandardAuditLogPayload) -> None:
        try:
            audit_log_id: Final = audit_log.get("id", "unknown")
            await self.sink.submit_audit(dict(audit_log), f"{audit_log_id}")
        except Exception as e:  # noqa: BLE001  # an audit log failure is logged and swallowed, as in S3Logger
            verbose_logger.exception("S3 audit log error: %s", e)

    async def async_log_batch_line_items(self, payload: StandardBatchLineItemsPayload) -> None:
        object_name: Final = batch_line_items_object_name(payload)
        s3_path: Final = self.s3_path
        s3_object_key: Final = (f"{s3_path.rstrip('/')}/" if s3_path else "") + object_name
        body: Final = await asyncio.to_thread(batch_line_items_ndjson, payload)
        delivered: Final = await self.sink.upload_line_items(
            s3_object_key, body, "-".join(object_name.rsplit("/", 2)[-2:])
        )
        if not delivered:
            verbose_logger.error(
                "s3 logging: the %d line items of part %d of batch %s were not stored at %s",
                len(payload["lines"]),
                payload["part"],
                payload["batch_id"],
                s3_object_key,
            )

    async def _download_object_from_s3(self, s3_object_key: str) -> object:
        try:
            downloaded: Final = await self.sink.download(s3_object_key)
            return None if downloaded is None else json.loads(downloaded)
        except Exception as e:  # noqa: BLE001  # a failed read is None, as in S3Logger
            verbose_logger.exception("Error downloading from S3: %s", e)
            return None

    async def get_proxy_server_request_from_cold_storage_with_object_key(self, object_key: str) -> object:
        try:
            return await self._download_object_from_s3(object_key)
        except Exception as e:  # noqa: BLE001  # a failed read is None, as in S3Logger
            verbose_logger.exception("Error retrieving object %s from cold storage: %s", object_key, e)
            return None


__all__ = [
    "S3BatchUploadError",
    "S3Logger",
    "legacy_should_log",
]
