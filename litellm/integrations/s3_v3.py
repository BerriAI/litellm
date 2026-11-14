import asyncio
import json
import os
from collections.abc import Mapping
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Final

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
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch
from litellm.integrations.custom_logger_v2.rust import RustLogger, rust_logger
from litellm.integrations.s3_v2 import S3Logger
from litellm.llms.bedrock.web_identity_session_policy import S3_LOGGING_SESSION_POLICY, session_policy_document
from litellm.secret_managers.main import get_secret_str
from litellm.types.integrations.s3_v2 import S3PartitionGranularity
from litellm.types.utils import StandardAuditLogPayload, StandardBatchLineItemsPayload, StandardLoggingPayload

if TYPE_CHECKING:
    from litellm.rust_bridge._native import NativeS3Log

NATIVE_S3_V2_FLAG: Final = "LITELLM_NATIVE_S3_V2"
CALLBACK_NAME: Final = "s3_v2"


@cache
def native_s3_v2_enabled() -> bool:
    return os.environ.get(NATIVE_S3_V2_FLAG, "").strip().lower() == "true"


class _Host:
    def __init__(self) -> None:
        self.failures: Final = CustomLogger()

    def callback_failed(self) -> None:
        self.failures.handle_callback_failure(callback_name="S3Logger")

    def cold_storage_is_s3_v2(self) -> bool:
        return litellm.cold_storage_custom_logger == "s3_v2"

    def region_from_environment(self) -> str | None:
        region_name: Final = get_secret_str("AWS_REGION_NAME", None)
        region: Final = get_secret_str("AWS_REGION", None)
        return region if region is not None else region_name

    def web_identity_token(self, reference: str) -> str:
        if reference.startswith("os.environ/") or reference in os.environ:
            raise ValueError("Invalid web identity token reference.")
        token: Final = get_secret_str(reference)
        if token is None:
            raise ValueError("OIDC token could not be retrieved from secret manager.")
        return token


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


def _constants() -> Mapping[str, object]:
    return {
        "batch_size": DEFAULT_S3_BATCH_SIZE or litellm.DEFAULT_BATCH_SIZE,
        "max_concurrent_uploads": DEFAULT_S3_MAX_CONCURRENT_UPLOADS,
        "max_adaptive_concurrency": DEFAULT_S3_MAX_ADAPTIVE_CONCURRENCY,
        "upload_attempt_timeout_seconds": S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS,
        "download_read_timeout_seconds": COMPLETION_HTTP_FALLBACK_SECONDS,
        "download_connect_timeout_seconds": HTTP_HANDLER_CONNECT_TIMEOUT_SECONDS,
        "web_identity_session_policy": session_policy_document(S3_LOGGING_SESSION_POLICY),
    }


def _flush_interval() -> float:
    return DEFAULT_S3_FLUSH_INTERVAL_SECONDS or litellm.DEFAULT_FLUSH_INTERVAL_SECONDS


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


class S3LoggerV3(RustLogger):
    sink: ClassVar["NativeS3Log"]
    s3_host: ClassVar[_Host]
    periodic_flush_task: ClassVar["asyncio.Task[None]"]

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
            if self.sink.strip_base64_files and isinstance(payload, Mapping):
                _STRIPPER.strip(payload)  # pyright: ignore[reportArgumentType]  # the shared StandardLoggingPayload, stripped in place as S3Logger does
            return await super().run_rust(hook, reads, view, ctx)
        except Exception as e:  # noqa: BLE001  # S3Logger counts a failure to build the element and moves on
            return self._failed(e)

    def _failed(self, error: Exception) -> None:
        verbose_logger.error("s3 Layer Error - %s", error, exc_info=error)
        self.s3_host.callback_failed()

    @property
    def s3_path(self) -> str | None:
        return self.sink.s3_path

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


S3_V2_LOGGER_TYPES: Final = (S3Logger, S3LoggerV3)


def new_s3_v2_logger(s3_callback_params_override: Mapping[str, object] | None = None) -> S3Logger | S3LoggerV3:
    if not native_s3_v2_enabled():
        return (
            S3Logger()
            if s3_callback_params_override is None
            else S3Logger(s3_callback_params_override=dict(s3_callback_params_override))
        )
    from litellm.rust_bridge._native import NativeS3Log

    host: Final = _Host()
    sink: Final = NativeS3Log(_resolved_params(s3_callback_params_override), _constants(), host)
    periodic: Final = asyncio.create_task(_periodic_flush(sink, _flush_interval()))
    generated: Final = type(rust_logger(sink.callback()))
    cls: Final = type(
        S3Logger.__name__, (S3LoggerV3, generated), {"sink": sink, "s3_host": host, "periodic_flush_task": periodic}
    )
    return cls(
        sink.callback(),
        {Field.deprecated_kwargs: lambda _kwargs: None},
    )


__all__ = [
    "NATIVE_S3_V2_FLAG",
    "S3_V2_LOGGER_TYPES",
    "S3LoggerV3",
    "legacy_should_log",
    "native_s3_v2_enabled",
    "new_s3_v2_logger",
]
