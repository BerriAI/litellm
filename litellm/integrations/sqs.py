"""SQS Logging Integration

This logger sends ``StandardLoggingPayload`` entries to an AWS SQS queue.

"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import traceback
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Annotated, ClassVar, Final

import httpx
from pydantic import BeforeValidator, ConfigDict, PositiveInt, TypeAdapter
from pydantic_core import PydanticCustomError

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.constants import (
    DEFAULT_SQS_BATCH_SIZE,
    DEFAULT_SQS_FLUSH_INTERVAL_SECONDS,
    DEFAULT_SQS_MAX_MESSAGE_BYTES,
    SQS_API_VERSION,
    SQS_SEND_MESSAGE_ACTION,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, run_aws_signing
from litellm.llms.bedrock.web_identity_session_policy import SQS_LOGGING_SESSION_POLICY, WebIdentitySessionPolicy
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.base_health_check import IntegrationHealthCheckStatus
from litellm.types.utils import StandardLoggingPayload

from .custom_batch_logger import CustomBatchLogger


def _reject_bool(value: object) -> object:
    if isinstance(value, bool):
        raise PydanticCustomError("bool_type", "expected a byte count, got a boolean")
    return value


_DECODED_JSON: Final = TypeAdapter(object)
_JSON_OBJECT: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_MAX_MESSAGE_BYTES: Final[TypeAdapter[int]] = TypeAdapter(
    Annotated[PositiveInt, BeforeValidator(_reject_bool)], config=ConfigDict(title="sqs_max_message_bytes")
)
_BASE64_INLINE_PATTERN: Final = re.compile(
    r"data:(?:application|image|audio|video)/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class _DroppableField:
    path: str
    is_set: Callable[[Mapping[str, object]], bool]
    drop: Callable[[Mapping[str, object]], Mapping[str, object]]


def _json_object(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value: Final = payload.get(key)
    return _JSON_OBJECT.validate_python(value) if isinstance(value, Mapping) else {}


def _without_tools(fields: Mapping[str, object]) -> Mapping[str, object]:
    return {key: value for key, value in fields.items() if key != "tools"}


_DROPPABLE_FIELDS: Final = (
    _DroppableField(
        path="response.tools",
        is_set=lambda payload: bool(_json_object(payload, "response").get("tools")),
        drop=lambda payload: {**payload, "response": _without_tools(_json_object(payload, "response"))},
    ),
    _DroppableField(
        path="model_map_information.model_map_value",
        is_set=lambda payload: _json_object(payload, "model_map_information").get("model_map_value") is not None,
        drop=lambda payload: {
            **payload,
            "model_map_information": {**_json_object(payload, "model_map_information"), "model_map_value": None},
        },
    ),
    _DroppableField(
        path="model_parameters.tools",
        is_set=lambda payload: bool(_json_object(payload, "model_parameters").get("tools")),
        drop=lambda payload: {
            **payload,
            "model_parameters": _without_tools(_json_object(payload, "model_parameters")),
        },
    ),
)


def _trimmed_payloads(
    payload: Mapping[str, object],
    fields: tuple[_DroppableField, ...],
    dropped: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], Mapping[str, object]]]:
    if not fields:
        return
    trimmed: Final = fields[0].drop(payload)
    now_dropped: Final = (*dropped, fields[0].path)
    yield now_dropped, trimmed
    yield from _trimmed_payloads(trimmed, fields[1:], now_dropped)


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


class SQSLogger(CustomBatchLogger, BaseAWSLLM):
    """Batching logger that writes logs to an AWS SQS queue, optionally encrypting the payload."""

    _web_identity_session_policy: ClassVar[WebIdentitySessionPolicy] = SQS_LOGGING_SESSION_POLICY

    def __init__(
        self,
        # --- Standard SQS params ---
        sqs_queue_url: str | None = None,
        sqs_region_name: str | None = None,
        sqs_api_version: str | None = None,
        sqs_use_ssl: bool = True,
        sqs_verify: bool | None = None,
        sqs_endpoint_url: str | None = None,
        sqs_aws_access_key_id: str | None = None,
        sqs_aws_secret_access_key: str | None = None,
        sqs_aws_session_token: str | None = None,
        sqs_aws_session_name: str | None = None,
        sqs_aws_profile_name: str | None = None,
        sqs_aws_role_name: str | None = None,
        sqs_aws_web_identity_token: str | None = None,
        sqs_aws_sts_endpoint: str | None = None,
        sqs_flush_interval: int | None = DEFAULT_SQS_FLUSH_INTERVAL_SECONDS,
        sqs_batch_size: int | None = DEFAULT_SQS_BATCH_SIZE,
        sqs_config=None,
        sqs_strip_base64_files: bool = False,
        sqs_max_message_bytes: int = DEFAULT_SQS_MAX_MESSAGE_BYTES,
        # --- 🔐 Application-level encryption params ---
        sqs_aws_use_application_level_encryption: bool = False,
        sqs_app_encryption_key_b64: str | None = None,
        sqs_app_encryption_aad: str | None = None,
        **kwargs,
    ) -> None:
        try:
            verbose_logger.debug("in init sqs logger - sqs_callback_params %s", litellm.aws_sqs_callback_params)

            self.async_httpx_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback,
            )

            self._init_sqs_params(
                sqs_queue_url=sqs_queue_url,
                sqs_region_name=sqs_region_name,
                sqs_api_version=sqs_api_version,
                sqs_use_ssl=sqs_use_ssl,
                sqs_verify=sqs_verify,
                sqs_endpoint_url=sqs_endpoint_url,
                sqs_aws_access_key_id=sqs_aws_access_key_id,
                sqs_aws_secret_access_key=sqs_aws_secret_access_key,
                sqs_aws_session_token=sqs_aws_session_token,
                sqs_aws_session_name=sqs_aws_session_name,
                sqs_aws_profile_name=sqs_aws_profile_name,
                sqs_aws_role_name=sqs_aws_role_name,
                sqs_aws_web_identity_token=sqs_aws_web_identity_token,
                sqs_aws_sts_endpoint=sqs_aws_sts_endpoint,
                sqs_strip_base64_files=sqs_strip_base64_files,
                sqs_max_message_bytes=sqs_max_message_bytes,
                sqs_aws_use_application_level_encryption=sqs_aws_use_application_level_encryption,
                sqs_app_encryption_key_b64=sqs_app_encryption_key_b64,
                sqs_app_encryption_aad=sqs_app_encryption_aad,
                sqs_config=sqs_config,
                **kwargs,
            )

            asyncio.create_task(self.periodic_flush())
            self.flush_lock = asyncio.Lock()

            verbose_logger.debug("sqs flush interval: %s, sqs batch size: %s", sqs_flush_interval, sqs_batch_size)

            CustomBatchLogger.__init__(
                self,
                flush_lock=self.flush_lock,
                flush_interval=sqs_flush_interval,
                batch_size=sqs_batch_size,
            )

            self.log_queue: list[StandardLoggingPayload] = []
            BaseAWSLLM.__init__(self)

        except Exception as e:
            print_verbose(f"Got exception on init sqs client {e}")
            raise e

    def _init_sqs_params(
        self,
        sqs_queue_url: str | None = None,
        sqs_region_name: str | None = None,
        sqs_api_version: str | None = None,
        sqs_use_ssl: bool = True,
        sqs_verify: bool | None = None,
        sqs_endpoint_url: str | None = None,
        sqs_aws_access_key_id: str | None = None,
        sqs_aws_secret_access_key: str | None = None,
        sqs_aws_session_token: str | None = None,
        sqs_aws_session_name: str | None = None,
        sqs_aws_profile_name: str | None = None,
        sqs_aws_role_name: str | None = None,
        sqs_aws_web_identity_token: str | None = None,
        sqs_aws_sts_endpoint: str | None = None,
        sqs_strip_base64_files: bool = False,
        sqs_max_message_bytes: int = DEFAULT_SQS_MAX_MESSAGE_BYTES,
        sqs_aws_use_application_level_encryption: bool = False,
        sqs_app_encryption_key_b64: str | None = None,
        sqs_app_encryption_aad: str | None = None,
        sqs_config=None,
    ) -> None:
        litellm.aws_sqs_callback_params = litellm.aws_sqs_callback_params or {}

        # read in .env variables - example os.environ/AWS_BUCKET_NAME
        for key, value in litellm.aws_sqs_callback_params.items():
            if isinstance(value, str) and value.startswith("os.environ/"):
                litellm.aws_sqs_callback_params[key] = litellm.get_secret(value)

        self.sqs_queue_url = litellm.aws_sqs_callback_params.get("sqs_queue_url") or sqs_queue_url
        self.sqs_region_name = litellm.aws_sqs_callback_params.get("sqs_region_name") or sqs_region_name
        self.sqs_api_version = litellm.aws_sqs_callback_params.get("sqs_api_version") or sqs_api_version
        self.sqs_use_ssl = litellm.aws_sqs_callback_params.get("sqs_use_ssl", True) or sqs_use_ssl
        self.sqs_verify = litellm.aws_sqs_callback_params.get("sqs_verify") or sqs_verify
        self.sqs_endpoint_url = litellm.aws_sqs_callback_params.get("sqs_endpoint_url") or sqs_endpoint_url
        self.sqs_aws_access_key_id = (
            litellm.aws_sqs_callback_params.get("sqs_aws_access_key_id") or sqs_aws_access_key_id
        )

        self.sqs_aws_secret_access_key = (
            litellm.aws_sqs_callback_params.get("sqs_aws_secret_access_key") or sqs_aws_secret_access_key
        )

        self.sqs_aws_session_token = (
            litellm.aws_sqs_callback_params.get("sqs_aws_session_token") or sqs_aws_session_token
        )

        self.sqs_aws_session_name = litellm.aws_sqs_callback_params.get("sqs_aws_session_name") or sqs_aws_session_name

        self.sqs_aws_profile_name = litellm.aws_sqs_callback_params.get("sqs_aws_profile_name") or sqs_aws_profile_name

        self.sqs_aws_role_name = litellm.aws_sqs_callback_params.get("sqs_aws_role_name") or sqs_aws_role_name

        self.sqs_aws_web_identity_token = (
            litellm.aws_sqs_callback_params.get("sqs_aws_web_identity_token") or sqs_aws_web_identity_token
        )

        self.sqs_aws_sts_endpoint = litellm.aws_sqs_callback_params.get("sqs_aws_sts_endpoint") or sqs_aws_sts_endpoint
        self.sqs_strip_base64_files = (
            litellm.aws_sqs_callback_params.get("sqs_strip_base64_files", False) or sqs_strip_base64_files
        )
        configured_max_message_bytes: Final = _DECODED_JSON.validate_python(
            litellm.aws_sqs_callback_params.get("sqs_max_message_bytes")
        )
        self.sqs_max_message_bytes: int = _MAX_MESSAGE_BYTES.validate_python(
            sqs_max_message_bytes if configured_max_message_bytes is None else configured_max_message_bytes
        )

        self.sqs_aws_use_application_level_encryption = (
            litellm.aws_sqs_callback_params.get("sqs_aws_use_application_level_encryption", False)
            or sqs_aws_use_application_level_encryption
        )
        self.sqs_app_encryption_key_b64 = (
            litellm.aws_sqs_callback_params.get("sqs_app_encryption_key_b64") or sqs_app_encryption_key_b64
        )
        self.sqs_app_encryption_aad: str | None = (
            litellm.aws_sqs_callback_params.get("sqs_app_encryption_aad") or sqs_app_encryption_aad
        )
        self.app_crypto: AppCrypto | None = None
        if self.sqs_aws_use_application_level_encryption:
            from litellm.litellm_core_utils.app_crypto import AppCrypto

            if not self.sqs_app_encryption_key_b64:
                raise ValueError("sqs_app_encryption_key_b64 is required when encryption is enabled.")
            key = base64.b64decode(self.sqs_app_encryption_key_b64)
            self.app_crypto = AppCrypto(key)
            verbose_logger.debug("SQSLogger: Application-level encryption enabled.")
        self.sqs_config = litellm.aws_sqs_callback_params.get("sqs_config") or sqs_config

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        try:
            verbose_logger.debug("SQS Logging - Enters logging function for model %s", kwargs)
            standard_logging_payload = kwargs.get("standard_logging_object")
            if self.sqs_strip_base64_files:
                standard_logging_payload = await self._strip_base64_from_messages(standard_logging_payload)
            if standard_logging_payload is None:
                raise ValueError("standard_logging_payload is None")

            self.log_queue.append(standard_logging_payload)
            verbose_logger.debug(
                "sqs logging: queue length %s, batch size %s",
                len(self.log_queue),
                self.batch_size,
            )
        except Exception as e:
            verbose_logger.exception("sqs Layer Error - %s", e)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            standard_logging_payload = kwargs.get("standard_logging_object")
            if standard_logging_payload is None:
                raise ValueError("standard_logging_payload is None")
            if self.sqs_strip_base64_files:
                standard_logging_payload = await self._strip_base64_from_messages(standard_logging_payload)

            self.log_queue.append(standard_logging_payload)
            verbose_logger.debug(
                "sqs logging: queue length %s, batch size %s",
                len(self.log_queue),
                self.batch_size,
            )

        except Exception as e:
            verbose_logger.exception("Datadog Layer Error - %s\n%s", e, traceback.format_exc())

    async def async_send_batch(self) -> None:
        verbose_logger.debug("sqs logger - sending batch of %s", len(self.log_queue))
        if not self.log_queue:
            return

        for payload in self.log_queue:
            asyncio.create_task(self.async_send_message(payload))

    def _message_body(self, payload: Mapping[str, object]) -> str:
        if self.app_crypto is None:
            return safe_dumps(payload)
        aad_bytes: Final = self.sqs_app_encryption_aad.encode("utf-8") if self.sqs_app_encryption_aad else None
        json_data: Final = _DECODED_JSON.validate_python(json.loads(safe_dumps(payload)))
        encrypted: Final = self.app_crypto.encrypt_json(json_data, aad=aad_bytes)
        return json.dumps({"__encrypted__": True, "payload": encrypted})

    def _message_body_within_limit(self, payload: Mapping[str, object], call_id: str | None) -> str | None:
        full_body: Final = self._message_body(payload)
        full_size: Final = _utf8_size(full_body)
        if full_size <= self.sqs_max_message_bytes:
            return full_body
        droppable: Final = tuple(field for field in _DROPPABLE_FIELDS if field.is_set(payload))
        trimmed_bodies: Final = (
            (dropped, self._message_body(trimmed)) for dropped, trimmed in _trimmed_payloads(payload, droppable)
        )
        fitting: Final = next(
            ((dropped, body) for dropped, body in trimmed_bodies if _utf8_size(body) <= self.sqs_max_message_bytes),
            None,
        )
        if fitting is None:
            verbose_logger.error(
                "Dropped the SQS message for litellm_call_id=%s. It is %s bytes, over sqs_max_message_bytes=%s, "
                "even after dropping the optional fields it had (%s)",
                call_id,
                full_size,
                self.sqs_max_message_bytes,
                ", ".join(field.path for field in droppable) or "none",
            )
            return None
        dropped, body = fitting
        verbose_logger.warning(
            "Dropped %s from the SQS message for litellm_call_id=%s to fit sqs_max_message_bytes=%s. "
            "It went from %s to %s bytes",
            ", ".join(dropped),
            call_id,
            self.sqs_max_message_bytes,
            full_size,
            _utf8_size(body),
        )
        return body

    async def async_send_message(self, payload: StandardLoggingPayload) -> None:
        call_id: Final = payload.get("litellm_call_id")
        try:
            from urllib.parse import quote

            import requests
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest

            from litellm.litellm_core_utils.asyncify import asyncify

            if self.sqs_queue_url is None:
                raise ValueError("sqs_queue_url not set")

            message_body: Final = self._message_body_within_limit(payload, call_id)
            if message_body is None:
                return

            asyncified_get_credentials: Final = asyncify(self.get_credentials)
            credentials: Final = await asyncified_get_credentials(
                aws_access_key_id=self.sqs_aws_access_key_id,
                aws_secret_access_key=self.sqs_aws_secret_access_key,
                aws_session_token=self.sqs_aws_session_token,
                aws_region_name=self.sqs_region_name,
                aws_session_name=self.sqs_aws_session_name,
                aws_profile_name=self.sqs_aws_profile_name,
                aws_role_name=self.sqs_aws_role_name,
                aws_web_identity_token=self.sqs_aws_web_identity_token,
                aws_sts_endpoint=self.sqs_aws_sts_endpoint,
            )

            body: Final = f"Action={SQS_SEND_MESSAGE_ACTION}&Version={SQS_API_VERSION}&MessageBody=" + quote(
                message_body, safe=""
            )

            headers: Final = {
                "Content-Type": "application/x-www-form-urlencoded",
            }

            req: Final = requests.Request("POST", self.sqs_queue_url, data=body, headers=headers)
            prepped: Final = req.prepare()

            aws_request: Final = AWSRequest(
                method=prepped.method,
                url=prepped.url,
                data=prepped.body,
                headers=prepped.headers,
            )
            await run_aws_signing(SigV4Auth(credentials, "sqs", self.sqs_region_name).add_auth, aws_request)

            signed_headers: Final = dict(aws_request.headers.items())

            await self.async_httpx_client.post(
                self.sqs_queue_url,
                data=body,
                headers=signed_headers,
            )
        except httpx.HTTPStatusError as e:
            verbose_logger.error(
                "SQS rejected the message for litellm_call_id=%s with status_code=%s: %s",
                call_id,
                e.response.status_code,
                " ".join(e.response.text.split()),
            )
        except Exception as e:
            verbose_logger.exception("Error sending to SQS: %s", e)

    async def async_health_check(self) -> IntegrationHealthCheckStatus:
        """
        Health check for SQS by sending a small test message to the configured queue.
        """
        try:
            from litellm.litellm_core_utils.litellm_logging import (
                create_dummy_standard_logging_payload,
            )

            # Create a minimal standard logging payload
            standard_logging_object: Final[StandardLoggingPayload] = create_dummy_standard_logging_payload()
            # Attempt to send a single message
            await self.async_send_message(standard_logging_object)
            return IntegrationHealthCheckStatus(status="healthy", error_message=None)
        except Exception as e:
            return IntegrationHealthCheckStatus(status="unhealthy", error_message=str(e))
