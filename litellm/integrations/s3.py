#### What this does ####
#    On success + failure, log events to Supabase

import hashlib
import os
from collections.abc import Mapping
from datetime import datetime
from typing import Final, cast

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.constants import (
    MAX_S3_OBJECT_DOWNLOAD_FILENAME_BYTES,
    MAX_S3_OBJECT_KEY_BYTES,
    S3_BOUNDED_OBJECT_KEY_HEAD_BYTES,
    S3_LOG_PROMPTS_ONLY_ENV_VAR,
    S3_PREFIX_DIGEST_CHARS,
)
from litellm.types.utils import StandardLoggingPayload

_S3_BOOL: Final = TypeAdapter(bool)
_UPLOAD_BOUND: Final = TypeAdapter(int)


def resolve_s3_log_prompts_only(configured: object, environ: Mapping[str, str] | None = None) -> bool:
    env: Final = os.environ if environ is None else environ
    raw: Final = env.get(S3_LOG_PROMPTS_ONLY_ENV_VAR) if configured is None else configured
    if raw is None or raw == "":
        return False
    try:
        return _S3_BOOL.validate_python(raw.strip() if isinstance(raw, str) else raw)
    except ValidationError:
        verbose_logger.warning("s3 logging: s3_log_prompts_only=%r is not a boolean, logging prompts only", raw)
        return True


def _resolve_positive_int(setting: str, configured: object, fallback: int, *, reject_bool: bool) -> int:
    if configured is None or configured == "":
        return fallback
    if reject_bool and isinstance(configured, bool):
        verbose_logger.warning(
            "s3 logging: %s=%r is a boolean, not an integer, using %s", setting, configured, fallback
        )
        return fallback
    try:
        bound: Final = _UPLOAD_BOUND.validate_python(configured.strip() if isinstance(configured, str) else configured)
    except ValidationError:
        verbose_logger.warning("s3 logging: %s=%r is not an integer, using %s", setting, configured, fallback)
        return fallback
    if bound < 1:
        verbose_logger.warning("s3 logging: %s=%r must be at least 1, using %s", setting, configured, fallback)
        return fallback
    return bound


def resolve_s3_max_concurrent_uploads(configured: object, fallback: int) -> int:
    return _resolve_positive_int("s3_max_concurrent_uploads", configured, fallback, reject_bool=False)


def resolve_s3_max_queue_size(configured: object, fallback: int) -> int:
    return _resolve_positive_int("s3_max_queue_size", configured, fallback, reject_bool=True)


def resolve_s3_max_retry_age_seconds(configured: object) -> int | None:
    if configured is None or configured == "":
        return None
    if isinstance(configured, bool):
        verbose_logger.warning(
            "s3 logging: s3_max_retry_age_seconds=%r is a boolean, not an integer, retry age budget disabled",
            configured,
        )
        return None
    try:
        bound: Final = _UPLOAD_BOUND.validate_python(configured.strip() if isinstance(configured, str) else configured)
    except ValidationError:
        verbose_logger.warning(
            "s3 logging: s3_max_retry_age_seconds=%r is not an integer, retry age budget disabled", configured
        )
        return None
    if bound < 0:
        verbose_logger.warning(
            "s3 logging: s3_max_retry_age_seconds=%r must be at least 0, retry age budget disabled", configured
        )
        return None
    return bound or None


def resolve_s3_max_adaptive_concurrency(configured: object, fallback: int) -> int:
    return _resolve_positive_int("s3_max_adaptive_concurrency", configured, fallback, reject_bool=True)


def resolve_s3_drop_on_terminal_error(configured: object) -> bool:
    if configured is None or configured == "":
        return False
    try:
        return _S3_BOOL.validate_python(configured.strip() if isinstance(configured, str) else configured)
    except ValidationError:
        verbose_logger.warning(
            "s3 logging: s3_drop_on_terminal_error=%r is not a boolean, keeping all failed uploads queued",
            configured,
        )
        return False


def resolve_s3_adaptive_concurrency(configured: object) -> bool:
    if configured is None or configured == "":
        return False
    try:
        return _S3_BOOL.validate_python(configured.strip() if isinstance(configured, str) else configured)
    except ValidationError:
        verbose_logger.warning(
            "s3 logging: s3_adaptive_concurrency=%r is not a boolean, keeping the fixed upload width",
            configured,
        )
        return False


def resolve_s3_batch_file_upload(configured: object) -> bool:
    if configured is None or configured == "":
        return False
    try:
        return _S3_BOOL.validate_python(configured.strip() if isinstance(configured, str) else configured)
    except ValidationError:
        verbose_logger.warning(
            "s3 logging: s3_batch_file_upload=%r is not a boolean, keeping per-request objects", configured
        )
        return False


def prompts_only_payload(payload: StandardLoggingPayload) -> StandardLoggingPayload:
    return {**payload, "response": None}


class S3Logger:
    # Class variables or attributes
    def __init__(
        self,
        s3_bucket_name=None,
        s3_path=None,
        s3_region_name=None,
        s3_api_version=None,
        s3_use_ssl=True,
        s3_verify=None,
        s3_endpoint_url=None,
        s3_aws_access_key_id=None,
        s3_aws_secret_access_key=None,
        s3_aws_session_token=None,
        s3_config=None,
        s3_server_side_encryption: str | None = None,
        s3_sse_kms_key_id: str | None = None,
        s3_log_prompts_only: bool | None = None,
        **kwargs,
    ):
        import boto3

        try:
            verbose_logger.debug("in init s3 logger - s3_callback_params %s", litellm.s3_callback_params)

            s3_use_team_prefix = False
            params: Final = {
                key: litellm.get_secret(value) if isinstance(value, str) and value.startswith("os.environ/") else value
                for key, value in (litellm.s3_callback_params or {}).items()
            }

            if litellm.s3_callback_params is not None:
                s3_bucket_name = params.get("s3_bucket_name")
                s3_region_name = params.get("s3_region_name")
                s3_api_version = params.get("s3_api_version")
                s3_use_ssl = params.get("s3_use_ssl", True)
                s3_verify = params.get("s3_verify")
                s3_endpoint_url = params.get("s3_endpoint_url")
                s3_aws_access_key_id = params.get("s3_aws_access_key_id")
                s3_aws_secret_access_key = params.get("s3_aws_secret_access_key")
                s3_aws_session_token = params.get("s3_aws_session_token")
                s3_config = params.get("s3_config")
                s3_path = params.get("s3_path")
                s3_server_side_encryption = params.get("s3_server_side_encryption")
                s3_sse_kms_key_id = params.get("s3_sse_kms_key_id")
                s3_use_team_prefix = bool(params.get("s3_use_team_prefix", False))
            self.s3_use_team_prefix = s3_use_team_prefix
            self.s3_log_prompts_only: object = (
                params.get("s3_log_prompts_only") if s3_log_prompts_only is None else s3_log_prompts_only
            )
            self.bucket_name = s3_bucket_name
            self.s3_path = s3_path
            self.s3_server_side_encryption, self.s3_sse_kms_key_id = resolve_sse_params(
                s3_server_side_encryption, s3_sse_kms_key_id
            )
            verbose_logger.debug("s3 logger using endpoint url %s", s3_endpoint_url)
            # Create an S3 client with custom endpoint URL
            self.s3_client = boto3.client(
                "s3",
                region_name=s3_region_name,
                endpoint_url=s3_endpoint_url,
                api_version=s3_api_version,
                use_ssl=s3_use_ssl,
                verify=s3_verify,
                aws_access_key_id=s3_aws_access_key_id,
                aws_secret_access_key=s3_aws_secret_access_key,
                aws_session_token=s3_aws_session_token,
                config=s3_config,
                **kwargs,
            )
        except Exception as e:
            print_verbose(f"Got exception on init s3 client {e}")
            raise e

    async def _async_log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        self.log_event(kwargs, response_obj, start_time, end_time, print_verbose)

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        try:
            verbose_logger.debug("s3 Logging - Enters logging function for model %s", kwargs)

            # construct payload to send to s3
            # follows the same params as langfuse.py
            litellm_params: Final = kwargs.get("litellm_params", {})
            metadata: Final = litellm_params.get("metadata", {}) or {}  # if litellm_params['metadata'] == None

            # Clean Metadata before logging - never log raw metadata
            # the raw metadata can contain circular references which leads to infinite recursion
            # we clean out all extra litellm metadata params before logging
            clean_metadata: Final = {}
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    # clean litellm metadata before logging
                    if key in [
                        "headers",
                        "endpoint",
                        "caching_groups",
                        "previous_models",
                    ]:
                        continue
                    else:
                        clean_metadata[key] = value

            # Ensure everything in the payload is converted to str
            payload: Final[StandardLoggingPayload | None] = cast(
                StandardLoggingPayload | None,
                kwargs.get("standard_logging_object", None),
            )

            if payload is None:
                return

            team_alias: Final = payload["metadata"].get("user_api_key_team_alias")

            team_alias_prefix = ""
            if litellm.enable_preview_features and self.s3_use_team_prefix and team_alias is not None:
                team_alias_prefix = f"{team_alias}/"

            s3_file_name: Final = litellm.utils.get_logging_id(start_time, payload) or ""
            s3_object_key: Final = get_s3_object_key(
                cast(str | None, self.s3_path) or "",
                team_alias_prefix,
                start_time,
                s3_file_name,
            )

            s3_object_download_filename: Final = get_s3_object_download_filename(start_time, payload["id"])

            from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

            payload_str: Final = safe_dumps(
                prompts_only_payload(payload) if resolve_s3_log_prompts_only(self.s3_log_prompts_only) else payload
            )

            print_verbose(f"\ns3 Logger - Logging payload = {payload_str}")

            sse_params: Final = {
                key: value
                for key, value in {
                    "ServerSideEncryption": self.s3_server_side_encryption,
                    "SSEKMSKeyId": self.s3_sse_kms_key_id,
                }.items()
                if value
            }

            response: Final = self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_object_key,
                Body=payload_str,
                ContentType="application/json",
                ContentLanguage="en",
                ContentDisposition=f'inline; filename="{s3_object_download_filename}"',
                CacheControl="private, immutable, max-age=31536000, s-maxage=0",
                **sse_params,
            )

            print_verbose(f"Response from s3:{response}")

            print_verbose(f"s3 Layer Logging - final response object: {response_obj}")
            return response
        except Exception as e:
            verbose_logger.exception("s3 Layer Error - %s", e)


def _validated_sse_value(name: str, value: str | None) -> str | None:
    if value is None or isinstance(value, str):
        return value
    verbose_logger.warning(
        "s3 logging: ignoring %s because it has invalid type %s; expected a string", name, type(value).__name__
    )
    return None


def resolve_sse_params(
    server_side_encryption: str | None,
    sse_kms_key_id: str | None,
) -> tuple[str | None, str | None]:
    valid_sse: Final = _validated_sse_value("s3_server_side_encryption", server_side_encryption)
    valid_key_id: Final = _validated_sse_value("s3_sse_kms_key_id", sse_kms_key_id)
    algorithm: Final = valid_sse or ("aws:kms" if valid_key_id else None)
    if algorithm is None:
        return None, None
    if valid_key_id and not algorithm.startswith("aws:kms"):
        verbose_logger.warning(
            "s3 logging: ignoring s3_sse_kms_key_id because s3_server_side_encryption is %s; set it to aws:kms to encrypt with the KMS key",
            algorithm,
        )
        return algorithm, None
    return algorithm, valid_key_id


S3_MIN_BOUNDED_FILE_NAME_BYTES: Final = 64


def _truncate_to_utf8_bytes(value: str, max_bytes: int) -> str:
    """Trim `value` so its UTF-8 encoding fits `max_bytes`, never splitting a character."""
    if max_bytes <= 0:
        return ""
    encoded: Final = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def get_s3_object_download_filename(start_time: datetime, response_id: str) -> str:
    """Content-Disposition filename for the uploaded object, bounded to the metadata header cap."""
    sanitized_response_id: Final = response_id.replace("/", "_").replace('"', "_")
    file_name: Final = f"time-{start_time.strftime('%Y-%m-%dT%H-%M-%S-%f')}_{response_id}"
    sanitized_file_name: Final = f"time-{start_time.strftime('%Y-%m-%dT%H-%M-%S-%f')}_{sanitized_response_id}"
    budget: Final = MAX_S3_OBJECT_DOWNLOAD_FILENAME_BYTES - len(b".json")
    if len(sanitized_file_name.encode("utf-8")) <= budget:
        return sanitized_file_name + ".json"
    return _bounded_s3_file_name(file_name, sanitized_file_name, budget) + ".json"


def _bounded_s3_file_name(s3_file_name: str, sanitized_s3_file_name: str, max_bytes: int) -> str:
    """As much of the file name as `max_bytes` allows, then the sha256 of the whole name."""
    digest: Final = hashlib.sha256(s3_file_name.encode("utf-8")).hexdigest()
    head_budget: Final = min(S3_BOUNDED_OBJECT_KEY_HEAD_BYTES, max_bytes - len(digest) - 1)
    head: Final = _truncate_to_utf8_bytes(sanitized_s3_file_name, head_budget)
    return f"{head}_{digest}" if head else digest


def _bounded_s3_prefix(configured_prefix: str, max_bytes: int) -> str:
    """As much of the configured prefix as fits, then a digest segment naming the full prefix."""
    digest_segment: Final = hashlib.sha256(configured_prefix.encode("utf-8")).hexdigest()[:S3_PREFIX_DIGEST_CHARS] + "/"
    if max_bytes < len(digest_segment):
        return ""
    head: Final = _truncate_to_utf8_bytes(configured_prefix, max_bytes - len(digest_segment) - 1).rstrip("/")
    return f"{head}/{digest_segment}" if head else digest_segment


def get_s3_object_key(
    s3_path: str,
    prefix: str,
    start_time: datetime,
    s3_file_name: str,
) -> str:
    sanitized_s3_file_name: Final = s3_file_name.replace("/", "_").replace(":", "_")
    configured_prefix: Final = (s3_path.rstrip("/") + "/" if s3_path else "") + prefix
    date_segment: Final = start_time.strftime("%Y-%m-%d") + "/"
    # we need the s3 key to include the time, so we log cache hits too
    s3_object_key: Final = configured_prefix + date_segment + sanitized_s3_file_name + ".json"
    if len(s3_object_key.encode("utf-8")) <= MAX_S3_OBJECT_KEY_BYTES:
        return s3_object_key

    # shorten the response id first and only trim the configured prefix if that is what does not
    # fit, so prefix scoped IAM policies and lifecycle rules keep matching
    budget: Final = MAX_S3_OBJECT_KEY_BYTES - len(date_segment.encode("utf-8")) - len(b".json")
    prefix_bytes: Final = len(configured_prefix.encode("utf-8"))
    if prefix_bytes + S3_MIN_BOUNDED_FILE_NAME_BYTES <= budget:
        bounded_file_name: Final = _bounded_s3_file_name(s3_file_name, sanitized_s3_file_name, budget - prefix_bytes)
        return configured_prefix + date_segment + bounded_file_name + ".json"

    shortest_file_name: Final = _bounded_s3_file_name(
        s3_file_name, sanitized_s3_file_name, S3_MIN_BOUNDED_FILE_NAME_BYTES
    )
    bounded_prefix: Final = _bounded_s3_prefix(configured_prefix, budget - len(shortest_file_name.encode("utf-8")))
    return bounded_prefix + date_segment + shortest_file_name + ".json"
