import posixpath
import re
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple, cast
from urllib.parse import quote, unquote

from litellm._uuid import uuid

VERTEX_AI_MANAGED_GCS_PREFIX = "litellm-vertex-files/"
BEDROCK_MANAGED_S3_BATCH_PREFIX = "litellm-bedrock-files-"
BEDROCK_MANAGED_S3_UPLOAD_PREFIX = "litellm-bedrock-files/"
BEDROCK_MANAGED_S3_OUTPUT_PREFIX = "litellm-batch-outputs/"
BEDROCK_MANAGED_S3_PREFIXES = (
    BEDROCK_MANAGED_S3_BATCH_PREFIX,
    BEDROCK_MANAGED_S3_UPLOAD_PREFIX,
    BEDROCK_MANAGED_S3_OUTPUT_PREFIX,
)
_MAPPING_PROXY_TYPE: type = type(MappingProxyType({}))

_SAFE_OBJECT_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_cloud_object_component(
    value: Optional[str], fallback: str = "file"
) -> str:
    if not isinstance(value, str):
        return fallback

    component = posixpath.basename(value.replace("\\", "/")).strip()
    if component in {"", ".", ".."}:
        return fallback

    component = "".join(
        "_" if ord(char) < 32 or ord(char) == 127 else char for char in component
    )
    component = _SAFE_OBJECT_COMPONENT_PATTERN.sub("_", component)
    component = component.strip("._")
    if not component:
        return fallback
    return component[:255]


def sanitize_cloud_object_path(value: Optional[str], fallback: str = "file") -> str:
    if not isinstance(value, str):
        return fallback

    segments = []
    for segment in value.replace("\\", "/").split("/"):
        sanitized_segment = sanitize_cloud_object_component(segment, fallback="")
        if sanitized_segment:
            segments.append(sanitized_segment)

    if not segments:
        return fallback
    return "/".join(segments)


def build_managed_cloud_object_name(
    prefix: str, filename: Optional[str], fallback_filename: str = "file"
) -> str:
    safe_filename = sanitize_cloud_object_component(
        filename, fallback=fallback_filename
    )
    return f"{prefix}{uuid.uuid4().hex}-{safe_filename}"


def _validate_cloud_object_path(object_name: str) -> None:
    if not object_name:
        raise ValueError("Cloud storage object name is required")
    if object_name.startswith("/"):
        raise ValueError("Cloud storage object name must be relative")
    if any(ord(char) < 32 or ord(char) == 127 for char in object_name):
        raise ValueError("Cloud storage object name contains control characters")
    segments = object_name.split("/")
    if any(segment in {".", ".."} for segment in segments):
        raise ValueError("Cloud storage object name contains an invalid path segment")
    if "" in segments[:-1]:
        raise ValueError("Cloud storage object name contains an invalid path segment")


def split_configured_cloud_bucket_name(bucket_name: str) -> Tuple[str, str]:
    if not isinstance(bucket_name, str) or not bucket_name.strip():
        raise ValueError("Cloud storage bucket name is required")

    bucket_name = bucket_name.strip()
    if "://" in bucket_name or "?" in bucket_name or "#" in bucket_name:
        raise ValueError(
            "Cloud storage bucket name must not include a URI scheme or query"
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in bucket_name):
        raise ValueError("Cloud storage bucket name contains control characters")

    bucket, _, prefix = bucket_name.partition("/")
    if not bucket:
        raise ValueError("Cloud storage bucket name is required")
    if "\\" in bucket:
        raise ValueError("Cloud storage bucket name contains an invalid separator")

    prefix = prefix.strip("/")
    if prefix:
        _validate_cloud_object_path(prefix)

    return bucket, prefix


def encode_gcs_object_name_for_url(object_name: str) -> str:
    return quote(unquote(object_name), safe="")


def encode_s3_object_key_for_url(object_key: str) -> str:
    return quote(unquote(object_key), safe="/")


def should_allow_legacy_cloud_file_ids(
    litellm_params: Optional[Mapping[str, Any]] = None,
) -> bool:
    value = None
    if isinstance(litellm_params, Mapping):
        trusted_model_credentials = litellm_params.get(
            "_litellm_internal_model_credentials"
        )
        if isinstance(trusted_model_credentials, _MAPPING_PROXY_TYPE):
            value = cast(Mapping[str, Any], trusted_model_credentials).get(
                "allow_legacy_cloud_file_ids"
            )

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def validate_managed_cloud_file_id(
    file_id: str,
    scheme: str,
    configured_bucket_name: str,
    allowed_object_prefixes: Sequence[str],
    allow_legacy_cloud_file_ids: bool = False,
) -> Tuple[str, str]:
    decoded_file_id = unquote(file_id)
    if not decoded_file_id.startswith(scheme):
        raise ValueError(f"file_id must be a {scheme} URI")

    full_path = decoded_file_id[len(scheme) :]
    if "/" not in full_path:
        raise ValueError("file_id must include a cloud storage object name")

    bucket_name, object_name = full_path.split("/", 1)
    configured_bucket, configured_prefix = split_configured_cloud_bucket_name(
        configured_bucket_name
    )
    if bucket_name != configured_bucket:
        raise ValueError("file_id bucket does not match the configured storage bucket")

    _validate_cloud_object_path(object_name)
    allowed_prefixes = tuple(allowed_object_prefixes)
    if configured_prefix:
        allowed_prefixes = tuple(
            f"{configured_prefix.rstrip('/')}/{prefix}" for prefix in allowed_prefixes
        )

    if object_name.startswith(allowed_prefixes):
        return bucket_name, object_name

    if allow_legacy_cloud_file_ids:
        if configured_prefix and not object_name.startswith(
            f"{configured_prefix.rstrip('/')}/"
        ):
            raise ValueError(
                "file_id object does not match the configured storage prefix"
            )
        return bucket_name, object_name

    raise ValueError("file_id must reference a LiteLLM-managed storage object")
