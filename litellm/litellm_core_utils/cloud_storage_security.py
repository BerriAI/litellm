import posixpath
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import accumulate, chain, repeat
from types import MappingProxyType
from typing import Any, Final, cast
from urllib.parse import quote, unquote

from litellm._uuid import uuid
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping

VERTEX_AI_MANAGED_GCS_PREFIX: Final = "litellm-vertex-files/"
BEDROCK_MANAGED_S3_BATCH_PREFIX: Final = "litellm-bedrock-files-"
BEDROCK_MANAGED_S3_UPLOAD_PREFIX: Final = "litellm-bedrock-files/"
BEDROCK_MANAGED_S3_OUTPUT_PREFIX: Final = "litellm-batch-outputs/"
BEDROCK_MANAGED_S3_PREFIXES: Final = (
    BEDROCK_MANAGED_S3_BATCH_PREFIX,
    BEDROCK_MANAGED_S3_UPLOAD_PREFIX,
    BEDROCK_MANAGED_S3_OUTPUT_PREFIX,
)
MANAGED_CLOUD_STORAGE_SCHEMES: Final = ("s3://", "gs://")
_MAPPING_PROXY_TYPE: Final[type] = type(MappingProxyType({}))
MAX_FILE_ID_DECODE_PASSES: Final = 8


def _unquote_once(current: str, _: str) -> str:
    return unquote(current)


def _decodings(value: str) -> tuple[str, ...]:
    return tuple(accumulate(repeat(value, MAX_FILE_ID_DECODE_PASSES + 2), _unquote_once))


def is_managed_cloud_storage_uri(file_id: str) -> bool:
    """
    True if file_id is a raw cloud-storage object URI (e.g. ``s3://bucket/key``).

    These are internal provider artifacts. On the multi-tenant proxy they must be
    retrieved through their managed unified file id so owner/team access is enforced;
    a raw URI supplied by a caller bypasses that check.
    """
    return isinstance(file_id, str) and file_id.startswith(MANAGED_CLOUD_STORAGE_SCHEMES)


_S3_SCHEME: Final = "s3://"
_S3_MEDIA_PART_TYPES: Final = frozenset(
    {"image_url", "file", "input_image", "input_file", "computer_screenshot", "image", "document"}
)
_GEMINI_MEDIA_PART_KEYS: Final = frozenset({"file_data", "fileData"})
_MEDIA_URI_KEYS: Final = frozenset({"url", "image_url", "file_id", "file_url", "file_data", "file_uri", "fileUri"})


def is_s3_uri(value: str) -> bool:
    return value.startswith(_S3_SCHEME)


def _media_part_uris(value: object, key: str | None = None) -> Iterator[str]:
    if isinstance(value, str):
        return iter((value,) if key in _MEDIA_URI_KEYS else ())
    fields: Final = as_str_mapping(value)
    if fields is not None:
        return chain.from_iterable(_media_part_uris(child, child_key) for child_key, child in fields.items())
    items: Final = as_object_list(value)
    return chain.from_iterable(map(_media_part_uris, items)) if items is not None else iter(())


def _is_media_part(fields: Mapping[str, object], key: str | None) -> bool:
    part_type: Final = fields.get("type")
    return (isinstance(part_type, str) and part_type in _S3_MEDIA_PART_TYPES) or key in _GEMINI_MEDIA_PART_KEYS


def _media_uris(value: object, key: str | None = None) -> Iterator[str]:
    fields: Final = as_str_mapping(value)
    if fields is not None and _is_media_part(fields, key):
        return _media_part_uris(fields)
    if fields is not None:
        return chain.from_iterable(_media_uris(child, child_key) for child_key, child in fields.items())
    items: Final = as_object_list(value)
    return chain.from_iterable(map(_media_uris, items)) if items is not None else iter(())


def _s3_bucket(uri: str) -> str | None:
    return uri.removeprefix(_S3_SCHEME).partition("/")[0] if uri.startswith(_S3_SCHEME) else None


def _s3_object_key(uri: str) -> str | None:
    if not uri.startswith(_S3_SCHEME):
        return None
    _, separator, object_key = uri.removeprefix(_S3_SCHEME).partition("/")
    return object_key if separator else None


def _outside_managed_prefixes(object_key: str) -> bool:
    return not any(prefix in object_key for prefix in BEDROCK_MANAGED_S3_PREFIXES)


def _refused_in_some_decoding(
    uri: str, s3_part: Callable[[str], str | None], is_allowed: Callable[[str], bool]
) -> bool:
    decodings: Final = _decodings(uri)
    parts: Final = tuple(part for part in map(s3_part, decodings) if part is not None)
    settled: Final = decodings[-1] == decodings[-2]
    return bool(parts) and (not settled or not all(map(is_allowed, parts)))


def s3_media_uris(request_body: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(filter(is_s3_uri, _media_uris(request_body))))


def is_managed_s3_uri(uri: str) -> bool:
    return _refused_in_some_decoding(uri, _s3_object_key, _outside_managed_prefixes)


def names_an_unlisted_s3_bucket(uri: str, is_listed: Callable[[str], bool]) -> bool:
    return _refused_in_some_decoding(uri, _s3_bucket, is_listed)


_SAFE_OBJECT_COMPONENT_PATTERN: Final = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_cloud_object_component(value: str | None, fallback: str = "file") -> str:
    if not isinstance(value, str):
        return fallback

    component = posixpath.basename(value.replace("\\", "/")).strip()
    if component in {"", ".", ".."}:
        return fallback

    component = "".join("_" if ord(char) < 32 or ord(char) == 127 else char for char in component)
    component = _SAFE_OBJECT_COMPONENT_PATTERN.sub("_", component)
    component = component.strip("._")
    if not component:
        return fallback
    return component[:255]


def sanitize_cloud_object_path(value: str | None, fallback: str = "file") -> str:
    if not isinstance(value, str):
        return fallback

    segments: Final = []
    for segment in value.replace("\\", "/").split("/"):
        sanitized_segment = sanitize_cloud_object_component(segment, fallback="")
        if sanitized_segment:
            segments.append(sanitized_segment)

    if not segments:
        return fallback
    return "/".join(segments)


def build_managed_cloud_object_name(prefix: str, filename: str | None, fallback_filename: str = "file") -> str:
    safe_filename: Final = sanitize_cloud_object_component(filename, fallback=fallback_filename)
    return f"{prefix}{uuid.uuid4().hex}-{safe_filename}"


def _validate_cloud_object_path(object_name: str) -> None:
    if not object_name:
        raise ValueError("Cloud storage object name is required")
    if object_name.startswith("/"):
        raise ValueError("Cloud storage object name must be relative")
    if any(ord(char) < 32 or ord(char) == 127 for char in object_name):
        raise ValueError("Cloud storage object name contains control characters")
    segments: Final = object_name.split("/")
    if any(segment in {".", ".."} for segment in segments):
        raise ValueError("Cloud storage object name contains an invalid path segment")
    if "" in segments[:-1]:
        raise ValueError("Cloud storage object name contains an invalid path segment")


def split_configured_cloud_bucket_name(bucket_name: str) -> tuple[str, str]:
    if not isinstance(bucket_name, str) or not bucket_name.strip():
        raise ValueError("Cloud storage bucket name is required")

    bucket_name = bucket_name.strip()
    if "://" in bucket_name or "?" in bucket_name or "#" in bucket_name:
        raise ValueError("Cloud storage bucket name must not include a URI scheme or query")
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
    litellm_params: Mapping[str, object] | None = None,
) -> bool:
    value = None
    if isinstance(litellm_params, Mapping):
        trusted_model_credentials: Final = litellm_params.get("_litellm_internal_model_credentials")
        if isinstance(trusted_model_credentials, _MAPPING_PROXY_TYPE):
            value = cast(Mapping[str, Any], trusted_model_credentials).get("allow_legacy_cloud_file_ids")

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
) -> tuple[str, str]:
    decoded_file_id: Final = unquote(file_id)
    if not decoded_file_id.startswith(scheme):
        raise ValueError(f"file_id must be a {scheme} URI")

    full_path: Final = decoded_file_id[len(scheme) :]
    if "/" not in full_path:
        raise ValueError("file_id must include a cloud storage object name")

    bucket_name, object_name = full_path.split("/", 1)
    configured_bucket, configured_prefix = split_configured_cloud_bucket_name(configured_bucket_name)
    if bucket_name != configured_bucket:
        raise ValueError("file_id bucket does not match the configured storage bucket")

    _validate_cloud_object_path(object_name)
    allowed_prefixes = tuple(allowed_object_prefixes)
    if configured_prefix:
        allowed_prefixes = tuple(f"{configured_prefix.rstrip('/')}/{prefix}" for prefix in allowed_prefixes)

    if object_name.startswith(allowed_prefixes):
        return bucket_name, object_name

    if allow_legacy_cloud_file_ids:
        if configured_prefix and not object_name.startswith(f"{configured_prefix.rstrip('/')}/"):
            raise ValueError("file_id object does not match the configured storage prefix")
        return bucket_name, object_name

    raise ValueError("file_id must reference a LiteLLM-managed storage object")
