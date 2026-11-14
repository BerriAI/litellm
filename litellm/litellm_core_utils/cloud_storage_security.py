import base64
import posixpath
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import accumulate, chain, repeat, takewhile
from typing import Final
from urllib.parse import quote, unquote

from pydantic import TypeAdapter

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
STORED_S3_BUCKET_PARAM: Final = "stored_s3_bucket_name"
REQUEST_S3_BUCKET_PARAMS: Final = frozenset({"s3_bucket_name", "s3_output_bucket_name"})


def stored_s3_bucket_kwargs(stored_file_id: str) -> Mapping[str, str]:
    """The bucket (and any prefix before LiteLLM's own path) of an ``s3://`` file id LiteLLM recorded itself,
    as the param that lets Bedrock read it back.

    Only pass ids from LiteLLM's own records (managed file rows, provider batch responses), never one a caller sent.
    """
    if not stored_file_id.startswith("s3://"):
        return {}
    bucket, _, object_key = stored_file_id.removeprefix("s3://").partition("/")
    if not bucket:
        return {}
    segments: Final = object_key.split("/")
    prefix_depth: Final = next(
        (depth for depth in range(len(segments)) if "/".join(segments[depth:]).startswith(BEDROCK_MANAGED_S3_PREFIXES)),
        0,
    )
    prefix: Final = "/".join(segments[:prefix_depth])
    return {STORED_S3_BUCKET_PARAM: f"{bucket}/{prefix}" if prefix else bucket}


MAX_FILE_ID_DECODE_PASSES: Final = 8


def _unquote_once(current: str, _: str) -> str:
    return unquote(current)


def _decodings(value: str) -> tuple[str, ...]:
    return tuple(accumulate(repeat(value, MAX_FILE_ID_DECODE_PASSES + 2), _unquote_once))


def is_managed_cloud_storage_uri(file_id: str) -> bool:
    """
    True if file_id is a raw cloud-storage object URI (e.g. ``s3://bucket/key``),
    up to ``MAX_FILE_ID_DECODE_PASSES`` layers of percent-encoding deep, or an id
    encoded deeper than that, which no client produces and which is treated as raw
    rather than decoded any further.

    These are internal provider artifacts. On the multi-tenant proxy they must be
    retrieved through their managed unified file id so owner/team access is enforced;
    a raw URI supplied by a caller bypasses that check.
    """
    if not isinstance(file_id, str):
        return False
    decodings: Final = _decodings(file_id)
    settled: Final = decodings[-1] == decodings[-2]
    return decodings[-1].startswith(MANAGED_CLOUD_STORAGE_SCHEMES) or not settled


_S3_SCHEME: Final = "s3://"
_S3_MEDIA_PART_TYPES: Final = frozenset(
    {"image_url", "file", "input_image", "input_file", "computer_screenshot", "image", "document"}
)
_GEMINI_MEDIA_PART_KEYS: Final = frozenset({"file_data", "fileData"})
_MEDIA_URI_KEYS: Final = frozenset({"url", "image_url", "file_id", "file_url", "file_data", "file_uri", "fileUri"})
_ENCODED_FILE_ID_PREFIXES: Final = ("file-", "batch_")
_EMBEDDED_S3_URI: Final = re.compile(r"s3://[^;]*")
_MODEL_FILE_ID_MAPPING_KEY: Final = "model_file_id_mapping"
_KEYS_THAT_MAY_HOLD_A_FILE_ID: Final = frozenset({"file_id", "file_url", "file_uri", "fileUri"})
_KEYS_THAT_MAY_HIDE_AN_S3_URI: Final = _KEYS_THAT_MAY_HOLD_A_FILE_ID | {_MODEL_FILE_ID_MAPPING_KEY}
_SCALAR_TYPES: Final = frozenset({int, float, bool, type(None)})
_OBJECT_TUPLE: Final = TypeAdapter(tuple[object, ...])


def is_s3_uri(value: str) -> bool:
    return value.startswith(_S3_SCHEME)


def _base64url_text(value: str) -> str | None:
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode()
    except ValueError:
        return None


def _s3_uris_in(text: str) -> Iterator[str]:
    return (match.group(0) for match in _EMBEDDED_S3_URI.finditer(text))


def _s3_uris_embedded_in_file_id(file_id: str) -> Iterator[str]:
    encodings: Final = dict.fromkeys((file_id, *(file_id.removeprefix(prefix) for prefix in _ENCODED_FILE_ID_PREFIXES)))
    return chain.from_iterable(map(_s3_uris_in, filter(None, map(_base64url_text, encodings))))


def _with_embedded_s3_uris(file_id: str) -> Iterator[str]:
    return chain((file_id,), _s3_uris_embedded_in_file_id(file_id))


def _non_scalar(values: Iterable[object]) -> Iterator[object]:
    return (value for value in values if type(value) not in _SCALAR_TYPES)


def _sequence_items(value: object) -> Sequence[object] | None:
    if isinstance(value, tuple):
        return _OBJECT_TUPLE.validate_python(value)
    return as_object_list(value)


@dataclass(frozen=True, slots=True)
class _BodyNode:
    value: object
    key: str | None = None
    inside_media_part: bool = False


def _is_media_part(fields: Mapping[str, object], key: str | None) -> bool:
    part_type: Final = fields.get("type")
    return (isinstance(part_type, str) and part_type in _S3_MEDIA_PART_TYPES) or key in _GEMINI_MEDIA_PART_KEYS


def _child_nodes(node: _BodyNode) -> Iterator[_BodyNode]:
    fields: Final = as_str_mapping(node.value)
    if fields is not None:
        inside_media_part: Final = node.inside_media_part or _is_media_part(fields, node.key)
        return (
            _BodyNode(child, child_key, inside_media_part)
            for child_key, child in fields.items()
            if type(child) not in _SCALAR_TYPES
        )
    items: Final = _sequence_items(node.value)
    if items is None:
        return iter(())
    return (_BodyNode(item, None, node.inside_media_part) for item in _non_scalar(items))


def _next_level(level: tuple[_BodyNode, ...], _: None) -> tuple[_BodyNode, ...]:
    return tuple(chain.from_iterable(map(_child_nodes, level)))


def _body_nodes(request_body: object) -> Iterator[_BodyNode]:
    levels: Final = accumulate(repeat(None), _next_level, initial=(_BodyNode(request_body),))
    return chain.from_iterable(takewhile(bool, levels))


def _string_leaf(node: _BodyNode) -> str | None:
    return node.value if isinstance(node.value, str) else None


def _media_file_id(node: _BodyNode) -> str | None:
    return _string_leaf(node) if node.inside_media_part and node.key in _KEYS_THAT_MAY_HOLD_A_FILE_ID else None


def _media_uri_candidates(trusted_file_ids: frozenset[str], node: _BodyNode) -> Iterator[str]:
    if not node.inside_media_part or not isinstance(node.value, str):
        return iter(())
    if node.key in _KEYS_THAT_MAY_HOLD_A_FILE_ID:
        return iter((node.value,)) if node.value in trusted_file_ids else _with_embedded_s3_uris(node.value)
    return iter((node.value,) if node.key in _MEDIA_URI_KEYS else ())


def _mapped_provider_file_ids(request_body: Mapping[str, object]) -> Iterator[str]:
    mapping_leaves: Final = filter(None, map(_string_leaf, _body_nodes(request_body.get(_MODEL_FILE_ID_MAPPING_KEY))))
    return chain.from_iterable(map(_with_embedded_s3_uris, mapping_leaves))


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


def _may_name_an_s3_uri(value: object) -> bool:
    if isinstance(value, str):
        return is_s3_uri(value)
    fields: Final = as_str_mapping(value)
    if fields is not None:
        return not _KEYS_THAT_MAY_HIDE_AN_S3_URI.isdisjoint(fields) or any(
            map(_may_name_an_s3_uri, _non_scalar(fields.values()))
        )
    items: Final = _sequence_items(value)
    return items is not None and any(map(_may_name_an_s3_uri, _non_scalar(items)))


def _may_carry_s3_media(request_body: Mapping[str, object]) -> bool:
    try:
        return _may_name_an_s3_uri(request_body)
    except RecursionError:
        return True


def media_file_ids(request_body: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(filter(None, map(_media_file_id, _body_nodes(request_body)))))


def file_id_embeds_s3_uri_matching(file_id: str, matches: Callable[[str], bool]) -> bool:
    return any(map(matches, _s3_uris_embedded_in_file_id(file_id)))


def s3_media_uris(
    request_body: Mapping[str, object], trusted_file_ids: frozenset[str] = frozenset()
) -> tuple[str, ...]:
    if not _may_carry_s3_media(request_body):
        return ()
    media_uris: Final = chain.from_iterable(
        map(partial(_media_uri_candidates, trusted_file_ids), _body_nodes(request_body))
    )
    return tuple(dict.fromkeys(filter(is_s3_uri, chain(media_uris, _mapped_provider_file_ids(request_body)))))


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
    value: Final = litellm_params.get("allow_legacy_cloud_file_ids") if litellm_params is not None else None
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
