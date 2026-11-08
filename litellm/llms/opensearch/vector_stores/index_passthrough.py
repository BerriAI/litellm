import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, chain, repeat, takewhile
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias
from urllib.parse import quote

from pydantic import TypeAdapter

from litellm.exceptions import AuthenticationError, BadRequestError, InternalServerError
from litellm.llms.bedrock.base_aws_llm import run_aws_signing

if TYPE_CHECKING:
    from litellm.llms.opensearch.vector_stores.transformation import OpenSearchVectorStoreConfig

IndexHttpMethod: TypeAlias = Literal["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"]
IndexPermission: TypeAlias = Literal["read", "write"]
BodyKind: TypeAlias = Literal["query", "mget", "bulk", "update", "unscanned"]

DEFAULT_CONTENT_TYPE: Final = "application/json"
NDJSON_CONTENT_TYPE: Final = "application/x-ndjson"
ACCEPTED_MEDIA_TYPES: Final = frozenset({DEFAULT_CONTENT_TYPE, NDJSON_CONTENT_TYPE})
_ACCEPTED_MEDIA_TYPE_PARAMETERS: Final = frozenset({"", "charset=utf-8"})
_INDEX_KEYS: Final = frozenset({"index", "_index", "indices"})
_OUT_OF_INDEX_KEYS: Final = _INDEX_KEYS | {"indexed_shape", "wrapper", "search_pipeline"}
_BULK_METADATA_KEYS: Final = _INDEX_KEYS | {"pipeline"}
_OUT_OF_INDEX_QUERY_PARAMETERS: Final = frozenset({"pipeline", "source", "source_content_type"})
_BULK_ACTIONS: Final = frozenset({"index", "create", "update", "delete"})
_DOCUMENT_READING_KEYS: Final = frozenset({"_source", "script"})
_JSON_BLANK: Final = b" \t\r\n"
_PATH_HELP: Final = "Use /opensearch/<index>/<api>, for example /opensearch/<index>/_search."


@dataclass(frozen=True, slots=True)
class IndexRefusal:
    status_code: int
    message: str


@dataclass(frozen=True, slots=True)
class IndexPath:
    index_name: str
    api: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IndexOperation:
    permission: IndexPermission
    methods: tuple[IndexHttpMethod, ...]
    api: str
    takes_id: bool
    body: BodyKind

    def matches(self, method: str, api: tuple[str, ...]) -> bool:
        named: Final = (self.api,) if self.api else ()
        return method in self.methods and api[: len(named)] == named and len(api) == len(named) + self.takes_id


@dataclass(frozen=True, slots=True)
class SignedIndexRequest:
    url: str
    media_type: str
    headers: Mapping[str, str]
    body: bytes


INDEX_OPERATIONS: Final = (
    IndexOperation("read", ("GET", "HEAD"), "", False, "unscanned"),
    IndexOperation("write", ("POST",), "_doc", False, "unscanned"),
    IndexOperation("write", ("PUT", "POST", "DELETE"), "_doc", True, "unscanned"),
    IndexOperation("write", ("PUT", "POST"), "_create", True, "unscanned"),
    IndexOperation("write", ("POST",), "_update", True, "update"),
    IndexOperation("write", ("POST", "PUT"), "_bulk", False, "bulk"),
    IndexOperation("write", ("POST",), "_delete_by_query", False, "query"),
    IndexOperation("write", ("POST",), "_update_by_query", False, "query"),
    IndexOperation("read", ("GET", "POST"), "_search", False, "query"),
    IndexOperation("read", ("GET", "POST"), "_count", False, "query"),
    IndexOperation("read", ("GET", "HEAD"), "_doc", True, "unscanned"),
    IndexOperation("read", ("GET", "HEAD"), "_source", True, "unscanned"),
    IndexOperation("read", ("GET", "POST"), "_mget", False, "mget"),
    IndexOperation("read", ("GET", "POST"), "_explain", True, "query"),
    IndexOperation("read", ("GET",), "_mapping", False, "unscanned"),
)


def index_operation(method: str, api: tuple[str, ...]) -> IndexOperation | None:
    return next((operation for operation in INDEX_OPERATIONS if operation.matches(method, api)), None)


def parse_index_path(endpoint: str) -> IndexPath | IndexRefusal:
    segments: Final = tuple(endpoint.split("/"))
    if any(segment in ("", ".", "..") for segment in segments):
        return IndexRefusal(400, f"The OpenSearch path '{endpoint}' has an empty, '.' or '..' segment. {_PATH_HELP}")
    return IndexPath(index_name=segments[0], api=segments[1:])


@dataclass(frozen=True, slots=True)
class _JsonObject:
    pairs: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class _Json:
    value: object


@dataclass(frozen=True, slots=True)
class _BulkAction:
    name: str
    metadata: _JsonObject
    has_source: bool


@dataclass(frozen=True, slots=True)
class _IndexScope:
    name: str
    provider_name: str
    can_read: bool


def _json_object(pairs: Sequence[tuple[str, object]]) -> _JsonObject:
    return _JsonObject(tuple(pairs))


_DECODER: Final = json.JSONDecoder(object_pairs_hook=_json_object)
_JSON_ARRAY: Final = TypeAdapter(list[object])
_JSON_VALUE: Final = TypeAdapter(object)


def _parse(raw: bytes) -> _Json | None:
    try:
        return _Json(_JSON_VALUE.validate_python(_DECODER.decode(raw.decode("utf-8"))))
    except ValueError:
        return None


def _own_keys(value: object) -> tuple[str, ...]:
    return tuple(key for key, _ in value.pairs) if isinstance(value, _JsonObject) else ()


def _children(value: object) -> Sequence[object]:
    if isinstance(value, _JsonObject):
        return tuple(child for _, child in value.pairs)
    if isinstance(value, list):
        return _JSON_ARRAY.validate_python(value)
    return ()


def _next_level(level: Sequence[object], _: None) -> Sequence[object]:
    return tuple(chain.from_iterable(map(_children, level)))


def _forbidden_key(value: object, forbidden: frozenset[str]) -> str | None:
    levels: Final = takewhile(bool, accumulate(repeat(None), _next_level, initial=(value,)))
    keys: Final[Iterable[str]] = chain.from_iterable(map(_own_keys, chain.from_iterable(levels)))
    return next((key for key in keys if key in forbidden), None)


def _plain(value: _JsonObject) -> Mapping[str, object]:
    return dict(value.pairs)


def _object_json(pairs: Sequence[tuple[str, object]]) -> str:
    fields: Final = ",".join(
        f"{json.dumps(key)}:{json.dumps(child, default=_plain, separators=(',', ':'))}" for key, child in pairs
    )
    return f"{{{fields}}}"


def _unparseable(api: str) -> IndexRefusal:
    return IndexRefusal(400, f"LiteLLM could not parse the {api} request body as JSON, so it did not send it.")


def _index_reference(value: object, forbidden: frozenset[str]) -> IndexRefusal | None:
    key: Final = _forbidden_key(value, forbidden)
    if key is None:
        return None
    return IndexRefusal(
        400,
        f"The request body uses '{key}', which can reach beyond the OpenSearch index in the URL path. "
        f"Remove '{key}' from the body.",
    )


def _read_needed(cause: str, index_name: str) -> IndexRefusal:
    return IndexRefusal(
        403,
        f"{cause}, so the key needs read on index '{index_name}' as well as write. "
        "Ask your administrator to add read to the key or its team.",
    )


def _document_reading_payload(api: str, payload: bytes, index: _IndexScope) -> IndexRefusal | None:
    if index.can_read:
        return None
    parsed: Final = _parse(payload)
    if parsed is None:
        return _unparseable(api)
    key: Final = next((key for key in _own_keys(parsed.value) if key in _DOCUMENT_READING_KEYS), None)
    return None if key is None else _read_needed(f"'{key}' in an {api} body reads stored documents", index.name)


def _line_refusal(api: str, line: bytes, forbidden: frozenset[str]) -> IndexRefusal | None:
    parsed: Final = _parse(line)
    if parsed is None:
        return _unparseable(api)
    return _index_reference(parsed.value, forbidden)


def _whole_body_refusal(api: str, body: bytes) -> IndexRefusal | None:
    if not body.strip(_JSON_BLANK):
        return None
    return _line_refusal(api, body, _OUT_OF_INDEX_KEYS)


def _pinned_index(metadata: _JsonObject, index: _IndexScope) -> _JsonObject | IndexRefusal | None:
    unnamed: Final = _JsonObject(tuple(pair for pair in metadata.pairs if pair[0] != "_index"))
    refusal: Final = _index_reference(unnamed, _BULK_METADATA_KEYS)
    if refusal is not None:
        return refusal
    named: Final = tuple(value for name, value in metadata.pairs if name == "_index")
    if not named:
        return None
    if any(value != index.name for value in named):
        return IndexRefusal(
            400,
            f"A _bulk action line sets '_index' to another index than '{index.name}' in the URL path. "
            f"Remove '_index' or set it to '{index.name}'.",
        )
    return _JsonObject(
        tuple((name, index.provider_name if name == "_index" else value) for name, value in metadata.pairs)
    )


def _bulk_action(line: bytes) -> _BulkAction | None:
    parsed: Final = _parse(line)
    if parsed is None:
        return None
    match parsed.value:
        case _JsonObject(pairs=((name, _JsonObject() as metadata),)) if name in _BULK_ACTIONS:
            return _BulkAction(name=name, metadata=metadata, has_source=name != "delete")
        case _:
            return None


def _bulk_action_line(action: _BulkAction | None, line: bytes, index: _IndexScope) -> bytes | IndexRefusal:
    if action is None:
        return IndexRefusal(
            400,
            "A _bulk action line must be one JSON object naming index, create, update, or delete. "
            "LiteLLM did not send the request.",
        )
    if not index.can_read and "_source" in _own_keys(action.metadata):
        return _read_needed("'_source' in a _bulk action line returns stored documents", index.name)
    pinned: Final = _pinned_index(action.metadata, index)
    if pinned is None:
        return line
    if isinstance(pinned, IndexRefusal):
        return pinned
    return f"{{{json.dumps(action.name)}:{_object_json(pinned.pairs)}}}".encode()


def _bulk_entry(line: bytes, lines: Iterator[bytes], index: _IndexScope) -> tuple[bytes | IndexRefusal, ...]:
    if not line.strip(_JSON_BLANK):
        return (line,)
    action: Final = _bulk_action(line)
    rendered: Final = _bulk_action_line(action, line, index)
    if isinstance(rendered, IndexRefusal) or action is None or not action.has_source:
        return (rendered,)
    source: Final = next(lines, None)
    if source is None:
        return (rendered,)
    if action.name != "update":
        return (rendered, source)
    return (rendered, _document_reading_payload("_bulk update", source, index) or source)


def _bulk_lines(lines: Iterator[bytes], index: _IndexScope) -> Iterator[bytes | IndexRefusal]:
    for line in lines:
        yield from (entry := _bulk_entry(line, lines, index))
        if any(isinstance(rendered, IndexRefusal) for rendered in entry):
            return


def _rendered_body(lines: Iterator[bytes | IndexRefusal]) -> bytes | IndexRefusal:
    rendered: Final = tuple(lines)
    refusal: Final = next((line for line in rendered if isinstance(line, IndexRefusal)), None)
    if refusal is not None:
        return refusal
    return b"\n".join(line for line in rendered if isinstance(line, bytes))


def _checked_body(operation: IndexOperation, body: bytes, index: _IndexScope) -> bytes | IndexRefusal:
    match operation.body:
        case "query" if operation.permission == "write" and not index.can_read:
            return _read_needed(f"{operation.api} searches stored documents", index.name)
        case "query" | "mget":
            return _whole_body_refusal(operation.api, body) or body
        case "update":
            return _document_reading_payload(operation.api, body, index) or body
        case "bulk":
            return _rendered_body(_bulk_lines(iter(body.split(b"\n")), index))
        case "unscanned":
            return body


def accepted_media_type(content_type: str | None) -> str | None:
    media, _, parameters = (content_type or DEFAULT_CONTENT_TYPE).partition(";")
    normalized: Final = media.strip().lower()
    if normalized in ACCEPTED_MEDIA_TYPES and parameters.strip().lower() in _ACCEPTED_MEDIA_TYPE_PARAMETERS:
        return normalized
    return None


def _checked_request_body(
    operation: IndexOperation | None,
    query: Sequence[tuple[str, str]],
    body: bytes,
    index: _IndexScope,
) -> bytes | IndexRefusal:
    if operation is None:
        return body
    parameter: Final = next((key for key, _ in query if key in _OUT_OF_INDEX_QUERY_PARAMETERS), None)
    if parameter is not None:
        return IndexRefusal(
            400, f"The {parameter} query parameter can reach beyond the OpenSearch index in the URL path. Remove it."
        )
    source_parameter: Final = next((key for key, _ in query if key.startswith("_source")), None)
    if operation.permission == "write" and not index.can_read and source_parameter is not None:
        return _read_needed(f"The {source_parameter} query parameter returns stored documents", index.name)
    try:
        return _checked_body(operation, body, index)
    except RecursionError:
        return IndexRefusal(400, f"The {operation.api} request body is nested too deeply for LiteLLM to check.")


def upstream_url(api_base: str, provider_index: str, api: tuple[str, ...], query: Sequence[tuple[str, str]]) -> str:
    path: Final = "/".join(quote(segment, safe="") for segment in (provider_index, *api))
    query_string: Final = "&".join(f"{quote(key, safe='')}={quote(value, safe='')}" for key, value in query)
    return f"{api_base}/{path}?{query_string}" if query_string else f"{api_base}/{path}"


async def prepare_index_request(
    config: "OpenSearchVectorStoreConfig",
    method: str,
    path: IndexPath,
    operation: IndexOperation | None,
    provider_index: str,
    content_type: str | None,
    query: Sequence[tuple[str, str]],
    body: bytes,
    store_params: Mapping[str, object],
    can_read: bool,
) -> SignedIndexRequest | IndexRefusal:
    media: Final = accepted_media_type(content_type)
    if media is None:
        return IndexRefusal(
            400, "The OpenSearch pass-through takes application/json or application/x-ndjson bodies in UTF-8."
        )
    checked: Final = _checked_request_body(
        operation, query, body, _IndexScope(name=path.index_name, provider_name=provider_index, can_read=can_read)
    )
    if isinstance(checked, IndexRefusal):
        return checked
    api_base: Final = store_params.get("api_base")
    try:
        url: Final = upstream_url(
            config.get_complete_url(api_base if isinstance(api_base, str) else None, store_params),
            provider_index,
            path.api,
            query,
        )
        signed: Final = await run_aws_signing(config.sign_http_request, method, url, checked, media, store_params)
    except (AuthenticationError, BadRequestError, InternalServerError) as error:
        return IndexRefusal(error.status_code, error.message)
    return SignedIndexRequest(
        url=url, media_type=media, headers=MappingProxyType(dict(signed.headers.items())), body=checked
    )
