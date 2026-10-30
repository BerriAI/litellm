import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias
from urllib.parse import quote

from pydantic import TypeAdapter

from litellm.exceptions import AuthenticationError, BadRequestError
from litellm.llms.bedrock.base_aws_llm import run_aws_signing
from litellm.types.vector_stores import VectorStoreIndexEndpoints

if TYPE_CHECKING:
    from litellm.llms.opensearch.vector_stores.transformation import OpenSearchVectorStoreConfig

IndexHttpMethod: TypeAlias = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
IndexPermission: TypeAlias = Literal["read", "write"]
BodyKind: TypeAlias = Literal["query", "msearch", "mget", "bulk", "unscanned"]

DEFAULT_CONTENT_TYPE: Final = "application/json"
NDJSON_CONTENT_TYPE: Final = "application/x-ndjson"
ACCEPTED_MEDIA_TYPES: Final = frozenset({DEFAULT_CONTENT_TYPE, NDJSON_CONTENT_TYPE})
_ACCEPTED_MEDIA_TYPE_PARAMETERS: Final = frozenset({"", "charset=utf-8"})
_INDEX_KEYS: Final = frozenset({"index", "_index", "indices"})
_OUT_OF_INDEX_KEYS: Final = _INDEX_KEYS | {"indexed_shape", "wrapper", "search_pipeline"}
_BULK_METADATA_KEYS: Final = _INDEX_KEYS | {"pipeline"}
_OUT_OF_INDEX_QUERY_PARAMETERS: Final = frozenset({"pipeline", "search_pipeline", "source", "source_content_type"})
_BULK_ACTIONS: Final = frozenset({"index", "create", "update", "delete"})
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

    @property
    def endpoint_pattern(self) -> str:
        return f"/{self.api}/" if self.takes_id else f"/{self.api}"

    def matches(self, method: str, api: tuple[str, ...]) -> bool:
        return method in self.methods and api[0] == self.api and len(api) == (2 if self.takes_id else 1)


@dataclass(frozen=True, slots=True)
class SignedIndexRequest:
    url: str
    media_type: str
    headers: Mapping[str, str]
    body: bytes


INDEX_OPERATIONS: Final = (
    IndexOperation("write", ("POST",), "_doc", False, "unscanned"),
    IndexOperation("write", ("PUT", "POST", "DELETE"), "_doc", True, "unscanned"),
    IndexOperation("write", ("PUT", "POST"), "_create", True, "unscanned"),
    IndexOperation("write", ("POST",), "_update", True, "unscanned"),
    IndexOperation("write", ("POST", "PUT"), "_bulk", False, "bulk"),
    IndexOperation("write", ("POST",), "_delete_by_query", False, "query"),
    IndexOperation("write", ("POST",), "_update_by_query", False, "query"),
    IndexOperation("read", ("GET", "POST"), "_search", False, "query"),
    IndexOperation("read", ("GET", "POST"), "_count", False, "query"),
    IndexOperation("read", ("GET",), "_doc", True, "unscanned"),
    IndexOperation("read", ("GET",), "_source", True, "unscanned"),
    IndexOperation("read", ("GET", "POST"), "_mget", False, "mget"),
    IndexOperation("read", ("GET", "POST"), "_msearch", False, "msearch"),
    IndexOperation("read", ("GET", "POST"), "_explain", True, "query"),
    IndexOperation("read", ("GET",), "_mapping", False, "unscanned"),
)


_NDJSON_APIS: Final = frozenset(
    operation.api for operation in INDEX_OPERATIONS if operation.body in ("bulk", "msearch")
)


def carries_ndjson_body(route: str, media_type: str) -> bool:
    return media_type == NDJSON_CONTENT_TYPE or route.rpartition("/")[2] in _NDJSON_APIS


def _endpoint_patterns(permission: IndexPermission) -> Iterator[tuple[IndexHttpMethod, str]]:
    for operation in INDEX_OPERATIONS:
        if operation.permission == permission:
            yield from zip(operation.methods, (operation.endpoint_pattern,) * len(operation.methods))


def index_endpoints() -> VectorStoreIndexEndpoints:
    return VectorStoreIndexEndpoints(read=list(_endpoint_patterns("read")), write=list(_endpoint_patterns("write")))


def index_operation(method: str, api: tuple[str, ...]) -> IndexOperation | None:
    return next((operation for operation in INDEX_OPERATIONS if operation.matches(method, api)), None)


def parse_index_path(endpoint: str) -> IndexPath | IndexRefusal:
    segments: Final = tuple(endpoint.split("/"))
    if any(segment in ("", ".", "..") for segment in segments):
        return IndexRefusal(400, f"The OpenSearch path '{endpoint}' has an empty, '.' or '..' segment. {_PATH_HELP}")
    if len(segments) < 2:
        return IndexRefusal(400, f"The OpenSearch path '{endpoint}' names no API after the index. {_PATH_HELP}")
    return IndexPath(index_name=segments[0], api=segments[1:])


@dataclass(frozen=True, slots=True)
class _JsonObject:
    pairs: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class _Json:
    value: object


@dataclass(frozen=True, slots=True)
class _BulkAction:
    metadata: _JsonObject
    has_source: bool


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


def _forbidden_key(level: Sequence[object], forbidden: frozenset[str]) -> str | None:
    if not level:
        return None
    keys: Final[Iterable[str]] = chain.from_iterable(map(_own_keys, level))
    key: Final = next((key for key in keys if key in forbidden), None)
    if key is not None:
        return key
    return _forbidden_key(tuple(chain.from_iterable(map(_children, level))), forbidden)


def _unparseable(api: str) -> IndexRefusal:
    return IndexRefusal(400, f"LiteLLM could not parse the {api} request body as JSON, so it did not send it.")


def _index_reference(value: object, forbidden: frozenset[str]) -> IndexRefusal | None:
    key: Final = _forbidden_key((value,), forbidden)
    if key is None:
        return None
    return IndexRefusal(
        400,
        f"The request body uses '{key}', which can reach beyond the OpenSearch index in the URL path. "
        f"Remove '{key}' from the body.",
    )


def _line_refusal(api: str, line: bytes, forbidden: frozenset[str]) -> IndexRefusal | None:
    parsed: Final = _parse(line)
    if parsed is None:
        return _unparseable(api)
    return _index_reference(parsed.value, forbidden)


def _whole_body_refusal(api: str, body: bytes) -> IndexRefusal | None:
    if not body.strip(_JSON_BLANK):
        return None
    return _line_refusal(api, body, _OUT_OF_INDEX_KEYS)


def _msearch_refusal(body: bytes) -> IndexRefusal | None:
    lines: Final = (line for line in body.split(b"\n") if line.strip(_JSON_BLANK))
    return next(filter(None, (_line_refusal("_msearch", line, _OUT_OF_INDEX_KEYS) for line in lines)), None)


def _bulk_action(line: bytes) -> _BulkAction | None:
    parsed: Final = _parse(line)
    if parsed is None:
        return None
    match parsed.value:
        case _JsonObject(pairs=((name, _JsonObject() as metadata),)) if name in _BULK_ACTIONS:
            return _BulkAction(metadata=metadata, has_source=name != "delete")
        case _:
            return None


def _bulk_actions(lines: Iterator[bytes]) -> Iterator[_BulkAction | None]:
    for line in lines:
        if not line.strip(_JSON_BLANK):
            continue
        if (action := _bulk_action(line)) is None:
            yield None
            return
        yield action
        if action.has_source:
            _ = next(lines, None)


def _bulk_action_refusal(action: _BulkAction | None) -> IndexRefusal | None:
    if action is None:
        return IndexRefusal(
            400,
            "A _bulk action line must be one JSON object naming index, create, update, or delete. "
            "LiteLLM did not send the request.",
        )
    return _index_reference(action.metadata, _BULK_METADATA_KEYS)


def _bulk_refusal(body: bytes) -> IndexRefusal | None:
    actions: Final = _bulk_actions(iter(body.split(b"\n")))
    return next(filter(None, (_bulk_action_refusal(action) for action in actions)), None)


def _body_refusal(operation: IndexOperation, body: bytes) -> IndexRefusal | None:
    match operation.body:
        case "query" | "mget":
            return _whole_body_refusal(operation.api, body)
        case "msearch":
            return _msearch_refusal(body)
        case "bulk":
            return _bulk_refusal(body)
        case "unscanned":
            return None


def accepted_media_type(content_type: str | None) -> str | None:
    media, _, parameters = (content_type or DEFAULT_CONTENT_TYPE).partition(";")
    normalized: Final = media.strip().lower()
    if normalized in ACCEPTED_MEDIA_TYPES and parameters.strip().lower() in _ACCEPTED_MEDIA_TYPE_PARAMETERS:
        return normalized
    return None


def request_refusal(
    operation: IndexOperation | None, query: Sequence[tuple[str, str]], body: bytes
) -> IndexRefusal | None:
    if operation is None:
        return None
    parameter: Final = next((key for key, _ in query if key in _OUT_OF_INDEX_QUERY_PARAMETERS), None)
    if parameter is not None:
        return IndexRefusal(
            400, f"The {parameter} query parameter can reach beyond the OpenSearch index in the URL path. Remove it."
        )
    try:
        return _body_refusal(operation, body)
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
) -> SignedIndexRequest | IndexRefusal:
    media: Final = accepted_media_type(content_type)
    if media is None:
        return IndexRefusal(
            400, "The OpenSearch pass-through takes application/json or application/x-ndjson bodies in UTF-8."
        )
    refusal: Final = request_refusal(operation, query, body)
    if refusal is not None:
        return refusal
    api_base: Final = store_params.get("api_base")
    try:
        url: Final = upstream_url(
            config.get_complete_url(api_base if isinstance(api_base, str) else None, store_params),
            provider_index,
            path.api,
            query,
        )
        signed: Final = await run_aws_signing(config.sign_http_request, method, url, body, media, store_params)
    except (AuthenticationError, BadRequestError) as error:
        return IndexRefusal(error.status_code, error.message)
    return SignedIndexRequest(
        url=url, media_type=media, headers=MappingProxyType(dict(signed.headers.items())), body=body
    )
