import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial, reduce
from itertools import accumulate
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Literal, NoReturn, TypeAlias
from urllib.parse import quote, urlencode, urlsplit

import httpx
from pydantic import ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.exceptions import AuthenticationError, BadRequestError, InternalServerError
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params
from litellm.litellm_core_utils.optional_imports import ensure_optional_import
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.vector_store.transformation import (
    BaseQueryEmbeddingVectorStoreConfig,
    VectorStoreEmbeddingExecutor,
)
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, SignsRequestsWithAWS, run_off_loop_if_aws, sign_aws_request
from litellm.llms.bedrock.web_identity_session_policy import (
    OPENSEARCH_INDEX_SESSION_POLICY,
    OPENSEARCH_SESSION_POLICY,
    WebIdentitySessionPolicy,
)
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.bedrock import AwsAuthParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import EmbeddingResponse
from litellm.types.vector_stores import (
    BaseVectorStoreAuthCredentials,
    VectorStoreCreateOptionalRequestParams,
    VectorStoreIndexEndpoints,
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from botocore.awsrequest import AWSPreparedRequest
    from botocore.credentials import Credentials

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

DEFAULT_EMBEDDING_FIELD: Final = "embedding"
DEFAULT_TEXT_FIELD: Final = "text"
DEFAULT_MAX_NUM_RESULTS: Final = 10
MAX_MAX_NUM_RESULTS: Final = 50
_PARAM_PREFIX: Final = "opensearch_"
_AWS_HOSTS: Final = (
    re.compile(r"\.(?P<region>[a-z0-9-]+)\.(?P<service>aoss|aoss-fips|es)\.amazonaws\.com(?:\.cn)?\Z"),
    re.compile(r"\.(?P<service>aoss|aos)\.(?P<region>[a-z0-9-]+)\.on\.aws\Z"),
)
_ACCOUNT_ID_LABEL: Final = re.compile(r"\A\d{12}\.")
_UNFILTERABLE_KNN: Final = ("does not support filters", "unknown token [START_OBJECT] after [filter]")
_ANALYZED_STRING_TYPES: Final = frozenset({"text", "match_only_text", "search_as_you_type"})
_STRING_TYPES: Final = _ANALYZED_STRING_TYPES | {"keyword", "wildcard", "constant_keyword"}
_CONTAINER_TYPES: Final = frozenset({"object", "nested"})
_FORBIDDEN: Final = 403
_QUERY_VECTOR: Final = TypeAdapter(tuple[float, ...])
_ATTRIBUTE_VALUE_TYPES: Final = (str, int, float, bool)
_SEARCH_ONLY_MESSAGE: Final = (
    "The OpenSearch vector store is search-only. Create the k-NN index in OpenSearch, "
    "then register it here with its name as opensearch_index_name or vector_store_id."
)
_PER_ACCOUNT_MESSAGE: Final = (
    "api_base is an OpenSearch Serverless per-account endpoint, which needs a collection header the vector store "
    "does not send. Use the collection endpoint instead, for example https://<collection-id>.<region>.aoss.amazonaws.com."
)
_SIGV4_ONLY_MESSAGE: Final = (
    "The OpenSearch vector store signs every request with AWS SigV4 and does not support api_key. "
    "Remove api_key and give the vector store AWS credentials instead."
)
_CREDENTIALS_HINT: Final = (
    "Set aws_access_key_id and aws_secret_access_key, aws_role_name, or aws_profile_name on the vector store, "
    "or run LiteLLM with an AWS identity."
)


def _bad_request(message: str) -> BadRequestError:
    return BadRequestError(message=message, model=None, llm_provider="opensearch")


def _with_signed_headers(headers: Mapping[str, object], signed: Mapping[str, str]) -> Mapping[str, object]:
    signed_names: Final = frozenset(name.lower() for name in signed)
    return {**{name: value for name, value in headers.items() if name.lower() not in signed_names}, **signed}


def _unauthenticated(reason: str) -> AuthenticationError:
    return AuthenticationError(
        message=f"No usable AWS credentials to sign the OpenSearch request ({reason}). {_CREDENTIALS_HINT}",
        llm_provider="opensearch",
        model=None,
    )


def _unavailable_credentials(error: Exception) -> InternalServerError:
    verbose_logger.warning("OpenSearch vector store AWS credentials could not be loaded or refreshed: %s", error)
    return InternalServerError(
        message="LiteLLM could not load or refresh the AWS credentials that sign the OpenSearch request, "
        "see the LiteLLM logs.",
        llm_provider="opensearch",
        model=None,
    )


def _usable_credentials(credentials: object) -> "Credentials":
    from botocore.credentials import Credentials

    if isinstance(credentials, Credentials):
        return credentials
    raise _unauthenticated("none were found")


class _OpenSearchParams(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    litellm_embedding_model: str | None = None
    opensearch_embedding_field: str | None = None
    opensearch_hybrid_search: bool | None = None
    opensearch_index_name: str | None = None
    opensearch_search_pipeline: str | None = None
    opensearch_text_field: str | None = None

    @property
    def embedding_field(self) -> str:
        return self.opensearch_embedding_field or DEFAULT_EMBEDDING_FIELD

    @property
    def text_field(self) -> str:
        return self.opensearch_text_field or DEFAULT_TEXT_FIELD

    def index_name(self, vector_store_id: str) -> str:
        return self.opensearch_index_name or vector_store_id

    def require_embedding_model(self) -> str:
        if not self.litellm_embedding_model:
            raise _bad_request(
                "litellm_embedding_model is required for the OpenSearch vector store. It must be the model that "
                f"produced the vectors in '{self.embedding_field}', for example openai/text-embedding-3-small."
            )
        return self.litellm_embedding_model


class _SigningRegion(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    aws_region_name: str | None = None


_FiniteFloat: TypeAlias = Annotated[StrictFloat, Field(allow_inf_nan=False)]


class _Equality(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["eq", "ne"]
    key: str
    value: StrictStr | StrictInt | _FiniteFloat | StrictBool


class _Range(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["gt", "gte", "lt", "lte"]
    key: str
    value: StrictStr | StrictInt | _FiniteFloat


class _Membership(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["in", "nin"]
    key: str
    value: tuple[StrictStr | StrictInt | _FiniteFloat, ...]


class _Compound(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["and", "or"]
    filters: "tuple[_Filter, ...]"


_Filter: TypeAlias = Annotated[_Equality | _Range | _Membership | _Compound, Field(discriminator="type")]
_Compound.model_rebuild()
_FILTER: Final = TypeAdapter[_Equality | _Range | _Membership | _Compound](_Filter)


class _FieldMapping(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    type: str = "object"
    ignore_above: int | None = None
    normalizer: str | None = None
    path: str | None = None
    include_in_parent: bool = False
    include_in_root: bool = False
    properties: "Mapping[str, _FieldMapping]" = Field(default_factory=dict)
    fields: "Mapping[str, _FieldMapping]" = Field(default_factory=dict)


class _IndexMapping(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    mappings: _FieldMapping


_INDEX_MAPPINGS: Final = TypeAdapter(Mapping[str, _IndexMapping])


@dataclass(frozen=True, slots=True)
class _ExactField:
    name: str
    ignore_above: int | None


@dataclass(frozen=True, slots=True)
class _InexactKey:
    key: str
    reason: str
    remedy: str | None = None


def _child_field(field: _FieldMapping | None, name: str) -> _FieldMapping | None:
    if field is None:
        return None
    return field.properties.get(name) or field.fields.get(name)


def _mapped_field(mapping: _FieldMapping, key: str) -> _FieldMapping | None:
    root: Final[_FieldMapping | None] = mapping
    return reduce(_child_field, key.split("."), root)


def _resolved_field(mapping: _FieldMapping, key: str) -> tuple[str, _FieldMapping] | None:
    field: Final = _mapped_field(mapping, key)
    if field is None:
        return None
    if field.type != "alias" or field.path is None:
        return key, field
    target: Final = _mapped_field(mapping, field.path)
    return None if target is None else (field.path, target)


def _nested_segments(mapping: _FieldMapping, path: str) -> tuple[tuple[str, _FieldMapping], ...]:
    parts: Final = path.split(".")
    prefixes: Final = (".".join(parts[:end]) for end in range(1, len(parts) + 1))
    segments: Final = ((prefix, _mapped_field(mapping, prefix)) for prefix in prefixes)
    return tuple((prefix, field) for prefix, field in segments if field is not None and field.type == "nested")


def _copied_to_root(parent_copied_to_root: bool, nested: _FieldMapping) -> bool:
    return nested.include_in_root or (nested.include_in_parent and parent_copied_to_root)


def _nested_path(mapping: _FieldMapping, path: str) -> str | None:
    segments: Final = _nested_segments(mapping, path)
    copied: Final = tuple(accumulate((field for _, field in segments), _copied_to_root, initial=True))[1:]
    if not segments or copied[-1]:
        return None
    return next(prefix for (prefix, _), reaches in zip(segments, copied, strict=True) if not reaches)


def _exact_field(
    key: str, field: _FieldMapping, comparison: _Equality | _Range | _Membership
) -> _ExactField | _InexactKey:
    if isinstance(comparison, _Range) and not isinstance(comparison.value, str) and field.type in _STRING_TYPES:
        return _InexactKey(
            key=key,
            reason=f"the index maps {key} as {field.type}, which compares numbers as text",
            remedy=f"Map {key} as a numeric field",
        )
    if field.type not in _ANALYZED_STRING_TYPES and field.normalizer is None:
        return _ExactField(name=key, ignore_above=field.ignore_above)
    keywords: Final = sorted(
        (name != "keyword", name, subfield.ignore_above)
        for name, subfield in field.fields.items()
        if subfield.type == "keyword" and subfield.normalizer is None
    )
    if keywords:
        _, name, ignore_above = keywords[0]
        return _ExactField(name=f"{key}.{name}", ignore_above=ignore_above)
    if field.normalizer is not None or any(subfield.type == "keyword" for subfield in field.fields.values()):
        return _InexactKey(key=key, reason=f"the index normalizes every keyword mapping of {key}")
    return _InexactKey(key=key, reason=f"the index maps {key} as {field.type} with no keyword subfield")


def _string_values(comparison: _Equality | _Range | _Membership) -> tuple[str, ...]:
    values: Final = comparison.value if isinstance(comparison, _Membership) else (comparison.value,)
    return tuple(value for value in values if isinstance(value, str))


def _queried_field(
    key: str, mapping: _FieldMapping, comparison: _Equality | _Range | _Membership
) -> _ExactField | _InexactKey | None:
    resolved: Final = _resolved_field(mapping, key)
    if resolved is None:
        return None
    path, field = resolved
    if field.type in _CONTAINER_TYPES:
        return _InexactKey(
            key=key,
            reason=f"the index maps {path} as {field.type}, which holds fields rather than values",
            remedy=f"Filter on a field inside {path}",
        )
    nested: Final = _nested_path(mapping, path)
    if nested is not None:
        return _InexactKey(
            key=key,
            reason=f"the index maps {nested} as nested, which needs a nested query",
            remedy=f"Map {nested} as object",
        )
    exact: Final = _exact_field(path, field, comparison)
    if isinstance(exact, _ExactField) and exact.name == path:
        return _ExactField(name=key, ignore_above=exact.ignore_above)
    return exact


def _exact_comparison(
    comparison: _Equality | _Range | _Membership, mappings: Mapping[str, _IndexMapping]
) -> _Equality | _Range | _Membership | _InexactKey:
    key: Final = comparison.key
    queried: Final = tuple(_queried_field(key, index.mappings, comparison) for index in mappings.values())
    exact: Final = tuple(field for field in queried if field is not None)
    differently: Final = _InexactKey(
        key=key,
        reason=f"the indexes behind this store map {key} differently",
        remedy=f"Map {key} to the same keyword field in every index behind the store",
    )
    if 0 < len(exact) < len(queried):
        return differently
    inexact: Final = next((field for field in exact if isinstance(field, _InexactKey)), None)
    if inexact is not None:
        return inexact
    fields: Final = tuple(field for field in exact if isinstance(field, _ExactField))
    names: Final = frozenset(field.name for field in fields)
    name: Final = next(iter(names), key)
    if len(names) > 1:
        return differently
    ignore_above: Final = min((field.ignore_above for field in fields if field.ignore_above), default=None)
    if ignore_above is not None and any(len(value) > ignore_above for value in _string_values(comparison)):
        return _InexactKey(key=key, reason=f"the index ignores {key} values longer than {ignore_above} characters")
    return comparison.model_copy(update={"key": name})


def _read_index_mappings(mapping: httpx.Response) -> Mapping[str, _IndexMapping] | BaseLLMException:
    if mapping.status_code == _FORBIDDEN:
        return BaseLLMException(
            status_code=mapping.status_code,
            message=(
                "OpenSearch refused the index mapping read that filters need. That read needs aoss:DescribeIndex on "
                "Serverless or es:ESHttpGet on a domain. Set drop_params to drop the filters instead."
            ),
            headers=mapping.headers,
        )
    if not mapping.is_success:
        return BaseLLMException(
            status_code=mapping.status_code,
            message=f"OpenSearch answered {mapping.status_code} to the index mapping read that filters need.",
            headers=mapping.headers,
        )
    try:
        return _INDEX_MAPPINGS.validate_json(mapping.content)
    except ValidationError:
        return BaseLLMException(
            status_code=502,
            message="OpenSearch answered the index mapping read without index mappings.",
            headers=mapping.headers,
        )


def _exact_filter(search_filter: _Filter, mappings: Mapping[str, _IndexMapping]) -> _Filter | _InexactKey:
    match search_filter:
        case _Compound(filters=filters):
            children: Final = tuple(_exact_filter(child, mappings) for child in filters)
            inexact: Final = next((child for child in children if isinstance(child, _InexactKey)), None)
            return inexact or search_filter.model_copy(update={"filters": children})
        case _:
            return _exact_comparison(search_filter, mappings)


def _opensearch_filter(search_filter: _Filter) -> Mapping[str, object]:
    match search_filter:
        case _Compound(type="and", filters=filters):
            return {"bool": {"filter": tuple(_opensearch_filter(child) for child in filters)}}
        case _Compound(filters=filters):
            return {
                "bool": {"should": tuple(_opensearch_filter(child) for child in filters), "minimum_should_match": 1}
            }
        case _Equality(type="eq", key=key, value=value):
            return {"term": {key: value}}
        case _Equality(key=key, value=value):
            return {"bool": {"must_not": ({"term": {key: value}},)}}
        case _Membership(type="in", key=key, value=values):
            return {"terms": {key: values}}
        case _Membership(key=key, value=values):
            return {"bool": {"must_not": ({"terms": {key: values}},)}}
        case _Range(type=operator, key=key, value=value):
            return {"range": {key: {operator: value}}}


class _SearchOptions(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    max_num_results: int | None = None
    filters: Mapping[str, object] | None = None
    ranking_options: Mapping[str, object] | None = None
    rewrite_query: bool | None = None

    @property
    def unsupported(self) -> tuple[str, ...]:
        requested: Final = (
            ("ranking_options", self.ranking_options),
            ("rewrite_query", self.rewrite_query),
        )
        return tuple(name for name, value in requested if value)

    def search_filter(self, drop_unsupported: bool) -> _Filter | None:
        if self.filters is None or self.filters == {}:
            return None
        try:
            return _FILTER.validate_python(self.filters)
        except ValidationError:
            if drop_unsupported:
                return None
            raise _bad_request(
                "filters must be an OpenAI vector store filter, either a comparison with a key, a value, and type eq, "
                'ne, gt, gte, lt, lte, in, or nin, or a compound with a filters list and type "and" or "or". Fix it, '
                "or set drop_params to drop it."
            ) from None

    @property
    def k(self) -> int:
        if self.max_num_results is None:
            return DEFAULT_MAX_NUM_RESULTS
        if not 1 <= self.max_num_results <= MAX_MAX_NUM_RESULTS:
            raise _bad_request(
                f"max_num_results must be between 1 and {MAX_MAX_NUM_RESULTS}, got {self.max_num_results}"
            )
        return self.max_num_results


_KNOWN_PARAMS: Final = frozenset(name for name in _OpenSearchParams.model_fields if name.startswith(_PARAM_PREFIX))


@dataclass(frozen=True, slots=True)
class _Search:
    params: _OpenSearchParams
    k: int
    search_filter: _Filter | None

    @property
    def hybrid(self) -> bool:
        return bool(self.params.opensearch_hybrid_search and self.params.opensearch_search_pipeline)

    @property
    def opensearch_filter(self) -> Mapping[str, object] | None:
        return None if self.search_filter is None else _opensearch_filter(self.search_filter)

    @property
    def path(self) -> str:
        pipeline: Final = self.params.opensearch_search_pipeline
        return f"/_search?{urlencode({'search_pipeline': pipeline}, quote_via=quote)}" if pipeline else "/_search"

    def query(self, query_text: str, vector: tuple[float, ...]) -> Mapping[str, object]:
        opensearch_filter: Final = self.opensearch_filter
        knn: Final[Mapping[str, object]] = {"vector": vector, "k": self.k}
        knn_query: Final[Mapping[str, object]] = {
            "knn": {
                self.params.embedding_field: knn if opensearch_filter is None else {**knn, "filter": opensearch_filter}
            }
        }
        if not self.hybrid:
            return knn_query
        keyword_query: Final[Mapping[str, object]] = {"match": {self.params.text_field: {"query": query_text}}}
        return {
            "hybrid": {
                "queries": (
                    keyword_query
                    if opensearch_filter is None
                    else {"bool": {"must": (keyword_query,), "filter": (opensearch_filter,)}},
                    knn_query,
                )
            }
        }


class _LoggedSearch(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    query: str | tuple[str, ...] = ""
    litellm_params: _OpenSearchParams = _OpenSearchParams()

    @property
    def search_query(self) -> str:
        return BaseQueryEmbeddingVectorStoreConfig.query_text(self.query)


class _Hit(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(alias="_id")
    score: float | None = Field(default=None, alias="_score")
    source: Mapping[str, object] = Field(default_factory=dict, alias="_source")


class _Hits(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    hits: Sequence[_Hit]


class _SearchResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    hits: _Hits


def aws_signing_target(url: str) -> tuple[Literal["aoss", "es"], str | None]:
    hostname: Final = urlsplit(url).hostname or ""
    match: Final = next(filter(None, (pattern.search(hostname) for pattern in _AWS_HOSTS)), None)
    if match is None:
        return "es", None
    return ("aoss" if match["service"].startswith("aoss") else "es"), match["region"]


def _search_result(hit: _Hit, params: _OpenSearchParams) -> VectorStoreSearchResult:
    text: Final = hit.source.get(params.text_field)
    filename: Final = hit.source.get("filename")
    return VectorStoreSearchResult(
        score=hit.score,
        content=[VectorStoreResultContent(type="text", text=text)] if isinstance(text, str) else [],
        file_id=hit.id,
        filename=filename if isinstance(filename, str) else None,
        attributes={
            key: value
            for key, value in hit.source.items()
            if key not in (params.text_field, params.embedding_field) and isinstance(value, _ATTRIBUTE_VALUE_TYPES)
        },
    )


class _OpenSearchAWS(BaseAWSLLM):
    _web_identity_session_policy: ClassVar[WebIdentitySessionPolicy] = OPENSEARCH_SESSION_POLICY


class OpenSearchIndexAWS(BaseAWSLLM):
    _web_identity_session_policy: ClassVar[WebIdentitySessionPolicy] = OPENSEARCH_INDEX_SESSION_POLICY


class OpenSearchVectorStoreConfig(BaseQueryEmbeddingVectorStoreConfig, SignsRequestsWithAWS):
    def __init__(self, aws: BaseAWSLLM | None = None) -> None:
        self._aws: Final = aws if aws is not None else _OpenSearchAWS()

    def get_auth_credentials(self, litellm_params: Mapping[str, object]) -> BaseVectorStoreAuthCredentials:
        return BaseVectorStoreAuthCredentials()

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return VectorStoreIndexEndpoints(read=[], write=[])

    def validate_environment(
        self, headers: Mapping[str, object], litellm_params: GenericLiteLLMParams | None
    ) -> dict[str, object]:  # mutable-ok: the shared HTTP handler requires writable headers
        params: Final = dict(litellm_params or {})
        self._signing_credentials(params, self._signing_region(params, str(params.get("api_base") or "")))
        return {**headers, "Content-Type": "application/json"}

    def _signing_region(self, params: Mapping[str, object], api_base: str) -> str:
        host_region: Final = aws_signing_target(api_base)[1]
        return self._aws.get_aws_region_name_for_non_llm_api_calls(
            host_region or _SigningRegion.model_validate(params).aws_region_name
        )

    def _signing_credentials(self, params: Mapping[str, object], region: str) -> "Credentials":
        if params.get("api_key"):
            raise _bad_request(_SIGV4_ONLY_MESSAGE)
        auth: Final = AwsAuthParams.model_validate(params)
        if bool(auth.aws_access_key_id) != bool(auth.aws_secret_access_key):
            raise _unauthenticated("aws_access_key_id and aws_secret_access_key must be set together")
        ensure_optional_import("botocore")
        from botocore.credentials import Credentials
        from botocore.exceptions import (
            CredentialRetrievalError,
            NoCredentialsError,
            PartialCredentialsError,
            ProfileNotFound,
        )

        try:
            credentials: Final = _usable_credentials(self._aws.resolve_credentials(auth, region))
        except CredentialRetrievalError as error:
            verbose_logger.warning("OpenSearch vector store AWS credential source failed: %s", error)
            raise _unauthenticated("the AWS credential source failed, see the LiteLLM logs") from None
        except OSError as error:
            raise _unavailable_credentials(error) from None
        except (NoCredentialsError, PartialCredentialsError, ProfileNotFound) as error:
            raise _unauthenticated(str(error)) from None
        try:
            frozen: Final = credentials.get_frozen_credentials()
        except (CredentialRetrievalError, OSError, RuntimeError) as error:
            raise _unavailable_credentials(error) from None
        if frozen.access_key is None or frozen.secret_key is None:
            raise _unauthenticated("none were found")
        return Credentials(frozen.access_key, frozen.secret_key, frozen.token, account_id=frozen.account_id)

    def get_complete_url(self, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        parsed: Final = urlsplit(api_base or "")
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise _bad_request(
                "api_base must be the OpenSearch endpoint URL, for example "
                "https://<collection-id>.<region>.aoss.amazonaws.com or https://<domain-endpoint>."
            )
        if _ACCOUNT_ID_LABEL.match(parsed.hostname) and aws_signing_target(api_base or "")[0] == "aoss":
            raise _bad_request(_PER_ACCOUNT_MESSAGE)
        return (api_base or "").rstrip("/")

    def get_vector_store_url(self, api_base: str, encoded_vector_store_id: str) -> NoReturn:
        raise _bad_request(_SEARCH_ONLY_MESSAGE)

    def get_vector_store_list_url(self, api_base: str) -> NoReturn:
        raise _bad_request(_SEARCH_ONLY_MESSAGE)

    @staticmethod
    def _params(
        litellm_params: Mapping[str, object],
        optional_params: VectorStoreSearchOptionalRequestParams,
        extra_body: Mapping[str, object] | None,
    ) -> _Search:
        unknown: Final = sorted(
            key for key in litellm_params if key.startswith(_PARAM_PREFIX) and key not in _KNOWN_PARAMS
        )
        if unknown:
            raise _bad_request(
                f"Unknown OpenSearch vector store parameter(s): {', '.join(unknown)}. "
                f"Supported: {', '.join(sorted(_KNOWN_PARAMS))}."
            )
        if extra_body:
            raise _bad_request("The OpenSearch vector store does not support extra_body.")
        try:
            options: Final = _SearchOptions.model_validate(optional_params)
        except ValidationError as error:
            raise _bad_request(f"Invalid vector store search parameters: {error}") from None
        drop_unsupported: Final = drops_unsupported_params(litellm_params.get("drop_params"))
        if options.unsupported and not drop_unsupported:
            raise _bad_request(
                f"The OpenSearch vector store does not support {', '.join(options.unsupported)} yet. "
                "Remove it from the request, or set drop_params to drop it."
            )
        try:
            params: Final = _OpenSearchParams.model_validate(litellm_params)
        except ValidationError:
            raise _bad_request(
                "opensearch_embedding_field, opensearch_index_name, opensearch_search_pipeline, opensearch_text_field, "
                "and litellm_embedding_model must be strings, and opensearch_hybrid_search must be a boolean."
            ) from None
        if params.opensearch_hybrid_search and not params.opensearch_search_pipeline and not drop_unsupported:
            raise _bad_request(
                "opensearch_hybrid_search needs opensearch_search_pipeline, the name of a search pipeline with a "
                "normalization-processor, since OpenSearch returns unnormalized hybrid scores without one. Set it, or "
                "set drop_params to run a k-NN search instead."
            )
        return _Search(params=params, k=options.k, search_filter=options.search_filter(drop_unsupported))

    @staticmethod
    def _search_request(
        vector_store_id: str,
        query_text: str,
        embedding: EmbeddingResponse,
        search: _Search,
        api_base: str,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: the provider contract returns a writable JSON request body
        vector: Final = _QUERY_VECTOR.validate_python(embedding.data[0]["embedding"])  # pyright: ignore[reportUnknownMemberType]  # EmbeddingResponse.data is an untyped list, the vector is validated here
        return (
            f"{api_base}/{quote(search.params.index_name(vector_store_id), safe='')}{search.path}",
            {
                "size": search.k,
                "query": search.query(query_text, vector),
                "_source": {"excludes": [search.params.embedding_field]},
            },
        )

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: "LiteLLMLoggingObj",
        litellm_params: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: the provider contract returns a writable JSON request body
        search: Final = self._params(litellm_params, vector_store_search_optional_params, extra_body)
        query_text: Final = self.query_text(query)
        embedding: Final = self.query_embedding_executor(embedding_executor, None).embed(
            search.params.require_embedding_model(), query_text, self.query_embedding_configuration(litellm_params)
        )
        return self._search_request(vector_store_id, query_text, embedding, search, api_base)

    async def atransform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: "LiteLLMLoggingObj",
        litellm_params: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: the provider contract returns a writable JSON request body
        search: Final = self._params(litellm_params, vector_store_search_optional_params, extra_body)
        query_text: Final = self.query_text(query)
        embedding: Final = await self.query_embedding_executor(embedding_executor, None).aembed(
            search.params.require_embedding_model(), query_text, self.query_embedding_configuration(litellm_params)
        )
        return self._search_request(vector_store_id, query_text, embedding, search, api_base)

    def _filter_mapping_url(
        self,
        vector_store_id: str,
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_params: Mapping[str, object],
    ) -> tuple[_Filter, str] | None:
        search: Final = self._params(litellm_params, vector_store_search_optional_params, None)
        if search.search_filter is None:
            return None
        return search.search_filter, f"{api_base}/{quote(search.params.index_name(vector_store_id), safe='')}/_mapping"

    def _signed_mapping_read_headers(
        self, url: str, headers: Mapping[str, object], litellm_params: Mapping[str, object]
    ) -> Mapping[str, object]:
        signed: Final = self.sign_http_request("GET", url, b"", "application/json", litellm_params)
        return _with_signed_headers(headers, dict(signed.headers.items()))

    def _exact_search_params(
        self,
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        search_filter: _Filter,
        mapping: httpx.Response,
        litellm_params: Mapping[str, object],
    ) -> VectorStoreSearchOptionalRequestParams:
        mappings: Final = _read_index_mappings(mapping)
        exact: Final = mappings if isinstance(mappings, BaseLLMException) else _exact_filter(search_filter, mappings)
        if not isinstance(exact, BaseLLMException | _InexactKey):
            return {**vector_store_search_optional_params, "filters": exact.model_dump()}
        if isinstance(exact, BaseLLMException) and exact.status_code != _FORBIDDEN:
            raise exact
        if drops_unsupported_params(litellm_params.get("drop_params")):
            return {**vector_store_search_optional_params, "filters": None}
        if isinstance(exact, BaseLLMException):
            raise exact
        remedy: Final = (
            exact.remedy or f"Map {exact.key} as keyword without a normalizer or give it such a keyword subfield"
        )
        raise _bad_request(
            f"filters cannot match {exact.key} exactly because {exact.reason}. {remedy}, or set drop_params to drop "
            "the filters."
        )

    def resolve_search_params(
        self,
        vector_store_id: str,
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
        get: Callable[[str, Mapping[str, object]], httpx.Response],
    ) -> VectorStoreSearchOptionalRequestParams:
        lookup: Final = self._filter_mapping_url(
            vector_store_id, vector_store_search_optional_params, api_base, litellm_params
        )
        if lookup is None:
            return vector_store_search_optional_params
        search_filter, url = lookup
        mapping: Final = get(url, self._signed_mapping_read_headers(url, headers, litellm_params))
        return self._exact_search_params(vector_store_search_optional_params, search_filter, mapping, litellm_params)

    async def aresolve_search_params(
        self,
        vector_store_id: str,
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
        get: Callable[[str, Mapping[str, object]], Awaitable[httpx.Response]],
    ) -> VectorStoreSearchOptionalRequestParams:
        lookup: Final = self._filter_mapping_url(
            vector_store_id, vector_store_search_optional_params, api_base, litellm_params
        )
        if lookup is None:
            return vector_store_search_optional_params
        search_filter, url = lookup
        signed: Final = await run_off_loop_if_aws(self, self._signed_mapping_read_headers, url, headers, litellm_params)
        mapping: Final = await get(url, signed)
        return self._exact_search_params(vector_store_search_optional_params, search_filter, mapping, litellm_params)

    def recover_rejected_search(
        self,
        rejection: httpx.Response,
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        litellm_params: Mapping[str, object],
    ) -> VectorStoreSearchOptionalRequestParams | BaseLLMException | None:
        if rejection.status_code != 400 or not any(marker in rejection.text for marker in _UNFILTERABLE_KNN):
            return None
        if drops_unsupported_params(litellm_params.get("drop_params")):
            return {**vector_store_search_optional_params, "filters": None}
        return BaseLLMException(
            status_code=400,
            message="This OpenSearch index cannot apply filters inside a k-NN query, which needs the Lucene engine "
            "on OpenSearch 2.4+ or Faiss on 2.9+. Remove the filters, or set drop_params to drop them.",
            headers=httpx.Headers(),
        )

    def sign_request(
        self,
        headers: Mapping[str, object],
        optional_params: Mapping[str, object],
        request_data: Mapping[str, object],
        api_base: str,
        api_key: str | None = None,
    ) -> tuple[dict[str, object], bytes]:  # mutable-ok: the shared HTTP handler requires writable headers
        body: Final = json.dumps(request_data).encode()
        signed: Final = self.sign_http_request("POST", api_base, body, "application/json", optional_params)
        return dict(_with_signed_headers(headers, dict(signed.headers.items()))), body

    def sign_http_request(
        self, method: str, url: str, body: bytes, content_type: str, litellm_params: Mapping[str, object]
    ) -> "AWSPreparedRequest":
        region: Final = self._signing_region(litellm_params, url)
        return sign_aws_request(
            method=method,
            get_credentials=partial(self._signing_credentials, litellm_params, region),
            service_name=aws_signing_target(url)[0],
            aws_region_name=region,
            url=url,
            body=body,
            headers={"Content-Type": content_type, "x-amz-content-sha256": hashlib.sha256(body).hexdigest()},
        )

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: "LiteLLMLoggingObj"
    ) -> VectorStoreSearchResponse:
        if not response.is_success:
            raise BaseLLMException(status_code=response.status_code, message=response.text, headers=response.headers)
        try:
            hits: Final = _SearchResponse.model_validate_json(response.content).hits.hits
        except ValidationError:
            raise BaseLLMException(
                status_code=502,
                message="OpenSearch answered the search without a hits.hits list.",
                headers=response.headers,
            ) from None
        logged: Final = _LoggedSearch.model_validate(litellm_logging_obj.model_call_details)
        return VectorStoreSearchResponse(
            object="vector_store.search_results.page",
            search_query=logged.search_query,
            data=[_search_result(hit, logged.litellm_params) for hit in hits],
        )

    def validate_create_vector_store(self) -> NoReturn:
        raise _bad_request(_SEARCH_ONLY_MESSAGE)

    def transform_create_vector_store_request(
        self, vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams, api_base: str
    ) -> NoReturn:
        raise _bad_request(_SEARCH_ONLY_MESSAGE)

    def transform_create_vector_store_response(self, response: httpx.Response) -> NoReturn:
        raise _bad_request(_SEARCH_ONLY_MESSAGE)
