import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, NoReturn
from urllib.parse import quote, urlsplit

import httpx
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.exceptions import AuthenticationError, BadRequestError
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params
from litellm.litellm_core_utils.optional_imports import ensure_optional_import
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.vector_store.transformation import (
    BaseQueryEmbeddingVectorStoreConfig,
    VectorStoreEmbeddingExecutor,
)
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, SignsRequestsWithAWS, sign_aws_json_post
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


def _unauthenticated(reason: str) -> AuthenticationError:
    return AuthenticationError(
        message=f"No usable AWS credentials to sign the OpenSearch request ({reason}). {_CREDENTIALS_HINT}",
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
    opensearch_index_name: str | None = None
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


class _SearchOptions(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    max_num_results: int | None = None
    filters: Mapping[str, object] | None = None
    ranking_options: Mapping[str, object] | None = None
    rewrite_query: bool | None = None

    @property
    def unsupported(self) -> tuple[str, ...]:
        requested: Final = (
            ("filters", self.filters),
            ("ranking_options", self.ranking_options),
            ("rewrite_query", self.rewrite_query),
        )
        return tuple(name for name, value in requested if value)

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


class OpenSearchVectorStoreConfig(BaseQueryEmbeddingVectorStoreConfig, SignsRequestsWithAWS):
    def __init__(self, aws: BaseAWSLLM | None = None) -> None:
        self._aws: Final = aws if aws is not None else BaseAWSLLM()

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
        from botocore.exceptions import (
            CredentialRetrievalError,
            NoCredentialsError,
            PartialCredentialsError,
            ProfileNotFound,
        )

        try:
            return _usable_credentials(self._aws.resolve_credentials(auth, region))
        except CredentialRetrievalError as error:
            verbose_logger.warning("OpenSearch vector store AWS credential source failed: %s", error)
            raise _unauthenticated("the AWS credential source failed, see the LiteLLM logs") from None
        except (NoCredentialsError, PartialCredentialsError, ProfileNotFound) as error:
            raise _unauthenticated(str(error)) from None

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
    ) -> tuple[_OpenSearchParams, int]:
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
        if options.unsupported and not drops_unsupported_params(litellm_params.get("drop_params")):
            raise _bad_request(
                f"The OpenSearch vector store does not support {', '.join(options.unsupported)} yet. "
                "Remove it from the request, or set drop_params to drop it."
            )
        try:
            return _OpenSearchParams.model_validate(litellm_params), options.k
        except ValidationError:
            raise _bad_request(
                "opensearch_embedding_field, opensearch_index_name, opensearch_text_field, and litellm_embedding_model "
                "must be strings."
            ) from None

    @staticmethod
    def _search_request(
        vector_store_id: str,
        embedding: EmbeddingResponse,
        k: int,
        params: _OpenSearchParams,
        api_base: str,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: the provider contract returns a writable JSON request body
        vector: Final = _QUERY_VECTOR.validate_python(embedding.data[0]["embedding"])  # pyright: ignore[reportUnknownMemberType]  # EmbeddingResponse.data is an untyped list, the vector is validated here
        return (
            f"{api_base}/{quote(params.index_name(vector_store_id), safe='')}/_search",
            {
                "size": k,
                "query": {"knn": {params.embedding_field: {"vector": vector, "k": k}}},
                "_source": {"excludes": [params.embedding_field]},
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
        params, k = self._params(litellm_params, vector_store_search_optional_params, extra_body)
        embedding: Final = self.query_embedding_executor(embedding_executor, None).embed(
            params.require_embedding_model(), self.query_text(query), self.query_embedding_configuration(litellm_params)
        )
        return self._search_request(vector_store_id, embedding, k, params, api_base)

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
        params, k = self._params(litellm_params, vector_store_search_optional_params, extra_body)
        embedding: Final = await self.query_embedding_executor(embedding_executor, None).aembed(
            params.require_embedding_model(), self.query_text(query), self.query_embedding_configuration(litellm_params)
        )
        return self._search_request(vector_store_id, embedding, k, params, api_base)

    def sign_request(
        self,
        headers: Mapping[str, str],
        optional_params: Mapping[str, object],
        request_data: Mapping[str, object],
        api_base: str,
        api_key: str | None = None,
    ) -> tuple[dict[str, str], bytes]:  # mutable-ok: the shared HTTP handler requires writable headers
        region: Final = self._signing_region(optional_params, api_base)
        body: Final = json.dumps(request_data)
        signed: Final = sign_aws_json_post(
            get_credentials=partial(self._signing_credentials, optional_params, region),
            service_name=aws_signing_target(api_base)[0],
            aws_region_name=region,
            url=api_base,
            body=body,
            headers={
                "Content-Type": "application/json",
                "x-amz-content-sha256": hashlib.sha256(body.encode()).hexdigest(),
            },
        )
        signed_headers: Final = dict(signed.headers.items())
        signed_names: Final = frozenset(name.lower() for name in signed_headers)
        return {
            **{name: value for name, value in headers.items() if name.lower() not in signed_names},
            **signed_headers,
        }, body.encode()

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
