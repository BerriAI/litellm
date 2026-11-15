import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Final
from unittest.mock import MagicMock

import httpcore
import httpx
import openai
import pytest
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials, RefreshableCredentials
from botocore.exceptions import CredentialRetrievalError
from h2.events import ConnectionTerminated

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.opensearch.vector_stores.index_passthrough import (
    IndexPath,
    IndexRefusal,
    index_operation,
    prepare_index_request,
)
from litellm.llms.opensearch.vector_stores.transformation import OpenSearchVectorStoreConfig
from litellm.types.llms.bedrock import AwsAuthParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import EmbeddingResponse
from litellm.types.vector_stores import VectorStoreSearchResponse
from litellm.vector_stores import main as vector_stores

ACCESS_KEY: Final = "AKIDOPENSEARCHTEST"
SECRET_KEY: Final = "opensearch-test-secret"
QUERY: Final = "How many days a year can I work abroad?"
QUERY_VECTOR: Final = [0.1, 0.2, 0.3]
CREDENTIAL_SOURCE_OUTPUT: Final = "credential-process-stderr-detail"
SERVERLESS: Final = "https://abc123.us-east-1.aoss.amazonaws.com"
BASE_PARAMS: Final = {
    "api_base": SERVERLESS,
    "aws_access_key_id": ACCESS_KEY,
    "aws_secret_access_key": SECRET_KEY,
    "litellm_embedding_model": "embedding-alias",
}
_AUTHORIZATION: Final = re.compile(
    r"\AAWS4-HMAC-SHA256 Credential=(?P<key>[^/]+)/(?P<date>\d{8})/(?P<region>[^/]+)/(?P<service>[^/]+)/aws4_request, "
    r"SignedHeaders=(?P<signed>[^,]+), Signature=(?P<signature>[0-9a-f]{64})\Z"
)


class RecordingEmbeddingExecutor:
    def __init__(self) -> None:
        self.call: Final = MagicMock(return_value=EmbeddingResponse(data=[{"embedding": QUERY_VECTOR}]))

    def embed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.call(model, query, configuration)

    async def aembed(self, model: str, query: str, configuration: Mapping[str, object]) -> EmbeddingResponse:
        return self.call(model, query, configuration)


def _signature_scope(request: httpx.Request) -> tuple[str, str]:
    authorization: Final = _AUTHORIZATION.match(request.headers["authorization"])
    assert authorization is not None, request.headers["authorization"]
    signed_names: Final = authorization["signed"].split(";")
    assert {"host", "x-amz-content-sha256", "x-amz-date"} <= set(signed_names), signed_names
    assert request.headers["x-amz-content-sha256"] == hashlib.sha256(request.content).hexdigest()
    signed_request: Final = AWSRequest(
        method=request.method,
        url=str(request.url),
        data=request.content,
        headers={name: request.headers[name] for name in signed_names},
    )
    signed_request.context["timestamp"] = request.headers["x-amz-date"]
    signer: Final = SigV4Auth(Credentials(ACCESS_KEY, SECRET_KEY), authorization["service"], authorization["region"])
    expected: Final = signer.signature(
        signer.string_to_sign(signed_request, signer.canonical_request(signed_request)), signed_request
    )
    assert authorization["key"] == ACCESS_KEY
    assert authorization["signature"] == expected, "SigV4 signature does not cover the request that was sent"
    return authorization["service"], authorization["region"]


def _sent(respond: MagicMock) -> tuple[httpx.Request, ...]:
    return tuple(call.args[0] for call in respond.call_args_list)


def _opensearch_hits(text_field: str, embedding_field: str) -> Mapping[str, object]:
    return {
        "took": 4,
        "timed_out": False,
        "hits": {
            "total": {"value": 2, "relation": "eq"},
            "max_score": 0.581,
            "hits": [
                {
                    "_index": "litellm-qa-docs",
                    "_id": "1%3A0%3Aw6ftW6EB",
                    "_score": 0.581,
                    "_source": {
                        text_field: "Employees may work abroad for at most 37 calendar days per year.",
                        embedding_field: [0.5, 0.5, 0.5],
                        "embedding_ada002": [0.25, 0.75],
                        "filename": "hr-remote-work-policy.md",
                        "category": "hr",
                        "revision": 7,
                        "published": True,
                        "tags": ["hr", "travel"],
                        "owner": {"team": "people-ops"},
                        "reviewed_by": None,
                    },
                },
                {"_index": "litellm-qa-docs", "_id": "doc-2", "_score": 0.42, "_source": {"category": "facilities"}},
            ],
        },
    }


EXPECTED_RESULTS: Final = [
    {
        "score": 0.581,
        "content": [{"type": "text", "text": "Employees may work abroad for at most 37 calendar days per year."}],
        "file_id": "1%3A0%3Aw6ftW6EB",
        "filename": "hr-remote-work-policy.md",
        "attributes": {"filename": "hr-remote-work-policy.md", "category": "hr", "revision": 7, "published": True},
    },
    {"score": 0.42, "content": [], "file_id": "doc-2", "filename": None, "attributes": {"category": "facilities"}},
]


async def _search(
    asynchronous: bool,
    respond: Callable[[httpx.Request], httpx.Response],
    executor: RecordingEmbeddingExecutor,
    kwargs: Mapping[str, object],
) -> VectorStoreSearchResponse:
    request_kwargs: Final = {
        **BASE_PARAMS,
        "vector_store_id": "litellm-qa-docs",
        "query": QUERY,
        "custom_llm_provider": "opensearch",
        "_direct_vector_store_embedding_executor": executor,
        **kwargs,
    }
    if asynchronous:
        return await litellm.vector_stores.asearch(
            **request_kwargs, client=AsyncHTTPHandler(transport=httpx.MockTransport(respond))
        )
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        return litellm.vector_stores.search(**request_kwargs, client=HTTPHandler(client=transport))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "fields",
    [{}, {"opensearch_embedding_field": "passage_vector", "opensearch_text_field": "passage"}],
)
@pytest.mark.parametrize(
    "api_base,region_params,service,region",
    [
        (SERVERLESS, {}, "aoss", "us-east-1"),
        (SERVERLESS, {"aws_region_name": "us-west-2"}, "aoss", "us-east-1"),
        ("https://search-docs-xyz.eu-west-1.es.amazonaws.com/", {}, "es", "eu-west-1"),
        ("https://vpc-docs-xyz.cn-north-1.es.amazonaws.com.cn", {}, "es", "cn-north-1"),
        ("https://abc123.us-gov-west-1.aoss-fips.amazonaws.com", {}, "aoss", "us-gov-west-1"),
        ("https://abc123.aoss.eu-central-1.on.aws", {}, "aoss", "eu-central-1"),
        ("https://search-docs-xyz.aos.ap-southeast-2.on.aws", {}, "es", "ap-southeast-2"),
        ("https://search.docs.example", {"aws_region_name": "ap-south-1"}, "es", "ap-south-1"),
    ],
)
@pytest.mark.asyncio
async def test_search_sends_a_signed_knn_query_and_returns_an_openai_results_page(
    asynchronous: bool,
    fields: Mapping[str, str],
    api_base: str,
    region_params: Mapping[str, str],
    service: str,
    region: str,
) -> None:
    executor: Final = RecordingEmbeddingExecutor()
    embedding_field: Final = fields.get("opensearch_embedding_field", "embedding")
    text_field: Final = fields.get("opensearch_text_field", "text")
    respond: Final = MagicMock(return_value=httpx.Response(200, json=_opensearch_hits(text_field, embedding_field)))

    response: Final = await _search(
        asynchronous,
        respond,
        executor,
        {**fields, **region_params, "api_base": api_base, "max_num_results": 3},
    )

    assert response == {"object": "vector_store.search_results.page", "search_query": QUERY, "data": EXPECTED_RESULTS}
    sent: Final = _sent(respond)
    assert len(sent) == 1
    assert str(sent[0].url) == f"{api_base.rstrip('/')}/litellm-qa-docs/_search"
    assert json.loads(sent[0].content) == {
        "size": 3,
        "query": {"knn": {embedding_field: {"vector": QUERY_VECTOR, "k": 3}}},
        "_source": {"excludes": [embedding_field]},
    }
    assert _signature_scope(sent[0]) == (service, region)
    executor.call.assert_called_once_with("embedding-alias", QUERY, {})


@pytest.mark.asyncio
async def test_search_defaults_to_ten_results_like_openai() -> None:
    respond: Final = MagicMock(return_value=httpx.Response(200, json={"hits": {"hits": []}}))

    response: Final = await _search(True, respond, RecordingEmbeddingExecutor(), {})

    assert response == {"object": "vector_store.search_results.page", "search_query": QUERY, "data": []}
    body: Final = json.loads(_sent(respond)[0].content)
    assert (body["size"], body["query"]["knn"]["embedding"]["k"]) == (10, 10)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_a_list_query_is_embedded_and_echoed_as_one_string(asynchronous: bool) -> None:
    executor: Final = RecordingEmbeddingExecutor()
    respond: Final = MagicMock(return_value=httpx.Response(200, json={"hits": {"hits": []}}))

    response: Final = await _search(asynchronous, respond, executor, {"query": ["remote work", "abroad"]})

    assert response["search_query"] == "remote work abroad"
    executor.call.assert_called_once_with("embedding-alias", "remote work abroad", {})


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [403, 404, 429, 500])
@pytest.mark.asyncio
async def test_opensearch_error_keeps_its_status(asynchronous: bool, status: int) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"status": status, "error": {"reason": "upstream refused", "type": "x"}})

    with pytest.raises(openai.APIStatusError) as error:
        await _search(asynchronous, respond, RecordingEmbeddingExecutor(), {})

    assert error.value.status_code == status


MALFORMED_FILTERS: Final = (
    {"filters": {"type": "between", "key": "revision", "value": [6, 8]}},
    {"filters": {"type": "gte", "key": "published", "value": True}},
    {"filters": {"type": "in", "key": "category", "value": "hr"}},
    {"filters": {"type": "eq", "key": "category"}},
    {"filters": {"type": "eq", "key": "category", "value": "hr", "case_sensitive": False}},
    {"filters": {"type": "and", "filters": [{"type": "eq", "key": "category", "value": "hr"}, {"type": "near"}]}},
    {"filters": {"type": "gt", "key": "score", "value": float("nan")}},
    {"filters": {"type": "in", "key": "score", "value": [0.5, float("inf")]}},
)
NON_OBJECT_FILTERS: Final = ({"filters": "category = hr"}, {"filters": []}, {"filters": 5})


@pytest.mark.parametrize(
    "overrides,message",
    [
        *((malformed, "filters must be an OpenAI vector store filter") for malformed in MALFORMED_FILTERS),
        *((non_object, "Invalid vector store search parameters") for non_object in NON_OBJECT_FILTERS),
        *(
            ({**non_object, "drop_params": True}, "Invalid vector store search parameters")
            for non_object in NON_OBJECT_FILTERS
        ),
        ({"opensearch_hybrid_search": True}, "opensearch_hybrid_search needs opensearch_search_pipeline"),
        ({"opensearch_hybrid_search": "sometimes"}, "opensearch_hybrid_search must be a boolean"),
        ({"opensearch_search_pipeline": ["hybrid-norm"]}, "opensearch_search_pipeline"),
        ({"ranking_options": {"score_threshold": 0.5}}, "ranking_options"),
        ({"rewrite_query": True}, "rewrite_query"),
        ({"max_num_results": 0}, "max_num_results"),
        ({"max_num_results": 51}, "max_num_results"),
        ({"max_num_results": "ten"}, "Invalid vector store search parameters"),
        ({"litellm_embedding_model": None}, "litellm_embedding_model"),
        ({"opensearch_text_feild": "passage"}, "opensearch_text_feild"),
        ({"opensearch_text_field": 7}, "opensearch_text_field"),
        ({"opensearch_index_name": ["docs"]}, "opensearch_index_name"),
        ({"extra_body": {"min_score": 0.5}}, "extra_body"),
        ({"api_base": None}, "api_base"),
        ({"api_base": "abc123.us-east-1.aoss.amazonaws.com"}, "api_base"),
        ({"api_base": "https://123456789012.aoss.us-east-1.on.aws"}, "per-account endpoint"),
        ({"api_key": "opensearch-basic-auth"}, "api_key"),
    ],
)
@pytest.mark.asyncio
async def test_unsupported_search_is_rejected_before_embedding_or_any_request(
    overrides: Mapping[str, object], message: str
) -> None:
    executor: Final = RecordingEmbeddingExecutor()
    respond: Final = MagicMock(return_value=httpx.Response(200, json={"hits": {"hits": []}}))

    with pytest.raises(litellm.BadRequestError, match=message) as error:
        await _search(True, respond, executor, overrides)

    assert error.value.status_code == 400
    respond.assert_not_called()
    executor.call.assert_not_called()


_AMBIENT_AWS_ENV: Final = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_PROFILE_NAME",
    "AWS_ROLE_NAME",
    "AWS_ROLE_ARN",
    "AWS_SESSION_NAME",
    "AWS_WEB_IDENTITY_TOKEN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
)


def _isolate_from_ambient_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in _AMBIENT_AWS_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in ("AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE", "BOTO_CONFIG"):
        config_file: Path = tmp_path / name
        config_file.write_text("")
        monkeypatch.setenv(name, str(config_file))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "credentials,env",
    [
        pytest.param({"aws_session_name": "opensearch-no-credentials"}, {}, id="none-anywhere"),
        pytest.param({"aws_profile_name": "opensearch-missing-profile"}, {}, id="missing-profile"),
        pytest.param({}, {"AWS_ACCESS_KEY_ID": "AKIDPARTIAL"}, id="key-without-secret"),
        pytest.param(
            {"aws_access_key_id": "AKIDSTOREONLY"},
            {"AWS_ACCESS_KEY_ID": "AKIDAMBIENT", "AWS_SECRET_ACCESS_KEY": "ambient-secret"},
            id="store-key-without-secret-beside-ambient-credentials",
        ),
        pytest.param(
            {"aws_secret_access_key": "store-secret-only"},
            {"AWS_ACCESS_KEY_ID": "AKIDAMBIENT", "AWS_SECRET_ACCESS_KEY": "ambient-secret"},
            id="store-secret-without-key-beside-ambient-credentials",
        ),
    ],
)
@pytest.mark.asyncio
async def test_missing_aws_credentials_are_a_401_before_embedding_or_any_request(
    asynchronous: bool,
    credentials: Mapping[str, str],
    env: Mapping[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_from_ambient_aws(monkeypatch, tmp_path)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    executor: Final = RecordingEmbeddingExecutor()
    respond: Final = MagicMock(return_value=httpx.Response(200, json={"hits": {"hits": []}}))

    with pytest.raises(litellm.AuthenticationError, match="No usable AWS credentials") as error:
        await _search(
            asynchronous,
            respond,
            executor,
            {"aws_access_key_id": None, "aws_secret_access_key": None, **credentials},
        )

    assert error.value.status_code == 401
    respond.assert_not_called()
    executor.call.assert_not_called()


class _FailingCredentialSource(BaseAWSLLM):
    def resolve_credentials(self, auth_params: AwsAuthParams, aws_region_name: str | None) -> Credentials:
        raise CredentialRetrievalError(provider="container-role", error_msg=CREDENTIAL_SOURCE_OUTPUT)


def test_failing_credential_source_is_a_401_that_keeps_its_output_out_of_the_reply() -> None:
    config: Final = OpenSearchVectorStoreConfig(aws=_FailingCredentialSource())

    with pytest.raises(litellm.AuthenticationError, match="the AWS credential source failed") as error:
        config.validate_environment({}, GenericLiteLLMParams(api_base=SERVERLESS))

    assert error.value.status_code == 401
    assert CREDENTIAL_SOURCE_OUTPUT not in str(error.value)


@pytest.mark.parametrize(
    "sign",
    [
        pytest.param(
            lambda config: config.validate_environment({}, GenericLiteLLMParams(api_base=SERVERLESS)),
            id="vector-store-search",
        ),
        pytest.param(
            lambda config: config.sign_http_request(
                "GET", f"{SERVERLESS}/docs/_mapping", b"", "application/json", {"api_base": SERVERLESS}
            ),
            id="index-pass-through",
        ),
    ],
)
def test_a_missing_container_credentials_token_file_is_a_500(
    sign: Callable[[OpenSearchVectorStoreConfig], object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_from_ambient_aws(monkeypatch, tmp_path)
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", "http://127.0.0.1:9/v1/credentials")
    monkeypatch.setenv("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE", str(tmp_path / "missing-token"))

    with pytest.raises(litellm.InternalServerError, match="could not load or refresh the AWS credentials") as error:
        sign(OpenSearchVectorStoreConfig())

    assert error.value.status_code == 500


EXPIRED_METADATA: Final = MappingProxyType(
    {"access_key": "AKIDEXAMPLE", "secret_key": "secret", "token": "token", "expiry_time": "2020-01-01T00:00:00Z"}
)


def _refresh_returning_expired_credentials() -> dict[str, str]:
    return dict(EXPIRED_METADATA)


def _refresh_after_the_token_file_is_gone() -> dict[str, str]:
    raise CredentialRetrievalError(provider="container-role", error_msg=CREDENTIAL_SOURCE_OUTPUT)


class _CredentialSourceThatCannotRefresh(BaseAWSLLM):
    def __init__(self, refresh: Callable[[], dict[str, str]]) -> None:
        super().__init__()
        self._refresh = refresh

    def resolve_credentials(self, auth_params: AwsAuthParams, aws_region_name: str | None) -> Credentials:
        return RefreshableCredentials.create_from_metadata(
            metadata=dict(EXPIRED_METADATA), refresh_using=self._refresh, method="container-role"
        )


@pytest.mark.parametrize("refresh", [_refresh_returning_expired_credentials, _refresh_after_the_token_file_is_gone])
@pytest.mark.parametrize(
    "sign",
    [
        pytest.param(
            lambda config: config.validate_environment({}, GenericLiteLLMParams(api_base=SERVERLESS)),
            id="vector-store-search",
        ),
        pytest.param(
            lambda config: config.sign_http_request(
                "GET", f"{SERVERLESS}/docs/_mapping", b"", "application/json", {"api_base": SERVERLESS}
            ),
            id="index-pass-through",
        ),
    ],
)
def test_credentials_that_fail_to_refresh_at_signing_are_a_500(
    sign: Callable[[OpenSearchVectorStoreConfig], object], refresh: Callable[[], dict[str, str]]
) -> None:
    config: Final = OpenSearchVectorStoreConfig(aws=_CredentialSourceThatCannotRefresh(refresh))

    with pytest.raises(litellm.InternalServerError, match="could not load or refresh the AWS credentials") as error:
        sign(config)

    assert error.value.status_code == 500
    assert CREDENTIAL_SOURCE_OUTPUT not in str(error.value)


@pytest.mark.asyncio
async def test_the_index_pass_through_answers_500_when_credentials_fail_to_refresh() -> None:
    refused: Final = await prepare_index_request(
        config=OpenSearchVectorStoreConfig(
            aws=_CredentialSourceThatCannotRefresh(_refresh_after_the_token_file_is_gone)
        ),
        method="GET",
        path=IndexPath(index_name="docs", api=("_mapping",)),
        operation=index_operation("GET", ("_mapping",)),
        provider_index="prod-docs",
        content_type=None,
        query=(),
        body=b"",
        store_params={"api_base": SERVERLESS},
        can_read=True,
    )

    assert isinstance(refused, IndexRefusal)
    assert (refused.status_code, CREDENTIAL_SOURCE_OUTPUT in refused.message) == (500, False)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_opensearch_index_name_picks_the_index_so_the_store_id_can_differ(asynchronous: bool) -> None:
    respond: Final = MagicMock(return_value=httpx.Response(200, json=_opensearch_hits("text", "embedding")))

    response: Final = await _search(
        asynchronous,
        respond,
        RecordingEmbeddingExecutor(),
        {"vector_store_id": "east-docs", "opensearch_index_name": "docs/v2"},
    )

    assert response["data"] == EXPECTED_RESULTS
    assert str(_sent(respond)[0].url) == f"{SERVERLESS}/docs%2Fv2/_search"


@pytest.mark.parametrize(
    "unsupported",
    [
        *MALFORMED_FILTERS,
        {"opensearch_hybrid_search": True},
        {"ranking_options": {"score_threshold": 0.5}},
        {"rewrite_query": True},
    ],
)
@pytest.mark.asyncio
async def test_unsupported_search_params_are_dropped_under_drop_params(unsupported: Mapping[str, object]) -> None:
    respond: Final = MagicMock(return_value=httpx.Response(200, json=_opensearch_hits("text", "embedding")))

    response: Final = await _search(
        True, respond, RecordingEmbeddingExecutor(), {**unsupported, "drop_params": True, "max_num_results": 3}
    )

    assert response["data"] == EXPECTED_RESULTS
    assert str(_sent(respond)[0].url) == f"{SERVERLESS}/litellm-qa-docs/_search"
    assert json.loads(_sent(respond)[0].content) == {
        "size": 3,
        "query": {"knn": {"embedding": {"vector": QUERY_VECTOR, "k": 3}}},
        "_source": {"excludes": ["embedding"]},
    }


@pytest.mark.asyncio
async def test_no_op_values_of_unsupported_params_are_accepted() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_opensearch_hits("text", "embedding"))

    response: Final = await _search(
        True, respond, RecordingEmbeddingExecutor(), {"filters": {}, "ranking_options": {}, "rewrite_query": False}
    )

    assert response["data"] == EXPECTED_RESULTS


DYNAMIC_MAPPING: Final = {
    "properties": {
        "category": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
        "published": {"type": "boolean"},
        "published_at": {"type": "date"},
        "revision": {"type": "long"},
        "score": {"type": "float"},
    }
}
HR_EXACT: Final = {"term": {"category.keyword": "hr"}}


def _index_mappings(mappings: Mapping[str, Mapping[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={index: {"mappings": mapping} for index, mapping in mappings.items()})


def _index(mapping: Mapping[str, object], *searches: httpx.Response) -> MagicMock:
    answers: Final = iter(searches)
    return MagicMock(
        side_effect=lambda request: (
            _index_mappings({"litellm-qa-docs": mapping}) if request.method == "GET" else next(answers)
        )
    )


def _searches(respond: MagicMock) -> tuple[httpx.Request, ...]:
    return tuple(request for request in _sent(respond) if request.method == "POST")


def _mapping_reads(respond: MagicMock) -> tuple[httpx.Request, ...]:
    return tuple(request for request in _sent(respond) if request.method == "GET")


@pytest.mark.parametrize(
    "search_filter,opensearch_filter",
    [
        pytest.param({"type": "eq", "key": "category", "value": "hr"}, HR_EXACT, id="eq"),
        pytest.param({"type": "eq", "key": "published", "value": True}, {"term": {"published": True}}, id="eq-bool"),
        pytest.param({"type": "ne", "key": "category", "value": "hr"}, {"bool": {"must_not": [HR_EXACT]}}, id="ne"),
        pytest.param(
            {"type": "ne", "key": "revision", "value": 7},
            {"bool": {"must_not": [{"term": {"revision": 7}}]}},
            id="ne-int",
        ),
        *(
            pytest.param(
                {"type": operator, "key": "revision", "value": 7}, {"range": {"revision": {operator: 7}}}, id=operator
            )
            for operator in ("gt", "gte", "lt", "lte")
        ),
        pytest.param({"type": "gte", "key": "score", "value": 0.5}, {"range": {"score": {"gte": 0.5}}}, id="gte-float"),
        pytest.param(
            {"type": "lt", "key": "published_at", "value": "2026-01-01"},
            {"range": {"published_at": {"lt": "2026-01-01"}}},
            id="lt-date",
        ),
        pytest.param(
            {"type": "in", "key": "category", "value": ["hr", "legal"]},
            {"terms": {"category.keyword": ["hr", "legal"]}},
            id="in",
        ),
        pytest.param({"type": "in", "key": "revision", "value": [6, 7]}, {"terms": {"revision": [6, 7]}}, id="in-int"),
        pytest.param(
            {"type": "nin", "key": "revision", "value": [6, 7]},
            {"bool": {"must_not": [{"terms": {"revision": [6, 7]}}]}},
            id="nin",
        ),
        pytest.param(
            {
                "type": "and",
                "filters": [
                    {"type": "eq", "key": "category", "value": "hr"},
                    {
                        "type": "or",
                        "filters": [
                            {"type": "gte", "key": "revision", "value": 7},
                            {"type": "eq", "key": "published", "value": False},
                        ],
                    },
                ],
            },
            {
                "bool": {
                    "filter": [
                        HR_EXACT,
                        {
                            "bool": {
                                "should": [{"range": {"revision": {"gte": 7}}}, {"term": {"published": False}}],
                                "minimum_should_match": 1,
                            }
                        },
                    ]
                }
            },
            id="and-of-or",
        ),
    ],
)
@pytest.mark.asyncio
async def test_filters_narrow_the_knn_query_with_the_matching_opensearch_filter(
    search_filter: Mapping[str, object], opensearch_filter: Mapping[str, object]
) -> None:
    respond: Final = _index(DYNAMIC_MAPPING, httpx.Response(200, json=_opensearch_hits("text", "embedding")))

    response: Final = await _search(
        True, respond, RecordingEmbeddingExecutor(), {"filters": search_filter, "max_num_results": 3}
    )

    assert response["data"] == EXPECTED_RESULTS
    (search,) = _searches(respond)
    assert json.loads(search.content) == {
        "size": 3,
        "query": {"knn": {"embedding": {"vector": QUERY_VECTOR, "k": 3, "filter": opensearch_filter}}},
        "_source": {"excludes": ["embedding"]},
    }


NESTED_TEAM: Final = {"team": {"type": "keyword"}}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "mapping,search_filter,opensearch_filter",
    [
        pytest.param(
            {"properties": {"tenant": {"type": "keyword"}}},
            {"type": "eq", "key": "tenant", "value": "acme"},
            {"term": {"tenant": "acme"}},
            id="keyword-field",
        ),
        pytest.param(
            {"properties": {"tenant": {"type": "text", "fields": {"raw": {"type": "keyword"}}}}},
            {"type": "ne", "key": "tenant", "value": "acme"},
            {"bool": {"must_not": [{"term": {"tenant.raw": "acme"}}]}},
            id="text-with-a-keyword-subfield",
        ),
        pytest.param(
            {"properties": {"tenant": {"type": "text", "fields": {"keyword": {"type": "keyword"}}}}},
            {"type": "gte", "key": "tenant", "value": "b"},
            {"range": {"tenant.keyword": {"gte": "b"}}},
            id="string-range",
        ),
        pytest.param(
            {"properties": {"owner": {"properties": {"team": {"type": "keyword"}}}}},
            {"type": "in", "key": "owner.team", "value": ["people-ops"]},
            {"terms": {"owner.team": ["people-ops"]}},
            id="object-subfield",
        ),
        pytest.param(
            {
                "properties": {
                    "tenant": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword", "normalizer": "lowercase"},
                            "raw": {"type": "keyword"},
                        },
                    }
                }
            },
            {"type": "eq", "key": "tenant", "value": "Acme"},
            {"term": {"tenant.raw": "Acme"}},
            id="skips-a-normalized-keyword-subfield",
        ),
        pytest.param(
            {
                "properties": {
                    "tenant": {"type": "keyword", "normalizer": "lowercase", "fields": {"raw": {"type": "keyword"}}}
                }
            },
            {"type": "eq", "key": "tenant", "value": "Acme"},
            {"term": {"tenant.raw": "Acme"}},
            id="normalized-keyword-with-a-plain-subfield",
        ),
        pytest.param(
            {"properties": {"category": {"type": "keyword"}}},
            {"type": "eq", "key": "tenant", "value": "acme"},
            {"term": {"tenant": "acme"}},
            id="unmapped",
        ),
        pytest.param(
            {"properties": {"ref": {"type": "text", "fields": {"keyword": {"type": "keyword"}}}}},
            {"type": "eq", "key": "ref", "value": 7},
            {"term": {"ref.keyword": 7}},
            id="number-on-text-with-a-keyword-subfield",
        ),
        pytest.param(
            {"properties": {"ref_code": {"type": "keyword"}, "ref": {"type": "alias", "path": "ref_code"}}},
            {"type": "eq", "key": "ref", "value": "FAC-017"},
            {"term": {"ref": "FAC-017"}},
            id="alias-to-a-keyword-field",
        ),
        pytest.param(
            {
                "properties": {
                    "owner": {"properties": {"team": {"type": "text", "fields": {"raw": {"type": "keyword"}}}}},
                    "team": {"type": "alias", "path": "owner.team"},
                }
            },
            {"type": "in", "key": "team", "value": ["People Ops"]},
            {"terms": {"owner.team.raw": ["People Ops"]}},
            id="alias-to-text-with-a-keyword-subfield",
        ),
        pytest.param(
            {"properties": {"owner": {"type": "nested", "include_in_root": True, "properties": NESTED_TEAM}}},
            {"type": "eq", "key": "owner.team", "value": "people-ops"},
            {"term": {"owner.team": "people-ops"}},
            id="nested-object-copied-to-the-root",
        ),
        pytest.param(
            {"properties": {"owner": {"type": "nested", "include_in_parent": True, "properties": NESTED_TEAM}}},
            {"type": "in", "key": "owner.team", "value": ["people-ops"]},
            {"terms": {"owner.team": ["people-ops"]}},
            id="top-level-nested-object-copied-to-its-parent",
        ),
        pytest.param(
            {
                "properties": {
                    "org": {
                        "type": "nested",
                        "properties": {"owner": {"type": "nested", "include_in_root": True, "properties": NESTED_TEAM}},
                    }
                }
            },
            {"type": "eq", "key": "org.owner.team", "value": "people-ops"},
            {"term": {"org.owner.team": "people-ops"}},
            id="nested-in-nested-copied-to-the-root",
        ),
        pytest.param(
            {
                "properties": {
                    "org": {
                        "type": "nested",
                        "include_in_parent": True,
                        "properties": {
                            "owner": {"type": "nested", "include_in_parent": True, "properties": NESTED_TEAM}
                        },
                    }
                }
            },
            {"type": "eq", "key": "org.owner.team", "value": "people-ops"},
            {"term": {"org.owner.team": "people-ops"}},
            id="nested-in-nested-copied-up-through-each-parent",
        ),
    ],
)
@pytest.mark.asyncio
async def test_filters_compare_the_exact_field_the_index_maps(
    asynchronous: bool,
    mapping: Mapping[str, object],
    search_filter: Mapping[str, object],
    opensearch_filter: Mapping[str, object],
) -> None:
    respond: Final = _index(mapping, httpx.Response(200, json=_opensearch_hits("text", "embedding")))

    await _search(asynchronous, respond, RecordingEmbeddingExecutor(), {"filters": search_filter, "max_num_results": 3})

    (mapping_read,) = _mapping_reads(respond)
    assert str(mapping_read.url) == f"{SERVERLESS}/litellm-qa-docs/_mapping"
    assert _signature_scope(mapping_read) == ("aoss", "us-east-1")
    (search,) = _searches(respond)
    assert json.loads(search.content)["query"]["knn"]["embedding"]["filter"] == opensearch_filter


TEXT_ONLY: Final = {"litellm-qa-docs": {"properties": {"tenant": {"type": "text"}}}}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "mappings,search_filter,reason",
    [
        pytest.param(
            TEXT_ONLY,
            {"type": "eq", "key": "tenant", "value": "team"},
            "the index maps tenant as text with no keyword subfield",
            id="text-only",
        ),
        pytest.param(
            TEXT_ONLY,
            {
                "type": "or",
                "filters": [
                    {"type": "eq", "key": "revision", "value": 7},
                    {"type": "nin", "key": "tenant", "value": ["team other"]},
                ],
            },
            "the index maps tenant as text with no keyword subfield",
            id="text-only-in-a-compound",
        ),
        pytest.param(
            {
                "litellm-qa-docs": {
                    "properties": {
                        "tenant": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 8}}}
                    }
                }
            },
            {"type": "eq", "key": "tenant", "value": "acme quarterly"},
            "the index ignores tenant values longer than 8 characters",
            id="longer-than-ignore-above",
        ),
        pytest.param(
            {
                "docs-2025": {"properties": {"tenant": {"type": "keyword"}}},
                "docs-2026": {"properties": {"tenant": {"type": "text", "fields": {"keyword": {"type": "keyword"}}}}},
            },
            {"type": "eq", "key": "tenant", "value": "acme"},
            "the indexes behind this store map tenant differently",
            id="alias-indexes-disagree",
        ),
        pytest.param(
            {
                "docs-2025": {
                    "properties": {
                        "owner": {"properties": {"team": {"type": "text", "fields": {"raw": {"type": "keyword"}}}}},
                        "team": {"type": "alias", "path": "owner.team"},
                    }
                },
                "docs-2026": {
                    "properties": {
                        "owner": {"properties": {"team": {"type": "text", "fields": {"raw": {"type": "keyword"}}}}}
                    }
                },
            },
            {"type": "eq", "key": "team", "value": "People Ops"},
            "the indexes behind this store map team differently",
            id="alias-rewrite-reaches-an-index-without-the-alias",
        ),
        pytest.param(
            {
                "docs-2025": {"properties": {"team": {"type": "keyword"}}},
                "docs-2026": {"properties": {"category": {"type": "keyword"}}},
            },
            {"type": "eq", "key": "team", "value": "HR"},
            "the indexes behind this store map team differently. Map team to the same keyword field in every "
            "index behind the store",
            id="only-some-indexes-map-the-key",
        ),
        pytest.param(
            {
                "docs-2025": {
                    "properties": {"category": {"type": "keyword"}, "team": {"type": "alias", "path": "category"}}
                },
                "docs-2026": {"properties": {"category": {"type": "keyword"}}},
            },
            {"type": "eq", "key": "team", "value": "HR"},
            "the indexes behind this store map team differently. Map team to the same keyword field in every "
            "index behind the store",
            id="only-some-indexes-alias-the-key",
        ),
        pytest.param(
            {"litellm-qa-docs": {"properties": {"tenant": {"type": "keyword", "normalizer": "lowercase"}}}},
            {"type": "eq", "key": "tenant", "value": "Acme"},
            "the index normalizes every keyword mapping of tenant",
            id="normalized-keyword",
        ),
        pytest.param(
            {
                "litellm-qa-docs": {
                    "properties": {
                        "tenant": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword", "normalizer": "lowercase"}},
                        }
                    }
                }
            },
            {"type": "eq", "key": "tenant", "value": "Acme"},
            "the index normalizes every keyword mapping of tenant",
            id="text-with-only-a-normalized-keyword-subfield",
        ),
        pytest.param(
            {"litellm-qa-docs": {"properties": {"year": {"type": "keyword"}}}},
            {"type": "gte", "key": "year", "value": 2026},
            "the index maps year as keyword, which compares numbers as text. Map year as a numeric field",
            id="number-range-on-a-keyword-field",
        ),
        pytest.param(
            {"litellm-qa-docs": {"properties": {"year": {"type": "text", "fields": {"keyword": {"type": "keyword"}}}}}},
            {"type": "lt", "key": "year", "value": 2026.5},
            "the index maps year as text, which compares numbers as text",
            id="number-range-on-text-with-a-keyword-subfield",
        ),
        pytest.param(
            TEXT_ONLY,
            {"type": "eq", "key": "tenant", "value": 7},
            "the index maps tenant as text with no keyword subfield",
            id="number-on-text-only",
        ),
        pytest.param(
            {"litellm-qa-docs": {"properties": {"title": {"type": "search_as_you_type"}}}},
            {"type": "eq", "key": "title", "value": "HR"},
            "the index maps title as search_as_you_type with no keyword subfield",
            id="search-as-you-type",
        ),
        pytest.param(
            {
                "litellm-qa-docs": {
                    "properties": {"tenant": {"type": "text"}, "org": {"type": "alias", "path": "tenant"}}
                }
            },
            {"type": "ne", "key": "org", "value": "acme"},
            "the index maps tenant as text with no keyword subfield",
            id="alias-to-text-only",
        ),
        pytest.param(
            {"litellm-qa-docs": {"properties": {"owner": {"type": "nested", "properties": NESTED_TEAM}}}},
            {"type": "eq", "key": "owner.team", "value": "people-ops"},
            "the index maps owner as nested, which needs a nested query. Map owner as object",
            id="field-inside-a-nested-object",
        ),
        pytest.param(
            {
                "litellm-qa-docs": {
                    "properties": {
                        "owner": {"type": "nested", "properties": NESTED_TEAM},
                        "team": {"type": "alias", "path": "owner.team"},
                    }
                }
            },
            {"type": "in", "key": "team", "value": ["people-ops"]},
            "the index maps owner as nested, which needs a nested query",
            id="alias-into-a-nested-object",
        ),
        pytest.param(
            {"litellm-qa-docs": {"properties": {"owner": {"type": "nested", "properties": NESTED_TEAM}}}},
            {"type": "ne", "key": "owner", "value": "people-ops"},
            "the index maps owner as nested, which holds fields rather than values. Filter on a field inside owner",
            id="the-nested-object-itself",
        ),
        pytest.param(
            {
                "litellm-qa-docs": {
                    "properties": {"org": {"properties": {"owner": {"type": "nested", "properties": NESTED_TEAM}}}}
                }
            },
            {"type": "eq", "key": "org.owner.team", "value": "people-ops"},
            "the index maps org.owner as nested, which needs a nested query. Map org.owner as object",
            id="nested-object-inside-a-plain-object",
        ),
        pytest.param(
            {
                "litellm-qa-docs": {
                    "properties": {
                        "org": {
                            "type": "nested",
                            "properties": {
                                "owner": {"type": "nested", "include_in_parent": True, "properties": NESTED_TEAM}
                            },
                        }
                    }
                }
            },
            {"type": "eq", "key": "org.owner.team", "value": "people-ops"},
            "the index maps org as nested, which needs a nested query. Map org as object",
            id="nested-object-copied-only-into-a-nested-parent",
        ),
        pytest.param(
            {
                "litellm-qa-docs": {
                    "properties": {
                        "org": {
                            "type": "nested",
                            "include_in_root": True,
                            "properties": {"owner": {"type": "nested", "properties": NESTED_TEAM}},
                        }
                    }
                }
            },
            {"type": "eq", "key": "org.owner.team", "value": "people-ops"},
            "the index maps org.owner as nested, which needs a nested query. Map org.owner as object",
            id="nested-object-inside-one-copied-to-the-root",
        ),
        pytest.param(
            {"litellm-qa-docs": {"properties": {"owner": {"properties": NESTED_TEAM}}}},
            {"type": "eq", "key": "owner", "value": "people-ops"},
            "the index maps owner as object, which holds fields rather than values. Filter on a field inside owner",
            id="an-object",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_filter_the_index_cannot_match_exactly_is_a_400_unless_drop_params_drops_the_filters(
    asynchronous: bool, mappings: Mapping[str, Mapping[str, object]], search_filter: Mapping[str, object], reason: str
) -> None:
    rejected: Final = MagicMock(return_value=_index_mappings(mappings))
    executor: Final = RecordingEmbeddingExecutor()

    with pytest.raises(openai.BadRequestError) as error:
        await _search(asynchronous, rejected, executor, {"filters": search_filter})

    assert reason in str(error.value)
    assert not _searches(rejected)
    executor.call.assert_not_called()

    dropped: Final = MagicMock(
        side_effect=[_index_mappings(mappings), httpx.Response(200, json=_opensearch_hits("text", "embedding"))]
    )
    response: Final = await _search(
        asynchronous, dropped, RecordingEmbeddingExecutor(), {"filters": search_filter, "drop_params": True}
    )

    assert response["data"] == EXPECTED_RESULTS
    (search,) = _searches(dropped)
    assert json.loads(search.content)["query"] == {"knn": {"embedding": {"vector": QUERY_VECTOR, "k": 10}}}


def _mapping_read_error(status: int) -> httpx.Response:
    return httpx.Response(status, json={"status": status, "error": {"reason": "no mapping for you", "type": "x"}})


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_a_refused_index_mapping_read_names_the_permission_it_needs_and_sends_no_search(
    asynchronous: bool,
) -> None:
    respond: Final = MagicMock(return_value=_mapping_read_error(403))
    executor: Final = RecordingEmbeddingExecutor()

    with pytest.raises(openai.APIStatusError) as error:
        await _search(asynchronous, respond, executor, {"filters": {"type": "eq", "key": "tenant", "value": "acme"}})

    assert error.value.status_code == 403
    assert "es:ESHttpGet" in str(error.value)
    assert "no mapping for you" not in str(error.value)
    assert not _searches(respond)
    executor.call.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("drop_params", [False, True])
@pytest.mark.parametrize(
    "mapping_read,status",
    [
        *(pytest.param(partial(_mapping_read_error, code), code, id=str(code)) for code in (404, 429, 500, 503)),
        pytest.param(partial(httpx.Response, 200, json={"acknowledged": True}), 502, id="unreadable"),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_index_mapping_read_keeps_the_filters_and_its_status_even_under_drop_params(
    asynchronous: bool, drop_params: bool, mapping_read: Callable[[], httpx.Response], status: int
) -> None:
    respond: Final = MagicMock(side_effect=lambda request: mapping_read())
    executor: Final = RecordingEmbeddingExecutor()

    with pytest.raises(openai.APIStatusError) as error:
        await _search(
            asynchronous,
            respond,
            executor,
            {"filters": {"type": "eq", "key": "tenant", "value": "acme"}, "drop_params": drop_params},
        )

    assert error.value.status_code == status
    assert "es:ESHttpGet" not in str(error.value)
    assert "no mapping for you" not in str(error.value)
    assert not _searches(respond)
    executor.call.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_a_refused_index_mapping_read_under_drop_params_drops_the_filters_and_searches_once(
    asynchronous: bool,
) -> None:
    respond: Final = MagicMock(
        side_effect=[_mapping_read_error(403), httpx.Response(200, json=_opensearch_hits("text", "embedding"))]
    )

    response: Final = await _search(
        asynchronous,
        respond,
        RecordingEmbeddingExecutor(),
        {"filters": {"type": "eq", "key": "tenant", "value": "acme"}, "drop_params": True},
    )

    assert response["data"] == EXPECTED_RESULTS
    (search,) = _searches(respond)
    assert json.loads(search.content)["query"] == {"knn": {"embedding": {"vector": QUERY_VECTOR, "k": 10}}}


def _rejection(reason: str) -> httpx.Response:
    return httpx.Response(
        400,
        json={
            "error": {"root_cause": [{"type": "query_shard_exception", "reason": reason}], "type": "x"},
            "status": 400,
        },
    )


ENGINE_REFUSES_FILTERS: Final = "failed to create query: Engine [NMSLIB] does not support filters"
KNN_PARSER_REFUSES_FILTERS: Final = "[knn] unknown token [START_OBJECT] after [filter]"
FILTERED_KNN: Final = {"knn": {"embedding": {"vector": QUERY_VECTOR, "k": 3, "filter": HR_EXACT}}}
UNFILTERED_KNN: Final = {"knn": {"embedding": {"vector": QUERY_VECTOR, "k": 3}}}


def _close_with_goaway(request: httpx.Request) -> httpx.Response:
    raise httpx.RemoteProtocolError("GOAWAY", request=request) from httpcore.RemoteProtocolError(ConnectionTerminated())


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_the_index_mapping_read_sends_the_caller_headers_and_retries_a_goaway(asynchronous: bool) -> None:
    answers: Final = iter(
        (
            _close_with_goaway,
            lambda request: _index_mappings({"litellm-qa-docs": DYNAMIC_MAPPING}),
            lambda request: httpx.Response(200, json=_opensearch_hits("text", "embedding")),
        )
    )
    respond: Final = MagicMock(side_effect=lambda request: next(answers)(request))

    response: Final = await _search(
        asynchronous,
        respond,
        RecordingEmbeddingExecutor(),
        {
            "filters": {"type": "eq", "key": "category", "value": "hr"},
            "max_num_results": 3,
            "extra_headers": {"x-gateway-key": "gateway-secret", "x-gateway-token": b"gateway-token"},
        },
    )

    assert response["data"] == EXPECTED_RESULTS
    goaway, mapping_read = _mapping_reads(respond)
    assert mapping_read.headers["x-gateway-key"] == "gateway-secret"
    assert mapping_read.headers["x-gateway-token"] == "gateway-token"
    assert _signature_scope(mapping_read) == ("aoss", "us-east-1")
    (search,) = _searches(respond)
    assert json.loads(search.content)["query"] == FILTERED_KNN


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("value", [None, 7, "café"])
@pytest.mark.asyncio
async def test_a_header_httpx_cannot_send_fails_a_filtered_search_like_an_unfiltered_one_before_embedding(
    asynchronous: bool, value: object
) -> None:
    respond: Final = _index(DYNAMIC_MAPPING, httpx.Response(200, json=_opensearch_hits("text", "embedding")))
    executor: Final = RecordingEmbeddingExecutor()
    headers: Final = {"extra_headers": {"x-trace-id": value}}

    with pytest.raises(openai.APIError) as unfiltered:
        await _search(asynchronous, respond, RecordingEmbeddingExecutor(), headers)
    with pytest.raises(openai.APIError) as filtered:
        await _search(
            asynchronous, respond, executor, {**headers, "filters": {"type": "eq", "key": "category", "value": "hr"}}
        )

    assert type(filtered.value) is type(unfiltered.value)
    assert str(filtered.value) == str(unfiltered.value)
    assert not _sent(respond)
    executor.call.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_a_connection_failure_on_the_index_mapping_read_raises_what_one_on_the_search_raises(
    asynchronous: bool,
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(openai.APIError) as on_search:
        await _search(asynchronous, MagicMock(side_effect=refuse), RecordingEmbeddingExecutor(), {})
    with pytest.raises(openai.APIError) as on_mapping_read:
        await _search(
            asynchronous,
            MagicMock(side_effect=refuse),
            RecordingEmbeddingExecutor(),
            {"filters": {"type": "eq", "key": "category", "value": "hr"}},
        )

    assert type(on_mapping_read.value) is type(on_search.value)
    assert getattr(on_mapping_read.value, "status_code", None) == getattr(on_search.value, "status_code", None)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_a_timeout_on_the_index_mapping_read_answers_408_like_one_on_the_search_without_embedding(
    asynchronous: bool,
) -> None:
    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    executor: Final = RecordingEmbeddingExecutor()
    with pytest.raises(openai.APIError) as on_search:
        await _search(asynchronous, MagicMock(side_effect=time_out), RecordingEmbeddingExecutor(), {})
    with pytest.raises(openai.APIError) as on_mapping_read:
        await _search(
            asynchronous,
            MagicMock(side_effect=time_out),
            executor,
            {"filters": {"type": "eq", "key": "category", "value": "hr"}},
        )

    assert getattr(on_search.value, "status_code", None) == 408
    assert type(on_mapping_read.value) is type(on_search.value)
    assert getattr(on_mapping_read.value, "status_code", None) == 408
    executor.call.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("refusal", [ENGINE_REFUSES_FILTERS, KNN_PARSER_REFUSES_FILTERS])
@pytest.mark.asyncio
async def test_a_filter_the_index_refuses_is_dropped_under_drop_params_without_embedding_again(
    asynchronous: bool, refusal: str
) -> None:
    respond: Final = _index(
        DYNAMIC_MAPPING, _rejection(refusal), httpx.Response(200, json=_opensearch_hits("text", "embedding"))
    )
    executor: Final = RecordingEmbeddingExecutor()

    response: Final = await _search(
        asynchronous,
        respond,
        executor,
        {"filters": {"type": "eq", "key": "category", "value": "hr"}, "drop_params": True, "max_num_results": 3},
    )

    assert response["data"] == EXPECTED_RESULTS
    sent: Final = _searches(respond)
    assert [json.loads(request.content)["query"] for request in sent] == [FILTERED_KNN, UNFILTERED_KNN]
    assert _signature_scope(sent[1]) == ("aoss", "us-east-1")
    executor.call.assert_called_once_with("embedding-alias", QUERY, {})


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("refusal", [ENGINE_REFUSES_FILTERS, KNN_PARSER_REFUSES_FILTERS])
@pytest.mark.asyncio
async def test_a_filter_the_index_refuses_without_drop_params_is_a_400_that_hides_the_opensearch_error(
    asynchronous: bool, refusal: str
) -> None:
    respond: Final = _index(DYNAMIC_MAPPING, _rejection(refusal))

    with pytest.raises(openai.BadRequestError) as error:
        await _search(
            asynchronous,
            respond,
            RecordingEmbeddingExecutor(),
            {"filters": {"type": "eq", "key": "category", "value": "hr"}, "max_num_results": 3},
        )

    assert "cannot apply filters inside a k-NN query" in str(error.value)
    assert "set drop_params to drop them" in str(error.value)
    assert refusal not in str(error.value)
    assert "query_shard_exception" not in str(error.value)
    assert len(_searches(respond)) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "drop_params,rejections,sent_queries,reason",
    [
        pytest.param(
            {"drop_params": True},
            ["failed to create query: unknown field [category]"],
            [FILTERED_KNN],
            "unknown field",
            id="other-400",
        ),
        pytest.param(
            {"drop_params": True},
            [ENGINE_REFUSES_FILTERS, "index is read-only"],
            [FILTERED_KNN, UNFILTERED_KNN],
            "index is read-only",
            id="retry-rejected",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_rejected_filtered_search_is_a_400_unless_drop_params_can_drop_the_filter(
    asynchronous: bool,
    drop_params: Mapping[str, object],
    rejections: list[str],
    sent_queries: list[Mapping[str, object]],
    reason: str,
) -> None:
    respond: Final = _index(DYNAMIC_MAPPING, *(_rejection(reason) for reason in rejections))

    with pytest.raises(openai.BadRequestError) as error:
        await _search(
            asynchronous,
            respond,
            RecordingEmbeddingExecutor(),
            {"filters": {"type": "eq", "key": "category", "value": "hr"}, "max_num_results": 3, **drop_params},
        )

    assert reason in str(error.value)
    assert [json.loads(request.content)["query"] for request in _searches(respond)] == sent_queries


HYBRID: Final = {
    "opensearch_hybrid_search": True,
    "opensearch_search_pipeline": "hybrid norm",
    "opensearch_text_field": "passage",
}
KEYWORD_QUERY: Final = {"match": {"passage": {"query": QUERY}}}
HR_ONLY: Final = {"type": "eq", "key": "category", "value": "hr"}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "filters,queries",
    [
        pytest.param({}, [KEYWORD_QUERY, {"knn": {"embedding": {"vector": QUERY_VECTOR, "k": 3}}}], id="unfiltered"),
        pytest.param(
            {"filters": HR_ONLY},
            [
                {"bool": {"must": [KEYWORD_QUERY], "filter": [HR_EXACT]}},
                {"knn": {"embedding": {"vector": QUERY_VECTOR, "k": 3, "filter": HR_EXACT}}},
            ],
            id="filtered",
        ),
    ],
)
@pytest.mark.asyncio
async def test_hybrid_search_sends_a_keyword_and_a_knn_query_through_the_signed_search_pipeline(
    asynchronous: bool, filters: Mapping[str, object], queries: list[Mapping[str, object]]
) -> None:
    executor: Final = RecordingEmbeddingExecutor()
    respond: Final = _index(DYNAMIC_MAPPING, httpx.Response(200, json=_opensearch_hits("passage", "embedding")))

    response: Final = await _search(asynchronous, respond, executor, {**HYBRID, **filters, "max_num_results": 3})

    assert response["data"] == EXPECTED_RESULTS
    sent: Final = _searches(respond)
    assert len(sent) == 1
    assert str(sent[0].url) == f"{SERVERLESS}/litellm-qa-docs/_search?search_pipeline=hybrid%20norm"
    assert json.loads(sent[0].content) == {
        "size": 3,
        "query": {"hybrid": {"queries": queries}},
        "_source": {"excludes": ["embedding"]},
    }
    assert _signature_scope(sent[0]) == ("aoss", "us-east-1")
    executor.call.assert_called_once_with("embedding-alias", QUERY, {})


@pytest.mark.parametrize("hybrid", [{}, {"opensearch_hybrid_search": False}])
@pytest.mark.asyncio
async def test_a_search_pipeline_without_hybrid_search_runs_the_knn_query_through_it(
    hybrid: Mapping[str, object],
) -> None:
    respond: Final = MagicMock(return_value=httpx.Response(200, json={"hits": {"hits": []}}))

    await _search(True, respond, RecordingEmbeddingExecutor(), {**hybrid, "opensearch_search_pipeline": "hybrid-norm"})

    sent: Final = _sent(respond)
    assert str(sent[0].url) == f"{SERVERLESS}/litellm-qa-docs/_search?search_pipeline=hybrid-norm"
    assert json.loads(sent[0].content)["query"] == {"knn": {"embedding": {"vector": QUERY_VECTOR, "k": 10}}}


def test_create_is_rejected_because_the_store_is_search_only() -> None:
    with pytest.raises(litellm.BadRequestError, match="search-only"):
        litellm.vector_stores.create(custom_llm_provider="opensearch")


async def _store_operation(
    asynchronous: bool, operation: str, respond: Callable[[httpx.Request], httpx.Response]
) -> object:
    request_kwargs: Final = {**BASE_PARAMS, "vector_store_id": "litellm-qa-docs", "custom_llm_provider": "opensearch"}
    if asynchronous:
        return await getattr(vector_stores, f"a{operation}")(
            **request_kwargs, client=AsyncHTTPHandler(transport=httpx.MockTransport(respond))
        )
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        return getattr(vector_stores, operation)(**request_kwargs, client=HTTPHandler(client=transport))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("operation", ["retrieve", "update", "delete"])
@pytest.mark.asyncio
async def test_store_operations_besides_search_send_nothing_to_the_index(asynchronous: bool, operation: str) -> None:
    respond: Final = MagicMock(return_value=httpx.Response(200, json={"acknowledged": True}))

    with pytest.raises(litellm.BadRequestError, match="search-only") as error:
        await _store_operation(asynchronous, operation, respond)

    assert error.value.status_code == 400
    respond.assert_not_called()


async def _list_stores(asynchronous: bool, respond: Callable[[httpx.Request], httpx.Response]) -> object:
    request_kwargs: Final = {**BASE_PARAMS, "custom_llm_provider": "opensearch"}
    if asynchronous:
        return await vector_stores.alist(
            **request_kwargs, client=AsyncHTTPHandler(transport=httpx.MockTransport(respond))
        )
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        return vector_stores.list(**request_kwargs, client=HTTPHandler(client=transport))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_listing_stores_sends_nothing_to_the_endpoint(asynchronous: bool) -> None:
    respond: Final = MagicMock(return_value=httpx.Response(200, json={"cluster_name": "docs", "tagline": "OpenSearch"}))

    with pytest.raises(litellm.BadRequestError, match="search-only") as error:
        await _list_stores(asynchronous, respond)

    assert error.value.status_code == 400
    respond.assert_not_called()


@pytest.mark.parametrize(
    "response,status",
    [
        (httpx.Response(403, json={"status": 403, "error": {"type": "Forbidden"}}), 403),
        (httpx.Response(200, json={"took": 1}), 502),
        (httpx.Response(200, text="<html>gateway</html>"), 502),
    ],
)
def test_response_without_search_hits_is_an_error_not_an_empty_page(response: httpx.Response, status: int) -> None:
    logging: Final = Logging(
        model="",
        messages=[],
        stream=False,
        call_type="vector_store_search",
        start_time=datetime(2026, 10, 20, tzinfo=timezone.utc),
        litellm_call_id="opensearch-test",
        function_id="opensearch-test",
    )

    with pytest.raises(BaseLLMException) as error:
        OpenSearchVectorStoreConfig().transform_search_vector_store_response(response, logging)

    assert error.value.status_code == status
