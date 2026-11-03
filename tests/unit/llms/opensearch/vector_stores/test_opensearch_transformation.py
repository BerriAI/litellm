import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from botocore.exceptions import CredentialRetrievalError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
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


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"filters": {"type": "eq", "key": "category", "value": "hr"}}, "filters"),
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
    for name in _AMBIENT_AWS_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in ("AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE", "BOTO_CONFIG"):
        config_file: Path = tmp_path / name
        config_file.write_text("")
        monkeypatch.setenv(name, str(config_file))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
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
        {"filters": {"type": "eq", "key": "category", "value": "hr"}},
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
