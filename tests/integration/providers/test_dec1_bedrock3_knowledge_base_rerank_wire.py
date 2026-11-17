"""Bedrock knowledge base retrieval through `/v1/vector_stores/{id}/search`, and Bedrock rerank through `/v1/rerank`.

A vector store registered with `custom_llm_provider: bedrock` is a Bedrock knowledge base. A search
must become one SigV4 signed `POST /knowledgebases/<id>/retrieve` on the Agents runtime with
`retrievalQuery.text` (or an IMAGE query), `retrievalConfiguration.vectorSearchConfiguration.numberOfResults`
from `max_num_results`, and OpenAI vector store `filters` rewritten into AWS `RetrievalFilter` operators
(`equals`, `greaterThanOrEquals`, `in`, `andAll`, `orAll`, ...), as landed in #51291
(`litellm/llms/bedrock/vector_stores/transformation.py:56`). Search by image landed in #51063: a
`query_image` turns into an IMAGE `retrievalQuery`, and only then are IMAGE hits returned. The answer is
an OpenAI `vector_store.search_results.page`.

A rerank deployment (`bedrock/<rerank model ARN>`) must send one SigV4 `POST /rerank` with
`queries[0].textQuery`, `sources[]` (TEXT for strings, JSON for objects) and
`rerankingConfiguration.bedrockRerankingConfiguration` naming the model ARN and `numberOfResults`
(`litellm/llms/bedrock/rerank/transformation.py:59`), and map `relevanceScore` back to `relevance_score`.
Existing files pin the user context field and the forwarded header signing; these tests cover the rest.
"""

from __future__ import annotations

import base64
import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_bedrock3_support import (
    KB_ACCESS_KEY,
    json_body,
    json_reply,
    knowledge_base,
    only_posts,
    response_cost,
    retrieve_reply,
    text_hit,
    unique_text,
)
from integration.providers.dec1_bedrock_support import aws_error, sigv4_scope
from pydantic import JsonValue

PNG_BASE64: Final = base64.b64encode(b"\x89PNG\r\n\x1a\nsynthetic").decode()
RERANK_ARN: Final = "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"
RERANK_ACCESS_KEY: Final = "AKIADEC1BEDROCK3RR01"


def _hit(_: Request) -> Reply:
    return retrieve_reply(text_hit("permitted text", 0.81, "s3://synthetic-bucket/docs/guide.pdf"))


def _filter_sent(request: Request) -> JsonValue:
    return json_body(request)["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]


# ------------------------------------------------------------------ Knowledge base request shape


def test_a_search_posts_one_signed_retrieve_with_the_text_query_and_result_count(gateway: Gateway) -> None:
    """A search with `query` and `max_num_results: 4` on a Bedrock knowledge base.

    Bedrock must receive one `POST /knowledgebases/<id>/retrieve`, SigV4 signed with the store's access key
    for `us-east-1` and service `bedrock`, whose body is exactly
    `{"retrievalQuery": {"text": <query>}, "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": 4}}}`.
    """
    query: Final = unique_text()
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(query=query, max_num_results=4)
        assert response.status_code == 200, response.text
        sent: Final = kb.one_post()
        assert sent.target == f"/knowledgebases/{kb.identity}/retrieve", sent.target
        assert sigv4_scope(sent) == (KB_ACCESS_KEY, "us-east-1", "bedrock")
        assert json_body(sent) == {
            "retrievalQuery": {"text": query},
            "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": 4}},
        }


def test_a_search_without_options_sends_no_retrieval_configuration(gateway: Gateway) -> None:
    """A search with only `query`.

    The body must be exactly `{"retrievalQuery": {"text": <query>}}`: an empty `retrievalConfiguration` is
    left out (`transformation.py:427`), since Bedrock validates its members.
    """
    query: Final = unique_text()
    with knowledge_base(gateway, _hit) as kb:
        assert kb.search(query=query).status_code == 200
        assert json_body(kb.one_post()) == {"retrievalQuery": {"text": query}}


def test_a_list_query_is_joined_with_spaces(gateway: Gateway) -> None:
    """`query: ["alpha", "beta"]` (a list).

    Bedrock takes one text, so the proxy must send `retrievalQuery.text == "alpha beta"`.
    """
    with knowledge_base(gateway, _hit) as kb:
        assert kb.search(query=["alpha", "beta"]).status_code == 200
        assert json_body(kb.one_post())["retrievalQuery"] == {"text": "alpha beta"}


@pytest.mark.parametrize(
    ("openai_filter", "aws_filter"),
    [
        ({"type": "eq", "key": "genre", "value": "guide"}, {"equals": {"key": "genre", "value": "guide"}}),
        ({"type": "ne", "key": "genre", "value": "spam"}, {"notEquals": {"key": "genre", "value": "spam"}}),
        ({"type": "gte", "key": "year", "value": 2024}, {"greaterThanOrEquals": {"key": "year", "value": 2024}}),
        ({"type": "lt", "key": "year", "value": 2026}, {"lessThan": {"key": "year", "value": 2026}}),
        ({"type": "in", "key": "tag", "value": ["a", "b"]}, {"in": {"key": "tag", "value": ["a", "b"]}}),
        ({"type": "nin", "key": "tag", "value": ["x"]}, {"notIn": {"key": "tag", "value": ["x"]}}),
    ],
    ids=["eq", "ne", "gte", "lt", "in_strings", "nin_strings"],
)
def test_an_openai_comparison_filter_becomes_the_aws_operator(
    gateway: Gateway, openai_filter: dict[str, JsonValue], aws_filter: dict[str, JsonValue]
) -> None:
    """An OpenAI vector store comparison filter is sent as `filters`.

    It must reach Bedrock as `retrievalConfiguration.vectorSearchConfiguration.filter` with the AWS operator
    name from `_AWS_COMPARISON_OPERATORS` (`eq` to `equals`, `gte` to `greaterThanOrEquals`, `nin` to `notIn`).
    """
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(query=unique_text(), filters=openai_filter)
        assert response.status_code == 200, response.text
        assert _filter_sent(kb.one_post()) == aws_filter


def test_an_openai_and_filter_becomes_and_all(gateway: Gateway) -> None:
    """`{"type": "and", "filters": [eq genre guide, gte year 2024]}`.

    Bedrock must receive `{"andAll": [{"equals": {...genre...}}, {"greaterThanOrEquals": {...year...}}]}` in order.
    """
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(
            query=unique_text(),
            filters={
                "type": "and",
                "filters": [
                    {"type": "eq", "key": "genre", "value": "guide"},
                    {"type": "gte", "key": "year", "value": 2024},
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert _filter_sent(kb.one_post()) == {
            "andAll": [
                {"equals": {"key": "genre", "value": "guide"}},
                {"greaterThanOrEquals": {"key": "year", "value": 2024}},
            ]
        }


def test_a_one_member_compound_filter_is_unwrapped(gateway: Gateway) -> None:
    """`{"type": "or", "filters": [eq genre guide]}` has one member.

    AWS requires `orAll` to hold at least two members, so the proxy must send just
    `{"equals": {"key": "genre", "value": "guide"}}` (`_aws_logical_filter`, `transformation.py:93`).
    """
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(
            query=unique_text(), filters={"type": "or", "filters": [{"type": "eq", "key": "genre", "value": "guide"}]}
        )
        assert response.status_code == 200, response.text
        assert _filter_sent(kb.one_post()) == {"equals": {"key": "genre", "value": "guide"}}


def test_a_numeric_in_filter_expands_to_or_all_of_equals(gateway: Gateway) -> None:
    """`{"type": "in", "key": "year", "value": [2024, 2025]}` holds numbers.

    AWS `in` takes only strings, so the proxy must expand it to
    `{"orAll": [{"equals": {"key": "year", "value": 2024}}, {"equals": {"key": "year", "value": 2025}}]}`.
    """
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(query=unique_text(), filters={"type": "in", "key": "year", "value": [2024, 2025]})
        assert response.status_code == 200, response.text
        assert _filter_sent(kb.one_post()) == {
            "orAll": [{"equals": {"key": "year", "value": 2024}}, {"equals": {"key": "year", "value": 2025}}]
        }


def test_a_legacy_operator_filter_is_still_mapped(gateway: Gateway) -> None:
    """The older shape `{"key": "genre", "value": "guide", "operator": "eq"}` is sent.

    `_map_operator_filter` must still produce `{"equals": {"key": "genre", "value": "guide"}}`.
    """
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(query=unique_text(), filters={"key": "genre", "value": "guide", "operator": "eq"})
        assert response.status_code == 200, response.text
        assert _filter_sent(kb.one_post()) == {"equals": {"key": "genre", "value": "guide"}}


def test_an_aws_native_filter_passes_through_unchanged(gateway: Gateway) -> None:
    """An AWS `RetrievalFilter` such as `{"startsWith": {"key": "path", "value": "docs/"}}` is sent as `filters`.

    It has no OpenAI `type` and names an AWS operator, so Bedrock must receive it byte for byte.
    """
    native: Final[dict[str, JsonValue]] = {"startsWith": {"key": "path", "value": "docs/"}}
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(query=unique_text(), filters=native)
        assert response.status_code == 200, response.text
        assert _filter_sent(kb.one_post()) == native


def test_an_invalid_openai_filter_is_a_400_before_bedrock(gateway: Gateway) -> None:
    """`{"type": "eq", "value": "guide"}` lacks `key`.

    Validation fails, so the caller must get HTTP 400 whose message contains
    `Invalid filters for the Bedrock vector store`, and Bedrock receives no POST.
    """
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(query=unique_text(), filters={"type": "eq", "value": "guide"})
        assert response.status_code == 400, response.text
        assert "Invalid filters for the Bedrock vector store" in response.text
        assert only_posts(kb.wire) == ()


def test_an_invalid_filter_under_drop_params_is_dropped_and_the_search_runs(gateway: Gateway) -> None:
    """The same invalid filter with `drop_params: true` on the request.

    `_invalid_filters` logs and returns nothing, so Bedrock must receive a retrieve with no `filter` member.
    UNVERIFIED that a request level `drop_params` reaches `map_openai_params` on the vector store route.
    """
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(query=unique_text(), filters={"type": "eq", "value": "guide"}, drop_params=True)
        assert response.status_code == 200, response.text
        body: Final = json_body(kb.one_post())
        assert "filter" not in json.dumps(body.get("retrievalConfiguration", {})), body


def test_a_caller_retrieval_query_is_refused_with_400(gateway: Gateway) -> None:
    """The caller sends a raw `retrievalQuery` object beside `query`.

    Bedrock vector stores take `query` or `query_image` only, so the caller must get HTTP 400 whose message
    starts `bedrock vector stores do not take retrievalQuery` and Bedrock receives no POST.
    """
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(query=unique_text(), retrievalQuery={"text": "smuggled"})
        assert response.status_code == 400, response.text
        assert "bedrock vector stores do not take retrievalQuery" in response.text
        assert only_posts(kb.wire) == ()


# ------------------------------------------------------------------ Knowledge base answers


def test_a_text_hit_becomes_an_openai_search_result_page(gateway: Gateway) -> None:
    """Bedrock answers one text hit (score 0.81, source `s3://synthetic-bucket/docs/guide.pdf`).

    The caller must get `object: vector_store.search_results.page`, `search_query` equal to the query, and
    one result with `content: [{"text": "permitted text", "type": "text"}]`, `file_id` the source URI,
    `filename: guide.pdf` and `score: 0.81`.
    """
    query: Final = unique_text()
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(query=query)
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["object"] == "vector_store.search_results.page"
        assert body["search_query"] == query
        (result,) = body["data"]
        assert result["content"] == [{"text": "permitted text", "type": "text"}]
        assert (result["file_id"], result["filename"], result["score"]) == (
            "s3://synthetic-bucket/docs/guide.pdf",
            "guide.pdf",
            0.81,
        )


def test_a_hit_without_a_source_uri_gets_chunk_and_data_source_ids(gateway: Gateway) -> None:
    """Bedrock answers a text hit whose metadata has only `x-amz-bedrock-kb-chunk-id: c9` and
    `x-amz-bedrock-kb-data-source-id: ds1`.

    The result must carry `file_id: bedrock-kb-c9` and `filename: bedrock-kb-document-ds1`.
    """

    def respond(_: Request) -> Reply:
        return retrieve_reply(
            {
                "content": {"text": "t", "type": "TEXT"},
                "score": 0.5,
                "metadata": {"x-amz-bedrock-kb-chunk-id": "c9", "x-amz-bedrock-kb-data-source-id": "ds1"},
            }
        )

    with knowledge_base(gateway, respond) as kb:
        response: Final = kb.search(query=unique_text())
        assert response.status_code == 200, response.text
        (result,) = response.json()["data"]
        assert (result["file_id"], result["filename"]) == ("bedrock-kb-c9", "bedrock-kb-document-ds1")


def test_a_query_image_becomes_an_image_retrieval_query(gateway: Gateway) -> None:
    """The caller sends `query_image: {"media_type": "image/png", "data": <base64>}` with a text `query`.

    Bedrock must receive `retrievalQuery == {"type": "IMAGE", "image": {"format": "png", "inlineContent": <base64>}}`
    (`_retrieval_query`, `transformation.py:183`), the text query not being sent.
    """
    with knowledge_base(gateway, _hit) as kb:
        response: Final = kb.search(
            query="similar pictures", query_image={"media_type": "image/png", "data": PNG_BASE64}
        )
        assert response.status_code == 200, response.text
        assert json_body(kb.one_post())["retrievalQuery"] == {
            "type": "IMAGE",
            "image": {"format": "png", "inlineContent": PNG_BASE64},
        }


def _image_and_text_hits(_: Request) -> Reply:
    return retrieve_reply(
        {
            "content": {"type": "IMAGE", "byteContent": f"data:image/png;base64,{PNG_BASE64}"},
            "score": 0.9,
            "metadata": {"x-amz-bedrock-kb-source-uri": "s3://synthetic-bucket/img/cat.png"},
        },
        text_hit("caption text", 0.4, "s3://synthetic-bucket/docs/captions.txt"),
    )


def test_an_image_search_returns_image_hits_as_base64_sources(gateway: Gateway) -> None:
    """With `query_image` set, Bedrock answers an IMAGE hit (`byteContent` a PNG data URI) and a text hit.

    Both must come back, the image as `{"type": "image", "source": {"type": "base64", "media_type": "image/png",
    "data": <base64 without the data URI prefix>}}`.
    """
    with knowledge_base(gateway, _image_and_text_hits) as kb:
        response: Final = kb.search(query="pics", query_image={"media_type": "image/png", "data": PNG_BASE64})
        assert response.status_code == 200, response.text
        data: Final = response.json()["data"]
        assert len(data) == 2, data
        assert data[0]["content"] == [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64}}
        ]


def test_a_text_search_drops_image_hits(gateway: Gateway) -> None:
    """The same mixed answer, but the caller searched by text only.

    `_result_content` skips IMAGE content unless the request had `query_image`, so only the text hit
    (`caption text`) may come back.
    """
    with knowledge_base(gateway, _image_and_text_hits) as kb:
        response: Final = kb.search(query=unique_text())
        assert response.status_code == 200, response.text
        data: Final = response.json()["data"]
        assert [result["content"] for result in data] == [[{"text": "caption text", "type": "text"}]]


def test_a_knowledge_base_access_denied_keeps_its_403(gateway: Gateway) -> None:
    """Bedrock answers the retrieve with HTTP 403, `x-amzn-ErrorType: AccessDeniedException` and a message only body.

    The caller must get HTTP 403, not a 500, so a missing `bedrock:Retrieve` grant is diagnosable.
    UNVERIFIED: the vector store error path maps through `BedrockError` with the upstream status.
    """
    with knowledge_base(
        gateway,
        lambda _: aws_error(403, "AccessDeniedException", "User is not authorized to perform: bedrock:Retrieve"),
    ) as kb:
        response: Final = kb.search(query=unique_text())
        assert response.status_code == 403, response.text


# ------------------------------------------------------------------ Rerank


def _rerank_reply(_: Request) -> Reply:
    return json_reply({"results": [{"index": 1, "relevanceScore": 0.75}, {"index": 0, "relevanceScore": 0.25}]})


def _rerank(gateway: Gateway, respond, **body: JsonValue):  # noqa: ANN001, ANN202
    """Register a rerank deployment on a new peer and send one `/v1/rerank`; returns (response, posts)."""
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/{RERANK_ARN}",
            api_key=None,
            api_base=None,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            aws_access_key_id=RERANK_ACCESS_KEY,
            aws_secret_access_key="synthetic-rerank-secret-dec1",
        )
        response: Final = gateway.request("POST", "/v1/rerank", {"model": model, **body})
        return response, only_posts(wire)


def test_rerank_object_documents_become_inline_json_sources(gateway: Gateway) -> None:
    """Two object documents `{"title": "a"}` and `{"title": "b"}` are reranked with `top_n: 1`.

    Bedrock must receive `sources == [{"inlineDocumentSource": {"jsonDocument": {"title": "a"}, "type": "JSON"},
    "type": "INLINE"}, ...]` and `numberOfResults: 1`.
    """
    response, posts = _rerank(gateway, _rerank_reply, query="pick", documents=[{"title": "a"}, {"title": "b"}], top_n=1)
    assert response.status_code == 200, response.text
    (sent,) = posts
    body: Final = json_body(sent)
    assert body["sources"] == [
        {"inlineDocumentSource": {"jsonDocument": {"title": "a"}, "type": "JSON"}, "type": "INLINE"},
        {"inlineDocumentSource": {"jsonDocument": {"title": "b"}, "type": "JSON"}, "type": "INLINE"},
    ]
    assert body["rerankingConfiguration"]["bedrockRerankingConfiguration"]["numberOfResults"] == 1  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]


def test_rerank_without_top_n_asks_for_every_document(gateway: Gateway) -> None:
    """Three string documents and no `top_n`.

    `numberOfResults` must be 3 (`top_n or len(documents)`), `rerankingConfiguration.type` must be
    `BEDROCK_RERANKING_MODEL`, and the model ARN must be `modelConfiguration.modelArn`.
    """
    response, posts = _rerank(gateway, _rerank_reply, query="pick", documents=["one", "two", "three"])
    assert response.status_code == 200, response.text
    configuration: Final = json_body(posts[0])["rerankingConfiguration"]
    assert configuration["type"] == "BEDROCK_RERANKING_MODEL"  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
    assert configuration["bedrockRerankingConfiguration"] == {  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        "modelConfiguration": {"modelArn": RERANK_ARN},
        "numberOfResults": 3,
    }


def test_rerank_is_signed_with_sigv4_even_when_a_bearer_style_key_is_configured(gateway: Gateway) -> None:
    """The rerank deployment has static keys; rerank never uses Bearer auth (`supports_bearer_token=False`).

    The request must carry a SigV4 `Authorization` with the deployment's access key, region `us-east-1` and
    service `bedrock`.
    """
    response, posts = _rerank(gateway, _rerank_reply, query="pick", documents=["one", "two"])
    assert response.status_code == 200, response.text
    assert sigv4_scope(posts[0]) == (RERANK_ACCESS_KEY, "us-east-1", "bedrock")


def test_rerank_answer_without_usage_bills_one_search_unit(gateway: Gateway) -> None:
    """Bedrock's rerank answer has `results` but no `usage`.

    `meta.billed_units.search_units` must be 1 (the `{"search_units": 1}` default), the results mapped to
    `relevance_score`, and the cost header must be one query at the `cohere.rerank-v3-5:0` row's
    `input_cost_per_query` of 0.002. UNVERIFIED: the cost header for rerank on this route.
    """
    response, _ = _rerank(gateway, _rerank_reply, query="pick", documents=["one", "two"])
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert body["meta"]["billed_units"]["search_units"] == 1
    assert body["results"] == [{"index": 1, "relevance_score": 0.75}, {"index": 0, "relevance_score": 0.25}]
    assert response_cost(response) == pytest.approx(0.002)


def test_a_rerank_validation_error_keeps_its_400(gateway: Gateway) -> None:
    """Bedrock answers the rerank with HTTP 400 `ValidationException` `numberOfResults must be <= 1000`.

    The handler raises `BedrockError` with the upstream status, so the caller must get HTTP 400 carrying
    that text.
    """
    response, posts = _rerank(
        gateway,
        lambda _: aws_error(400, "ValidationException", "numberOfResults must be <= 1000"),
        query="pick",
        documents=["one"],
    )
    assert response.status_code == 400, response.text
    assert "numberOfResults must be <= 1000" in response.text
    assert len(posts) == 1
