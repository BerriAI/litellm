from typing import Final

import anthropic
import openai
import pytest
from integration._support.client import Gateway, Scenario, list_value, object_value
from integration._support.wire import Reply, Request, wire_server
from integration.providers._api_base_trailing_slash_support import (
    KEY,
    TRAILING,
    TRAILING_IDS,
    VECTOR,
    Row,
    assert_control_answers,
    assert_one_request,
    body_of,
    deployment,
    first_item,
    new_marker,
    provider_peer,
    reply_text,
    request_json,
    result_url,
)
from pydantic import JsonValue

_CHAT_ROWS: Final = (
    Row("deepseek/deepseek-chat", "/ds", "/chat/completions"),
    Row("heroku/claude-4-sonnet", "/hk", "/v1/chat/completions"),
    Row("gradient_ai/llama3.3-70b-instruct", "/gr", "/api/v1/chat/completions", drops_params=True),
    Row("gigachat/GigaChat", "/gc/v1", "/chat/completions", token=True),
    Row("ollama_chat/llama3.1", "/ol", "/api/chat"),
    Row("ollama/llama3.1", "/ol", "/api/generate"),
)
_BRIDGED_ROWS: Final = (
    Row("deepseek/deepseek-chat", "/ds", "/chat/completions"),
    Row("ollama_chat/llama3.1", "/ol", "/api/chat"),
)
_MESSAGES_ROWS: Final = (
    Row("deepseek/deepseek-chat", "/ds", "/anthropic/v1/messages"),
    Row("heroku/claude-4-sonnet", "/hk", "/v1/chat/completions"),
    Row("ollama_chat/llama3.1", "/ol", "/api/chat"),
)
_EMBEDDING_ROWS: Final = (
    Row("ollama/nomic-embed-text", "/ol", "/api/embed", keyed=False),
    Row("jina_ai/jina-embeddings-v3", "/jn/v1", "/embeddings"),
    Row("voyage/voyage-3", "/vy/v1", "/embeddings"),
    Row("voyage/voyage-multimodal-3", "/vy/v1", "/multimodalembeddings"),
    Row("perplexity/pplx-embed-v1-0.6b", "/px", "/v1/embeddings"),
    Row("volcengine/doubao-embedding", "/ve/api/v3", "/embeddings"),
)
_SEARCH_ROWS: Final = (
    Row("perplexity", "/px", "/search"),
    Row("tavily", "/tv", "/search"),
    Row("exa_ai", "/ex", "/search"),
    Row("firecrawl", "/fc", "/search"),
    Row("linkup", "/lk", "/search"),
    Row("fastcrw", "/cr", "/search"),
)
_FULL_ROUTE_ROWS: Final = (
    Row("deepseek/deepseek-chat", "/ds/chat/completions", ""),
    Row("ollama_chat/llama3.1", "/ol/api/chat", ""),
)
_FULL_ROUTE_EMBEDDING_ROWS: Final = (
    Row("jina_ai/jina-embeddings-v3", "/jn/v1/embeddings", ""),
    Row("voyage/voyage-3", "/vy/v1/embeddings", ""),
    Row("perplexity/pplx-embed-v1-0.6b", "/px/v1/embeddings", ""),
)


def _ids(rows: tuple[Row, ...]) -> tuple[str, ...]:
    return tuple(f"{row.model}{row.route}" for row in rows)


def _openai(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _chat(gateway: Gateway, model: str, marker: str) -> str | None:
    completion: Final = _openai(gateway).chat.completions.create(
        model=model, messages=[{"role": "user", "content": marker}]
    )
    return completion.choices[0].message.content


def _embedding(gateway: Gateway, model: str, marker: str) -> JsonValue:
    response: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": marker})
    return first_item(body_of(response)["data"])["embedding"]


def _search_tool(gateway: Gateway, scenario: Scenario, provider: str, api_base: str) -> str:
    name: Final = f"slash-tool-{new_marker()}"
    created: Final = gateway.post(
        "/search_tools",
        {
            "search_tool": {
                "search_tool_name": name,
                "litellm_params": {"search_provider": provider, "api_key": KEY, "api_base": api_base},
            }
        },
    )
    scenario.cleanups.callback(gateway.request, "DELETE", f"/search_tools/{created['search_tool_id']}")
    return name


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
@pytest.mark.parametrize("row", _CHAT_ROWS, ids=_ids(_CHAT_ROWS))
def test_chat_completion_reaches_one_single_slash_url(gateway: Gateway, row: Row, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(provider_peer) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, row, f"{wire.url}{row.base}{trailing}")
        assert _chat(gateway, model, marker) == reply_text(marker)
        assert_one_request(wire.drain(), row.target, marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
@pytest.mark.parametrize("row", _BRIDGED_ROWS, ids=_ids(_BRIDGED_ROWS))
def test_streamed_chat_reaches_one_single_slash_url(gateway: Gateway, row: Row, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(provider_peer) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, row, f"{wire.url}{row.base}{trailing}")
        stream: Final = _openai(gateway).chat.completions.create(
            model=model, messages=[{"role": "user", "content": marker}], stream=True
        )
        text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
        assert text == reply_text(marker)
        assert request_json(assert_one_request(wire.drain(), row.target, marker))["stream"] is True


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
@pytest.mark.parametrize("row", _MESSAGES_ROWS, ids=_ids(_MESSAGES_ROWS))
def test_messages_reaches_one_single_slash_url(gateway: Gateway, row: Row, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(provider_peer) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, row, f"{wire.url}{row.base}{trailing}")
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
        message: Final = client.messages.create(
            model=model, max_tokens=16, messages=[{"role": "user", "content": marker}]
        )
        assert [block.text for block in message.content if block.type == "text"] == [reply_text(marker)]
        assert_one_request(wire.drain(), row.target, marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
@pytest.mark.parametrize("row", _BRIDGED_ROWS, ids=_ids(_BRIDGED_ROWS))
def test_responses_bridge_reaches_one_single_slash_url(gateway: Gateway, row: Row, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(provider_peer) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, row, f"{wire.url}{row.base}{trailing}")
        response: Final = _openai(gateway).responses.create(model=model, input=marker)
        assert response.output_text == reply_text(marker)
        assert_one_request(wire.drain(), row.target, marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
@pytest.mark.parametrize("row", _EMBEDDING_ROWS, ids=_ids(_EMBEDDING_ROWS))
def test_embedding_reaches_one_single_slash_url(gateway: Gateway, row: Row, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(provider_peer) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, row, f"{wire.url}{row.base}{trailing}")
        assert _embedding(gateway, model, marker) == list(VECTOR)
        assert_one_request(wire.drain(), row.target, marker, keyed=row.keyed)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
def test_voyage_contextual_embedding_reaches_one_single_slash_url(gateway: Gateway, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(provider_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="voyage/voyage-context-3", api_base=f"{wire.url}/vy/v1{trailing}", api_key=KEY
        )
        response: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": [[marker, "chunk b"]]})
        assert body_of(response)["data"] == [
            {"object": "list", "index": 0, "data": [{"object": "embedding", "index": 0, "embedding": list(VECTOR)}]}
        ]
        assert_one_request(wire.drain(), "/vy/v1/contextualizedembeddings", marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
@pytest.mark.parametrize("row", _SEARCH_ROWS, ids=_ids(_SEARCH_ROWS))
def test_search_reaches_one_single_slash_url(gateway: Gateway, row: Row, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(provider_peer) as wire, gateway.scenario() as scenario:
        tool: Final = _search_tool(gateway, scenario, row.model, f"{wire.url}{row.base}{trailing}")
        response: Final = gateway.request("POST", f"/v1/search/{tool}", {"query": marker})
        assert [object_value(result)["url"] for result in list_value(body_of(response)["results"])] == [
            result_url(marker)
        ]
        assert_one_request(wire.drain(), row.target, marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
def test_gigachat_pass_through_with_a_router_model_reaches_one_single_slash_url(
    gateway: Gateway, trailing: str
) -> None:
    marker: Final = new_marker()
    with wire_server(provider_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="gigachat/GigaChat", api_base=f"{wire.url}/gc/v1{trailing}", api_key=KEY, gigachat_access_token=KEY
        )
        response: Final = gateway.request(
            "POST", "/gigachat/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}
        )
        assert first_item(body_of(response)["choices"])["message"] == {
            "role": "assistant",
            "content": reply_text(marker),
        }
        forwarded: Final = assert_one_request(wire.drain(), "/gc/v1/chat/completions", marker)
        assert request_json(forwarded)["model"] == "GigaChat", forwarded.body


@pytest.mark.parametrize(
    ("row", "trailing"),
    (
        (Row("deepseek/deepseek-chat", "/ds", "/chat/completions"), "//"),
        (Row("deepseek/deepseek-chat", f"/ds/{'p' * 5000}", "/chat/completions"), "/"),
    ),
    ids=("two-slashes", "five-kb-prefix"),
)
def test_chat_base_with_repeated_slashes_or_a_long_prefix_reaches_one_url(
    gateway: Gateway, row: Row, trailing: str
) -> None:
    marker: Final = new_marker()
    with wire_server(provider_peer) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, row, f"{wire.url}{row.base}{trailing}")
        assert _chat(gateway, model, marker) == reply_text(marker)
        assert_one_request(wire.drain(), row.target, marker)


def test_embedding_and_search_bases_ending_in_two_slashes_reach_one_url(gateway: Gateway) -> None:
    embed_marker: Final = new_marker()
    search_marker: Final = new_marker()
    with wire_server(provider_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="jina_ai/jina-embeddings-v3", api_base=f"{wire.url}/jn/v1//", api_key=KEY)
        assert _embedding(gateway, model, embed_marker) == list(VECTOR)
        assert_one_request(wire.drain(), "/jn/v1/embeddings", embed_marker)
        tool: Final = _search_tool(gateway, scenario, "tavily", f"{wire.url}/tv//")
        response: Final = gateway.request("POST", f"/v1/search/{tool}", {"query": search_marker})
        assert response.status_code == 200, response.text
        assert_one_request(wire.drain(), "/tv/search", search_marker)


@pytest.mark.parametrize("api_base", (12345, ["http://127.0.0.1:9/ds/"]), ids=("int", "list"))
def test_non_string_api_base_is_rejected_and_other_deployments_keep_answering(
    gateway: Gateway, api_base: JsonValue
) -> None:
    with gateway.scenario() as scenario:
        response: Final = gateway.request(
            "POST",
            "/model/new",
            {
                "model_name": f"integration-{new_marker()}",
                "litellm_params": {"model": "deepseek/deepseek-chat", "api_key": KEY, "api_base": api_base},
            },
        )
        assert response.status_code == 422, response.text
        assert "api_base" in response.text, response.text
        assert_control_answers(gateway, scenario)


@pytest.mark.parametrize(
    ("row", "status"),
    (
        (Row("deepseek/deepseek-chat", "/ds", "/chat/completions"), 401),
        (Row("jina_ai/jina-embeddings-v3", "/jn/v1", "/embeddings"), 404),
    ),
    ids=("deepseek-401", "jina-404"),
)
def test_upstream_error_on_a_slash_base_reaches_the_caller_from_one_url(
    gateway: Gateway, row: Row, status: int
) -> None:
    marker: Final = new_marker()
    error: Final = f'{{"error": {{"message": "scripted {status} {marker}", "type": "scripted"}}}}'.encode()

    def respond(_: Request) -> Reply:
        return Reply(status=status, body=error)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, row, f"{wire.url}{row.base}/")
        body: Final[dict[str, JsonValue]] = (
            {"model": model, "input": marker}
            if row.route == "/embeddings"
            else {"model": model, "messages": [{"role": "user", "content": marker}]}
        )
        path: Final = "/v1/embeddings" if row.route == "/embeddings" else "/v1/chat/completions"
        response: Final = gateway.request("POST", path, body)
        assert response.status_code == status, response.text
        assert marker in response.text, response.text
        assert_one_request(wire.drain(), row.target, marker)
        assert_control_answers(gateway, scenario)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
@pytest.mark.parametrize("row", _FULL_ROUTE_ROWS, ids=_ids(_FULL_ROUTE_ROWS))
def test_chat_base_that_already_names_the_route_is_sent_as_is(gateway: Gateway, row: Row, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(provider_peer) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, row, f"{wire.url}{row.base}{trailing}")
        assert _chat(gateway, model, marker) == reply_text(marker)
        assert_one_request(wire.drain(), row.target, marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
@pytest.mark.parametrize("row", _FULL_ROUTE_EMBEDDING_ROWS, ids=_ids(_FULL_ROUTE_EMBEDDING_ROWS))
def test_embedding_base_that_already_names_the_route_is_sent_as_is(gateway: Gateway, row: Row, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(provider_peer) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, row, f"{wire.url}{row.base}{trailing}")
        assert _embedding(gateway, model, marker) == list(VECTOR)
        assert_one_request(wire.drain(), row.target, marker)


@pytest.mark.parametrize("trailing", TRAILING, ids=TRAILING_IDS)
def test_search_base_that_already_names_the_route_is_sent_as_is(gateway: Gateway, trailing: str) -> None:
    marker: Final = new_marker()
    with wire_server(provider_peer) as wire, gateway.scenario() as scenario:
        tool: Final = _search_tool(gateway, scenario, "tavily", f"{wire.url}/tv/search{trailing}")
        response: Final = gateway.request("POST", f"/v1/search/{tool}", {"query": marker})
        assert response.status_code == 200, response.text
        assert_one_request(wire.drain(), "/tv/search", marker)
