"""Contract: what a chat caller sees when Anthropic answers with its newer response fields.

`POST /v1/chat/completions` on an `anthropic/` deployment turns Anthropic's Messages response into an OpenAI chat
completion (`AnthropicConfig.transform_parsed_response`, `extract_response_content`, `calculate_usage` and
`_build_provider_specific_fields` in `litellm/llms/anthropic/chat/transformation.py`). Fields with no OpenAI twin
travel in `choices[0].message.provider_specific_fields` (citations, web fetch results, the code execution
container, applied context management edits), Anthropic's `usage.service_tier` and server tool counters stay on
`usage`, and Anthropic's rate limit headers are relayed both with the `llm_provider-` prefix and under the OpenAI
`x-ratelimit-*` names (`process_anthropic_headers`, `common_utils.py:2886`). A rewrite of the response side must
keep every one of these, because clients read them to bill, to resume a container and to back off.
"""

from __future__ import annotations

from typing import Final

from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_anthropic2_support import (
    CONTAINER_ID,
    provider_fields,
    replying,
    usage_of,
)
from integration.providers.dec1_anthropic_support import (
    Call,
    chat,
    chat_stream_lines,
    default_respond,
    deployment,
    first_message,
    marker,
    peer,
    served,
    text_stream_chunks,
    usage_block,
)
from pydantic import JsonValue

_FETCH_CALL: Final[dict[str, JsonValue]] = {
    "type": "server_tool_use",
    "id": "srvtoolu_dec1fetch",
    "name": "web_fetch",
    "input": {"url": "https://kb.dec1.invalid/articles/7"},
}
_FETCH_RESULT: Final[dict[str, JsonValue]] = {
    "type": "web_fetch_tool_result",
    "tool_use_id": "srvtoolu_dec1fetch",
    "content": {
        "type": "web_fetch_result",
        "url": "https://kb.dec1.invalid/articles/7",
        "content": {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": "The synthetic widget ships in blue."},
            "title": "Synthetic article seven",
        },
        "retrieved_at": "2026-11-17T08:30:00Z",
    },
}
_ANSWER: Final[dict[str, JsonValue]] = {"type": "text", "text": "It ships in blue."}


def test_usage_service_tier_is_relayed_on_the_chat_usage(gateway: Gateway) -> None:
    """The provider answers with `usage.service_tier: "priority"`.

    The caller's `usage.service_tier` must be `"priority"`: `calculate_usage` reads it and `Usage` keeps it as an
    extra attribute (`types/utils.py:2177`). Billing dashboards and the priority price lookup both read it.
    """
    with peer(replying(usage=usage_block(service_tier="priority"))) as provider, gateway.scenario() as scenario:
        body: Final = served(chat(gateway, deployment(scenario, provider)))
        assert usage_of(body).get("service_tier") == "priority"


def test_a_web_fetch_result_is_returned_with_the_web_search_results(gateway: Gateway) -> None:
    """The provider answers a `server_tool_use` web_fetch call, its `web_fetch_tool_result` and a text answer.

    The caller's message content must be the text `It ships in blue.`, and
    `provider_specific_fields.web_search_results` must hold the `web_fetch_tool_result` block unchanged:
    `extract_response_content` collects both `web_search_tool_result` and `web_fetch_tool_result` there
    (`transformation.py:2911`).
    """
    with peer(replying([_FETCH_CALL, _FETCH_RESULT, _ANSWER])) as provider, gateway.scenario() as scenario:
        body: Final = served(chat(gateway, deployment(scenario, provider)))
        assert first_message(body)["content"] == "It ships in blue."
        assert list_value(provider_fields(body)["web_search_results"]) == [_FETCH_RESULT]


def test_a_server_web_fetch_call_is_not_returned_as_a_client_tool_call(gateway: Gateway) -> None:
    """Same web fetch turn, ending with `stop_reason: end_turn`.

    Anthropic already ran the fetch, so the caller must get `finish_reason: "stop"` and no OpenAI `tool_calls`
    for `web_fetch`; the call is listed under `provider_specific_fields.server_tool_calls` instead
    (`transformation.py:3284`). A client that sees a `web_fetch` tool call would try to run it itself.
    UNVERIFIED: that `message.tool_calls` is absent or empty rather than holding the server call.
    """
    with peer(replying([_FETCH_CALL, _FETCH_RESULT, _ANSWER])) as provider, gateway.scenario() as scenario:
        body: Final = served(chat(gateway, deployment(scenario, provider)))
        assert object_value(list_value(body["choices"])[0])["finish_reason"] == "stop"
        assert not first_message(body).get("tool_calls")
        server_calls: Final = list_value(provider_fields(body)["server_tool_calls"])
        assert [object_value(call)["id"] for call in server_calls] == ["srvtoolu_dec1fetch"]


def test_web_fetch_requests_are_not_reported_as_web_search_requests(gateway: Gateway) -> None:
    """The provider reports `usage.server_tool_use: {"web_fetch_requests": 2}` and no web searches.

    `calculate_usage` reads only `web_search_requests` and `tool_search_requests`, so the caller's usage must not
    carry `server_tool_use` at all (`Usage` deletes the field when both are None). A rewrite that folds fetches into
    `web_search_requests` would bill them at the per search fee.
    """
    usage: Final = usage_block(server_tool_use={"web_fetch_requests": 2})
    with peer(replying([_FETCH_CALL, _FETCH_RESULT, _ANSWER], usage=usage)) as provider, gateway.scenario() as scenario:
        body: Final = served(chat(gateway, deployment(scenario, provider)))
        assert "server_tool_use" not in usage_of(body)


def test_web_search_requests_next_to_web_fetch_requests_keep_only_the_search_count(gateway: Gateway) -> None:
    """The provider reports `server_tool_use: {"web_search_requests": 3, "web_fetch_requests": 2}`.

    The caller's `usage.server_tool_use` must be `{"web_search_requests": 3}` with no fetch counter
    (`ServerToolUse` has no `web_fetch_requests` field, `types/utils.py:2020`). UNVERIFIED: whether the
    serialized object also carries `tool_search_requests: null`; only the two named keys are pinned.
    """
    usage: Final = usage_block(server_tool_use={"web_search_requests": 3, "web_fetch_requests": 2})
    with peer(replying(usage=usage)) as provider, gateway.scenario() as scenario:
        counters: Final = object_value(
            usage_of(served(chat(gateway, deployment(scenario, provider))))["server_tool_use"]
        )
        assert counters.get("web_search_requests") == 3
        assert "web_fetch_requests" not in counters


def test_the_code_execution_container_is_returned_in_provider_specific_fields(gateway: Gateway) -> None:
    """The provider answers a text block and a top-level `container: {"id": "container_dec1reuse", "expires_at":
    "2026-11-17T09:30:00Z"}`.

    The caller's `provider_specific_fields.container` must be that object unchanged (`transformation.py:3280`).
    Clients pass the id back as `container` to reuse the sandbox and its files.
    """
    container: Final[dict[str, JsonValue]] = {"id": CONTAINER_ID, "expires_at": "2026-11-17T09:30:00Z"}
    with peer(replying(container=container)) as provider, gateway.scenario() as scenario:
        body: Final = served(chat(gateway, deployment(scenario, provider)))
        assert provider_fields(body).get("container") == container


def test_a_bash_code_execution_result_becomes_a_code_interpreter_result_with_the_container_id(
    gateway: Gateway,
) -> None:
    """The provider answers a `server_tool_use` `bash_code_execution` call with input `{"command": "ls"}`, its
    `bash_code_execution_tool_result` (stdout `report.csv`), a text block, and `container.id:
    container_dec1reuse`.

    `provider_specific_fields.code_interpreter_results` must hold one entry of type `code_interpreter_call` with
    `id: srvtoolu_dec1bash`, `container_id: container_dec1reuse` and `status: completed`
    (`_build_code_interpreter_results`, `transformation.py:3196`).
    """
    content: Final[list[JsonValue]] = [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_dec1bash",
            "name": "bash_code_execution",
            "input": {"command": "ls"},
        },
        {
            "type": "bash_code_execution_tool_result",
            "tool_use_id": "srvtoolu_dec1bash",
            "content": {"type": "bash_code_execution_result", "stdout": "report.csv\n", "stderr": "", "return_code": 0},
        },
        {"type": "text", "text": "One file."},
    ]
    with peer(replying(content, container={"id": CONTAINER_ID})) as provider, gateway.scenario() as scenario:
        body: Final = served(chat(gateway, deployment(scenario, provider)))
        results: Final = list_value(provider_fields(body)["code_interpreter_results"])
        assert len(results) == 1
        result: Final = object_value(results[0])
        assert result.get("type") == "code_interpreter_call"
        assert result.get("id") == "srvtoolu_dec1bash"
        assert result.get("container_id") == CONTAINER_ID
        assert result.get("status") == "completed"


def test_applied_context_management_edits_are_returned(gateway: Gateway) -> None:
    """The provider answers with a top-level `context_management: {"applied_edits": [{"type":
    "clear_tool_uses_20250919", "cleared_tool_uses": 4, "cleared_input_tokens": 12000}]}`.

    The caller's `provider_specific_fields.context_management` must be that object unchanged
    (`transformation.py:3253`), so a client can tell how much of its history Anthropic cleared.
    """
    applied: Final[dict[str, JsonValue]] = {
        "applied_edits": [{"type": "clear_tool_uses_20250919", "cleared_tool_uses": 4, "cleared_input_tokens": 12000}]
    }
    with peer(replying(context_management=applied)) as provider, gateway.scenario() as scenario:
        body: Final = served(chat(gateway, deployment(scenario, provider)))
        assert provider_fields(body).get("context_management") == applied


def test_a_search_result_location_citation_is_returned_unchanged(gateway: Gateway) -> None:
    """The provider answers a text block citing a search result: one citation of type `search_result_location`
    with `source`, `title`, `search_result_index: 0`, `start_block_index: 0`, `end_block_index: 1` and
    `cited_text`.

    `provider_specific_fields.citations` must be `[[<that citation>]]`, one inner list per cited text block
    (`transformation.py:2937`), with every key kept, since RAG clients link the answer back by `source`.

    Pinned current (run 44176682961): each citation also gains `supported_text`, the text of the block that cites
    it (litellm/llms/anthropic/chat/transformation.py:2936-2948).
    """
    citation: Final[dict[str, JsonValue]] = {
        "type": "search_result_location",
        "source": "https://kb.dec1.invalid/articles/7",
        "title": "Synthetic article seven",
        "search_result_index": 0,
        "start_block_index": 0,
        "end_block_index": 1,
        "cited_text": "The synthetic widget ships in blue.",
    }
    block: Final[dict[str, JsonValue]] = {"type": "text", "text": "It ships in blue.", "citations": [citation]}
    with peer(replying([block])) as provider, gateway.scenario() as scenario:
        body: Final = served(chat(gateway, deployment(scenario, provider)))
        assert provider_fields(body).get("citations") == [[{**citation, "supported_text": "It ships in blue."}]]


def test_a_page_location_citation_on_a_pdf_is_returned_unchanged(gateway: Gateway) -> None:
    """The provider answers a text block with one `page_location` citation (`document_index: 0`,
    `document_title`, `start_page_number: 2`, `end_page_number: 3`, `cited_text`).

    `provider_specific_fields.citations` must be `[[<that citation>]]` with the page numbers intact, plus the
    `supported_text` of the citing block (pinned current, litellm/llms/anthropic/chat/transformation.py:2936-2948).
    """
    citation: Final[dict[str, JsonValue]] = {
        "type": "page_location",
        "document_index": 0,
        "document_title": "Widget sheet",
        "start_page_number": 2,
        "end_page_number": 3,
        "cited_text": "The synthetic widget ships in blue.",
    }
    block: Final[dict[str, JsonValue]] = {"type": "text", "text": "Blue.", "citations": [citation]}
    with peer(replying([block])) as provider, gateway.scenario() as scenario:
        body: Final = served(chat(gateway, deployment(scenario, provider)))
        assert provider_fields(body).get("citations") == [[{**citation, "supported_text": "Blue."}]]


def test_anthropic_rate_limit_headers_are_also_relayed_under_the_openai_names(gateway: Gateway) -> None:
    """The provider answers with `anthropic-ratelimit-requests-limit: 50`, `-requests-remaining: 49`,
    `-tokens-limit: 80000` and `-tokens-remaining: 79000`.

    The caller must get `x-ratelimit-limit-requests: 50`, `x-ratelimit-remaining-requests: 49`,
    `x-ratelimit-limit-tokens: 80000` and `x-ratelimit-remaining-tokens: 79000` (`process_anthropic_headers`), the
    names OpenAI SDK clients read to back off.
    """
    headers: Final = {
        "anthropic-ratelimit-requests-limit": "50",
        "anthropic-ratelimit-requests-remaining": "49",
        "anthropic-ratelimit-tokens-limit": "80000",
        "anthropic-ratelimit-tokens-remaining": "79000",
    }
    with peer(replying(headers=headers)) as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider))
        served(response)
        assert response.headers.get("x-ratelimit-limit-requests") == "50"
        assert response.headers.get("x-ratelimit-remaining-requests") == "49"
        assert response.headers.get("x-ratelimit-limit-tokens") == "80000"
        assert response.headers.get("x-ratelimit-remaining-tokens") == "79000"


def test_input_and_output_token_rate_limit_headers_are_relayed_with_the_provider_prefix(gateway: Gateway) -> None:
    """The provider answers with the newer split headers `anthropic-ratelimit-input-tokens-remaining: 39000`,
    `anthropic-ratelimit-output-tokens-remaining: 7900` and `anthropic-ratelimit-tokens-reset:
    2026-11-17T08:31:00Z`, which have no OpenAI twin.

    They must reach the caller as `llm_provider-anthropic-ratelimit-input-tokens-remaining: 39000`,
    `llm_provider-anthropic-ratelimit-output-tokens-remaining: 7900` and
    `llm_provider-anthropic-ratelimit-tokens-reset: 2026-11-17T08:31:00Z` (every provider header is prefixed).
    """
    headers: Final = {
        "anthropic-ratelimit-input-tokens-remaining": "39000",
        "anthropic-ratelimit-output-tokens-remaining": "7900",
        "anthropic-ratelimit-tokens-reset": "2026-11-17T08:31:00Z",
    }
    with peer(replying(headers=headers)) as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider))
        served(response)
        for name, value in headers.items():
            assert response.headers.get(f"llm_provider-{name}") == value, name


def test_a_streamed_chat_relays_the_request_id_and_rate_limit_headers(gateway: Gateway) -> None:
    """The caller streams; the provider's event stream response carries `request-id: req_dec1stream` and
    `anthropic-ratelimit-requests-remaining: 48`.

    The streaming response headers must include `llm_provider-request-id: req_dec1stream` and
    `llm_provider-anthropic-ratelimit-requests-remaining: 48`, the same as a non-streamed call: the chat handler
    builds the stream wrapper with `process_anthropic_headers(headers)` (`chat/handler.py:393`). The request id is
    what Anthropic support asks for.
    """

    def respond(call: Call) -> Reply:
        if call.method != "POST":
            return default_respond(call)
        return Reply(
            content_type="text/event-stream",
            chunks=text_stream_chunks(call.body),
            headers={"request-id": "req_dec1stream", "anthropic-ratelimit-requests-remaining": "48"},
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider)
        headers, _ = chat_stream_lines(
            gateway, {"model": model, "messages": [{"role": "user", "content": marker()}], "stream": True}
        )
        assert headers.get("llm_provider-request-id") == "req_dec1stream"
        assert headers.get("llm_provider-anthropic-ratelimit-requests-remaining") == "48"
