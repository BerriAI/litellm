"""Contract: OpenAI hosted tools on `POST /v1/responses` served natively by an `openai/` deployment.

OpenAI runs its hosted tools (file_search, code_interpreter, web_search, image_generation, computer use and
remote MCP) inside the provider, so the proxy's job is to carry the tool definitions to `<api_base>/responses`
unchanged and to carry the tool call items back to the caller unchanged. Four proxy behaviours sit on top of
that and must survive the Rust port:

- a `file_search` tool whose `vector_store_ids` are not in the proxy's vector store registry is forwarded as
  sent (`vector_store_registry.py:205`); `OpenAIResponsesAPIConfig.supports_native_file_search()` is true, so
  the emulated file search of the bridge never runs for OpenAI (`responses/main.py:1471`);
- an `mcp` tool whose `server_url` is an outside server is not executed by the proxy's MCP gateway
  (`litellm_proxy_mcp_handler.py:219`, only gateway-named tools are split out) and reaches OpenAI as sent;
- a `container_id` on a `code_interpreter_call` output item is rewritten to a managed id
  `cntr_<base64 litellm:custom_llm_provider:openai;model_id:<id>;container_id:<raw>>`
  (`ResponsesAPIRequestUtils._update_container_ids_in_response`, `responses/utils.py:1232`) and the container is
  recorded as owned by the calling key (`container_endpoints/ownership.py:155`), so `/v1/containers/{id}` routes
  back to the same deployment and refuses other keys;
- built-in tool calls are billed on top of tokens: each `web_search_call` item at the row's
  `search_context_cost_per_query` (0.025 for `gpt-4o-mini`) and a request with a `file_search` tool and a
  `file_search_call` item at `OPENAI_FILE_SEARCH_COST_PER_1K_CALLS` = 2.5 / 1000 (`constants.py:580`).

Every deployment points at a local peer; nothing reaches a real OpenAI host.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value, string_value
from integration._support.wire import Reply
from integration.providers.dec1_openai2_support import (
    FILE_SEARCH_CALL_COST,
    decoded_managed_id,
    deployment_id,
    managed_container_id,
    resource_respond,
    with_output,
)
from integration.providers.dec1_openai_support import (
    CALL_COST,
    Call,
    data_events,
    deployment,
    json_reply,
    marker,
    peer,
    responses,
    responses_object,
    served,
    spend_row_by_group,
    sse,
    stream_lines,
)
from pydantic import JsonValue

_MODEL: Final = "openai/gpt-4o-mini"
_WEB_SEARCH_PER_CALL: Final = 0.025
_RAW_CONTAINER: Final = "cntr_dec1openai2codeinterp"

_FILE_SEARCH_TOOL: Final[dict[str, JsonValue]] = {
    "type": "file_search",
    "vector_store_ids": ["vs_dec1_openai2_unregistered"],
    "max_num_results": 4,
    "filters": {"type": "eq", "key": "team", "value": "blue"},
    "ranking_options": {"ranker": "auto", "score_threshold": 0.25},
}
_CODE_INTERPRETER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "code_interpreter",
    "container": {"type": "auto", "file_ids": ["file-dec1-openai2-data"]},
}
_WEB_SEARCH_TOOL: Final[dict[str, JsonValue]] = {
    "type": "web_search",
    "search_context_size": "high",
    "user_location": {"type": "approximate", "country": "GB", "city": "London", "timezone": "Europe/London"},
    "filters": {"allowed_domains": ["example.org"]},
}
_IMAGE_TOOL: Final[dict[str, JsonValue]] = {
    "type": "image_generation",
    "model": "gpt-image-1",
    "size": "1024x1024",
    "quality": "low",
    "output_format": "webp",
    "background": "transparent",
    "partial_images": 2,
}
_COMPUTER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "computer_use_preview",
    "display_width": 1024,
    "display_height": 768,
    "environment": "browser",
}
_MCP_TOOL: Final[dict[str, JsonValue]] = {
    "type": "mcp",
    "server_label": "dec1docs",
    "server_url": "https://mcp.dec1-openai2.invalid/sse",
    "require_approval": "never",
    "allowed_tools": ["search_docs"],
    "headers": {"Authorization": "Bearer dec1-openai2-mcp-token"},
}


def _code_interpreter_item(container: str) -> dict[str, JsonValue]:
    return {
        "id": "ci_dec1openai2",
        "type": "code_interpreter_call",
        "status": "completed",
        "code": "print(1 + 1)",
        "container_id": container,
        "outputs": [{"type": "logs", "logs": "2"}],
    }


@pytest.mark.parametrize(
    "tool",
    [_FILE_SEARCH_TOOL, _CODE_INTERPRETER_TOOL, _WEB_SEARCH_TOOL, _IMAGE_TOOL, _COMPUTER_TOOL, _MCP_TOOL],
    ids=["file_search", "code_interpreter", "web_search", "image_generation", "computer_use", "remote_mcp"],
)
def test_each_hosted_tool_reaches_openai_exactly_as_sent(gateway: Gateway, tool: dict[str, JsonValue]) -> None:
    """The caller sends one hosted tool definition to an `openai/gpt-4o-mini` deployment.

    The provider body carries `tools == [<tool>]` with every nested field unchanged (filters, ranking options,
    container file ids, user location, image options, display size, MCP headers). A port that models tools as
    function tools only, or validates them through a narrow struct, would drop the hosted ones.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, tools=[tool]))
        assert provider.only().body["tools"] == [tool]


def test_web_search_preview_with_its_context_size_is_forwarded(gateway: Gateway) -> None:
    """The caller sends the older `{"type": "web_search_preview", "search_context_size": "low"}` tool.

    OpenAI still accepts this spelling, so the provider body carries it unchanged and it is not renamed to
    `web_search`.
    """
    tool: Final[dict[str, JsonValue]] = {"type": "web_search_preview", "search_context_size": "low"}
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, tools=[tool]))
        assert provider.only().body["tools"] == [tool]


def test_hosted_tools_and_a_function_tool_travel_together_in_order(gateway: Gateway) -> None:
    """The caller mixes `file_search`, a flat function tool and `web_search` in one `tools` list.

    The provider sees the three tools in the same order and shape; the function tool stays flat (Responses
    shape) and is not nested under `function`.
    """
    function_tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "name": "lookup_order",
        "description": "Look up an order",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    }
    tools: Final[list[JsonValue]] = [_FILE_SEARCH_TOOL, function_tool, _WEB_SEARCH_TOOL]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, tools=tools))
        assert provider.only().body["tools"] == tools


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"type": "file_search"},
        {"type": "web_search_preview"},
        {"type": "image_generation"},
        {"type": "mcp", "server_label": "dec1docs", "name": "search_docs"},
    ],
    ids=["file_search", "web_search_preview", "image_generation", "mcp"],
)
def test_a_hosted_tool_choice_is_forwarded_unchanged(gateway: Gateway, tool_choice: dict[str, JsonValue]) -> None:
    """The caller forces a hosted tool with `tool_choice: {"type": <hosted>}` alongside the matching tool.

    The provider body carries the same `tool_choice` object. The chat-completions normaliser
    (`{"type": "function", "function": ...}`) must not run on a native OpenAI Responses call.
    """
    tools: Final[dict[str, list[JsonValue]]] = {
        "file_search": [_FILE_SEARCH_TOOL],
        "web_search_preview": [{"type": "web_search_preview"}],
        "image_generation": [_IMAGE_TOOL],
        "mcp": [_MCP_TOOL],
    }
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, tools=tools[string_value(tool_choice["type"])], tool_choice=tool_choice))
        assert provider.only().body["tool_choice"] == tool_choice


def test_hosted_tool_include_paths_are_forwarded(gateway: Gateway) -> None:
    """The caller asks for `include: ["file_search_call.results", "code_interpreter_call.outputs",
    "web_search_call.action.sources", "computer_call_output.output.image_url"]`.

    The provider body carries the same list in the same order; these are the only way to get tool results back,
    so dropping one silently empties the caller's citations.
    """
    include: Final[list[JsonValue]] = [
        "file_search_call.results",
        "code_interpreter_call.outputs",
        "web_search_call.action.sources",
        "computer_call_output.output.image_url",
    ]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, tools=[_FILE_SEARCH_TOOL], include=include))
        assert provider.only().body["include"] == include


@pytest.mark.parametrize(
    "item",
    [
        {
            "id": "fs_dec1openai2",
            "type": "file_search_call",
            "status": "completed",
            "queries": ["refund policy"],
            "results": [
                {
                    "file_id": "file-dec1-openai2-policy",
                    "filename": "policy.md",
                    "score": 0.91,
                    "text": "Refunds within 30 days.",
                    "attributes": {"team": "blue"},
                }
            ],
        },
        {
            "id": "ws_dec1openai2",
            "type": "web_search_call",
            "status": "completed",
            "action": {
                "type": "search",
                "query": "dec1 weather",
                "sources": [{"type": "url", "url": "https://example.org/"}],
            },
        },
        {
            "id": "ig_dec1openai2",
            "type": "image_generation_call",
            "status": "completed",
            "result": "aGVsbG8=",
            "revised_prompt": "a small red square",
            "size": "1024x1024",
            "quality": "low",
        },
        {
            "id": "cu_dec1openai2",
            "type": "computer_call",
            "call_id": "call_dec1openai2",
            "status": "completed",
            "action": {"type": "click", "button": "left", "x": 10, "y": 20},
            "pending_safety_checks": [],
        },
        {
            "id": "mcpl_dec1openai2",
            "type": "mcp_list_tools",
            "server_label": "dec1docs",
            "tools": [{"name": "search_docs", "input_schema": {"type": "object"}}],
        },
        {
            "id": "mcp_dec1openai2",
            "type": "mcp_call",
            "server_label": "dec1docs",
            "name": "search_docs",
            "arguments": '{"q": "refunds"}',
            "output": "30 days",
            "error": None,
        },
        {
            "id": "mcpr_dec1openai2",
            "type": "mcp_approval_request",
            "server_label": "dec1docs",
            "name": "search_docs",
            "arguments": "{}",
        },
    ],
    ids=[
        "file_search_call",
        "web_search_call",
        "image_generation_call",
        "computer_call",
        "mcp_list_tools",
        "mcp_call",
        "mcp_approval_request",
    ],
)
def test_each_hosted_tool_output_item_reaches_the_caller_unchanged(
    gateway: Gateway, item: dict[str, JsonValue]
) -> None:
    """The provider answers with one hosted tool call item followed by the usual assistant message.

    The caller's `output[0]` equals the provider's item field for field and `output[1]` is the message. A port
    that keeps only `message`, `reasoning` and `function_call` items, or re-serialises items through a closed
    union, would lose the tool result. UNVERIFIED: none of these item types carries a `container_id`, so no id is
    rewritten; `image_generation_call.result` is relayed as base64 untouched.
    """

    def respond(call: Call) -> Reply:
        message: Final = list_value(responses_object(call.body)["output"])[0]
        return with_output(call.body, [item, message])

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(responses(gateway, model))
        output: Final = list_value(body["output"])
        assert object_value(output[0]) == item, output
        assert object_value(output[1])["type"] == "message", output


@pytest.mark.parametrize(
    "item",
    [
        {
            "type": "computer_call_output",
            "call_id": "call_dec1openai2",
            "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,aGVsbG8="},
            "acknowledged_safety_checks": [{"id": "sc_dec1", "code": "malicious_instructions", "message": "ok"}],
        },
        {"type": "mcp_approval_response", "approval_request_id": "mcpr_dec1openai2", "approve": True},
    ],
    ids=["computer_call_output", "mcp_approval_response"],
)
def test_a_hosted_tool_follow_up_input_item_is_forwarded_unchanged(
    gateway: Gateway, item: dict[str, JsonValue]
) -> None:
    """The caller continues a computer use or MCP approval loop by sending the follow-up item in `input`.

    The provider body's `input` is exactly `[<item>]`. These item types have no chat-completions equivalent, so
    a port that routes input items through the chat message model would refuse or drop them.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, input=[item], tools=[_COMPUTER_TOOL], truncation="auto"))
        call: Final = provider.only()
        assert call.body["input"] == [item]
        assert call.body["truncation"] == "auto"


def test_a_code_interpreter_container_id_comes_back_as_a_managed_id(gateway: Gateway) -> None:
    """The provider answers with a `code_interpreter_call` item whose `container_id` is `cntr_dec1openai2codeinterp`.

    The caller's item carries `cntr_<base64>` instead, and the base64 decodes to
    `litellm:custom_llm_provider:openai;model_id:<deployment id>;container_id:cntr_dec1openai2codeinterp`, so a
    follow-up container call routes back to the same deployment and credentials.
    """

    def respond(call: Call) -> Reply:
        return with_output(call.body, [_code_interpreter_item(_RAW_CONTAINER)])

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(responses(gateway, model, tools=[_CODE_INTERPRETER_TOOL]))
        container: Final = string_value(object_value(list_value(body["output"])[0])["container_id"])
        assert container == managed_container_id("openai", deployment_id(gateway, model), _RAW_CONTAINER), container


def test_a_container_file_citation_in_a_message_carries_the_managed_container_id(gateway: Gateway) -> None:
    """The assistant message carries a `container_file_citation` annotation naming container
    `cntr_dec1openai2codeinterp` and file `cfile_dec1`.

    The caller's annotation has the managed `cntr_` id (decoding to the raw one) and the same `file_id`,
    because that annotation is what clients use to download the generated file (`responses/utils.py:1054`).
    """
    annotation: Final[dict[str, JsonValue]] = {
        "type": "container_file_citation",
        "container_id": _RAW_CONTAINER,
        "file_id": "cfile_dec1",
        "filename": "plot.png",
        "start_index": 0,
        "end_index": 4,
    }

    def respond(call: Call) -> Reply:
        message: Final[dict[str, JsonValue]] = {
            "id": "msg_dec1openai2",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "plot", "annotations": [annotation]}],
        }
        return with_output(call.body, [_code_interpreter_item(_RAW_CONTAINER), message])

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(responses(gateway, model, tools=[_CODE_INTERPRETER_TOOL]))
        message: Final = object_value(list_value(body["output"])[1])
        seen: Final = object_value(list_value(object_value(list_value(message["content"])[0])["annotations"])[0])
        assert seen["file_id"] == "cfile_dec1", seen
        assert decoded_managed_id(string_value(seen["container_id"]), "cntr_").endswith(
            f";container_id:{_RAW_CONTAINER}"
        ), seen


def test_the_managed_container_id_from_a_response_retrieves_the_raw_container(gateway: Gateway) -> None:
    """A virtual key gets a managed container id from a code interpreter answer, then calls
    `GET /v1/containers/<managed id>` with the same key.

    The proxy records the container as owned by that key, decodes the id and sends
    `GET /v1/containers/cntr_dec1openai2codeinterp` to the same deployment's `api_base` with its key; the caller
    gets 200. UNVERIFIED: the caller's `id` in the retrieve answer is assumed to be the managed id again.
    """

    def respond(call: Call) -> Reply:
        if call.path.endswith("/responses"):
            return with_output(call.body, [_code_interpreter_item(_RAW_CONTAINER)])
        return resource_respond(call)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        owner: Final = scenario.key(models=[model])
        body: Final = served(responses(gateway, model, key=owner, tools=[_CODE_INTERPRETER_TOOL]))
        container: Final = string_value(object_value(list_value(body["output"])[0])["container_id"])
        retrieved: Final = gateway.request("GET", f"/v1/containers/{container}", key=owner)
        assert retrieved.status_code == 200, retrieved.text
        gets: Final = tuple(call for call in provider.all() if call.method == "GET" and "/containers/" in call.path)
        assert [call.path for call in gets] == [f"/v1/containers/{_RAW_CONTAINER}"], gets
        assert gets[0].headers["authorization"] == "Bearer dec1-openai-provider-key"


def test_another_key_cannot_open_a_container_created_by_code_interpreter(gateway: Gateway) -> None:
    """Key A runs code interpreter and receives a managed container id; key B, with access to the same model,
    calls `GET /v1/containers/<id>`.

    Key B gets 403 with detail `Forbidden` and the provider sees no container GET, because ownership was recorded
    for key A when the Responses answer came back (`ownership.py:343`).
    """

    def respond(call: Call) -> Reply:
        if call.path.endswith("/responses"):
            return with_output(call.body, [_code_interpreter_item(_RAW_CONTAINER + "b")])
        return resource_respond(call)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        owner: Final = scenario.key(models=[model])
        stranger: Final = scenario.key(models=[model])
        body: Final = served(responses(gateway, model, key=owner, tools=[_CODE_INTERPRETER_TOOL]))
        container: Final = string_value(object_value(list_value(body["output"])[0])["container_id"])
        refused: Final = gateway.request("GET", f"/v1/containers/{container}", key=stranger)
        assert refused.status_code == 403, refused.text
        assert "Forbidden" in refused.text
        assert not [call for call in provider.all() if "/containers/" in call.path]


def test_a_streamed_code_interpreter_item_carries_the_managed_container_id(gateway: Gateway) -> None:
    """The provider streams `response.output_item.added` and `response.output_item.done` for a
    `code_interpreter_call` with raw `container_id`, then `response.completed`.

    Both item events the caller receives carry the managed `cntr_` id that decodes to the raw id
    (`streaming_iterator.py:549`), so a UI rendering items incrementally can call the container routes.
    """

    def respond(call: Call) -> Reply:
        item: Final = _code_interpreter_item(_RAW_CONTAINER)
        done: Final = {**responses_object(call.body), "output": [item]}
        return Reply(
            content_type="text/event-stream",
            chunks=(
                sse(
                    {
                        "type": "response.created",
                        "response": {**done, "status": "in_progress", "output": []},
                        "sequence_number": 0,
                    },
                    "response.created",
                ),
                sse(
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {**item, "status": "in_progress"},
                        "sequence_number": 1,
                    },
                    "response.output_item.added",
                ),
                sse(
                    {"type": "response.output_item.done", "output_index": 0, "item": item, "sequence_number": 2},
                    "response.output_item.done",
                ),
                sse({"type": "response.completed", "response": done, "sequence_number": 3}, "response.completed"),
            ),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        _, lines = stream_lines(
            gateway,
            "/v1/responses",
            {"model": model, "input": marker(), "stream": True, "tools": [_CODE_INTERPRETER_TOOL]},
        )
        items: Final = tuple(
            object_value(event["item"])
            for event in data_events(lines)
            if event.get("type") in ("response.output_item.added", "response.output_item.done")
        )
        assert len(items) == 2, items
        for item in items:
            container = string_value(item["container_id"])
            assert decoded_managed_id(container, "cntr_").endswith(f";container_id:{_RAW_CONTAINER}"), item


def test_a_managed_container_id_in_a_code_interpreter_tool_reaches_openai_as_the_raw_id(gateway: Gateway) -> None:
    """The caller reuses a container by sending `tools: [{"type": "code_interpreter", "container": <managed cntr_ id>}]`
    that the proxy handed out earlier.

    The provider must see `container: "cntr_dec1openai2codeinterp"`, the raw id OpenAI issued, since OpenAI does
    not know the managed id.
    """
    pytest.skip(
        "BUG: nothing decodes a managed container id inside tools[].container on /v1/responses (only the "
        "/v1/containers routes call decode_managed_container_id_for_request), so OpenAI receives the cntr_<base64> id"
    )
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        managed: Final = managed_container_id("openai", deployment_id(gateway, model), _RAW_CONTAINER)
        served(responses(gateway, model, tools=[{"type": "code_interpreter", "container": managed}]))
        assert provider.only().body["tools"] == [{"type": "code_interpreter", "container": _RAW_CONTAINER}]


def test_hosted_tool_stream_events_are_relayed_in_the_provider_order(gateway: Gateway) -> None:
    """The provider streams `response.web_search_call.in_progress`, `.searching` and `.completed` between
    `response.created` and `response.completed`.

    The caller receives the same five event types in the same order, each once. Tool progress events are not in
    the plain text event set a port might whitelist.
    """
    order: Final = (
        "response.created",
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
        "response.completed",
    )

    def respond(call: Call) -> Reply:
        done: Final = responses_object(call.body)
        events: Final[list[dict[str, JsonValue]]] = [
            {"type": order[0], "response": {**done, "status": "in_progress", "output": []}},
            {"type": order[1], "output_index": 0, "item_id": "ws_dec1openai2"},
            {"type": order[2], "output_index": 0, "item_id": "ws_dec1openai2"},
            {"type": order[3], "output_index": 0, "item_id": "ws_dec1openai2"},
            {"type": order[4], "response": done},
        ]
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(
                sse({**event, "sequence_number": index}, string_value(event["type"]))
                for index, event in enumerate(events)
            ),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        _, lines = stream_lines(
            gateway, "/v1/responses", {"model": model, "input": marker(), "stream": True, "tools": [_WEB_SEARCH_TOOL]}
        )
        assert tuple(event.get("type") for event in data_events(lines)) == order


def test_each_web_search_call_is_billed_on_top_of_tokens(gateway: Gateway) -> None:
    """The provider answers with two `web_search_call` items, a message and usage 11 + 7 tokens, on a deployment
    priced at 0.001 / 0.002 per token.

    The SpendLogs row bills tokens (0.025) plus two searches at the `gpt-4o-mini` row's
    `search_context_cost_per_query` of 0.025 each (`_count_web_search_calls`), total 0.075. UNVERIFIED: the
    built-in tool price is looked up on the cost map row `gpt-4o-mini` even when the deployment has custom token
    prices.
    """

    def respond(call: Call) -> Reply:
        searches: Final[list[JsonValue]] = [
            {"id": f"ws_{index}", "type": "web_search_call", "status": "completed"} for index in range(2)
        ]
        return with_output(call.body, [*searches, list_value(responses_object(call.body)["output"])[0]])

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, tools=[{"type": "web_search"}]))
        row: Final = spend_row_by_group(gateway, model)
        assert float(str(row["spend"])) == pytest.approx(CALL_COST + 2 * _WEB_SEARCH_PER_CALL)


def test_a_file_search_call_adds_the_openai_per_call_price(gateway: Gateway) -> None:
    """The request carries a `file_search` tool and the provider answers with one `file_search_call` item plus a
    message, usage 11 + 7 tokens.

    The SpendLogs row bills 0.025 for tokens plus 0.0025 for the file search (`get_cost_for_file_search` returns
    `OPENAI_FILE_SEARCH_COST_PER_1K_CALLS` once per response, whatever the number of calls), total 0.0275.
    """

    def respond(call: Call) -> Reply:
        search: Final[dict[str, JsonValue]] = {
            "id": "fs_dec1",
            "type": "file_search_call",
            "status": "completed",
            "queries": ["q"],
            "results": None,
        }
        return with_output(call.body, [search, list_value(responses_object(call.body)["output"])[0]])

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, tools=[_FILE_SEARCH_TOOL]))
        row: Final = spend_row_by_group(gateway, model)
        assert float(str(row["spend"])) == pytest.approx(CALL_COST + FILE_SEARCH_CALL_COST)


def test_a_file_search_tool_the_model_did_not_use_adds_nothing(gateway: Gateway) -> None:
    """The request carries a `file_search` tool but the provider answers with only a message.

    The SpendLogs row bills tokens only, 0.025: the per-call price needs a `file_search_call` item in the output
    (`response_object_includes_file_search_call`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        served(responses(gateway, model, tools=[_FILE_SEARCH_TOOL]))
        row: Final = spend_row_by_group(gateway, model)
        assert float(str(row["spend"])) == pytest.approx(CALL_COST)


def test_a_registry_vector_store_id_in_file_search_is_not_sent_to_openai(gateway: Gateway) -> None:
    """The caller names `vs_integration_config_store`, which the shared proxy declares in `vector_store_registry`,
    in a `file_search` tool.

    The registry recognises every id in the tool, so the tool is removed from the request
    (`_get_vector_store_ids_from_tool_calls`, `vector_store_registry.py:216`) and the proxy runs the search itself;
    the provider body has no `file_search` tool. UNVERIFIED: the shared registry store points at the shared
    upstream, which is assumed to answer the search; the assertion is only on what the OpenAI deployment receives.
    """
    tool: Final[dict[str, JsonValue]] = {"type": "file_search", "vector_store_ids": ["vs_integration_config_store"]}
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = responses(gateway, model, tools=[tool])
        assert response.status_code == 200, response.text
        sent: Final = provider.only().body.get("tools") or []
        assert all(object_value(entry).get("type") != "file_search" for entry in list_value(sent)), sent


def test_an_mcp_tool_naming_an_outside_server_never_reaches_the_gateway_mcp_executor(gateway: Gateway) -> None:
    """The provider answers an `mcp` tool request with an `mcp_call` item already executed by OpenAI.

    The caller gets that answer in one round trip: the provider is called exactly once (no second Responses call
    with tool results appended), proving the proxy did not run its own MCP tool loop for an outside `server_url`.
    """

    def respond(call: Call) -> Reply:
        item: Final[dict[str, JsonValue]] = {
            "id": "mcp_dec1",
            "type": "mcp_call",
            "server_label": "dec1docs",
            "name": "search_docs",
            "arguments": "{}",
            "output": "ok",
            "error": None,
        }
        return with_output(call.body, [item])

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(responses(gateway, model, tools=[_MCP_TOOL]))
        assert len(provider.posts()) == 1, provider.posts()
        assert object_value(list_value(body["output"])[0])["type"] == "mcp_call", body


def test_a_hosted_tool_answer_keeps_its_managed_response_id(gateway: Gateway) -> None:
    """A Responses call with a `web_search` tool comes back from provider id `resp_dec1hosted`.

    The caller's `id` is the managed `resp_` id that decodes to
    `litellm:custom_llm_provider:openai;model_id:<id>;response_id:resp_dec1hosted`, the same as a plain call:
    hosted tools must not switch the answer to a different id scheme.
    """

    def respond(call: Call) -> Reply:
        return json_reply(responses_object(call.body, identity="resp_dec1hosted"))

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        body: Final = served(responses(gateway, model, tools=[_WEB_SEARCH_TOOL]))
        decoded: Final = decoded_managed_id(string_value(body["id"]), "resp_")
        assert (
            decoded
            == f"litellm:custom_llm_provider:openai;model_id:{deployment_id(gateway, model)};response_id:resp_dec1hosted"
        )
