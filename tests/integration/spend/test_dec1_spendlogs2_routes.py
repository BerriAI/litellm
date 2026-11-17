"""Contract: the spend log row each route writes, route by route, ahead of the Spend Logs V2 port.

The Rust callback subscribes to `attempt_ended` for every route (plan, "V2 hooks and fields"), so every surface
that today reaches `_ProxyDBLogger` must keep producing exactly one row with the same identity columns. Round 1
and `test_dec1_spend_log_row_surfaces.py` pin chat, embeddings, Anthropic `/v1/messages`, `/v1/responses` and text
completions on the scripted OpenAI and Anthropic providers; the `test_dec1_msgprov_*` files pin the provider name
and price of each `/v1/messages` backend. This file adds, per route:

1. `/v1/messages` (Rust native front door on Anthropic, Bedrock InvokeModel, Vertex AI and Azure AI Foundry,
   chat bridge on Azure OpenAI): `request_id` is the id the caller received, `model_group` and `api_key` are the
   caller's, one row per call streamed or not, and a provider failure row with `error_information.error_code`.
2. `/v1/responses` native (OpenAI deployment) and bridged (Anthropic deployment): both rows have `call_type`
   `aresponses` and the daily endpoint `/responses`.
3. Embeddings with a list input: one row for the whole batch of inputs.
4. A configured pass-through route: `call_type` `pass_through_endpoint`, one row per call, failure rows.
5. MCP tool calls over REST: `call_type` `call_mcp_tool`, `mcp_namespaced_tool_name`, a failure row for a tool error.
6. A batch charged on completion: the `aretrieve_batch` cost row is settle-once (`<batch id>_batch_cost`, one
   delivery keyed by the request id, `is_batch` true), however often the batch is retrieved.
7. Realtime sessions: one `_arealtime` row per session summing its turns.

Expected values are read from code; where a column could only be guessed the docstring says `UNVERIFIED`.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Final

import pytest
import websockets
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.mcp import call_tool, mcp_peer, register_mcp, tool_names
from tests.integration._support.upstream import delete_scenario, register_scenario
from tests.integration._support.wire import Reply, Request, wire_server
from tests.integration.cost_calculation.cost_tracking_case import RealtimeResponse, RealtimeTurn
from tests.integration.messages_endpoint.dec1_messages_support import Rig, echo_message
from tests.integration.messages_endpoint.dec1_msgprov_support import (
    AZURE_AI_BACKEND,
    AZURE_AI_KEY,
    AZURE_API_VERSION,
    AZURE_DEPLOYMENT,
    AZURE_KEY,
    BEDROCK_MODEL_ID,
    BEDROCK_TOKEN,
    PRICES,
    VERTEX_BACKEND,
    VERTEX_LOCATION,
    VERTEX_PROJECT,
    anthropic_body_echo,
    anthropic_error_reply,
    aws_error,
    chat_echo,
    invoke_echo,
    only_spend_row,
    openai_error_reply,
    provider_deployment,
    vertex_credentials,
    vertex_echo,
)
from tests.integration.spend.dec1_spend_support import (
    CHAT_COST,
    EMBEDDING_PROMPT_TOKENS,
    LANDING_SECONDS,
    RESPONSES_COST,
    chat_model,
    daily_rows,
    embedding_model,
    hashed,
    messages_model,
    number,
    row_by_id,
    rows_of_key,
    scripted_upstream,
    served,
)
from tests.integration.spend.dec1_spendlogs2_support import error_information, failure_rows, metadata_dict
from tests.integration.spend.dec1_spendlogs_support import failing_upstream
from tests.integration.spend.test_batch_completion_accounting import batch_input_file, batch_routes

pytestmark: Final = pytest.mark.shares_worker_state

MESSAGES_SPEND: Final = 11 * 0.001 + 7 * 0.002


def _anthropic(gateway: Gateway, respond: Callable[[Request], Reply] | None = None) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        respond or echo_message,
        model="anthropic/claude-sonnet-4-5-20250929",
        api_key="synthetic-anthropic-key",
        **PRICES,
    )


def _bedrock(gateway: Gateway, respond: Callable[[Request], Reply] | None = None) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        respond or invoke_echo,
        model=f"bedrock/invoke/{BEDROCK_MODEL_ID}",
        api_key=BEDROCK_TOKEN,
        aws_region_name="us-east-1",
        **PRICES,
    )


def _vertex(gateway: Gateway, respond: Callable[[Request], Reply] | None = None) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        respond or vertex_echo,
        model=f"vertex_ai/{VERTEX_BACKEND}",
        api_key=None,
        vertex_project=VERTEX_PROJECT,
        vertex_location=VERTEX_LOCATION,
        vertex_credentials=vertex_credentials(gateway),
        **PRICES,
    )


def _foundry(gateway: Gateway, respond: Callable[[Request], Reply] | None = None) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway, respond or anthropic_body_echo, model=f"azure_ai/{AZURE_AI_BACKEND}", api_key=AZURE_AI_KEY, **PRICES
    )


def _azure(gateway: Gateway, respond: Callable[[Request], Reply] | None = None) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        respond or chat_echo(AZURE_DEPLOYMENT),
        model=f"azure/{AZURE_DEPLOYMENT}",
        api_key=AZURE_KEY,
        api_version=AZURE_API_VERSION,
        **PRICES,
    )


PROVIDERS: Final = {
    "anthropic": _anthropic,
    "bedrock": _bedrock,
    "vertex_ai": _vertex,
    "azure_ai": _foundry,
    "azure": _azure,
}
FAILURES: Final[Mapping[str, Reply]] = {
    "anthropic": anthropic_error_reply(500, "api_error", "dec1 sl2 upstream failure"),
    "bedrock": aws_error(500, "InternalServerException", "dec1 sl2 upstream failure"),
    "vertex_ai": anthropic_error_reply(500, "api_error", "dec1 sl2 upstream failure"),
    "azure_ai": anthropic_error_reply(500, "api_error", "dec1 sl2 upstream failure"),
    "azure": openai_error_reply(500, "dec1 sl2 upstream failure", "server_error"),
}
PROVIDER_IDS: Final = tuple(PROVIDERS)


def _keyed_post(rig: Rig, **fields: JsonValue) -> tuple[str, dict[str, JsonValue]]:
    key: Final = rig.scenario.key(models=[rig.model])
    response: Final = rig.post(rig.body(**fields), key=key)
    assert response.status_code == 200, response.text
    return key, JSON_OBJECT.validate_json(response.content)


@pytest.mark.parametrize("provider", PROVIDER_IDS)
def test_messages_row_is_filed_under_the_id_the_caller_received(gateway: Gateway, provider: str) -> None:
    """One `/v1/messages` call per backend. The single row of the model group has `request_id` equal to the
    `id` in the response body, so the id a client logs is the id the spend log lookup takes. For the Azure
    OpenAI chat bridge the body id is whatever the bridge returns (UNVERIFIED whether it is the `chatcmpl-` id
    or a minted `msg_` id); the contract is only that both agree."""
    with PROVIDERS[provider](gateway) as rig:
        _, body = _keyed_post(rig)
        assert only_spend_row(rig)["request_id"] == body["id"]


@pytest.mark.parametrize("provider", PROVIDER_IDS)
def test_messages_row_names_the_calling_key_and_the_requested_group(gateway: Gateway, provider: str) -> None:
    """The same call made with a virtual key: the row's `api_key` is that key's sha256, `model_group` is the
    model name the caller sent, `call_type` is `anthropic_messages`, and `custom_llm_provider` is the backend's
    provider name."""
    with PROVIDERS[provider](gateway) as rig:
        key, _ = _keyed_post(rig)
        row: Final = only_spend_row(rig)
        assert (row["api_key"], row["model_group"], row["call_type"]) == (hashed(key), rig.model, "anthropic_messages")
        assert row["custom_llm_provider"] == provider, row


@pytest.mark.parametrize("provider", PROVIDER_IDS)
def test_streamed_messages_write_one_row_with_the_non_streamed_spend(gateway: Gateway, provider: str) -> None:
    """A streamed `/v1/messages` call on each backend writes exactly one success row with `spend` 0.025
    (11 input and 7 output tokens at 0.001 / 0.002), the same as the non-streamed call."""
    with PROVIDERS[provider](gateway) as rig:
        response, _ = rig.stream(rig.body(stream=True))
        assert response.status_code == 200
        row: Final = only_spend_row(rig)
        assert row["status"] == "success", row
        assert number(row["spend"]) == pytest.approx(MESSAGES_SPEND), row


@pytest.mark.parametrize("provider", PROVIDER_IDS)
def test_messages_provider_failure_row_records_the_upstream_status(gateway: Gateway, provider: str) -> None:
    """Each backend answers 500. The caller gets an error, and one failure row for the model group lands with
    `spend` 0 and `error_information.error_code` "500". UNVERIFIED: the shared proxy's retries do not change the
    status the row records (round 1 found one row per failed request when retries also fail)."""
    reply: Final = FAILURES[provider]
    with PROVIDERS[provider](gateway, lambda _request: reply) as rig:
        assert rig.post(rig.body()).status_code >= 500
        row: Final = only_spend_row(rig)
        assert (row["status"], number(row["spend"])) == ("failure", 0.0), row
        assert error_information(row)["error_code"] == "500", row


def test_native_and_bridged_responses_rows_share_call_type_and_daily_endpoint(gateway: Gateway) -> None:
    """`/v1/responses` on an OpenAI deployment (native) and on an Anthropic deployment (bridged through
    `/v1/messages`) each write one row with `call_type` `aresponses`, and both daily user rows carry endpoint
    `/responses` (`ROUTE_ENDPOINT_MAPPING`). Native spend is 0.017 (9 + 4 tokens) and bridged spend 0.018
    (12 + 3 tokens from the scripted Anthropic answer)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        native: Final = chat_model(scenario, wire)
        bridged: Final = messages_model(scenario, wire)
        key: Final = scenario.key(models=[native, bridged])
        for model in (native, bridged):
            response = gateway.request(
                "POST", "/v1/responses", {"model": model, "input": f"dec1 {uuid.uuid4()}"}, key=key
            )
            served(response)
        rows: Final = rows_of_key(gateway, key, 2)
        assert {row["call_type"] for row in rows} == {"aresponses"}, rows
        spends: Final = sorted(number(row["spend"]) for row in rows)
        assert spends == [pytest.approx(RESPONSES_COST), pytest.approx(0.018)], rows
        endpoints: Final = {
            object_value(row["row"])["endpoint"] for row in daily_rows(gateway, "LiteLLM_DailyUserSpend", key, 2)
        }
        assert endpoints == {"/responses"}, endpoints


def test_bridged_responses_row_names_the_anthropic_provider(gateway: Gateway) -> None:
    """The bridged `/v1/responses` row's `custom_llm_provider` is `anthropic` and its `request_id` is the `resp_`
    id the caller received, not the Anthropic `msg_` id, so `previous_response_id` lookups find it."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        body: Final = served(
            gateway.request("POST", "/v1/responses", {"model": model, "input": "dec1 bridge"}, key=key)
        )
        row: Final = row_by_id(gateway, string_value(body["id"]))
        assert row["custom_llm_provider"] == "anthropic", row
        assert str(row["request_id"]).startswith("resp_"), row


def test_an_embedding_batch_of_inputs_writes_one_row(gateway: Gateway) -> None:
    """`/v1/embeddings` with a list of three inputs is one provider call and writes one `aembedding` row with
    the provider's 7 prompt tokens and `spend` 0.007, not one row per input."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = embedding_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", "/v1/embeddings", {"model": model, "input": ["dec1 a", "dec1 b", f"dec1 {uuid.uuid4()}"]}, key=key
        )
        served(response)
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (row["call_type"], row["prompt_tokens"]) == ("aembedding", EMBEDDING_PROMPT_TOKENS), row
        assert number(row["spend"]) == pytest.approx(0.007), row


def test_an_embedding_failure_writes_one_failure_row(gateway: Gateway) -> None:
    """An embedding deployment whose provider answers 500 writes one failure row for the key with `call_type`
    `aembedding` and `spend` 0."""
    with failing_upstream(500) as wire, gateway.scenario() as scenario:
        model: Final = embedding_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": "dec1 x"}, key=key)
        assert response.status_code >= 500, response.text
        row: Final = failure_rows(gateway, key, 1)[0]
        assert (row["call_type"], number(row["spend"])) == ("aembedding", 0.0), row


@contextmanager
def _passthrough(gateway: Gateway, scenario: Scenario, status: int = 200) -> Iterator[str]:
    def answer(request: Request) -> Reply:
        if status != 200:
            return Reply(status=status, body=b'{"error":"dec1 sl2 passthrough failure"}')
        return Reply(body=json.dumps({"ok": True, "echo": request.target}).encode())

    with wire_server(answer) as wire:
        path: Final = f"/dec1-sl2-pt-{uuid.uuid4().hex[:10]}"
        created: Final = gateway.post("/config/pass_through_endpoint", {"path": path, "target": wire.url, "auth": True})
        endpoints: Final = TypeAdapter(list[JsonValue]).validate_python(created["endpoints"])
        identity: Final = str(object_value(endpoints[0])["id"])
        scenario.cleanups.callback(
            gateway.request, "DELETE", "/config/pass_through_endpoint", params={"endpoint_id": identity}
        )
        yield path


def test_each_pass_through_call_writes_one_row(gateway: Gateway) -> None:
    """Two calls to a configured pass-through route (auth on) with a virtual key write two rows for that key,
    each with `call_type` `pass_through_endpoint` and `status` `success`. The V2 gate keeps tracking pass-through
    calls even without identity (`_UNATTRIBUTED_TRACKABLE_CALL_TYPES`)."""
    with gateway.scenario() as scenario, _passthrough(gateway, scenario) as path:
        key: Final = scenario.key()
        for _ in range(2):
            assert gateway.request("POST", path, {"dec1": "sl2"}, key=key).status_code == 200
        rows: Final = rows_of_key(gateway, key, 2)
        assert {(row["call_type"], row["status"]) for row in rows} == {("pass_through_endpoint", "success")}, rows


def test_a_pass_through_call_to_an_unpriced_target_costs_nothing(gateway: Gateway) -> None:
    """The generic pass-through target returns JSON with no usage, so its row has `spend` 0 and 0 tokens.
    UNVERIFIED: no fallback pricing applies to a generic target."""
    with gateway.scenario() as scenario, _passthrough(gateway, scenario) as path:
        key: Final = scenario.key()
        assert gateway.request("POST", path, {"dec1": "sl2"}, key=key).status_code == 200
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (number(row["spend"]), row["total_tokens"]) == (0.0, 0), row


def test_a_failed_pass_through_call_writes_a_failure_row(gateway: Gateway) -> None:
    """The pass-through target answers 503. The caller gets 503 and one failure row lands for the key with
    `call_type` `pass_through_endpoint`. UNVERIFIED: the status is passed through unchanged."""
    with gateway.scenario() as scenario, _passthrough(gateway, scenario, status=503) as path:
        key: Final = scenario.key()
        assert gateway.request("POST", path, {"dec1": "sl2"}, key=key).status_code == 503
        row: Final = failure_rows(gateway, key, 1)[0]
        assert row["call_type"] == "pass_through_endpoint", row


def _mcp_rows(gateway: Gateway, key: str, count: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: gateway.rows(
            "SELECT request_id, call_type, status, spend, model, mcp_namespaced_tool_name, metadata "
            'FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s ORDER BY "startTime"',
            (hashed(key), "call_mcp_tool"),
        ),
        lambda found: len(found) == count,
        seconds=LANDING_SECONDS,
    )


def test_an_mcp_tool_call_row_names_the_namespaced_tool(gateway: Gateway) -> None:
    """A REST tool call of `add` on a fresh MCP server writes one `call_mcp_tool` row whose
    `mcp_namespaced_tool_name` is the namespaced name the caller used (`<alias>-add`) and whose `model` is
    `MCP: <alias>-add`. UNVERIFIED: the column holds the namespaced, not the bare, tool name."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "sl2" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        full: Final = tool_names(gateway, key, identity)["add"]
        assert call_tool(gateway, key, identity, full, {"a": 2, "b": 3}).status_code == 200
        row: Final = _mcp_rows(gateway, key, 1)[0]
        assert row["mcp_namespaced_tool_name"] == full, row
        assert row["model"] == f"MCP: {full}", row


def test_an_unpriced_mcp_tool_call_is_free(gateway: Gateway) -> None:
    """Without `mcp_server_cost_info` the tool call row has `spend` 0 and `status` `success`."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "sl2" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        call_tool(gateway, key, identity, tool_names(gateway, key, identity)["add"], {"a": 1, "b": 1})
        row: Final = _mcp_rows(gateway, key, 1)[0]
        assert (row["status"], number(row["spend"])) == ("success", 0.0), row


def test_a_failing_mcp_tool_writes_a_failure_row(gateway: Gateway) -> None:
    """The math service's `fail` tool raises. Its call writes one `call_mcp_tool` row with `status` `failure`.
    UNVERIFIED: a tool error result (as opposed to a transport error) is logged as a failure."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "sl2" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        call_tool(gateway, key, identity, tool_names(gateway, key, identity)["fail"], {})
        assert _mcp_rows(gateway, key, 1)[0]["status"] == "failure"


def _completed_batch(gateway: Gateway, scenario: Scenario, key: str) -> str:
    scenario_id: Final = f"sl2-batch-{uuid.uuid4().hex[:12]}"
    handle: Final = register_scenario(scenario_id, batch_routes("gpt-4o-mini"))
    scenario.cleanups.callback(delete_scenario, handle)
    model: Final = scenario.model(api_base=handle.api_base())
    uploaded: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "model": model},
        {"file": ("in.jsonl", batch_input_file(model), "application/jsonl")},
        key=key,
    )
    assert uploaded.status_code == 200, uploaded.text
    created: Final = gateway.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": string_value(JSON_OBJECT.validate_json(uploaded.content)["id"]),
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": model,
        },
        key=key,
    )
    assert created.status_code == 200, created.text
    return string_value(JSON_OBJECT.validate_json(created.content)["id"])


def _batch_cost_rows(gateway: Gateway, key: str) -> list[dict[str, JsonValue]]:
    return gateway.rows(
        'SELECT request_id, spend FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s',
        (hashed(key), "aretrieve_batch"),
    )


def test_a_completed_batch_is_charged_by_one_settle_once_row(gateway: Gateway) -> None:
    """A batch whose provider reports `completed` is retrieved three times. Exactly one `aretrieve_batch` row
    exists for the key, and its `request_id` ends with `_batch_cost` (`BATCH_COST_REQUEST_ID_SUFFIX`,
    litellm_core_utils/batch_utils.py:33). A second charge on a later poll is the plan's named double-charge
    risk."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        batch_id: Final = _completed_batch(gateway, scenario, key)
        for _ in range(3):
            retrieval = gateway.request("GET", f"/v1/batches/{batch_id}", key=key)
            assert retrieval.status_code == 200, retrieval.text
        rows: Final = eventually(lambda: _batch_cost_rows(gateway, key), lambda found: len(found) >= 1, seconds=70)
        assert len(rows) == 1, rows
        assert str(rows[0]["request_id"]).endswith("_batch_cost"), rows


def test_a_batch_cost_delivery_is_keyed_by_its_request_id(gateway: Gateway) -> None:
    """The batch cost row's delivery has `delivery_id` equal to the row's `request_id` and `is_batch` true
    (`_delivery_id` and `_settles_once`, table_repositories.py:154): that is what makes a replayed settle-once
    charge a no-op."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        batch_id: Final = _completed_batch(gateway, scenario, key)
        assert gateway.request("GET", f"/v1/batches/{batch_id}", key=key).status_code == 200
        request_id: Final = str(
            eventually(lambda: _batch_cost_rows(gateway, key), lambda found: len(found) == 1, seconds=70)[0][
                "request_id"
            ]
        )
        found: Final = gateway.rows(
            'SELECT delivery_id, is_batch FROM "LiteLLM_SpendLogDelivery" WHERE request_id = %s', (request_id,)
        )
        assert found == [{"delivery_id": request_id, "is_batch": True}], found


def _realtime_deployment(scenario: Scenario, turns: int) -> str:
    scenario_id: Final = f"sl2-rt-{uuid.uuid4().hex[:12]}"
    turn: Final = RealtimeTurn(
        input_tokens=10,
        output_tokens=5,
        input_text_tokens=10,
        input_audio_tokens=0,
        input_cached_tokens=0,
        output_text_tokens=5,
        output_audio_tokens=0,
    )
    response: Final = RealtimeResponse(
        content_type="application/x-realtime",
        events=tuple(turn.render(index, "$REQUEST_ID") for index in range(1, turns + 1)),
    )
    handle: Final = register_scenario(scenario_id, response)
    scenario.cleanups.callback(delete_scenario, handle)
    return scenario.model(
        model="openai/gpt-4o-realtime-preview",
        api_key=scenario_id,
        api_base=os.environ["INTEGRATION_UPSTREAM_URL"].rstrip("/"),
        input_cost_per_token=0.001,
        output_cost_per_token=0.002,
    )


async def _realtime_session(url: str, key: str, model: str, turns: int) -> None:
    async with websockets.connect(
        f"{url.replace('http://', 'ws://').replace('https://', 'wss://')}/v1/realtime?model={model}",
        additional_headers={"Authorization": f"Bearer {key}"},
    ) as websocket:
        await websocket.recv()
        for _ in range(turns):
            await websocket.send(json.dumps({"type": "response.create"}))
            while JSON_OBJECT.validate_json(await websocket.recv()).get("type") != "response.done":
                pass


def test_a_realtime_session_writes_one_row_for_all_its_turns(gateway: Gateway) -> None:
    """A realtime websocket session with two `response.done` turns of 10 input and 5 output text tokens each
    writes exactly one `_arealtime` row for the key with `prompt_tokens` 20 and `completion_tokens` 10 (the same
    one-row-per-session rule `cost_calculation/test_batch_realtime_cost.py` relies on). UNVERIFIED: token sums
    across turns; prices given on the deployment apply to text tokens."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        model: Final = _realtime_deployment(scenario, 2)
        asyncio.run(_realtime_session(os.environ["INTEGRATION_PROXY_URL"].rstrip("/"), key, model, 2))
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert row["call_type"] == "_arealtime", row
        assert (row["prompt_tokens"], row["completion_tokens"]) == (20, 10), row


def test_a_realtime_session_row_is_priced_from_the_summed_usage(gateway: Gateway) -> None:
    """The same two-turn session at 0.001 / 0.002 per token costs 20 * 0.001 + 10 * 0.002 = 0.04, written as the
    row's `spend` with `status` `success`. UNVERIFIED as above."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        model: Final = _realtime_deployment(scenario, 2)
        asyncio.run(_realtime_session(os.environ["INTEGRATION_PROXY_URL"].rstrip("/"), key, model, 2))
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (row["status"], number(row["spend"])) == ("success", pytest.approx(0.04)), row


def test_a_chat_row_and_a_messages_row_on_one_key_keep_their_own_call_types(gateway: Gateway) -> None:
    """One chat completion (0.02) and one Anthropic `/v1/messages` call (0.018) on the same key write two rows
    with `call_type` `acompletion` and `anthropic_messages`, and `metadata.user_api_key_hash` equal to the key hash
    on both: the route decides the call type, the key decides the attribution."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        chat: Final = chat_model(scenario, wire)
        messages: Final = messages_model(scenario, wire)
        key: Final = scenario.key(models=[chat, messages])
        served(
            gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": chat, "messages": [{"role": "user", "content": f"dec1 {uuid.uuid4()}"}]},
                key=key,
            )
        )
        served(
            gateway.request(
                "POST",
                "/v1/messages",
                {
                    "model": messages,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"dec1 {uuid.uuid4()}"}],
                },
                key=key,
            )
        )
        rows: Final = rows_of_key(gateway, key, 2)
        assert {(row["call_type"], round(number(row["spend"]), 6)) for row in rows} == {
            ("acompletion", CHAT_COST),
            ("anthropic_messages", 0.018),
        }, rows
        assert {metadata_dict(row).get("user_api_key_hash") for row in rows} == {hashed(key)}, rows
