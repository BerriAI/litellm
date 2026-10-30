"""/v1/messages on Databricks Claude names reaches the workspace's native Anthropic route with the caller's
``safeguards``, while every other Databricks name and api_base keeps the merge base's chat completions bridge or
opted-in passthrough."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import list_value, object_value, string_value
from integration._support.process import graceful_stop_seconds
from integration._support.safeguards_rig import (
    CLAUDE_MODEL,
    CLAUDE_TARGET,
    DANGEROUS_TOOL_USE_BETA,
    NATIVE_MODEL,
    NATIVE_TARGET,
    OMITTED,
    OTHER_NATIVE_MODELS,
    SAFEGUARDS,
    Rig,
    auto_mode_body,
    chat_backend,
    deployment,
    flagged,
    json_body,
    messages_body,
    native_backend,
    native_message,
    native_stream_frames,
    nonce,
    request_json,
    safeguards_rig,
    sse,
    sse_events,
    stream_results,
    text_block,
    tool_id,
    tool_use_block,
)
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)

DATABRICKS_MODEL: Final = "safeguards-databricks"
UNITY_CATALOG_MODEL: Final = "safeguards-databricks-unity-catalog"
UNITY_CATALOG_ROOT_MODEL: Final = "safeguards-databricks-unity-catalog-root"
SDK_MODEL: Final = "safeguards-databricks-sdk"
DROP_PARAMS_MODEL: Final = "safeguards-databricks-drop-params"
ENVIRONMENT_MODEL: Final = "safeguards-databricks-environment"
GPT_MODEL: Final = "safeguards-databricks-gpt"
CUSTOM_CLAUDE_MODEL: Final = "safeguards-databricks-custom-claude"
GATEWAY_MODEL: Final = "safeguards-databricks-behind-gateway"
OPTED_IN_GATEWAY_MODEL: Final = "safeguards-databricks-opted-in-gateway"
AZURE_MODEL: Final = "safeguards-azure-claude"
EDENAI_MODEL: Final = "safeguards-edenai"

SERVING_TARGET: Final = "/dbx/serving-endpoints/anthropic/v1/messages"
UNITY_CATALOG_TARGET: Final = "/dbxuc/ai-gateway/anthropic/v1/messages"
UNITY_CATALOG_ROOT_TARGET: Final = "/ai-gateway/anthropic/v1/messages"
SDK_TARGET: Final = "/dbxsdk/serving-endpoints/anthropic/v1/messages"
ENVIRONMENT_TARGET: Final = "/dbxenv/serving-endpoints/anthropic/v1/messages"
SERVING_CHAT_TARGET: Final = "/dbx/serving-endpoints/chat/completions"
GATEWAY_CHAT_TARGET: Final = "/gw/dbx/chat/completions"
OPTED_IN_GATEWAY_TARGET: Final = "/gw/optin/v1/messages"
AZURE_TARGET: Final = "/azure/anthropic/v1/messages"
DATABRICKS_KEY: Final = "synthetic-databricks-key"
ENVIRONMENT_KEY: Final = "synthetic-databricks-environment-key"
SDK_TOKEN: Final = "synthetic-databricks-sdk-token"
PARTNER_USER_AGENT: Final = "partnerapp/2.0"
INVOCATION_METRICS: Final = "amazon-bedrock-invocationMetrics"
BILLING_BLOCK: Final[Mapping[str, JsonValue]] = {
    "type": "text",
    "text": "x-anthropic-billing-header: cc_version=2.1.283.00; cc_entrypoint=sdk-cli;",
}


def environment(wire_url: str) -> Mapping[str, str]:
    return {
        "DATABRICKS_API_BASE": f"{wire_url}/dbxenv/serving-endpoints",
        "DATABRICKS_API_KEY": ENVIRONMENT_KEY,
        "DATABRICKS_USER_AGENT": PARTNER_USER_AGENT,
        "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS": "True",
    }


def sdk_environment(wire_url: str) -> Mapping[str, str]:
    return {
        "DATABRICKS_HOST": f"{wire_url}/dbxsdk",
        "DATABRICKS_TOKEN": SDK_TOKEN,
        "DATABRICKS_AUTH_TYPE": "pat",
        "DATABRICKS_CONFIG_FILE": "/dev/null",
        "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS": "True",
    }


def databricks(name: str, model: str, api_base: str, **extra: JsonValue) -> Mapping[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": f"databricks/{model}", "api_base": api_base, "api_key": DATABRICKS_KEY, **extra},
    }


def extra_models(wire_url: str) -> Sequence[Mapping[str, JsonValue]]:
    return (
        databricks(DATABRICKS_MODEL, "databricks-claude-sonnet-4-6", f"{wire_url}/dbx/serving-endpoints"),
        databricks(
            UNITY_CATALOG_MODEL,
            "system.ai.claude-sonnet-4-6",
            f"{wire_url}/dbxuc/serving-endpoints/anthropic/v1/messages",
        ),
        databricks(UNITY_CATALOG_ROOT_MODEL, "system.ai.claude-sonnet-4-6", wire_url),
        databricks(
            DROP_PARAMS_MODEL, "databricks-claude-opus-4-6", f"{wire_url}/dbx/serving-endpoints", drop_params=True
        ),
        {"model_name": ENVIRONMENT_MODEL, "litellm_params": {"model": "databricks/databricks-claude-haiku-4-5"}},
        databricks(GPT_MODEL, "databricks-gpt-oss-120b", f"{wire_url}/dbx/serving-endpoints"),
        databricks(CUSTOM_CLAUDE_MODEL, "my-claude-endpoint", f"{wire_url}/dbx/serving-endpoints"),
        databricks(GATEWAY_MODEL, "databricks-claude-sonnet-4-6", f"{wire_url}/gw/dbx"),
        {
            **databricks(OPTED_IN_GATEWAY_MODEL, "databricks-claude-sonnet-4-6", f"{wire_url}/gw/optin"),
            "model_info": {"supported_endpoints": ["/v1/messages"]},
        },
        deployment(AZURE_MODEL, "azure_ai/claude-sonnet-4-6", f"{wire_url}/azure"),
    )


def sdk_models(wire_url: str) -> Sequence[Mapping[str, JsonValue]]:
    return (
        {
            "model_name": SDK_MODEL,
            "litellm_params": {
                "model": "databricks/databricks-claude-haiku-4-5",
                "api_base": f"{wire_url}/dbxsdk/serving-endpoints",
            },
        },
    )


@pytest.fixture(scope="module")
def databricks_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with safeguards_rig(
        tmp_path_factory.mktemp("databricks-native-messages"), environment=environment, extra_models=extra_models
    ) as rig:
        yield rig


@pytest.fixture(scope="module")
def sdk_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with safeguards_rig(
        tmp_path_factory.mktemp("databricks-sdk-auth"),
        environment=sdk_environment,
        remove_environment=(
            "DATABRICKS_API_KEY",
            "DATABRICKS_API_BASE",
            "DATABRICKS_CLIENT_ID",
            "DATABRICKS_CLIENT_SECRET",
        ),
        extra_models=sdk_models,
    ) as rig:
        yield rig


def beta_map_without_the_databricks_safeguards_beta() -> bytes:
    beta_map: Final = object_value(json.loads(Path("litellm/anthropic_beta_headers_config.json").read_text()))
    databricks_column: Final = {**object_value(beta_map["databricks"]), DANGEROUS_TOOL_USE_BETA: None}
    return json.dumps({**beta_map, "databricks": databricks_column}).encode()


def serving_endpoint_model(wire_url: str) -> Sequence[Mapping[str, JsonValue]]:
    return (databricks(DATABRICKS_MODEL, "databricks-claude-sonnet-4-6", f"{wire_url}/dbx/serving-endpoints"),)


@pytest.fixture(scope="module")
def remote_beta_map_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Rig, Wire]]:
    served_map: Final = beta_map_without_the_databricks_safeguards_beta()
    with (
        wire_server(lambda _: Reply(body=served_map)) as beta_map,
        safeguards_rig(
            tmp_path_factory.mktemp("databricks-remote-beta-map"),
            environment=lambda _: {"LITELLM_ANTHROPIC_BETA_HEADERS_URL": f"{beta_map.url}/beta.json"},
            remove_environment=("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS",),
            classifier_model=OMITTED,
            extra_models=serving_endpoint_model,
        ) as rig,
    ):
        yield rig, beta_map


def scripted_message(
    identity: str, model: str = "databricks-claude-sonnet-4-6", **extra: JsonValue
) -> Mapping[str, JsonValue]:
    return native_message(identity, content=(text_block("Deleting."), tool_use_block(identity)), model=model, **extra)


def workspace_verdict(identity: str) -> Sequence[JsonValue]:
    return [
        {
            "type": "dangerous_tool_use",
            "status": {
                "type": "available",
                "tool_uses": {identity: {"type": "evaluated", "outcome": "allowed", "explanation": "workspace"}},
            },
        }
    ]


def only_backend_call(rig: Rig, request_nonce: str) -> Request:
    received: Final = rig.received(request_nonce)
    assert rig.classifier_calls(received) == (), [request.target for request in received]
    assert len(received) == 1, [request.target for request in received]
    return received[0]


def beta_values(request: Request) -> frozenset[str]:
    return frozenset(value.strip() for value in request.headers.get("anthropic-beta", "").split(",") if value.strip())


def tool_types_and_names(body: Mapping[str, JsonValue]) -> tuple[tuple[JsonValue, JsonValue], ...]:
    tools: Final = tuple(object_value(tool) for tool in list_value(body.get("tools", [])))
    return tuple((tool.get("type"), tool.get("name")) for tool in tools)


def frame_data(frame: bytes) -> dict[str, JsonValue]:
    line: Final = next(line for line in frame.splitlines() if line.startswith(b"data: "))
    return object_value(json.loads(line.removeprefix(b"data: ")))


def test_claude_serving_endpoint_reaches_the_native_route_and_keeps_the_workspace_verdict(
    databricks_rig: Rig,
) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    message: Final = scripted_message(identity, safeguard_results=list(workspace_verdict(identity)))
    databricks_rig.register(request_nonce, native_backend(message, stream=False))
    request: Final = auto_mode_body(DATABRICKS_MODEL, request_nonce)
    body: Final = json_body(databricks_rig.post(request, beta=None))
    assert body == {**message, "model": DATABRICKS_MODEL}, body
    sent: Final = only_backend_call(databricks_rig, request_nonce)
    assert sent.target == SERVING_TARGET, sent.target
    assert sent.headers.get("authorization") == f"Bearer {DATABRICKS_KEY}", sent.headers
    assert beta_values(sent) == {DANGEROUS_TOOL_USE_BETA}, sent.headers
    outbound: Final = request_json(sent)
    assert outbound["safeguards"] == request["safeguards"], outbound
    assert outbound["messages"] == request["messages"] and outbound["tools"] == request["tools"], outbound


def odd_chunks(frames: Sequence[bytes]) -> tuple[bytes, ...]:
    return tuple(bytes((byte,)) for byte in b"".join(frames))


def whole_frames(frames: Sequence[bytes]) -> tuple[bytes, ...]:
    return tuple(frames)


@pytest.mark.parametrize("reshape", (whole_frames, odd_chunks), ids=("whole_frames", "one_byte_chunks"))
def test_claude_stream_drops_invocation_metrics_and_keeps_every_other_frame(
    databricks_rig: Rig, reshape: Callable[[Sequence[bytes]], tuple[bytes, ...]]
) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    verdict: Final = workspace_verdict(identity)
    metrics: Final[Mapping[str, JsonValue]] = {"inputTokenCount": 40, "outputTokenCount": 9, "invocationLatency": 812}
    sent_frames: Final = (
        *native_stream_frames(scripted_message(identity), delta_extra={"safeguard_results": list(verdict)})[:-1],
        sse({"type": "message_stop", INVOCATION_METRICS: dict(metrics)}),
    )
    databricks_rig.register(
        request_nonce, native_backend(scripted_message(identity), stream=True, chunks=reshape(sent_frames))
    )
    status, raw = databricks_rig.stream(auto_mode_body(DATABRICKS_MODEL, request_nonce))
    assert status == 200, raw
    received_events: Final = sse_events(raw)
    sent_events: Final = tuple(frame_data(frame) for frame in sent_frames)
    assert len(received_events) == len(sent_events), received_events
    assert received_events[-1] == {"type": "message_stop"}, received_events[-1]
    assert received_events[1:-1] == sent_events[1:-1], received_events
    opening: Final = object_value(received_events[0]["message"])
    assert opening == {**object_value(sent_events[0]["message"]), "model": DATABRICKS_MODEL}, opening
    assert stream_results(raw) == verdict, raw
    assert only_backend_call(databricks_rig, request_nonce).target == SERVING_TARGET


@pytest.mark.parametrize(
    ("model", "target"),
    ((UNITY_CATALOG_MODEL, UNITY_CATALOG_TARGET), (UNITY_CATALOG_ROOT_MODEL, UNITY_CATALOG_ROOT_TARGET)),
    ids=("full_messages_url", "workspace_root"),
)
def test_unity_catalog_claude_name_reaches_the_ai_gateway_route(databricks_rig: Rig, model: str, target: str) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    message: Final = scripted_message(identity, safeguard_results=list(workspace_verdict(identity)))
    databricks_rig.register(request_nonce, native_backend(message, stream=False))
    request: Final = auto_mode_body(model, request_nonce)
    body: Final = json_body(databricks_rig.post(request))
    assert body["safeguard_results"] == workspace_verdict(identity), body
    sent: Final = only_backend_call(databricks_rig, request_nonce)
    assert sent.target == target, sent.target
    assert request_json(sent)["safeguards"] == request["safeguards"], sent.body


def test_environment_only_deployment_reaches_the_native_route_with_the_environment_key_and_partner_agent(
    databricks_rig: Rig,
) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    databricks_rig.register(request_nonce, native_backend(scripted_message(identity), stream=False))
    json_body(databricks_rig.post(auto_mode_body(ENVIRONMENT_MODEL, request_nonce)))
    sent: Final = only_backend_call(databricks_rig, request_nonce)
    assert sent.target == ENVIRONMENT_TARGET, sent.target
    assert sent.headers.get("authorization") == f"Bearer {ENVIRONMENT_KEY}", sent.headers
    assert sent.headers.get("user-agent", "").startswith("partnerapp_litellm/"), sent.headers


def test_client_anthropic_headers_leave_sdk_auth_in_charge_of_the_native_route(sdk_rig: Rig) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    sdk_rig.register(
        request_nonce, native_backend(scripted_message(identity, model="databricks-claude-haiku-4-5"), stream=False)
    )
    json_body(
        sdk_rig.post(
            auto_mode_body(SDK_MODEL, request_nonce),
            beta="context-1m-2025-08-07,fast-mode-2026-02-01",
            headers={"anthropic-version": "2023-06-01"},
        )
    )
    sent: Final = only_backend_call(sdk_rig, request_nonce)
    assert sent.target == SDK_TARGET, sent.target
    assert sent.headers.get("authorization") == f"Bearer {SDK_TOKEN}", sent.headers
    assert beta_values(sent) == {"context-1m-2025-08-07", DANGEROUS_TOOL_USE_BETA}, sent.headers
    assert sent.headers.get("anthropic-version") == "2023-06-01", sent.headers


@pytest.mark.parametrize(
    ("model", "target"),
    (
        (GPT_MODEL, SERVING_CHAT_TARGET),
        (CUSTOM_CLAUDE_MODEL, SERVING_CHAT_TARGET),
        (GATEWAY_MODEL, GATEWAY_CHAT_TARGET),
    ),
    ids=("gpt_oss", "custom_endpoint_named_claude", "api_base_outside_workspace_routes"),
)
def test_other_names_and_api_bases_keep_the_chat_completions_bridge(
    databricks_rig: Rig, model: str, target: str
) -> None:
    request_nonce: Final = nonce()
    databricks_rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
    body: Final = json_body(
        databricks_rig.post(
            auto_mode_body(model, request_nonce),
            beta=f"context-1m-2025-08-07,{DANGEROUS_TOOL_USE_BETA}",
            headers={"anthropic-version": "2023-06-01"},
        )
    )
    assert body["type"] == "message", body
    received: Final = databricks_rig.received(request_nonce)
    backend_calls: Final = databricks_rig.backend_calls(received)
    assert [call.target for call in backend_calls] == [target], received
    assert "safeguards" not in request_json(backend_calls[0]), backend_calls[0].body
    assert "anthropic-beta" not in backend_calls[0].headers, backend_calls[0].headers
    assert "anthropic-version" not in backend_calls[0].headers, backend_calls[0].headers


def gateway_rejecting_the_prompt_caching_scope_beta(message: Mapping[str, JsonValue]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if "prompt-caching-scope-2026-01-05" not in beta_values(request):
            return Reply(body=json.dumps(message).encode())
        rejection: Final = "Unexpected value(s) `prompt-caching-scope-2026-01-05` for the `anthropic-beta` header."
        return Reply(status=400, body=json.dumps({"error_code": "BAD_REQUEST", "message": rejection}).encode())

    return respond


def test_opted_in_passthrough_api_base_gets_no_client_anthropic_beta(databricks_rig: Rig) -> None:
    request_nonce: Final = nonce()
    message: Final = native_message(
        tool_id(request_nonce),
        content=(text_block("Done."),),
        stop_reason="end_turn",
        model="databricks-claude-sonnet-4-6",
    )
    databricks_rig.register(request_nonce, gateway_rejecting_the_prompt_caching_scope_beta(message))
    body: Final = json_body(
        databricks_rig.post(
            messages_body(OPTED_IN_GATEWAY_MODEL, request_nonce),
            beta="prompt-caching-scope-2026-01-05,context-1m-2025-08-07",
            headers={"anthropic-version": "2023-06-01"},
        )
    )
    assert body["id"] == message["id"] and body["content"] == message["content"], body
    sent: Final = only_backend_call(databricks_rig, request_nonce)
    assert sent.target == OPTED_IN_GATEWAY_TARGET, sent.target
    assert "anthropic-beta" not in sent.headers, sent.headers


def test_chat_completions_on_a_databricks_model_carries_no_client_anthropic_headers(databricks_rig: Rig) -> None:
    request_nonce: Final = nonce()
    databricks_rig.register(
        request_nonce, chat_backend(tool_id(request_nonce), stream=False, text="ok", finish_reason="stop")
    )
    response: Final = databricks_rig.gateway.client.post(
        "/v1/chat/completions",
        json={"model": GPT_MODEL, "messages": [{"role": "user", "content": f"say ok {request_nonce}"}]},
        headers={
            "Authorization": f"Bearer {databricks_rig.gateway.key}",
            "anthropic-beta": f"context-1m-2025-08-07,{DANGEROUS_TOOL_USE_BETA}",
            "anthropic-version": "2023-06-01",
        },
        timeout=120,
    )
    assert response.status_code == 200, response.text
    sent: Final = only_backend_call(databricks_rig, request_nonce)
    assert sent.target == SERVING_CHAT_TARGET, sent.target
    assert "anthropic-beta" not in sent.headers, sent.headers
    assert "anthropic-version" not in sent.headers, sent.headers


def test_client_betas_reach_the_workspace_filtered_by_its_beta_map(databricks_rig: Rig) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    databricks_rig.register(request_nonce, native_backend(scripted_message(identity), stream=False))
    client_betas: Final = (
        "context-1m-2025-08-07",
        "prompt-caching-scope-2026-01-05",
        "fast-mode-2026-02-01",
        DANGEROUS_TOOL_USE_BETA,
    )
    json_body(databricks_rig.post(auto_mode_body(DATABRICKS_MODEL, request_nonce), beta=",".join(client_betas)))
    sent: Final = only_backend_call(databricks_rig, request_nonce)
    assert beta_values(sent) == {"context-1m-2025-08-07", DANGEROUS_TOOL_USE_BETA}, sent.headers


def test_native_route_drops_container_hoists_system_turns_and_keeps_the_billing_block(databricks_rig: Rig) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    databricks_rig.register(request_nonce, native_backend(scripted_message(identity), stream=False))
    scoped_system: Final[Mapping[str, JsonValue]] = {
        "type": "text",
        "text": "Synthetic agent instructions.",
        "cache_control": {"type": "ephemeral", "scope": "global"},
    }
    request: Final = {
        **auto_mode_body(DATABRICKS_MODEL, request_nonce),
        "system": [dict(BILLING_BLOCK), dict(scoped_system)],
        "messages": [
            {"role": "system", "content": "leading operator note"},
            {"role": "user", "content": f"clean the build directory {request_nonce}"},
        ],
        "container": "container_synthetic_0001",
    }
    json_body(databricks_rig.post(request))
    outbound: Final = request_json(only_backend_call(databricks_rig, request_nonce))
    assert "container" not in outbound, outbound
    system: Final = tuple(object_value(block) for block in list_value(outbound["system"]))
    assert system[0] == BILLING_BLOCK, system
    assert {**scoped_system, "cache_control": {"type": "ephemeral"}} in system, system
    assert "leading operator note" in json.dumps(system), system
    roles: Final = tuple(string_value(object_value(message)["role"]) for message in list_value(outbound["messages"]))
    assert roles == ("user",), outbound["messages"]


SERVER_TOOLS: Final[Sequence[JsonValue]] = (
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 1},
    {"type": "web_fetch_20250910", "name": "web_fetch"},
    {"type": "code_execution_20250825", "name": "code_execution"},
)
BASH_SERVER_TOOL: Final[Mapping[str, JsonValue]] = {"type": "bash_20250124", "name": "bash"}


@pytest.mark.parametrize(
    ("tool_choice", "kept_tool_choice"),
    (
        ({"type": "tool", "name": "web_search"}, None),
        ({"type": "tool", "name": "Bash"}, {"type": "tool", "name": "Bash"}),
        ({"type": "auto"}, {"type": "auto"}),
    ),
    ids=("forced_dropped_tool", "forced_kept_tool", "auto"),
)
def test_drop_params_drops_only_the_server_tools_the_workspace_rejects(
    databricks_rig: Rig, tool_choice: Mapping[str, JsonValue], kept_tool_choice: Mapping[str, JsonValue] | None
) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    databricks_rig.register(request_nonce, native_backend(scripted_message(identity), stream=False))
    request: Final = auto_mode_body(DROP_PARAMS_MODEL, request_nonce)
    tools: Final = [*list_value(request["tools"]), *SERVER_TOOLS, dict(BASH_SERVER_TOOL)]
    json_body(databricks_rig.post({**request, "tools": tools, "tool_choice": dict(tool_choice)}))
    outbound: Final = request_json(only_backend_call(databricks_rig, request_nonce))
    assert tool_types_and_names(outbound) == ((None, "Bash"), ("bash_20250124", "bash")), outbound["tools"]
    assert outbound.get("tool_choice") == kept_tool_choice, outbound


def test_drop_params_with_only_rejected_server_tools_sends_no_tools_and_no_tool_choice(databricks_rig: Rig) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    databricks_rig.register(
        request_nonce,
        native_backend(native_message(identity, content=(text_block("Done."),), stop_reason="end_turn"), stream=False),
    )
    request: Final = auto_mode_body(DROP_PARAMS_MODEL, request_nonce)
    json_body(databricks_rig.post({**request, "tools": list(SERVER_TOOLS), "tool_choice": {"type": "any"}}))
    outbound: Final = request_json(only_backend_call(databricks_rig, request_nonce))
    assert "tools" not in outbound and "tool_choice" not in outbound, outbound


def workspace_rejecting_server_tools(request: Request) -> Reply:
    message: Final = "tool type 'web_search_20250305' is not supported for this model"
    return Reply(status=400, body=json.dumps({"error_code": "BAD_REQUEST", "message": message}).encode())


def test_without_drop_params_server_tools_reach_the_workspace_as_sent(databricks_rig: Rig) -> None:
    request_nonce: Final = nonce()
    databricks_rig.register(request_nonce, workspace_rejecting_server_tools)
    request: Final = auto_mode_body(DATABRICKS_MODEL, request_nonce)
    tools: Final = [*list_value(request["tools"]), *SERVER_TOOLS]
    response: Final = databricks_rig.post({**request, "tools": tools})
    assert response.status_code == 400, response.text
    assert "not supported for this model" in response.text, response.text
    outbound: Final = request_json(only_backend_call(databricks_rig, request_nonce))
    assert outbound["tools"] == tools, outbound["tools"]


DERIVES_THE_BETA: Final[Mapping[str, tuple[str, bool]]] = {
    CLAUDE_MODEL: (CLAUDE_TARGET, True),
    AZURE_MODEL: (AZURE_TARGET, True),
    DATABRICKS_MODEL: (SERVING_TARGET, True),
    NATIVE_MODEL: (NATIVE_TARGET, False),
    "safeguards-tencent": (OTHER_NATIVE_MODELS["safeguards-tencent"], False),
    "safeguards-minimax": (OTHER_NATIVE_MODELS["safeguards-minimax"], False),
    "safeguards-openai-like": (OTHER_NATIVE_MODELS["safeguards-openai-like"], False),
}


@pytest.mark.parametrize(("model", "expected"), tuple(DERIVES_THE_BETA.items()), ids=tuple(DERIVES_THE_BETA))
def test_safeguards_without_a_client_beta_derive_it_only_where_the_beta_map_allows(
    databricks_rig: Rig, model: str, expected: tuple[str, bool]
) -> None:
    target, derives = expected
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    databricks_rig.register(
        request_nonce,
        native_backend(scripted_message(identity, safeguard_results=list(workspace_verdict(identity))), stream=False),
    )
    json_body(databricks_rig.post(auto_mode_body(model, request_nonce), beta=None))
    backend_calls: Final = databricks_rig.backend_calls(databricks_rig.received(request_nonce))
    assert [call.target for call in backend_calls] == [target], backend_calls
    assert request_json(backend_calls[0])["safeguards"] == list(SAFEGUARDS), backend_calls[0].body
    assert (DANGEROUS_TOOL_USE_BETA in beta_values(backend_calls[0])) is derives, backend_calls[0].headers


def workspace_rejecting_safeguards(message: Mapping[str, JsonValue]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if "safeguards" not in request_json(request) and DANGEROUS_TOOL_USE_BETA not in beta_values(request):
            return Reply(body=json.dumps(message).encode())
        rejection: Final = "Unknown field `safeguards` or beta `dangerous-tool-use-2026-09-03` for this workspace."
        return Reply(status=400, body=json.dumps({"error_code": "BAD_REQUEST", "message": rejection}).encode())

    return respond


@pytest.mark.parametrize("beta", (DANGEROUS_TOOL_USE_BETA, None), ids=("client_beta", "no_client_beta"))
def test_a_loaded_beta_map_without_the_databricks_safeguards_beta_drops_safeguards_and_the_beta(
    remote_beta_map_rig: tuple[Rig, Wire], beta: str | None
) -> None:
    rig, beta_map = remote_beta_map_rig
    request_nonce: Final = nonce()
    message: Final = scripted_message(tool_id(request_nonce))
    rig.register(request_nonce, workspace_rejecting_safeguards(message))
    body: Final = json_body(rig.post(auto_mode_body(DATABRICKS_MODEL, request_nonce), beta=beta))
    assert body == {**message, "model": DATABRICKS_MODEL}, body
    sent: Final = only_backend_call(rig, request_nonce)
    assert sent.target == SERVING_TARGET, sent.target
    assert "safeguards" not in request_json(sent), sent.body
    assert DANGEROUS_TOOL_USE_BETA not in beta_values(sent), sent.headers
    assert beta_map.received.qsize() >= 1, "the proxy never fetched the served beta map"


@pytest.mark.parametrize("beta", (DANGEROUS_TOOL_USE_BETA, None), ids=("client_beta", "no_client_beta"))
def test_json_provider_gets_neither_safeguards_nor_the_beta_and_the_proxy_classifies(
    databricks_rig: Rig, beta: str | None
) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    databricks_rig.register(request_nonce, native_backend(scripted_message(identity, model="gpt-4o"), stream=False))
    body: Final = json_body(databricks_rig.post(auto_mode_body(EDENAI_MODEL, request_nonce), beta=beta))
    assert body["safeguard_results"] == flagged((identity,)), body
    received: Final = databricks_rig.received(request_nonce)
    assert len(databricks_rig.classifier_calls(received)) == 1, [request.target for request in received]
    backend_calls: Final = databricks_rig.backend_calls(received)
    assert [call.target for call in backend_calls] == [OTHER_NATIVE_MODELS[EDENAI_MODEL]], backend_calls
    assert "safeguards" not in request_json(backend_calls[0]), backend_calls[0].body
    assert DANGEROUS_TOOL_USE_BETA not in beta_values(backend_calls[0]), backend_calls[0].headers
