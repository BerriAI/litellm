import json
import sys
from collections.abc import AsyncIterator
from types import ModuleType, SimpleNamespace
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.databricks.chat.transformation import DatabricksConfig
from litellm.llms.databricks.messages.transformation import DatabricksAnthropicMessagesConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager

DATABRICKS_API_BASE: Final = "https://my.workspace.cloud.databricks.com/serving-endpoints"
DATABRICKS_API_KEY: Final = "dapimykey"
CLAUDE_MODEL: Final = "databricks-claude-sonnet-4-6"
UNITY_CATALOG_CLAUDE_MODEL: Final = "system.ai.claude-sonnet-4-6"
WORKSPACE_URL: Final = "https://my.workspace.cloud.databricks.com"
MESSAGES_URL: Final = f"{DATABRICKS_API_BASE}/anthropic/v1/messages"
ANTHROPIC_ROUTE_API_BASE: Final = f"{DATABRICKS_API_BASE}/anthropic"
OPTED_IN_MODEL_INFO: Final = {"supported_endpoints": ["/v1/messages"]}
AI_GATEWAY_MESSAGES_URL: Final = f"{WORKSPACE_URL}/ai-gateway/anthropic/v1/messages"
DANGEROUS_TOOL_USE_BETA: Final = "dangerous-tool-use-2026-09-03"
AUTO_MODE_SAFEGUARDS: Final = [
    {"type": "dangerous_tool_use", "classifier_context": {"v": 1, "permission_mode": "auto"}}
]
SAFEGUARD_RESULTS: Final = [
    {
        "type": "dangerous_tool_use",
        "status": {"type": "available", "tool_uses": {"toolu_bdrk_01": {"type": "evaluated", "outcome": "not_flagged"}}},
    }
]
BASH_TOOL: Final = {
    "name": "Bash",
    "description": "Run a shell command",
    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
}
USER_TURN: Final = [{"role": "user", "content": "Use the Bash tool to run: echo hello"}]


@pytest.mark.parametrize(
    ("model", "messages_url"),
    [(CLAUDE_MODEL, MESSAGES_URL), (UNITY_CATALOG_CLAUDE_MODEL, AI_GATEWAY_MESSAGES_URL)],
)
def test_foundation_claude_names_dispatch_to_a_native_messages_route(model: str, messages_url: str) -> None:
    config: Final = ProviderConfigManager.get_provider_anthropic_messages_config(
        model=model, provider=litellm.LlmProviders.DATABRICKS
    )

    assert config is not None
    assert (
        config.get_complete_url(
            api_base=DATABRICKS_API_BASE, api_key=None, model=model, optional_params={}, litellm_params={}
        )
        == messages_url
    )


@pytest.mark.parametrize("model", ["databricks-gpt-oss-120b", "my-claude-finetune", "main.default.claude-proxy"])
def test_names_outside_the_foundation_claude_models_keep_the_chat_completions_bridge(model: str) -> None:
    config: Final = ProviderConfigManager.get_provider_anthropic_messages_config(
        model=model, provider=litellm.LlmProviders.DATABRICKS
    )

    assert config is None


@pytest.mark.parametrize(
    "api_base",
    [DATABRICKS_API_BASE, f"{DATABRICKS_API_BASE}/", ANTHROPIC_ROUTE_API_BASE, MESSAGES_URL],
)
def test_complete_url_targets_the_workspace_anthropic_messages_route(api_base: str) -> None:
    url: Final = DatabricksAnthropicMessagesConfig().get_complete_url(
        api_base=api_base, api_key=None, model=CLAUDE_MODEL, optional_params={}, litellm_params={}
    )

    assert url == MESSAGES_URL


def test_validate_environment_signs_with_the_pat_and_keeps_the_workspace_base() -> None:
    headers, api_base = DatabricksAnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers={},
        model=CLAUDE_MODEL,
        messages=USER_TURN,
        optional_params={},
        litellm_params={},
        api_key=DATABRICKS_API_KEY,
        api_base=DATABRICKS_API_BASE,
    )

    assert api_base == DATABRICKS_API_BASE
    assert headers["Authorization"] == f"Bearer {DATABRICKS_API_KEY}"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"
    assert headers["User-Agent"].startswith("litellm/")


def test_validate_environment_reads_the_workspace_credentials_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_API_KEY", DATABRICKS_API_KEY)
    monkeypatch.setenv("DATABRICKS_API_BASE", DATABRICKS_API_BASE)
    monkeypatch.delenv("DATABRICKS_CLIENT_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_CLIENT_SECRET", raising=False)
    config: Final = DatabricksAnthropicMessagesConfig()

    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model=CLAUDE_MODEL,
        messages=USER_TURN,
        optional_params={},
        litellm_params={},
    )
    url: Final = config.get_complete_url(
        api_base=api_base, api_key=None, model=CLAUDE_MODEL, optional_params={}, litellm_params={}
    )

    assert headers["Authorization"] == f"Bearer {DATABRICKS_API_KEY}"
    assert url == MESSAGES_URL


@pytest.mark.parametrize(
    "api_base",
    [DATABRICKS_API_BASE, f"{DATABRICKS_API_BASE}/", ANTHROPIC_ROUTE_API_BASE, MESSAGES_URL],
)
def test_validation_then_url_building_appends_the_messages_route_once(api_base: str) -> None:
    config: Final = DatabricksAnthropicMessagesConfig()
    _, validated_api_base = config.validate_anthropic_messages_environment(
        headers={},
        model=CLAUDE_MODEL,
        messages=USER_TURN,
        optional_params={},
        litellm_params={},
        api_key=DATABRICKS_API_KEY,
        api_base=api_base,
    )

    url: Final = config.get_complete_url(
        api_base=validated_api_base, api_key=DATABRICKS_API_KEY, model=CLAUDE_MODEL, optional_params={}, litellm_params={}
    )

    assert url == MESSAGES_URL


@pytest.mark.parametrize(
    "api_base",
    [
        DATABRICKS_API_BASE,
        WORKSPACE_URL,
        f"{WORKSPACE_URL}/",
        f"{WORKSPACE_URL}/ai-gateway/mlflow/v1",
        f"{WORKSPACE_URL}/ai-gateway/anthropic",
        AI_GATEWAY_MESSAGES_URL,
        ANTHROPIC_ROUTE_API_BASE,
        MESSAGES_URL,
        f"{MESSAGES_URL}/",
    ],
)
def test_unity_catalog_claude_names_route_to_the_ai_gateway_messages_route(api_base: str) -> None:
    config: Final = DatabricksAnthropicMessagesConfig()
    _, validated_api_base = config.validate_anthropic_messages_environment(
        headers={},
        model=UNITY_CATALOG_CLAUDE_MODEL,
        messages=USER_TURN,
        optional_params={},
        litellm_params={},
        api_key=DATABRICKS_API_KEY,
        api_base=api_base,
    )

    url: Final = config.get_complete_url(
        api_base=validated_api_base,
        api_key=DATABRICKS_API_KEY,
        model=UNITY_CATALOG_CLAUDE_MODEL,
        optional_params={},
        litellm_params={},
    )

    assert url == AI_GATEWAY_MESSAGES_URL


@pytest.mark.parametrize(
    ("model", "chat_url", "messages_url"),
    [
        (CLAUDE_MODEL, f"{DATABRICKS_API_BASE}/chat/completions", MESSAGES_URL),
        (UNITY_CATALOG_CLAUDE_MODEL, f"{WORKSPACE_URL}/ai-gateway/mlflow/v1/chat/completions", AI_GATEWAY_MESSAGES_URL),
    ],
)
def test_messages_and_chat_routes_agree_on_which_names_use_the_ai_gateway(
    model: str, chat_url: str, messages_url: str
) -> None:
    chat: Final = DatabricksConfig().get_complete_url(
        api_base=DATABRICKS_API_BASE, api_key=DATABRICKS_API_KEY, model=model, optional_params={}, litellm_params={}
    )
    messages: Final = DatabricksAnthropicMessagesConfig().get_complete_url(
        api_base=DATABRICKS_API_BASE, api_key=DATABRICKS_API_KEY, model=model, optional_params={}, litellm_params={}
    )

    assert (chat, messages) == (chat_url, messages_url)


def test_partner_user_agent_matches_the_chat_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_USER_AGENT", "partner/1.0")
    messages_headers, _ = DatabricksAnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers={},
        model=CLAUDE_MODEL,
        messages=USER_TURN,
        optional_params={},
        litellm_params={},
        api_key=DATABRICKS_API_KEY,
        api_base=DATABRICKS_API_BASE,
    )
    chat_headers: Final = DatabricksConfig().validate_environment(
        headers={},
        model=CLAUDE_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        api_key=DATABRICKS_API_KEY,
        api_base=DATABRICKS_API_BASE,
    )

    assert messages_headers["User-Agent"].startswith("partner_litellm/")
    assert messages_headers["User-Agent"] == chat_headers["User-Agent"]


def test_request_transform_normalizes_system_turns_and_strips_cache_scope() -> None:
    request: Final = DatabricksAnthropicMessagesConfig().transform_anthropic_messages_request(
        model=CLAUDE_MODEL,
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "system", "content": "mid-turn instruction"},
            {"role": "user", "content": "second"},
        ],
        anthropic_messages_optional_request_params={
            "max_tokens": 64,
            "system": [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral", "scope": "global"}}],
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert [message["role"] for message in request["messages"]] == ["user", "assistant", "user", "user"]
    assert "mid-turn instruction" in json.dumps(request["messages"][2]["content"])
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_request_transform_keeps_the_billing_header_block() -> None:
    billing_block: Final = {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.296; cc_entrypoint=cli"}
    request: Final = DatabricksAnthropicMessagesConfig().transform_anthropic_messages_request(
        model=CLAUDE_MODEL,
        messages=[{"role": "user", "content": "first"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 64,
            "system": [billing_block, {"type": "text", "text": "be brief"}],
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["system"] == [billing_block, {"type": "text", "text": "be brief"}]


def _reply(request: httpx.Request, body: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        json=body,
        request=request,
    )


SERVER_TOOLS: Final = [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
    {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1},
    {"type": "code_execution_20250825", "name": "code_execution"},
]


async def _workspace_request_body(**request_params: object) -> dict[str, object]:
    received: Final[list[httpx.Request]] = []  # mutable-ok: the fake workspace records what litellm sent

    def workspace_answers(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return _reply(
            request,
            {
                "id": "msg_bdrk_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 6},
            },
        )

    upstream: Final = AsyncHTTPHandler(transport=httpx.MockTransport(workspace_answers))
    try:
        await litellm.anthropic.messages.acreate(
            max_tokens=64,
            messages=USER_TURN,
            model=f"databricks/{CLAUDE_MODEL}",
            api_key=DATABRICKS_API_KEY,
            api_base=DATABRICKS_API_BASE,
            client=upstream,
            **request_params,
        )
    finally:
        await upstream.close()
    assert len(received) == 1
    return json.loads(received[0].content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tools", "tool_choice", "expected_tools", "expected_tool_choice"),
    [
        ([*SERVER_TOOLS, BASH_TOOL], {"type": "any"}, [BASH_TOOL], {"type": "any"}),
        ([*SERVER_TOOLS, BASH_TOOL], {"type": "tool", "name": "web_search"}, [BASH_TOOL], None),
        (SERVER_TOOLS[:1], {"type": "auto"}, None, None),
        ([{"type": "web_search_20250305", "max_uses": 8}, BASH_TOOL], {"type": "any"}, [BASH_TOOL], {"type": "any"}),
    ],
)
async def test_drop_params_drops_the_server_tools_the_workspace_rejects(
    tools: list[dict[str, object]],
    tool_choice: dict[str, object],
    expected_tools: list[dict[str, object]] | None,
    expected_tool_choice: dict[str, object] | None,
) -> None:
    body: Final = await _workspace_request_body(tools=tools, tool_choice=tool_choice, drop_params=True)

    assert body.get("tools") == expected_tools
    assert body.get("tool_choice") == expected_tool_choice


@pytest.mark.asyncio
async def test_server_tools_reach_the_workspace_as_sent_without_drop_params() -> None:
    body: Final = await _workspace_request_body(tools=[*SERVER_TOOLS, BASH_TOOL], tool_choice={"type": "any"})

    assert body["tools"] == [*SERVER_TOOLS, BASH_TOOL]
    assert body["tool_choice"] == {"type": "any"}


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_params", [True, False])
@pytest.mark.parametrize("container", ["cntr_abc", {"id": "cntr_abc"}])
async def test_container_never_reaches_the_workspace(drop_params: bool, container: object) -> None:
    body: Final = await _workspace_request_body(
        container=container, metadata={"user_id": "user-1"}, drop_params=drop_params
    )

    assert "container" not in body
    assert body["metadata"] == {"user_id": "user-1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("user_agent_param", ["databricks_user_agent", "user_agent"])
async def test_per_call_partner_user_agent_reaches_the_workspace_like_the_chat_route(user_agent_param: str) -> None:
    received: Final[list[httpx.Request]] = []  # mutable-ok: the fake workspace records what litellm sent

    def workspace_answers(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return _reply(
            request,
            {
                "id": "msg_bdrk_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 6},
            },
        )

    upstream: Final = AsyncHTTPHandler(transport=httpx.MockTransport(workspace_answers))
    try:
        await litellm.anthropic.messages.acreate(
            max_tokens=64,
            messages=USER_TURN,
            model=f"databricks/{CLAUDE_MODEL}",
            api_key=DATABRICKS_API_KEY,
            api_base=DATABRICKS_API_BASE,
            client=upstream,
            **{user_agent_param: "partnerx/1.0"},
        )
    finally:
        await upstream.close()
    chat_headers: Final = DatabricksConfig().validate_environment(
        headers={},
        model=CLAUDE_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        optional_params={user_agent_param: "partnerx/1.0"},
        litellm_params={},
        api_key=DATABRICKS_API_KEY,
        api_base=DATABRICKS_API_BASE,
    )

    assert len(received) == 1
    assert received[0].headers["user-agent"].startswith("partnerx_litellm/")
    assert received[0].headers["user-agent"] == chat_headers["User-Agent"]
    assert user_agent_param not in json.loads(received[0].content)


@pytest.mark.asyncio
async def test_safeguards_reach_the_workspace_and_safeguard_results_come_back(
    local_beta_headers_config: None,
) -> None:
    received: Final[list[httpx.Request]] = []  # mutable-ok: the fake workspace records what the proxy sent

    def workspace_answers(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return _reply(
            request,
            {
                "id": "msg_bdrk_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [
                    {"type": "tool_use", "id": "toolu_bdrk_01", "name": "Bash", "input": {"command": "echo hello"}}
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 6},
                "safeguard_results": SAFEGUARD_RESULTS,
            },
        )

    upstream: Final = AsyncHTTPHandler(transport=httpx.MockTransport(workspace_answers))
    try:
        response = await litellm.anthropic.messages.acreate(
            max_tokens=256,
            messages=USER_TURN,
            model=f"databricks/{CLAUDE_MODEL}",
            api_key=DATABRICKS_API_KEY,
            api_base=DATABRICKS_API_BASE,
            tools=[BASH_TOOL],
            safeguards=AUTO_MODE_SAFEGUARDS,
            client=upstream,
        )
    finally:
        await upstream.close()

    assert [str(request.url) for request in received] == [MESSAGES_URL]
    sent: Final = received[0]
    sent_body: Final = json.loads(sent.content)
    assert sent.headers["authorization"] == f"Bearer {DATABRICKS_API_KEY}"
    assert sent.headers["anthropic-beta"] == DANGEROUS_TOOL_USE_BETA
    assert sent_body["safeguards"] == AUTO_MODE_SAFEGUARDS
    assert sent_body["model"] == CLAUDE_MODEL
    assert response["safeguard_results"] == SAFEGUARD_RESULTS
    assert response["content"][0]["id"] == "toolu_bdrk_01"


@pytest.mark.asyncio
async def test_streaming_message_delta_keeps_safeguard_results(local_beta_headers_config: None) -> None:
    message_delta: Final = {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None, "safeguard_results": SAFEGUARD_RESULTS},
        "usage": {"output_tokens": 6},
    }
    events: Final = (
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_bdrk_01",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 12, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_bdrk_01", "name": "Bash", "input": {}},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"command": "echo hello"}'},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", message_delta),
        ("message_stop", {"type": "message_stop"}),
    )
    sse_body: Final = "".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events).encode()
    received: Final[list[httpx.Request]] = []  # mutable-ok: the fake workspace records what the proxy sent

    def workspace_streams(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=sse_body,
            request=request,
        )

    upstream: Final = AsyncHTTPHandler(transport=httpx.MockTransport(workspace_streams))
    try:
        stream = await litellm.anthropic.messages.acreate(
            max_tokens=256,
            messages=USER_TURN,
            model=f"databricks/{CLAUDE_MODEL}",
            api_key=DATABRICKS_API_KEY,
            api_base=DATABRICKS_API_BASE,
            tools=[BASH_TOOL],
            safeguards=AUTO_MODE_SAFEGUARDS,
            stream=True,
            client=upstream,
        )
        relayed: Final = b"".join([chunk async for chunk in stream])
    finally:
        await upstream.close()

    assert [str(request.url) for request in received] == [MESSAGES_URL]
    assert received[0].headers["anthropic-beta"] == DANGEROUS_TOOL_USE_BETA
    sent_body: Final = json.loads(received[0].content)
    assert sent_body["safeguards"] == AUTO_MODE_SAFEGUARDS
    assert sent_body["stream"] is True
    relayed_deltas: Final = [
        json.loads(line.removeprefix(b"data: "))
        for line in relayed.splitlines()
        if line.startswith(b'data: {"type": "message_delta"')
    ]
    assert len(relayed_deltas) == 1, relayed.decode()
    assert relayed_deltas[0]["delta"]["safeguard_results"] == SAFEGUARD_RESULTS
    assert b'"id": "toolu_bdrk_01"' in relayed


CHAT_COMPLETION: Final = {
    "id": "chatcmpl_01",
    "object": "chat.completion",
    "created": 1792227137,
    "model": "claude-sonnet-4-6",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
}
MESSAGE: Final = {
    "id": "msg_bdrk_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 6},
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "api_base", "request_params", "expected_url"),
    [
        (UNITY_CATALOG_CLAUDE_MODEL, "http://gateway.internal/dbx", {}, "http://gateway.internal/dbx/chat/completions"),
        (CLAUDE_MODEL, "http://gateway.internal/dbx", {}, "http://gateway.internal/dbx/chat/completions"),
        (
            CLAUDE_MODEL,
            "http://gateway.internal/dbx-anthropic",
            {"model_info": OPTED_IN_MODEL_INFO},
            "http://gateway.internal/dbx-anthropic/v1/messages",
        ),
        (CLAUDE_MODEL, ANTHROPIC_ROUTE_API_BASE, {"model_info": OPTED_IN_MODEL_INFO}, MESSAGES_URL),
        (CLAUDE_MODEL, "http://gateway.internal", {}, "http://gateway.internal/chat/completions"),
        (UNITY_CATALOG_CLAUDE_MODEL, WORKSPACE_URL, {}, AI_GATEWAY_MESSAGES_URL),
        (UNITY_CATALOG_CLAUDE_MODEL, f"{WORKSPACE_URL}/", {}, AI_GATEWAY_MESSAGES_URL),
    ],
)
async def test_each_api_base_reaches_a_route_the_workspace_serves(
    model: str, api_base: str, request_params: dict[str, object], expected_url: str
) -> None:
    received: Final[list[httpx.Request]] = []  # mutable-ok: the fake upstream records what litellm sent

    def upstream_answers(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return _reply(request, CHAT_COMPLETION if request.url.path.endswith("/chat/completions") else MESSAGE)

    upstream: Final = AsyncHTTPHandler(transport=httpx.MockTransport(upstream_answers))
    try:
        await litellm.anthropic.messages.acreate(
            max_tokens=64,
            messages=USER_TURN,
            model=f"databricks/{model}",
            api_key=DATABRICKS_API_KEY,
            api_base=api_base,
            client=upstream,
            **request_params,
        )
    finally:
        await upstream.close()

    assert [str(request.url) for request in received] == [expected_url]


SDK_TOKEN: Final = "sdk-test-token"


@pytest.mark.asyncio
async def test_client_anthropic_headers_keep_the_sdk_credentials_on_the_native_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("DATABRICKS_API_BASE", "DATABRICKS_API_KEY", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    sdk_config: Final = SimpleNamespace(host=WORKSPACE_URL, authenticate=lambda: {"Authorization": f"Bearer {SDK_TOKEN}"})
    sdk_module: Final = ModuleType("databricks.sdk")
    setattr(sdk_module, "WorkspaceClient", lambda: SimpleNamespace(config=sdk_config))
    setattr(sdk_module, "useragent", SimpleNamespace(with_partner=lambda _: None))
    package_module: Final = ModuleType("databricks")
    package_module.__path__ = []
    setattr(package_module, "sdk", sdk_module)
    monkeypatch.setitem(sys.modules, "databricks", package_module)
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk_module)
    received: Final[list[httpx.Request]] = []  # mutable-ok: the fake workspace records what litellm sent

    def workspace_answers(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return _reply(request, MESSAGE)

    upstream: Final = AsyncHTTPHandler(transport=httpx.MockTransport(workspace_answers))
    try:
        await litellm.anthropic.messages.acreate(
            max_tokens=64,
            messages=USER_TURN,
            model=f"databricks/{CLAUDE_MODEL}",
            api_base=DATABRICKS_API_BASE,
            client=upstream,
            provider_specific_header={
                "custom_llm_provider": "databricks",
                "extra_headers": {"anthropic-beta": "context-1m-2025-08-07", "anthropic-version": "2023-06-01"},
                "native_anthropic_messages_only": True,
            },
        )
    finally:
        await upstream.close()

    assert [str(request.url) for request in received] == [MESSAGES_URL]
    assert received[0].headers["Authorization"] == f"Bearer {SDK_TOKEN}"
    assert received[0].headers["anthropic-beta"] == "context-1m-2025-08-07"


@pytest.mark.asyncio
async def test_client_anthropic_headers_skip_an_opted_in_passthrough_api_base() -> None:
    received: Final[list[httpx.Request]] = []  # mutable-ok: the fake upstream records what litellm sent

    def upstream_answers(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return _reply(request, MESSAGE)

    upstream: Final = AsyncHTTPHandler(transport=httpx.MockTransport(upstream_answers))
    try:
        await litellm.anthropic.messages.acreate(
            max_tokens=64,
            messages=USER_TURN,
            model=f"databricks/{CLAUDE_MODEL}",
            api_key=DATABRICKS_API_KEY,
            api_base="http://gateway.internal/dbx-anthropic",
            client=upstream,
            model_info=OPTED_IN_MODEL_INFO,
            provider_specific_header={
                "custom_llm_provider": "databricks",
                "extra_headers": {"anthropic-beta": "prompt-caching-scope-2026-01-05", "anthropic-version": "2023-01-01"},
                "native_anthropic_messages_only": True,
            },
        )
    finally:
        await upstream.close()

    assert [str(request.url) for request in received] == ["http://gateway.internal/dbx-anthropic/v1/messages"]
    assert "anthropic-beta" not in received[0].headers
    assert received[0].headers["anthropic-version"] != "2023-01-01"


class _ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes, chunk_size: int) -> None:
        self._chunks: Final = tuple(body[start : start + chunk_size] for start in range(0, len(body), chunk_size))

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [7, 4096])
async def test_streamed_message_stop_drops_the_bedrock_invocation_metrics(chunk_size: int) -> None:
    events: Final = (
        ("message_start", {"type": "message_start", "message": {**MESSAGE, "content": [], "stop_reason": None}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}}),
        (
            "message_stop",
            {
                "type": "message_stop",
                "amazon-bedrock-invocationMetrics": {
                    "inputTokenCount": 18,
                    "outputTokenCount": 4,
                    "invocationLatency": 496,
                    "firstByteLatency": 487,
                },
            },
        ),
    )
    sse_body: Final = "".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events).encode()

    def workspace_streams(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream(sse_body, chunk_size),
            request=request,
        )

    upstream: Final = AsyncHTTPHandler(transport=httpx.MockTransport(workspace_streams))
    try:
        stream = await litellm.anthropic.messages.acreate(
            max_tokens=64,
            messages=USER_TURN,
            model=f"databricks/{CLAUDE_MODEL}",
            api_key=DATABRICKS_API_KEY,
            api_base=DATABRICKS_API_BASE,
            stream=True,
            client=upstream,
        )
        relayed: Final = b"".join([chunk async for chunk in stream])
    finally:
        await upstream.close()

    relayed_events: Final = [
        json.loads(line.removeprefix(b"data:")) for line in relayed.splitlines() if line.startswith(b"data:")
    ]
    assert relayed_events == [*(payload for _, payload in events[:-1]), {"type": "message_stop"}], relayed.decode()
