import json
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


def test_claude_serving_endpoints_dispatch_to_the_native_messages_config() -> None:
    config: Final = ProviderConfigManager.get_provider_anthropic_messages_config(
        model=CLAUDE_MODEL, provider=litellm.LlmProviders.DATABRICKS
    )

    assert isinstance(config, DatabricksAnthropicMessagesConfig)
    assert config.custom_llm_provider == "databricks"


def test_unity_catalog_claude_names_dispatch_to_the_native_messages_config() -> None:
    config: Final = ProviderConfigManager.get_provider_anthropic_messages_config(
        model=UNITY_CATALOG_CLAUDE_MODEL, provider=litellm.LlmProviders.DATABRICKS
    )

    assert isinstance(config, DatabricksAnthropicMessagesConfig)


@pytest.mark.parametrize("model", ["databricks-gpt-oss-120b", "my-claude-finetune", "main.default.claude-proxy"])
def test_names_outside_the_foundation_claude_models_keep_the_chat_completions_bridge(model: str) -> None:
    config: Final = ProviderConfigManager.get_provider_anthropic_messages_config(
        model=model, provider=litellm.LlmProviders.DATABRICKS
    )

    assert config is None


@pytest.mark.parametrize(
    "api_base",
    [DATABRICKS_API_BASE, f"{DATABRICKS_API_BASE}/", MESSAGES_URL],
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
    [DATABRICKS_API_BASE, f"{DATABRICKS_API_BASE}/", MESSAGES_URL],
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
