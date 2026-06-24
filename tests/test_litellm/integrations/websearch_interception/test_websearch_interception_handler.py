"""
Unit tests for WebSearch Interception Handler

Tests the WebSearchInterceptionLogger class and helper functions.
"""

import builtins
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    ProxyException,
)
from litellm.router_utils.search_api_router import SearchAPIRouter
from litellm.types.utils import LlmProviders


def test_initialize_from_proxy_config():
    """Test initialization from proxy config with litellm_settings"""
    litellm_settings = {
        "websearch_interception_params": {
            "enabled_providers": ["bedrock", "vertex_ai"],
            "search_tool_name": "my-search",
        }
    }
    callback_specific_params = {}

    logger = WebSearchInterceptionLogger.initialize_from_proxy_config(
        litellm_settings=litellm_settings,
        callback_specific_params=callback_specific_params,
    )

    assert LlmProviders.BEDROCK.value in logger.enabled_providers
    assert LlmProviders.VERTEX_AI.value in logger.enabled_providers
    assert logger.search_tool_name == "my-search"


def test_initialize_from_proxy_config_ignores_non_dict_callback_specific_params():
    """Regression (#29590): a non-dict value under
    callback_settings.websearch_interception must not crash initialization.

    Forwarding callback_settings as callback_specific_params activates this
    branch; without the isinstance(dict) guard a non-dict value reached
    from_config_yaml(...).get(...) and raised AttributeError at proxy startup.
    The value is ignored and the logger falls back to defaults.
    """
    logger = WebSearchInterceptionLogger.initialize_from_proxy_config(
        litellm_settings={},
        callback_specific_params={"websearch_interception": True},
    )

    assert logger.search_tool_name is None


def test_initialize_from_proxy_config_honors_dict_callback_specific_params():
    """A valid dict under callback_settings.websearch_interception is applied."""
    logger = WebSearchInterceptionLogger.initialize_from_proxy_config(
        litellm_settings={},
        callback_specific_params={
            "websearch_interception": {"search_tool_name": "ws-tool"}
        },
    )

    assert logger.search_tool_name == "ws-tool"


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop():
    """Test that agentic loop is NOT triggered for wrong provider or missing WebSearch tool"""
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    # Test 1: Wrong provider (not in enabled_providers)
    response = Mock()
    should_run, tools_dict = await logger.async_should_run_agentic_loop(
        response=response,
        model="gpt-4",
        messages=[],
        tools=[{"name": "WebSearch"}],
        stream=False,
        custom_llm_provider="openai",  # Not in enabled_providers
        kwargs={},
    )

    assert should_run is False
    assert tools_dict == {}

    # Test 2: No WebSearch tool in request
    should_run, tools_dict = await logger.async_should_run_agentic_loop(
        response=response,
        model="bedrock/claude",
        messages=[],
        tools=[{"name": "SomeOtherTool"}],  # No WebSearch
        stream=False,
        custom_llm_provider="bedrock",
        kwargs={},
    )

    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_async_build_agentic_loop_plan_returns_request_patch():
    """Callback should return a typed patch for base handler reruns."""
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
    logger._execute_search = AsyncMock(  # type: ignore
        return_value="Title: LiteLLM\nURL: docs\nSnippet: test"
    )

    tools_dict = {
        "tool_calls": [
            {
                "id": "toolu_123",
                "type": "tool_use",
                "name": "litellm_web_search",
                "input": {"query": "what is litellm"},
            }
        ],
        "response_format": "anthropic",
    }
    logging_obj = MagicMock()
    logging_obj.model_call_details = {
        "agentic_loop_params": {"model": "bedrock/invoke/claude-3-5-sonnet"}
    }
    kwargs = {
        "temperature": 0.2,
        "_websearch_interception_converted_stream": True,
        "litellm_logging_obj": object(),
    }

    plan = await logger.async_build_agentic_loop_plan(
        tools=tools_dict,
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "search LiteLLM"}],
        response=None,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={
            "max_tokens": 1024,
            "tools": [{"name": "litellm_web_search"}],
        },
        logging_obj=logging_obj,
        stream=False,
        kwargs=kwargs,
    )

    assert plan.run_agentic_loop is True
    assert plan.request_patch is not None
    assert plan.request_patch.model == "bedrock/invoke/claude-3-5-sonnet"
    assert plan.request_patch.max_tokens == 1024
    assert plan.request_patch.messages is not None
    assert len(plan.request_patch.messages) == 3
    assert "_websearch_interception_converted_stream" not in plan.request_patch.kwargs
    assert "litellm_logging_obj" not in plan.request_patch.kwargs
    assert plan.request_patch.kwargs["temperature"] == 0.2


@pytest.mark.asyncio
async def test_execute_search_passes_configured_search_tool_credentials(monkeypatch):
    """Search interception should use credentials from the selected router search tool."""
    monkeypatch.setenv("FOO_BAR", "custom-perplexity-key")
    proxy_server_module = ModuleType("litellm.proxy.proxy_server")
    proxy_server_module.llm_router = SimpleNamespace(
        search_tools=[
            {
                "search_tool_name": "perplexity-search",
                "litellm_params": {
                    "search_provider": "perplexity",
                    "api_key": "os.environ/FOO_BAR",
                    "api_base": "https://custom.perplexity.example",
                },
            }
        ]
    )
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_module)

    search_response = SearchResponse(
        object="search",
        results=[
            SearchResult(
                title="LiteLLM",
                url="https://docs.litellm.ai",
                snippet="LiteLLM search result",
            )
        ],
    )
    asearch = AsyncMock(return_value=search_response)
    monkeypatch.setattr(
        "litellm.integrations.websearch_interception.handler.litellm.asearch",
        asearch,
    )

    logger = WebSearchInterceptionLogger(search_tool_name="perplexity-search")

    search_text, structured_response = await logger._execute_search("what is litellm")

    asearch.assert_awaited_once_with(
        query="what is litellm",
        search_provider="perplexity",
        api_key="custom-perplexity-key",
        api_base="https://custom.perplexity.example",
    )
    assert structured_response is search_response
    assert "LiteLLM" in search_text


def _search_tool_auth(
    *,
    key_search_tools: list[str] | None = None,
    team_search_tools: list[str] | None = None,
):
    key_permission = (
        LiteLLM_ObjectPermissionTable(
            object_permission_id="key-permission",
            search_tools=key_search_tools,
        )
        if key_search_tools is not None
        else None
    )
    team_permission = (
        LiteLLM_ObjectPermissionTable(
            object_permission_id="team-permission",
            search_tools=team_search_tools,
        )
        if team_search_tools is not None
        else None
    )
    return SimpleNamespace(
        api_key="hashed-key",
        object_permission=key_permission,
        team_id="team-1" if team_permission is not None else None,
        team_object_permission=team_permission,
    )


def _set_proxy_search_tools(monkeypatch, search_tools):
    proxy_server_module = ModuleType("litellm.proxy.proxy_server")
    proxy_server_module.llm_router = SimpleNamespace(search_tools=search_tools)
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_module)


def _mock_asearch(monkeypatch):
    asearch = AsyncMock(return_value=SearchResponse(object="search", results=[]))
    monkeypatch.setattr(
        "litellm.integrations.websearch_interception.handler.litellm.asearch",
        asearch,
    )
    return asearch


def _perplexity_search_tool(
    *,
    search_tool_name: str = "perplexity-search",
    api_key: str = "should-not-be-used",
):
    return {
        "search_tool_name": search_tool_name,
        "litellm_params": {
            "search_provider": "perplexity",
            "api_key": api_key,
        },
    }


@pytest.mark.asyncio
async def test_execute_search_checks_key_search_tool_permission(monkeypatch):
    """Search interception should not use configured credentials denied to the key."""
    _set_proxy_search_tools(monkeypatch, [_perplexity_search_tool()])
    asearch = _mock_asearch(monkeypatch)

    logger = WebSearchInterceptionLogger(search_tool_name="perplexity-search")
    user_api_key_auth = _search_tool_auth(key_search_tools=["other-search"])

    with pytest.raises(ProxyException) as exc_info:
        await logger._execute_search(
            "what is litellm",
            kwargs={"metadata": {"user_api_key_auth": user_api_key_auth}},
        )

    assert "not allowed to access search tool" in exc_info.value.message
    asearch.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_search_allows_key_search_tool_permission(monkeypatch):
    """Allowed keys should keep the configured search credential path."""
    monkeypatch.setenv("FOO_BAR", "custom-perplexity-key")
    _set_proxy_search_tools(
        monkeypatch, [_perplexity_search_tool(api_key="os.environ/FOO_BAR")]
    )
    search_response = SearchResponse(object="search", results=[])
    asearch = AsyncMock(return_value=search_response)
    monkeypatch.setattr(
        "litellm.integrations.websearch_interception.handler.litellm.asearch",
        asearch,
    )

    logger = WebSearchInterceptionLogger(search_tool_name="perplexity-search")
    user_api_key_auth = _search_tool_auth(key_search_tools=["perplexity-search"])

    _, structured_response = await logger._execute_search(
        "what is litellm",
        kwargs={"metadata": {"user_api_key_auth": user_api_key_auth}},
    )

    asearch.assert_awaited_once_with(
        query="what is litellm",
        search_provider="perplexity",
        api_key="custom-perplexity-key",
        api_base=None,
    )
    assert structured_response is search_response


@pytest.mark.asyncio
async def test_execute_search_checks_fallback_search_tool_permission(monkeypatch):
    """Fallback to the first router tool should authorize that actual tool name."""
    _set_proxy_search_tools(
        monkeypatch,
        [_perplexity_search_tool(search_tool_name="default-perplexity-search")],
    )
    asearch = _mock_asearch(monkeypatch)

    logger = WebSearchInterceptionLogger(search_tool_name="missing-search-tool")
    user_api_key_auth = _search_tool_auth(key_search_tools=["missing-search-tool"])

    with pytest.raises(ProxyException) as exc_info:
        await logger._execute_search(
            "what is litellm",
            kwargs={"metadata": {"user_api_key_auth": user_api_key_auth}},
        )

    assert "default-perplexity-search" in exc_info.value.message
    asearch.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_search_checks_team_search_tool_permission(monkeypatch):
    """Team search-tool allowlists should still apply when the key allows access."""
    _set_proxy_search_tools(monkeypatch, [_perplexity_search_tool()])
    asearch = _mock_asearch(monkeypatch)

    logger = WebSearchInterceptionLogger(search_tool_name="perplexity-search")
    user_api_key_auth = _search_tool_auth(
        key_search_tools=["perplexity-search"],
        team_search_tools=["other-search"],
    )

    with pytest.raises(ProxyException) as exc_info:
        await logger._execute_search(
            "what is litellm",
            kwargs={"metadata": {"user_api_key_auth": user_api_key_auth}},
        )

    assert "Team not allowed" in exc_info.value.message
    asearch.assert_not_awaited()


@pytest.mark.asyncio
async def test_short_circuit_search_propagates_search_tool_permission_error(
    monkeypatch,
):
    """Short-circuit requests should fail authorization instead of returning error text."""
    _set_proxy_search_tools(monkeypatch, [_perplexity_search_tool()])
    asearch = _mock_asearch(monkeypatch)

    logger = WebSearchInterceptionLogger(
        enabled_providers=["test_provider"],
        search_tool_name="perplexity-search",
    )
    user_api_key_auth = _search_tool_auth(key_search_tools=["other-search"])

    with pytest.raises(ProxyException):
        await logger.try_short_circuit_search(
            model="test-model",
            messages=[{"role": "user", "content": "what is litellm"}],
            tools=[{"name": "litellm_web_search"}],
            custom_llm_provider="test_provider",
            kwargs={
                "litellm_params": {"metadata": {"user_api_key_auth": user_api_key_auth}}
            },
        )

    asearch.assert_not_awaited()


@pytest.mark.asyncio
async def test_anthropic_agentic_loop_propagates_search_tool_permission_error(
    monkeypatch,
):
    """Anthropic agentic-loop searches should not turn auth failures into tool text."""
    _set_proxy_search_tools(monkeypatch, [_perplexity_search_tool()])
    asearch = _mock_asearch(monkeypatch)

    logger = WebSearchInterceptionLogger(search_tool_name="perplexity-search")
    user_api_key_auth = _search_tool_auth(key_search_tools=["other-search"])

    with pytest.raises(ProxyException):
        await logger._build_anthropic_request_patch(
            model="bedrock/claude",
            messages=[{"role": "user", "content": "what is litellm"}],
            tool_calls=[
                {
                    "id": "toolu_123",
                    "type": "tool_use",
                    "name": "litellm_web_search",
                    "input": {"query": "what is litellm"},
                }
            ],
            thinking_blocks=[],
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            kwargs={"metadata": {"user_api_key_auth": user_api_key_auth}},
        )

    asearch.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_completion_agentic_loop_propagates_search_tool_permission_error(
    monkeypatch,
):
    """Chat-completion agentic-loop searches should preserve auth failures."""
    _set_proxy_search_tools(monkeypatch, [_perplexity_search_tool()])
    asearch = _mock_asearch(monkeypatch)

    logger = WebSearchInterceptionLogger(search_tool_name="perplexity-search")
    user_api_key_auth = _search_tool_auth(key_search_tools=["other-search"])

    with pytest.raises(ProxyException):
        await logger._build_chat_completion_request_patch(
            model="gpt-4o",
            messages=[{"role": "user", "content": "what is litellm"}],
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "litellm_web_search",
                        "arguments": {"query": "what is litellm"},
                    },
                }
            ],
            optional_params={},
            kwargs={"metadata": {"user_api_key_auth": user_api_key_auth}},
        )

    asearch.assert_not_awaited()


def test_search_tool_auth_helpers_cover_fallback_shapes():
    """Auth helpers should handle proxy object and dict shapes used in tests/hooks."""
    user_api_key_auth = _search_tool_auth(key_search_tools=["perplexity-search"])
    auth_dict = {
        "object_permission": {"search_tools": ["dict-search"]},
        "team_id": "team-1",
    }

    assert (
        WebSearchInterceptionLogger._get_user_api_key_auth_from_kwargs(
            {"user_api_key_auth": user_api_key_auth}
        )
        is user_api_key_auth
    )
    assert (
        WebSearchInterceptionLogger._get_user_api_key_auth_from_kwargs(
            {"user_api_key_dict": user_api_key_auth}
        )
        is user_api_key_auth
    )
    assert WebSearchInterceptionLogger._get_auth_attr(auth_dict, "team_id") == "team-1"
    assert (
        WebSearchInterceptionLogger._get_search_tool_names_from_object_permission(None)
        == []
    )
    assert WebSearchInterceptionLogger._get_search_tool_names_from_object_permission(
        {"search_tools": ["dict-search"]}
    ) == ["dict-search"]
    assert (
        WebSearchInterceptionLogger._get_search_tool_names_from_object_permission(
            {"search_tools": []}
        )
        == []
    )


@pytest.mark.asyncio
async def test_authorize_search_tool_access_loads_team_permission(monkeypatch):
    """Team permissions should be loaded when auth context has only a team id."""
    team_permission = LiteLLM_ObjectPermissionTable(
        object_permission_id="team-permission",
        search_tools=["perplexity-search"],
    )

    async def get_team_object(**kwargs):
        assert kwargs["team_id"] == "team-1"
        assert kwargs["prisma_client"] == "prisma"
        assert kwargs["user_api_key_cache"] == "cache"
        assert kwargs["proxy_logging_obj"] == "proxy-logging"
        return SimpleNamespace(object_permission=team_permission)

    auth_checks_module = ModuleType("litellm.proxy.auth.auth_checks")
    auth_checks_module.get_team_object = get_team_object
    monkeypatch.setitem(
        sys.modules, "litellm.proxy.auth.auth_checks", auth_checks_module
    )

    proxy_server_module = ModuleType("litellm.proxy.proxy_server")
    proxy_server_module.prisma_client = "prisma"
    proxy_server_module.proxy_logging_obj = "proxy-logging"
    proxy_server_module.user_api_key_cache = "cache"
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_module)

    user_api_key_auth = _search_tool_auth(key_search_tools=["perplexity-search"])
    user_api_key_auth.team_id = "team-1"
    user_api_key_auth.team_object_permission = None

    await WebSearchInterceptionLogger._authorize_search_tool_access(
        "perplexity-search",
        {"user_api_key_auth": user_api_key_auth},
    )
    await WebSearchInterceptionLogger._authorize_search_tool_access(
        None,
        {"user_api_key_auth": user_api_key_auth},
    )


@pytest.mark.asyncio
async def test_short_circuit_search_returns_text_for_non_auth_search_errors():
    """Non-auth search failures should keep the existing synthetic error response."""
    logger = WebSearchInterceptionLogger(enabled_providers=["test_provider"])
    logger._execute_search = AsyncMock(side_effect=RuntimeError("search down"))  # type: ignore

    response = await logger.try_short_circuit_search(
        model="test-model",
        messages=[{"role": "user", "content": "what is litellm"}],
        tools=[{"name": "litellm_web_search"}],
        custom_llm_provider="test_provider",
    )

    assert response is not None
    assert response["content"] == [
        {"type": "text", "text": "Search failed: search down"}
    ]


@pytest.mark.asyncio
async def test_anthropic_agentic_loop_keeps_non_auth_search_errors_as_tool_text():
    """Non-auth search failures should still be passed to the follow-up model as text."""
    logger = WebSearchInterceptionLogger()
    logger._execute_search = AsyncMock(side_effect=RuntimeError("search down"))  # type: ignore
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}

    patch, structured_results = await logger._build_anthropic_request_patch(
        model="bedrock/claude",
        messages=[{"role": "user", "content": "what is litellm"}],
        tool_calls=[
            {
                "id": "toolu_123",
                "type": "tool_use",
                "name": "litellm_web_search",
                "input": {"query": "what is litellm"},
            },
            {
                "id": "toolu_empty",
                "type": "tool_use",
                "name": "litellm_web_search",
                "input": {},
            },
        ],
        thinking_blocks=[],
        anthropic_messages_optional_request_params={},
        logging_obj=logging_obj,
        kwargs={},
    )

    assert patch.messages is not None
    assert "Search failed: search down" in str(patch.messages)
    assert structured_results == [None, None]


@pytest.mark.asyncio
async def test_chat_completion_agentic_loop_keeps_non_auth_search_errors_as_tool_text():
    """Chat-completion search failures should remain ordinary tool-result text."""
    logger = WebSearchInterceptionLogger()
    logger._execute_search = AsyncMock(side_effect=RuntimeError("search down"))  # type: ignore

    patch = await logger._build_chat_completion_request_patch(
        model="gpt-4o",
        messages=[{"role": "user", "content": "what is litellm"}],
        tool_calls=[
            {
                "id": "call_123",
                "type": "function",
                "name": "litellm_web_search",
                "input": {"query": "what is litellm"},
                "function": {
                    "name": "litellm_web_search",
                    "arguments": {"query": "what is litellm"},
                },
            },
            {
                "id": "call_empty",
                "type": "function",
                "name": "litellm_web_search",
                "input": {},
                "function": {
                    "name": "litellm_web_search",
                    "arguments": {},
                },
            },
        ],
        optional_params={},
        kwargs={},
    )

    assert patch.messages is not None
    assert "Search failed: search down" in str(patch.messages)


@pytest.mark.asyncio
async def test_execute_search_falls_back_to_first_search_tool_credentials(monkeypatch):
    """If the requested search tool is missing, use the first configured tool's credentials."""
    monkeypatch.setenv("CUSTOM_SEARCH_BASE", "https://fallback.perplexity.example")
    proxy_server_module = ModuleType("litellm.proxy.proxy_server")
    proxy_server_module.llm_router = SimpleNamespace(
        search_tools=[
            {
                "search_tool_name": "default-perplexity-search",
                "litellm_params": {
                    "search_provider": "perplexity",
                    "api_key": "literal-perplexity-key",
                    "api_base": "os.environ/CUSTOM_SEARCH_BASE",
                },
            }
        ]
    )
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_module)

    search_response = SearchResponse(object="search", results=[])
    asearch = AsyncMock(return_value=search_response)
    monkeypatch.setattr(
        "litellm.integrations.websearch_interception.handler.litellm.asearch",
        asearch,
    )

    logger = WebSearchInterceptionLogger(search_tool_name="missing-search-tool")

    search_text, structured_response = await logger._execute_search("what is litellm")

    asearch.assert_awaited_once_with(
        query="what is litellm",
        search_provider="perplexity",
        api_key="literal-perplexity-key",
        api_base="https://fallback.perplexity.example",
    )
    assert structured_response is search_response
    assert "search" in search_text


@pytest.mark.asyncio
async def test_execute_search_uses_default_perplexity_without_router_tools(monkeypatch):
    """Search interception should not pass stale credentials when no router tools exist."""
    proxy_server_module = ModuleType("litellm.proxy.proxy_server")
    proxy_server_module.llm_router = SimpleNamespace(search_tools=[])
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_module)

    search_response = SearchResponse(object="search", results=[])
    asearch = AsyncMock(return_value=search_response)
    monkeypatch.setattr(
        "litellm.integrations.websearch_interception.handler.litellm.asearch",
        asearch,
    )

    logger = WebSearchInterceptionLogger()

    search_text, structured_response = await logger._execute_search("what is litellm")

    asearch.assert_awaited_once_with(
        query="what is litellm",
        search_provider="perplexity",
        api_key=None,
        api_base=None,
    )
    assert structured_response is search_response
    assert "search" in search_text


@pytest.mark.asyncio
async def test_execute_search_uses_default_perplexity_when_router_import_fails(
    monkeypatch,
):
    """Search interception should still work when proxy router import is unavailable."""
    real_import = builtins.__import__

    def raise_for_proxy_server(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "litellm.proxy.proxy_server":
            raise ImportError("proxy unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", raise_for_proxy_server)

    search_response = SearchResponse(object="search", results=[])
    asearch = AsyncMock(return_value=search_response)
    monkeypatch.setattr(
        "litellm.integrations.websearch_interception.handler.litellm.asearch",
        asearch,
    )

    logger = WebSearchInterceptionLogger()

    search_text, structured_response = await logger._execute_search("what is litellm")

    asearch.assert_awaited_once_with(
        query="what is litellm",
        search_provider="perplexity",
        api_key=None,
        api_base=None,
    )
    assert structured_response is search_response
    assert "search" in search_text


def test_search_provider_credentials_resolve_configured_values(monkeypatch):
    """Credential resolver should resolve supported values and ignore unsupported ones."""
    monkeypatch.setenv("FOO_BAR", "custom-perplexity-key")

    api_key, api_base = SearchAPIRouter._resolve_search_provider_credentials(
        tool_litellm_params={
            "api_key": "os.environ/FOO_BAR",
            "api_base": "https://custom.perplexity.example",
        }
    )

    assert api_key == "custom-perplexity-key"
    assert api_base == "https://custom.perplexity.example"

    api_key, api_base = SearchAPIRouter._resolve_search_provider_credentials(
        tool_litellm_params={"api_key": 123, "api_base": None}
    )

    assert api_key is None
    assert api_base is None


@pytest.mark.asyncio
async def test_execute_search_reraises_search_errors(monkeypatch):
    """Search interception should not hide provider search failures."""
    proxy_server_module = ModuleType("litellm.proxy.proxy_server")
    proxy_server_module.llm_router = SimpleNamespace(search_tools=[])
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server_module)

    asearch = AsyncMock(side_effect=RuntimeError("search failed"))
    monkeypatch.setattr(
        "litellm.integrations.websearch_interception.handler.litellm.asearch",
        asearch,
    )

    logger = WebSearchInterceptionLogger()

    with pytest.raises(RuntimeError, match="search failed"):
        await logger._execute_search("what is litellm")


@pytest.mark.asyncio
async def test_internal_flags_filtered_from_followup_kwargs():
    """Test that internal _websearch_interception flags are filtered from follow-up request kwargs.

    Regression test for bug where _websearch_interception_converted_stream was passed
    to the follow-up LLM request, causing "Extra inputs are not permitted" errors
    from providers like Bedrock that use strict parameter validation.
    """
    # Simulate kwargs that would be passed during agentic loop execution
    kwargs_with_internal_flags = {
        "_websearch_interception_converted_stream": True,
        "_websearch_interception_other_flag": "test",
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    # Apply the same filtering logic used in _execute_agentic_loop
    kwargs_for_followup = {
        k: v
        for k, v in kwargs_with_internal_flags.items()
        if not k.startswith("_websearch_interception")
    }

    # Verify internal flags are filtered out
    assert "_websearch_interception_converted_stream" not in kwargs_for_followup
    assert "_websearch_interception_other_flag" not in kwargs_for_followup

    # Verify regular kwargs are preserved
    assert kwargs_for_followup["temperature"] == 0.7
    assert kwargs_for_followup["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_provider_from_top_level_kwargs():
    """Test that async_pre_call_deployment_hook finds custom_llm_provider at top-level kwargs.

    Regression test for bug where the hook only checked kwargs["litellm_params"]["custom_llm_provider"]
    but the router places custom_llm_provider at the top level of kwargs.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    # Simulate kwargs as they arrive from the router path:
    # custom_llm_provider is at the TOP LEVEL (not nested under litellm_params)
    kwargs = {
        "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "messages": [{"role": "user", "content": "Search the web for LiteLLM"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            {"type": "function", "function": {"name": "other_tool", "parameters": {}}},
        ],
        "custom_llm_provider": "bedrock",
        "api_key": "fake-key",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

    # Should NOT be None — the hook should have triggered
    assert result is not None
    # The web_search tool should be converted to litellm_web_search (OpenAI format)
    assert any(
        t.get("type") == "function"
        and t.get("function", {}).get("name") == "litellm_web_search"
        for t in result["tools"]
    )
    # The non-web-search tool should be preserved
    assert any(
        t.get("type") == "function"
        and t.get("function", {}).get("name") == "other_tool"
        for t in result["tools"]
    )


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_returns_full_kwargs():
    """Test that async_pre_call_deployment_hook returns the full kwargs dict, not a partial one.

    Regression test for bug where the hook returned {"tools": converted_tools} instead of
    the full kwargs dict, causing model/messages/api_key/etc. to be lost.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=["openai"])

    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Search for something"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search"},
        ],
        "custom_llm_provider": "openai",
        "api_key": "fake-key-for-testing",
        "temperature": 0.7,
        "metadata": {"user": "test"},
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

    assert result is not None
    # All original keys must be preserved
    assert result["model"] == "gpt-4o"
    assert result["messages"] == [{"role": "user", "content": "Search for something"}]
    assert result["api_key"] == "fake-key-for-testing"
    assert result["temperature"] == 0.7
    assert result["metadata"] == {"user": "test"}
    assert result["custom_llm_provider"] == "openai"
    # Tools should be converted
    assert any(
        t.get("type") == "function"
        and t.get("function", {}).get("name") == "litellm_web_search"
        for t in result["tools"]
    )


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_skips_disabled_provider():
    """Test that the hook returns None for providers not in enabled_providers."""
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "custom_llm_provider": "openai",  # Not in enabled_providers
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)
    assert result is None


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_skips_no_websearch_tools():
    """Test that the hook returns None when no web search tools are present."""
    logger = WebSearchInterceptionLogger(enabled_providers=["openai"])

    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [
            {"type": "function", "function": {"name": "calculator", "parameters": {}}},
        ],
        "custom_llm_provider": "openai",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)
    assert result is None


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_nested_litellm_params_fallback():
    """Test that the hook still works when custom_llm_provider is in nested litellm_params.

    This is the Anthropic experimental pass-through path where litellm_params is
    explicitly constructed with custom_llm_provider inside it.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    kwargs = {
        "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "litellm_params": {
            "custom_llm_provider": "bedrock",
        },
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

    assert result is not None
    assert any(
        t.get("type") == "function"
        and t.get("function", {}).get("name") == "litellm_web_search"
        for t in result["tools"]
    )
    # Full kwargs preserved
    assert result["model"] == "anthropic.claude-haiku-4-5-20251001-v1:0"


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_provider_derived_from_model_name():
    """Test that async_pre_call_deployment_hook derives custom_llm_provider from the model name.

    Regression test for the router _acompletion path where custom_llm_provider is NOT
    in kwargs at all — neither at top-level nor in litellm_params. The hook must derive
    the provider from the model name (e.g., "openai/gpt-4o-mini" → "openai").
    """
    logger = WebSearchInterceptionLogger(enabled_providers=["openai"])

    # Simulate kwargs as they arrive from router._acompletion:
    # NO custom_llm_provider key anywhere — only model name contains the provider
    kwargs = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Search the web for LiteLLM"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
        ],
        "api_key": "fake-key",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

    # Should NOT be None — the hook should derive "openai" from "openai/gpt-4o-mini"
    assert result is not None
    assert any(
        t.get("type") == "function"
        and t.get("function", {}).get("name") == "litellm_web_search"
        for t in result["tools"]
    )
    # Full kwargs preserved
    assert result["model"] == "openai/gpt-4o-mini"
    assert result["api_key"] == "fake-key"


@pytest.mark.asyncio
async def test_deployment_hook_converts_stream_and_logging_obj_syncs():
    """
    Regression test: websearch interception with stream=True must not skip logging.

    Before the fix, the stream conversion only happened in async_pre_request_hook
    (inside the anthropic_messages function scope). wrapper_async still saw
    stream=True, took the streaming early-return path, and skipped all spend/cost
    logging.  The fix moves stream conversion into the deployment hook so
    wrapper_async sees stream=False, and then syncs logging_obj.stream.

    This test verifies:
    1. The deployment hook sets stream=False and the converted flag.
    2. wrapper_async syncs logging_obj.stream after the hook runs.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    kwargs = {
        "model": "anthropic.claude-opus-4-6-20250219-v1:0",
        "messages": [{"role": "user", "content": "Search for LiteLLM"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
        ],
        "custom_llm_provider": "bedrock",
        "stream": True,
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

    assert result is not None
    assert result["stream"] is False
    assert result["_websearch_interception_converted_stream"] is True

    # Simulate what wrapper_async does after the deployment hook:
    # logging_obj.stream was set to True during function_setup (before hook).
    # After the hook, wrapper_async must sync it.
    logging_obj = MagicMock()
    logging_obj.stream = True  # original value from function_setup

    _hook_stream = result.get("stream")
    if _hook_stream is not None and logging_obj.stream != _hook_stream:
        logging_obj.stream = _hook_stream

    assert logging_obj.stream is False
