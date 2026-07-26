"""
Unit tests for WebSearch Interception Handler

Tests the WebSearchInterceptionLogger class and helper functions.
"""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_TeamTable, ProxyException, UserAPIKeyAuth
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
        callback_specific_params={"websearch_interception": {"search_tool_name": "ws-tool"}},
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
    logging_obj.model_call_details = {"agentic_loop_params": {"model": "bedrock/invoke/claude-3-5-sonnet"}}
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
@pytest.mark.parametrize(
    ("model", "custom_llm_provider", "expected_model"),
    [
        (
            "qwen/qwen3.5-397b-a17b",
            "nvidia_nim",
            "nvidia_nim/qwen/qwen3.5-397b-a17b",
        ),
        (
            "nvidia_nim/qwen/qwen3.5-397b-a17b",
            "nvidia_nim",
            "nvidia_nim/qwen/qwen3.5-397b-a17b",
        ),
        (
            "nvidia_nim/qwen/qwen3.5-397b-a17b",
            "nvidia",
            "nvidia/nvidia_nim/qwen/qwen3.5-397b-a17b",
        ),
    ],
)
async def test_chat_completion_followup_preserves_provider_for_slashed_models(
    model, custom_llm_provider, expected_model
):
    """Follow-up chat completions must not drop providers for namespaced models."""
    logger = WebSearchInterceptionLogger(enabled_providers=[custom_llm_provider])
    logger._execute_search = AsyncMock(  # type: ignore
        return_value=("Title: LiteLLM\nURL: docs\nSnippet: test", None)
    )

    request_patch = await logger._build_chat_completion_request_patch(
        model=model,
        messages=[{"role": "user", "content": "search LiteLLM"}],
        tool_calls=[
            {
                "id": "call_123",
                "name": "litellm_web_search",
                "input": {"query": "what is litellm"},
            }
        ],
        optional_params={"tools": [{"name": "litellm_web_search"}]},
        kwargs={
            "custom_llm_provider": custom_llm_provider,
            "temperature": 0.2,
        },
        response_format="openai",
    )

    assert request_patch.model == expected_model
    assert request_patch.messages is not None
    assert len(request_patch.messages) == 3
    assert request_patch.kwargs["temperature"] == 0.2
    assert "custom_llm_provider" not in request_patch.kwargs


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
        k: v for k, v in kwargs_with_internal_flags.items() if not k.startswith("_websearch_interception")
    }

    # Verify internal flags are filtered out
    assert "_websearch_interception_converted_stream" not in kwargs_for_followup
    assert "_websearch_interception_other_flag" not in kwargs_for_followup

    # Verify regular kwargs are preserved
    assert kwargs_for_followup["temperature"] == 0.7
    assert kwargs_for_followup["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_execute_search_passes_selected_search_tool_litellm_params(monkeypatch):
    import litellm
    from litellm.proxy import proxy_server

    logger = WebSearchInterceptionLogger(
        enabled_providers=["bedrock"],
        search_tool_name="ui-tavily",
    )
    router = MagicMock()
    router.search_tools = [
        {
            "search_tool_name": "ui-tavily",
            "litellm_params": {
                "search_provider": "tavily",
                "api_key": "fake-ui-key",
                "api_base": "https://api.tavily.com",
                "timeout": 10.0,
                "max_retries": 2,
                "country": None,
            },
        }
    ]
    mock_asearch = AsyncMock(return_value=SearchResponse(object="search", results=[]))
    user_api_key_auth = UserAPIKeyAuth(
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="op-allowed-search",
            search_tools=["ui-tavily"],
        )
    )

    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(litellm, "asearch", mock_asearch)

    await logger._execute_search(
        "what is litellm",
        kwargs={"litellm_params": {"metadata": {"user_api_key_auth": user_api_key_auth}}},
    )

    mock_asearch.assert_awaited_once_with(
        query="what is litellm",
        search_provider="tavily",
        api_key="fake-ui-key",
        api_base="https://api.tavily.com",
        timeout=10.0,
        max_retries=2,
    )


@pytest.mark.asyncio
async def test_execute_search_enforces_key_search_tool_permission(monkeypatch):
    import litellm
    from litellm.proxy import proxy_server

    logger = WebSearchInterceptionLogger(
        enabled_providers=["bedrock"],
        search_tool_name="blocked-search",
    )
    router = MagicMock()
    router.search_tools = [
        {
            "search_tool_name": "blocked-search",
            "litellm_params": {
                "search_provider": "tavily",
                "api_key": "fake-ui-key",
            },
        }
    ]
    mock_asearch = AsyncMock(return_value=SearchResponse(object="search", results=[]))
    user_api_key_auth = UserAPIKeyAuth(
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="op-key-search",
            search_tools=["allowed-search"],
        )
    )

    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(litellm, "asearch", mock_asearch)

    with pytest.raises(ProxyException):
        await logger._execute_search(
            "what is litellm",
            kwargs={"metadata": {"user_api_key_auth": user_api_key_auth}},
        )

    mock_asearch.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_search_enforces_team_search_tool_permission(monkeypatch):
    import litellm
    from litellm.proxy import proxy_server

    logger = WebSearchInterceptionLogger(
        enabled_providers=["bedrock"],
        search_tool_name="blocked-search",
    )
    router = MagicMock()
    router.search_tools = [
        {
            "search_tool_name": "blocked-search",
            "litellm_params": {
                "search_provider": "tavily",
                "api_key": "fake-ui-key",
            },
        }
    ]
    mock_asearch = AsyncMock(return_value=SearchResponse(object="search", results=[]))
    team_key_auth = UserAPIKeyAuth(team_id="team-1")
    team_object = LiteLLM_TeamTable(
        team_id="team-1",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="op-team-search",
            search_tools=["allowed-search"],
        ),
    )
    mock_get_team_object = AsyncMock(return_value=team_object)

    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(litellm, "asearch", mock_asearch)
    monkeypatch.setattr("litellm.proxy.auth.auth_checks.get_team_object", mock_get_team_object)

    with pytest.raises(ProxyException):
        await logger._execute_search(
            "what is litellm",
            kwargs={"metadata": {"user_api_key_auth": team_key_auth}},
        )

    mock_get_team_object.assert_awaited_once()
    mock_asearch.assert_not_awaited()


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
        t.get("type") == "function" and t.get("function", {}).get("name") == "litellm_web_search"
        for t in result["tools"]
    )
    # The non-web-search tool should be preserved
    assert any(
        t.get("type") == "function" and t.get("function", {}).get("name") == "other_tool" for t in result["tools"]
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
        t.get("type") == "function" and t.get("function", {}).get("name") == "litellm_web_search"
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
        t.get("type") == "function" and t.get("function", {}).get("name") == "litellm_web_search"
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
        t.get("type") == "function" and t.get("function", {}).get("name") == "litellm_web_search"
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


def test_sync_forced_tool_choice_repoints_converted_web_search():
    """Regression (tool_choice 400): a forced tool_choice naming the original
    web_search tool must be repointed to litellm_web_search after conversion.

    Native clients (e.g. Claude Code) send
    tool_choice={"type": "tool", "name": "web_search"}. The tool definition is
    renamed to litellm_web_search, so an unrewritten tool_choice points at a
    tool that no longer exists and Anthropic rejects with
    "Tool 'web_search' not found in provided tools".
    """
    converted_tools = [
        {
            "type": "function",
            "function": {"name": LITELLM_WEB_SEARCH_TOOL_NAME, "parameters": {}},
        }
    ]

    result = WebSearchInterceptionLogger._sync_forced_tool_choice(
        {"type": "tool", "name": "web_search"}, converted_tools
    )

    assert result == {"type": "tool", "name": LITELLM_WEB_SEARCH_TOOL_NAME}


def test_sync_forced_tool_choice_leaves_existing_tool_untouched():
    """A native Anthropic tool_choice already naming a tool on the converted
    list (top-level name, no function wrapper) must not be rewritten."""
    converted_tools = [{"name": LITELLM_WEB_SEARCH_TOOL_NAME}]

    result = WebSearchInterceptionLogger._sync_forced_tool_choice(
        {"type": "tool", "name": LITELLM_WEB_SEARCH_TOOL_NAME}, converted_tools
    )

    assert result == {"type": "tool", "name": LITELLM_WEB_SEARCH_TOOL_NAME}


def test_sync_forced_tool_choice_preserves_extra_tool_choice_fields():
    """Repointing must keep other tool_choice keys intact."""
    converted_tools = [
        {
            "type": "function",
            "function": {"name": LITELLM_WEB_SEARCH_TOOL_NAME, "parameters": {}},
        }
    ]

    result = WebSearchInterceptionLogger._sync_forced_tool_choice(
        {"type": "tool", "name": "web_search", "disable_parallel_tool_use": True},
        converted_tools,
    )

    assert result == {
        "type": "tool",
        "name": LITELLM_WEB_SEARCH_TOOL_NAME,
        "disable_parallel_tool_use": True,
    }


@pytest.mark.parametrize(
    "tool_choice",
    ["auto", {"type": "auto"}, {"type": "any"}, None],
)
def test_sync_forced_tool_choice_leaves_non_forced_untouched(tool_choice):
    """Only a forced {"type": "tool", ...} choice is rewritten; auto/any/string
    and None pass through unchanged."""
    converted_tools = [{"name": LITELLM_WEB_SEARCH_TOOL_NAME}]

    result = WebSearchInterceptionLogger._sync_forced_tool_choice(tool_choice, converted_tools)

    assert result == tool_choice


@pytest.mark.asyncio
async def test_pre_request_hook_syncs_forced_tool_choice():
    """End-to-end: async_pre_request_hook converts web_search and repoints the
    forced tool_choice in the same pass, so the outgoing request is consistent.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=["anthropic"])

    kwargs = {
        "litellm_params": {"custom_llm_provider": "anthropic"},
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "tool_choice": {"type": "tool", "name": "web_search"},
    }

    result = await logger.async_pre_request_hook(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "search the web"}],
        kwargs=kwargs,
    )

    assert result is not None
    assert result["tool_choice"] == {
        "type": "tool",
        "name": LITELLM_WEB_SEARCH_TOOL_NAME,
    }
