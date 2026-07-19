"""
Tests for the MCP sampling completion pipeline.

Covers building the internal `acompletion` kwargs from MCP request params
(messages, sampling options, tools, tool choice, metadata), routing the call
through the proxy router / guardrails, and the end-to-end
`handle_sampling_create_message` success and error-propagation behaviour.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp.types import CreateMessageResult, ErrorData

from litellm.proxy._experimental.mcp_server.sampling_handler import (
    _build_completion_kwargs,
    _run_guardrails_and_call_llm,
    handle_sampling_create_message,
)


def _params(**overrides):
    base = dict(
        messages=[
            SimpleNamespace(
                role="user", content=SimpleNamespace(type="text", text="hi")
            )
        ],
        systemPrompt="be concise",
        maxTokens=128,
        temperature=None,
        stopSequences=None,
        tools=None,
        toolChoice=None,
        metadata=None,
        modelPreferences=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _passthrough_add_data():
    async def _add(data, **kwargs):
        return data

    return _add


class TestBuildCompletionKwargs:
    async def test_should_include_sampling_options_and_tools(self):
        params = _params(
            temperature=0.3,
            stopSequences=["STOP"],
            tools=[
                SimpleNamespace(
                    name="search", description="d", inputSchema={"type": "object"}
                )
            ],
            toolChoice=SimpleNamespace(mode="required"),
            metadata={"trace": "abc"},
        )
        with patch(
            "litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request",
            side_effect=_passthrough_add_data(),
        ):
            kwargs = await _build_completion_kwargs(
                params=params,
                model="gpt-4o",
                user_api_key_auth=SimpleNamespace(user_id="u1"),
                raw_headers=None,
                client_ip=None,
            )

        assert kwargs["model"] == "gpt-4o"
        assert kwargs["max_tokens"] == 128
        assert kwargs["temperature"] == 0.3
        assert kwargs["stop"] == ["STOP"]
        assert kwargs["tools"][0]["function"]["name"] == "search"
        assert kwargs["tool_choice"] == "required"
        assert kwargs["metadata"]["mcp_metadata"] == {"trace": "abc"}
        assert kwargs["user"] == "u1"
        assert kwargs["messages"][0] == {"role": "system", "content": "be concise"}

    async def test_should_omit_optional_fields_when_unset(self):
        with patch(
            "litellm.proxy.litellm_pre_call_utils.add_litellm_data_to_request",
            side_effect=_passthrough_add_data(),
        ):
            kwargs = await _build_completion_kwargs(
                params=_params(),
                model="gpt-4o",
                user_api_key_auth=SimpleNamespace(user_id=None),
                raw_headers=None,
                client_ip=None,
            )

        assert "temperature" not in kwargs
        assert "stop" not in kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs
        assert kwargs["metadata"] == {}


class TestRunGuardrailsAndCallLlm:
    async def test_should_route_through_llm_router_when_available(self):
        router = MagicMock()
        router.acompletion = AsyncMock(return_value="router-response")
        with (
            patch("litellm.proxy.proxy_server.proxy_logging_obj", None),
            patch("litellm.proxy.proxy_server.llm_router", router),
        ):
            result = await _run_guardrails_and_call_llm(
                completion_kwargs={"model": "gpt-4o", "messages": []},
                user_api_key_auth=SimpleNamespace(),
            )

        assert result == "router-response"
        router.acompletion.assert_awaited_once()

    async def test_should_propagate_guardrail_rejection(self):
        plo = MagicMock()
        plo.pre_call_hook = AsyncMock(side_effect=ValueError("blocked by guardrail"))
        with patch("litellm.proxy.proxy_server.proxy_logging_obj", plo):
            with pytest.raises(ValueError, match="blocked by guardrail"):
                await _run_guardrails_and_call_llm(
                    completion_kwargs={"model": "gpt-4o", "messages": []},
                    user_api_key_auth=SimpleNamespace(),
                )


class TestHandleSamplingCreateMessagePipeline:
    async def test_should_return_message_result_on_success(self):
        auth = SimpleNamespace(user_id="u1", api_key="sk-test", token="tok")
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="the answer is 42", tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            model="gpt-4o",
        )
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._resolve_model_from_preferences",
                return_value="gpt-4o",
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._check_model_access",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._run_budget_checks",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._build_completion_kwargs",
                new_callable=AsyncMock,
                return_value={"model": "gpt-4o", "messages": []},
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._run_guardrails_and_call_llm",
                new_callable=AsyncMock,
                return_value=response,
            ),
        ):
            result = await handle_sampling_create_message(
                context=MagicMock(),
                params=_params(),
                default_model="gpt-4o",
                user_api_key_auth=auth,
            )

        assert isinstance(result, CreateMessageResult)
        assert result.content.text == "the answer is 42"
        assert result.stopReason == "endTurn"

    async def test_should_reraise_known_proxy_exceptions(self):
        from litellm.exceptions import RateLimitError

        auth = SimpleNamespace(user_id="u1", api_key="sk-test", token="tok")
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._resolve_model_from_preferences",
                return_value="gpt-4o",
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._check_model_access",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._run_budget_checks",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._build_completion_kwargs",
                new_callable=AsyncMock,
                side_effect=RateLimitError(
                    "rate limited", llm_provider="openai", model="gpt-4o"
                ),
            ),
        ):
            with pytest.raises(RateLimitError):
                await handle_sampling_create_message(
                    context=MagicMock(),
                    params=_params(),
                    default_model="gpt-4o",
                    user_api_key_auth=auth,
                )

    async def test_should_return_error_data_on_unexpected_failure(self):
        auth = SimpleNamespace(user_id="u1", api_key="sk-test", token="tok")
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._resolve_model_from_preferences",
                return_value="gpt-4o",
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._check_model_access",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._run_budget_checks",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.sampling_handler._build_completion_kwargs",
                new_callable=AsyncMock,
                side_effect=RuntimeError("kaboom"),
            ),
        ):
            result = await handle_sampling_create_message(
                context=MagicMock(),
                params=_params(),
                default_model="gpt-4o",
                user_api_key_auth=auth,
            )

        assert isinstance(result, ErrorData)
        assert "kaboom" in result.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
