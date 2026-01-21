"""
Unit tests for Qualifire guardrail integration.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.types.guardrails import GuardrailEventHooks


class TestQualifireGuardrailInit:
    """Tests for QualifireGuardrail initialization."""

    def test_init_with_default_prompt_injections(self):
        """Test that prompt_injections defaults to True when no checks are specified."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="test_guardrail",
        )

        assert guardrail.prompt_injections is True
        assert guardrail.qualifire_api_key == "test_key"

    def test_init_with_evaluation_id_no_default_checks(self):
        """Test that no default checks are enabled when evaluation_id is provided."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            evaluation_id="eval_123",
            guardrail_name="test_guardrail",
        )

        # prompt_injections should remain None since evaluation_id is provided
        assert guardrail.prompt_injections is None
        assert guardrail.evaluation_id == "eval_123"

    def test_init_with_explicit_checks(self):
        """Test initialization with explicit check flags."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            pii_check=True,
            hallucinations_check=True,
            guardrail_name="test_guardrail",
        )

        assert guardrail.pii_check is True
        assert guardrail.hallucinations_check is True
        # prompt_injections should not be set to True if other checks are provided
        assert guardrail.prompt_injections is None

    def test_init_with_on_flagged_monitor(self):
        """Test initialization with monitor mode."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            on_flagged="monitor",
            guardrail_name="test_guardrail",
        )

        assert guardrail.on_flagged == "monitor"

    def test_init_with_default_api_base(self):
        """Test that default API base is set when not provided."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            DEFAULT_QUALIFIRE_API_BASE,
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="test_guardrail",
        )

        assert guardrail.qualifire_api_base == DEFAULT_QUALIFIRE_API_BASE

    def test_init_with_custom_api_base(self):
        """Test initialization with custom API base URL."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            api_base="https://custom.qualifire.ai",
            guardrail_name="test_guardrail",
        )

        assert guardrail.qualifire_api_base == "https://custom.qualifire.ai"


class TestQualifireGuardrailMessageConversion:
    """Tests for message conversion to API format."""

    def test_convert_simple_messages(self):
        """Test conversion of simple text messages."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="test_guardrail",
        )

        messages = [
            {"role": "user", "content": "Hello, world!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = guardrail._convert_messages_to_api_format(messages)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, world!"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hi there!"

    def test_convert_multimodal_messages(self):
        """Test conversion of multimodal messages with text parts."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="test_guardrail",
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"},
                ],
            },
        ]

        result = guardrail._convert_messages_to_api_format(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "First part\nSecond part"

    def test_convert_messages_with_tool_calls(self):
        """Test conversion of messages with tool calls."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="test_guardrail",
        )

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            },
        ]

        result = guardrail._convert_messages_to_api_format(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "call_123"
        assert result[0]["tool_calls"][0]["name"] == "get_weather"
        assert result[0]["tool_calls"][0]["arguments"] == {"location": "NYC"}


class TestQualifireGuardrailToolConversion:
    """Tests for tool definition conversion."""

    def test_convert_openai_function_tools(self):
        """Test conversion of OpenAI function tool format."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="test_guardrail",
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = guardrail._convert_tools_to_api_format(tools)

        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather for a location"

    def test_convert_empty_tools(self):
        """Test that empty tools returns None."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="test_guardrail",
        )

        result = guardrail._convert_tools_to_api_format(None)
        assert result is None

        result = guardrail._convert_tools_to_api_format([])
        assert result is None


class TestQualifireGuardrailAPICall:
    """Tests for API call with httpx client."""

    @pytest.mark.asyncio
    async def test_evaluate_called_with_prompt_injections(self):
        """Test that evaluate endpoint is called with prompt_injections enabled."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            prompt_injections=True,
            guardrail_name="test_guardrail",
        )

        # Mock the async HTTP handler
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "score": 100,
            "status": "completed",
            "evaluationResults": [],
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello, world!"}]

        await guardrail._run_qualifire_check(
            messages=messages, output=None, dynamic_params={}
        )

        # Verify the API was called
        guardrail.async_handler.post.assert_called_once()
        call_kwargs = guardrail.async_handler.post.call_args[1]

        assert "json" in call_kwargs
        payload = call_kwargs["json"]
        assert payload["prompt_injections"] is True
        assert "messages" in payload
        assert call_kwargs["url"].endswith("/api/evaluation/evaluate")

    @pytest.mark.asyncio
    async def test_evaluate_called_with_multiple_checks(self):
        """Test that evaluate is called with multiple checks enabled."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            prompt_injections=True,
            pii_check=True,
            hallucinations_check=True,
            assertions=["Output must be valid JSON"],
            guardrail_name="test_guardrail",
        )

        # Mock the async HTTP handler
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "score": 100,
            "status": "completed",
            "evaluationResults": [],
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello, world!"}]

        await guardrail._run_qualifire_check(
            messages=messages, output="Test output", dynamic_params={}
        )

        # Verify the API was called with correct payload
        guardrail.async_handler.post.assert_called_once()
        call_kwargs = guardrail.async_handler.post.call_args[1]

        payload = call_kwargs["json"]
        assert payload["prompt_injections"] is True
        assert payload["pii_check"] is True
        assert payload["hallucinations_check"] is True
        assert payload["assertions"] == ["Output must be valid JSON"]
        assert payload["output"] == "Test output"

    @pytest.mark.asyncio
    async def test_invoke_endpoint_used_with_evaluation_id(self):
        """Test that invoke endpoint is used when evaluation_id is provided."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            evaluation_id="eval_123",
            guardrail_name="test_guardrail",
        )

        # Mock the async HTTP handler
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "score": 100,
            "status": "completed",
            "evaluationResults": [],
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello, world!"}]

        await guardrail._run_qualifire_check(
            messages=messages, output="Test output", dynamic_params={}
        )

        # Verify the invoke endpoint was called
        guardrail.async_handler.post.assert_called_once()
        call_kwargs = guardrail.async_handler.post.call_args[1]

        assert call_kwargs["url"].endswith("/api/evaluation/invoke")
        payload = call_kwargs["json"]
        assert payload["evaluation_id"] == "eval_123"
        assert payload["input"] == "Hello, world!"
        assert payload["output"] == "Test output"

    @pytest.mark.asyncio
    async def test_correct_headers_sent(self):
        """Test that correct headers are sent with the API request."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="my_api_key",
            guardrail_name="test_guardrail",
        )

        # Mock the async HTTP handler
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "score": 100,
            "status": "completed",
            "evaluationResults": [],
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello!"}]

        await guardrail._run_qualifire_check(
            messages=messages, output=None, dynamic_params={}
        )

        call_kwargs = guardrail.async_handler.post.call_args[1]
        headers = call_kwargs["headers"]

        assert headers["X-Qualifire-API-Key"] == "my_api_key"
        assert headers["Content-Type"] == "application/json"


class TestQualifireGuardrailCheckIfFlagged:
    """Tests for the _check_if_flagged method."""

    def test_check_if_flagged_returns_false_for_success(self):
        """Test that _check_if_flagged returns False for successful evaluations."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="test_guardrail",
        )

        # Result with completed status and no flagged items (dict format)
        result = {
            "status": "completed",
            "score": 100,
            "evaluationResults": [],
        }

        assert guardrail._check_if_flagged(result) is False

    def test_check_if_flagged_returns_true_for_flagged_content(self):
        """Test that _check_if_flagged returns True when content is flagged."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="test_guardrail",
        )

        # Result with flagged item (dict format matching API response)
        result = {
            "status": "completed",
            "score": 15,
            "evaluationResults": [
                {
                    "type": "prompt_injection",
                    "results": [
                        {
                            "flagged": True,
                            "score": 0.15,
                            "reason": "Prompt injection detected",
                        }
                    ],
                }
            ],
        }

        assert guardrail._check_if_flagged(result) is True

    def test_check_if_flagged_returns_false_when_no_flagged_items(self):
        """Test that _check_if_flagged returns False when no items are flagged."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="test_guardrail",
        )

        # Result with evaluation results but nothing flagged
        result = {
            "status": "completed",
            "score": 95,
            "evaluationResults": [
                {
                    "type": "prompt_injection",
                    "results": [
                        {
                            "flagged": False,
                            "score": 0.95,
                            "reason": "No issues detected",
                        }
                    ],
                }
            ],
        }

        assert guardrail._check_if_flagged(result) is False


class TestQualifireGuardrailShouldRun:
    """Tests for should_run_guardrail method."""

    def test_should_run_guardrail_with_guardrail_in_metadata(self):
        """Test that guardrail runs when specified in metadata."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="qualifire-guard",
            event_hook=GuardrailEventHooks.pre_call,
        )

        data = {
            "messages": [{"role": "user", "content": "test"}],
            "metadata": {"guardrails": ["qualifire-guard"]},
        }

        result = guardrail.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.pre_call
        )

        assert result is True

    def test_should_not_run_guardrail_when_not_in_metadata(self):
        """Test that guardrail doesn't run when not specified in metadata."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="qualifire-guard",
            event_hook=GuardrailEventHooks.pre_call,
        )

        data = {
            "messages": [{"role": "user", "content": "test"}],
            "metadata": {"guardrails": ["other-guardrail"]},
        }

        result = guardrail.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.pre_call
        )

        assert result is False

    def test_should_run_guardrail_with_default_on(self):
        """Test that guardrail runs when default_on is True."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="qualifire-guard",
            event_hook=GuardrailEventHooks.pre_call,
            default_on=True,
        )

        data = {
            "messages": [{"role": "user", "content": "test"}],
        }

        result = guardrail.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.pre_call
        )

        assert result is True


class TestQualifireGuardrailHooks:
    """Tests for guardrail hook methods."""

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_returns_none_when_disabled(self):
        """Test that async_pre_call_hook returns None when guardrail is disabled."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="qualifire-guard",
            event_hook=GuardrailEventHooks.pre_call,
        )

        data = {
            "messages": [{"role": "user", "content": "test"}],
            "metadata": {"guardrails": ["other-guardrail"]},
        }

        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=MagicMock(),
            cache=MagicMock(),
            data=data,
            call_type="completion",
        )

        # When guardrail doesn't run (not in metadata), it returns None
        assert result is None

    @pytest.mark.asyncio
    async def test_async_moderation_hook_returns_when_no_messages(self):
        """Test that async_moderation_hook returns when no messages in data."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="qualifire-guard",
            event_hook=GuardrailEventHooks.during_call,
            default_on=True,
        )

        data = {
            "model": "gpt-4",
            # No messages
        }

        result = await guardrail.async_moderation_hook(
            data=data,
            user_api_key_dict=MagicMock(),
            call_type="completion",
        )

        assert result is None


class TestQualifireGuardrailConfigModel:
    """Tests for QualifireGuardrailConfigModel."""

    def test_config_model_ui_friendly_name(self):
        """Test that config model has correct UI friendly name."""
        from litellm.types.proxy.guardrails.guardrail_hooks.qualifire import (
            QualifireGuardrailConfigModel,
        )

        assert QualifireGuardrailConfigModel.ui_friendly_name() == "Qualifire"

    def test_config_model_fields(self):
        """Test that config model has expected fields."""
        from litellm.types.proxy.guardrails.guardrail_hooks.qualifire import (
            QualifireGuardrailConfigModel,
        )

        model = QualifireGuardrailConfigModel()

        # Check default values
        assert model.on_flagged == "block"
        assert model.evaluation_id is None
        assert model.prompt_injections is None


class TestQualifireGuardrailRegistry:
    """Tests for guardrail registry integration."""

    def test_qualifire_in_supported_integrations(self):
        """Test that QUALIFIRE is in SupportedGuardrailIntegrations enum."""
        from litellm.types.guardrails import SupportedGuardrailIntegrations

        assert hasattr(SupportedGuardrailIntegrations, "QUALIFIRE")
        assert SupportedGuardrailIntegrations.QUALIFIRE.value == "qualifire"

    def test_initialize_guardrail_function_exists(self):
        """Test that initialize_guardrail function is properly exported."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire import (
            guardrail_initializer_registry,
            initialize_guardrail,
        )

        assert initialize_guardrail is not None
        assert "qualifire" in guardrail_initializer_registry

    def test_guardrail_class_registry_exists(self):
        """Test that guardrail_class_registry is properly exported."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire import (
            guardrail_class_registry,
        )
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        assert "qualifire" in guardrail_class_registry
        assert guardrail_class_registry["qualifire"] == QualifireGuardrail
