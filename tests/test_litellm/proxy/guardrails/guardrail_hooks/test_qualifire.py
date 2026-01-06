"""
Unit tests for Qualifire guardrail integration.
"""

import sys
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


class TestQualifireGuardrailMessageConversion:
    """Tests for message conversion to Qualifire format."""

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

        # Create mock LLMMessage class
        mock_llm_message = MagicMock()

        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire.QualifireGuardrail._convert_messages_to_qualifire_format"
        ) as mock_convert:
            mock_convert.return_value = [mock_llm_message, mock_llm_message]
            result = guardrail._convert_messages_to_qualifire_format(messages)
            assert len(result) == 2

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

        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire.QualifireGuardrail._convert_messages_to_qualifire_format"
        ) as mock_convert:
            mock_convert.return_value = [MagicMock()]
            result = guardrail._convert_messages_to_qualifire_format(messages)
            assert len(result) == 1


class TestQualifireGuardrailEvaluateKwargs:
    """Tests for evaluate kwargs passed to Qualifire client."""

    @pytest.mark.asyncio
    async def test_evaluate_called_with_prompt_injections(self):
        """Test that evaluate is called with prompt_injections enabled."""
        # Mock the qualifire module and its types
        mock_qualifire_types = MagicMock()
        mock_llm_message = MagicMock()
        mock_llm_tool_call = MagicMock()
        mock_message_instance = MagicMock()
        mock_llm_message.return_value = mock_message_instance
        
        mock_qualifire_types.LLMMessage = mock_llm_message
        mock_qualifire_types.LLMToolCall = mock_llm_tool_call
        
        with patch.dict('sys.modules', {'qualifire': MagicMock(), 'qualifire.types': mock_qualifire_types}):
            from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
                QualifireGuardrail,
            )

            guardrail = QualifireGuardrail(
                api_key="test_key",
                prompt_injections=True,
                guardrail_name="test_guardrail",
            )

            # Mock the client
            mock_client = MagicMock()
            mock_result = MagicMock()
            mock_result.score = 100
            mock_result.status = "completed"
            mock_result.evaluationResults = []
            mock_client.evaluate.return_value = mock_result
            guardrail._client = mock_client

            messages = [{"role": "user", "content": "Hello, world!"}]

            await guardrail._run_qualifire_check(
                messages=messages, output=None, dynamic_params={}
            )

            # Verify evaluate was called with correct kwargs
            mock_client.evaluate.assert_called_once()
            call_kwargs = mock_client.evaluate.call_args[1]
            assert call_kwargs["prompt_injections"] is True
            assert "messages" in call_kwargs

    @pytest.mark.asyncio
    async def test_evaluate_called_with_multiple_checks(self):
        """Test that evaluate is called with multiple checks enabled."""
        # Mock the qualifire module and its types
        mock_qualifire_types = MagicMock()
        mock_llm_message = MagicMock()
        mock_llm_tool_call = MagicMock()
        mock_message_instance = MagicMock()
        mock_llm_message.return_value = mock_message_instance
        
        mock_qualifire_types.LLMMessage = mock_llm_message
        mock_qualifire_types.LLMToolCall = mock_llm_tool_call
        
        with patch.dict('sys.modules', {'qualifire': MagicMock(), 'qualifire.types': mock_qualifire_types}):
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

            # Mock the client
            mock_client = MagicMock()
            mock_result = MagicMock()
            mock_result.score = 100
            mock_result.status = "completed"
            mock_result.evaluationResults = []
            mock_client.evaluate.return_value = mock_result
            guardrail._client = mock_client

            messages = [{"role": "user", "content": "Hello, world!"}]

            await guardrail._run_qualifire_check(
                messages=messages, output="Test output", dynamic_params={}
            )

            # Verify evaluate was called with correct kwargs
            mock_client.evaluate.assert_called_once()
            call_kwargs = mock_client.evaluate.call_args[1]
            assert call_kwargs["prompt_injections"] is True
            assert call_kwargs["pii_check"] is True
            assert call_kwargs["hallucinations_check"] is True
            assert call_kwargs["assertions"] == ["Output must be valid JSON"]
            assert call_kwargs["output"] == "Test output"


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

        # Mock result with completed status and no flagged items
        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.evaluationResults = []

        assert guardrail._check_if_flagged(mock_result) is False

    def test_check_if_flagged_returns_true_for_flagged_content(self):
        """Test that _check_if_flagged returns True when content is flagged."""
        from litellm.proxy.guardrails.guardrail_hooks.qualifire.qualifire import (
            QualifireGuardrail,
        )

        guardrail = QualifireGuardrail(
            api_key="test_key",
            guardrail_name="test_guardrail",
        )

        # Mock result with flagged item
        mock_inner_result = MagicMock()
        mock_inner_result.flagged = True

        mock_eval_result = MagicMock()
        mock_eval_result.results = [mock_inner_result]

        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.evaluationResults = [mock_eval_result]

        assert guardrail._check_if_flagged(mock_result) is True

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
        mock_inner_result = MagicMock()
        mock_inner_result.flagged = False

        mock_eval_result = MagicMock()
        mock_eval_result.results = [mock_inner_result]

        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.evaluationResults = [mock_eval_result]

        assert guardrail._check_if_flagged(mock_result) is False


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
