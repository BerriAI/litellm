"""
Unit tests for PointGuardAI guardrail integration.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.pointguardai.pointguardai import (
    PointGuardAIGuardrail,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs


class TestPointGuardAIGuardrailInit:
    """Tests for PointGuardAIGuardrail initialization."""

    def test_init_with_required_params(self):
        """Test initialization with all required parameters."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_api_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
        )

        assert guardrail.pointguardai_api_key == "test_api_key"
        assert guardrail.pointguardai_org_code == "test-org"
        assert guardrail.pointguardai_policy_config_name == "test-policy"
        assert guardrail.guardrail_name == "pointguardai-guard"

    def test_init_with_missing_api_key(self):
        """Test that initialization fails without api_key."""
        with pytest.raises(HTTPException) as exc_info:
            PointGuardAIGuardrail(
                api_key="",
                api_base="https://api.appsoc.com",
                org_code="test-org",
                policy_config_name="test-policy",
            )

        assert exc_info.value.status_code == 401
        assert "api_key" in str(exc_info.value.detail)

    def test_init_with_missing_org_code(self):
        """Test that initialization fails without org_code."""
        with pytest.raises(HTTPException) as exc_info:
            PointGuardAIGuardrail(
                api_key="test_key",
                api_base="https://api.appsoc.com",
                org_code="",
                policy_config_name="test-policy",
            )

        assert exc_info.value.status_code == 401
        assert "org_code" in str(exc_info.value.detail)

    def test_init_with_missing_policy_config_name(self):
        """Test that initialization fails without policy_config_name."""
        with pytest.raises(HTTPException) as exc_info:
            PointGuardAIGuardrail(
                api_key="test_key",
                api_base="https://api.appsoc.com",
                org_code="test-org",
                policy_config_name="",
            )

        assert exc_info.value.status_code == 401
        assert "policy_config_name" in str(exc_info.value.detail)

    def test_init_with_default_api_base(self):
        """Test that default API base is set and path is appended."""
        # When api_base is provided without the full path, it should append the path
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        # Should append path to provided base
        assert "https://api.appsoc.com" in guardrail.pointguardai_api_base
        assert "/policies/inspect" in guardrail.pointguardai_api_base
        assert guardrail.pointguardai_api_base.endswith("/policies/inspect")

    def test_init_with_custom_api_base(self):
        """Test initialization with custom API base URL."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
            api_base="https://custom.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        assert "https://custom.appsoc.com" in guardrail.pointguardai_api_base
        assert "/policies/inspect" in guardrail.pointguardai_api_base

    def test_init_with_org_code_template_replacement(self):
        """Test that {{org}} template is replaced with org_code."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="my-org-123",
            policy_config_name="test-policy",
        )

        # URL should have org_code filled in
        assert "my-org-123" in guardrail.pointguardai_api_base
        assert "{{org}}" not in guardrail.pointguardai_api_base

    def test_init_headers_configuration(self):
        """Test that headers are correctly configured for v2 API."""
        guardrail = PointGuardAIGuardrail(
            api_key="my_secret_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        assert guardrail.headers["X-appsoc-api-key"] == "my_secret_key"
        assert guardrail.headers["Content-Type"] == "application/json"


class TestPointGuardAIGuardrailMessageTransformation:
    """Tests for message transformation to API format."""

    def test_transform_messages_with_supported_roles(self):
        """Test transformation of messages with supported roles."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = guardrail.transform_messages(messages)

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_transform_messages_with_tool_role(self):
        """Test that unsupported roles are converted to 'user'."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        messages = [
            {"role": "user", "content": "Get weather"},
            {"role": "tool", "content": "Weather is sunny", "tool_call_id": "123"},
        ]

        result = guardrail.transform_messages(messages)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        # Tool role should be converted to user
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Weather is sunny"

    def test_transform_messages_preserves_content(self):
        """Test that message content is preserved during transformation."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        messages = [
            {"role": "user", "content": "This is my message", "extra_field": "value"},
        ]

        result = guardrail.transform_messages(messages)

        assert result[0]["content"] == "This is my message"
        # Extra fields should be preserved
        assert result[0]["extra_field"] == "value"


class TestPointGuardAIGuardrailRequestPreparation:
    """Tests for API request preparation."""

    @pytest.mark.asyncio
    async def test_prepare_request_with_input_only(self):
        """Test request preparation with input messages only (pre_call)."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="my-policy",
        )

        messages = [{"role": "user", "content": "Hello"}]

        result = await guardrail.prepare_pointguard_ai_runtime_scanner_request(
            new_messages=messages,
            response_string=None,
        )

        assert result is not None
        assert result["configName"] == "my-policy"
        assert "input" in result
        assert result["input"] == messages
        assert "output" not in result

    @pytest.mark.asyncio
    async def test_prepare_request_with_output_only(self):
        """Test request preparation with output only (post_call response)."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="my-policy",
        )

        result = await guardrail.prepare_pointguard_ai_runtime_scanner_request(
            new_messages=[],
            response_string="This is the response",
        )

        assert result is not None
        assert result["configName"] == "my-policy"
        assert "output" in result
        assert result["output"][0]["role"] == "assistant"
        assert result["output"][0]["content"] == "This is the response"

    @pytest.mark.asyncio
    async def test_prepare_request_with_both_input_output(self):
        """Test request preparation with both input and output."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="my-policy",
        )

        messages = [{"role": "user", "content": "Hello"}]

        result = await guardrail.prepare_pointguard_ai_runtime_scanner_request(
            new_messages=messages,
            response_string="Hi there!",
        )

        assert result is not None
        assert "input" in result
        assert "output" in result
        assert result["input"] == messages
        assert result["output"][0]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_prepare_request_with_model_metadata(self):
        """Test that model metadata is included when provided."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="my-policy",
            model_provider_name="openai",
            model_name="gpt-4",
        )

        messages = [{"role": "user", "content": "Hello"}]

        result = await guardrail.prepare_pointguard_ai_runtime_scanner_request(
            new_messages=messages,
            response_string=None,
        )

        assert result is not None
        assert result["modelProviderName"] == "openai"
        assert result["modelName"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_prepare_request_without_model_metadata(self):
        """Test that model metadata is not included when not provided."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="my-policy",
        )

        messages = [{"role": "user", "content": "Hello"}]

        result = await guardrail.prepare_pointguard_ai_runtime_scanner_request(
            new_messages=messages,
            response_string=None,
        )

        assert result is not None
        assert "modelProviderName" not in result
        assert "modelName" not in result

    @pytest.mark.asyncio
    async def test_prepare_request_returns_none_for_empty_data(self):
        """Test that None is returned when no messages or response provided."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="my-policy",
        )

        result = await guardrail.prepare_pointguard_ai_runtime_scanner_request(
            new_messages=[],
            response_string=None,
        )

        assert result is None


class TestPointGuardAIGuardrailResponseProcessing:
    """Tests for response processing and violation detection."""

    def test_check_sections_present_with_input(self):
        """Test detection of input section in response."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        response_data = {
            "input": {
                "blocked": False,
                "content": [{"originalContent": "Hello"}],
            }
        }
        messages = [{"role": "user", "content": "Hello"}]

        input_present, output_present = guardrail._check_sections_present(
            response_data, messages, None
        )

        assert input_present is True
        assert output_present is False

    def test_check_sections_present_with_output(self):
        """Test detection of output section in response."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        response_data = {
            "output": {
                "blocked": False,
                "content": [{"originalContent": "Hi there"}],
            }
        }

        input_present, output_present = guardrail._check_sections_present(
            response_data, [], "Hi there"
        )

        assert input_present is False
        assert output_present is True

    def test_extract_status_flags_input_blocked(self):
        """Test extraction of input blocked flag."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        response_data = {
            "input": {"blocked": True, "modified": False},
            "output": {"blocked": False, "modified": False},
        }

        (
            input_blocked,
            output_blocked,
            input_modified,
            output_modified,
        ) = guardrail._extract_status_flags(response_data, True, False)

        assert input_blocked is True
        assert output_blocked is False
        assert input_modified is False
        assert output_modified is False

    def test_extract_status_flags_output_modified(self):
        """Test extraction of output modified flag."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        response_data = {
            "input": {"blocked": False, "modified": False},
            "output": {"blocked": False, "modified": True},
        }

        (
            input_blocked,
            output_blocked,
            input_modified,
            output_modified,
        ) = guardrail._extract_status_flags(response_data, False, True)

        assert input_blocked is False
        assert output_blocked is False
        assert input_modified is False
        assert output_modified is True

    def test_extract_violations_from_input(self):
        """Test extraction of violations from input section."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        response_data = {
            "input": {
                "blocked": True,
                "content": [
                    {
                        "violations": [
                            {"severity": "HIGH", "categories": ["pii"]},
                            {"severity": "MEDIUM", "categories": ["toxic"]},
                        ]
                    }
                ],
            }
        }

        violations = guardrail._extract_violations(response_data, True, False)

        assert len(violations) == 2
        assert violations[0]["severity"] == "HIGH"
        assert violations[1]["severity"] == "MEDIUM"

    def test_create_violation_details(self):
        """Test creation of violation detail objects."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        violations = [
            {
                "severity": "HIGH",
                "scanner": "pii_scanner",
                "inspector": "regex",
                "categories": ["credit_card", "ssn"],
                "confidenceScore": 0.95,
                "mode": "BLOCKING",
            }
        ]

        details = guardrail._create_violation_details(violations)

        assert len(details) == 1
        assert details[0]["severity"] == "HIGH"
        assert details[0]["scanner"] == "pii_scanner"
        assert details[0]["categories"] == ["credit_card", "ssn"]
        assert details[0]["confidenceScore"] == 0.95

    def test_handle_modifications_input(self):
        """Test handling of input modifications."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        response_data = {
            "input": {
                "modified": True,
                "content": [
                    {
                        "originalContent": "My SSN is 123-45-6789",
                        "modifiedContent": "My SSN is [REDACTED]",
                    }
                ],
            }
        }

        result = guardrail._handle_modifications(response_data, True, False)

        assert result is not None
        assert len(result) == 1
        assert result[0]["modifiedContent"] == "My SSN is [REDACTED]"


class TestPointGuardAIGuardrailAPICall:
    """Tests for API call with httpx client."""

    @pytest.mark.asyncio
    async def test_api_call_no_violations(self):
        """Test API call when no violations are detected."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        # Mock successful response with no violations
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "input": {"blocked": False, "modified": False, "content": []},
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]

        result = await guardrail.make_pointguard_api_request(
            request_data={},
            new_messages=messages,
            response_string=None,
        )

        assert result is None  # No modifications
        guardrail.async_handler.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_call_with_blocked_content(self):
        """Test API call when content is blocked."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        # Mock blocked response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "input": {
                "blocked": True,
                "content": [
                    {
                        "violations": [
                            {
                                "severity": "HIGH",
                                "scanner": "test",
                                "inspector": "test",
                                "categories": ["prohibited"],
                                "confidenceScore": 0.95,
                                "mode": "BLOCKING",
                            }
                        ]
                    }
                ],
            }
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Bad content"}]

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.make_pointguard_api_request(
                request_data={},
                new_messages=messages,
                response_string=None,
            )

        assert exc_info.value.status_code == 400
        assert "Violated PointGuardAI policy" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_api_call_with_modified_content(self):
        """Test API call when content is modified."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        # Mock modified response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "input": {
                "blocked": False,
                "modified": True,
                "content": [
                    {
                        "originalContent": "My email is test@example.com",
                        "modifiedContent": "My email is [REDACTED]",
                    }
                ],
            }
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "My email is test@example.com"}]

        result = await guardrail.make_pointguard_api_request(
            request_data={},
            new_messages=messages,
            response_string=None,
        )

        assert result is not None
        assert len(result) == 1
        assert result[0]["modifiedContent"] == "My email is [REDACTED]"

    @pytest.mark.asyncio
    async def test_api_call_correct_headers(self):
        """Test that correct headers are sent with API request."""
        guardrail = PointGuardAIGuardrail(
            api_key="my_secret_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "input": {"blocked": False, "modified": False},
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]

        await guardrail.make_pointguard_api_request(
            request_data={},
            new_messages=messages,
            response_string=None,
        )

        call_kwargs = guardrail.async_handler.post.call_args[1]
        assert call_kwargs["headers"]["X-appsoc-api-key"] == "my_secret_key"
        assert call_kwargs["headers"]["X-appsoc-api-email"] == "admin@example.com"


class TestPointGuardAIGuardrailApplyGuardrail:
    """Tests for the unified apply_guardrail method."""

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_no_violations(self):
        """Test apply_guardrail for request input with no violations."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
        )

        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "input": {"blocked": False, "modified": False},
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        inputs = GenericGuardrailAPIInputs(
            texts=["Hello, world!"],
            structured_messages=[{"role": "user", "content": "Hello, world!"}],
        )

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"messages": [{"role": "user", "content": "Hello, world!"}]},
            input_type="request",
            logging_obj=None,
        )

        assert result == inputs  # No modifications

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_blocked(self):
        """Test apply_guardrail for request that gets blocked."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
        )

        # Mock blocked response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "input": {
                "blocked": True,
                "content": [
                    {
                        "violations": [
                            {
                                "severity": "HIGH",
                                "scanner": "test",
                                "inspector": "test",
                                "categories": ["prohibited"],
                                "confidenceScore": 0.95,
                                "mode": "BLOCKING",
                            }
                        ]
                    }
                ],
            }
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        inputs = GenericGuardrailAPIInputs(
            texts=["Bad content"],
            structured_messages=[{"role": "user", "content": "Bad content"}],
        )

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"messages": [{"role": "user", "content": "Bad content"}]},
                input_type="request",
                logging_obj=None,
            )

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_modified(self):
        """Test apply_guardrail for request with content modification."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
        )

        # Mock modified response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "input": {
                "blocked": False,
                "modified": True,
                "content": [
                    {
                        "originalContent": "My SSN is 123-45-6789",
                        "modifiedContent": "My SSN is [REDACTED]",
                    }
                ],
            }
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "My SSN is 123-45-6789"}]
        inputs = GenericGuardrailAPIInputs(
            texts=["My SSN is 123-45-6789"],
            structured_messages=messages,
        )

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"messages": messages},
            input_type="request",
            logging_obj=None,
        )

        # Content should be modified
        assert result["structured_messages"][0]["content"] == "My SSN is [REDACTED]"

    @pytest.mark.asyncio
    async def test_apply_guardrail_response_no_violations(self):
        """Test apply_guardrail for response with no violations."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
        )

        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {"blocked": False, "modified": False},
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        inputs = GenericGuardrailAPIInputs(
            texts=["I'm doing well, thanks!"],
        )

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="response",
            logging_obj=None,
        )

        assert result == inputs

    @pytest.mark.asyncio
    async def test_apply_guardrail_response_blocked(self):
        """Test apply_guardrail for response that gets blocked."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
        )

        # Mock blocked response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {
                "blocked": True,
                "content": [
                    {
                        "violations": [
                            {
                                "severity": "MEDIUM",
                                "scanner": "test",
                                "inspector": "test",
                                "categories": ["sensitive"],
                                "confidenceScore": 0.85,
                                "mode": "BLOCKING",
                            }
                        ]
                    }
                ],
            }
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        inputs = GenericGuardrailAPIInputs(
            texts=["Sensitive response"],
        )

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="response",
                logging_obj=None,
            )

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_apply_guardrail_response_modified(self):
        """Test apply_guardrail for response with content modification."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
        )

        # Mock modified response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {
                "blocked": False,
                "modified": True,
                "content": [
                    {
                        "originalContent": "Contact me at test@example.com",
                        "modifiedContent": "Contact me at [EMAIL_REDACTED]",
                    }
                ],
            }
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        inputs = GenericGuardrailAPIInputs(
            texts=["Contact me at test@example.com"],
        )

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="response",
            logging_obj=None,
        )

        # Content should be modified
        assert result["texts"][0] == "Contact me at [EMAIL_REDACTED]"

    @pytest.mark.asyncio
    async def test_apply_guardrail_extracts_messages_from_request_data(self):
        """Test that apply_guardrail extracts messages from request_data when not in inputs."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
        )

        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "input": {"blocked": False, "modified": False},
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        # No structured_messages in inputs
        inputs = GenericGuardrailAPIInputs(
            texts=["Hello"],
        )

        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"messages": [{"role": "user", "content": "Hello"}]},
            input_type="request",
            logging_obj=None,
        )

        # Should have called the API with transformed messages
        guardrail.async_handler.post.assert_called_once()


class TestPointGuardAIGuardrailErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handle_http_status_error_401(self):
        """Test handling of 401 authentication error."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        # Mock 401 error
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        def raise_status_error():
            raise Exception("HTTP 401")

        mock_response.raise_for_status = raise_status_error

        import httpx

        error = httpx.HTTPStatusError(
            "401 error", request=MagicMock(), response=mock_response
        )

        with pytest.raises(HTTPException) as exc_info:
            guardrail._handle_http_status_error(error)

        assert exc_info.value.status_code == 401
        assert "authentication failed" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_handle_network_timeout(self):
        """Test handling of timeout error."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        import httpx

        timeout_error = httpx.TimeoutException("Request timeout")

        with pytest.raises(HTTPException) as exc_info:
            guardrail._handle_network_errors(timeout_error)

        assert exc_info.value.status_code == 504
        assert "timeout" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_handle_connection_error(self):
        """Test handling of connection error."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        import httpx

        conn_error = httpx.ConnectError("Connection refused")

        with pytest.raises(HTTPException) as exc_info:
            guardrail._handle_network_errors(conn_error)

        assert exc_info.value.status_code == 503
        assert "unavailable" in str(exc_info.value.detail).lower()


class TestPointGuardAIGuardrailShouldRun:
    """Tests for should_run_guardrail method."""

    def test_should_run_guardrail_with_guardrail_in_metadata(self):
        """Test that guardrail runs when specified in metadata."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
            event_hook=GuardrailEventHooks.pre_call,
        )

        data = {
            "messages": [{"role": "user", "content": "test"}],
            "metadata": {"guardrails": ["pointguardai-guard"]},
        }

        result = guardrail.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.pre_call
        )

        assert result is True

    def test_should_not_run_guardrail_when_not_in_metadata(self):
        """Test that guardrail doesn't run when not specified in metadata."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
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
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
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

    def test_should_run_guardrail_with_wrong_event_hook(self):
        """Test that guardrail doesn't run with mismatched event hook."""
        guardrail = PointGuardAIGuardrail(
            api_key="test_key",
                        api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
            event_hook=GuardrailEventHooks.pre_call,
            default_on=True,
        )

        data = {
            "messages": [{"role": "user", "content": "test"}],
        }

        # Trying to run during_call when configured for pre_call
        result = guardrail.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.during_call
        )

        assert result is False


class TestPointGuardAIGuardrailConfigModel:
    """Tests for PointGuardAIGuardrailConfigModel."""

    def test_config_model_ui_friendly_name(self):
        """Test that config model has correct UI friendly name."""
        from litellm.types.proxy.guardrails.guardrail_hooks.pointguardai import (
            PointGuardAIGuardrailConfigModel,
        )

        assert PointGuardAIGuardrailConfigModel.ui_friendly_name() == "PointGuard AI"

    def test_config_model_fields(self):
        """Test that config model has expected fields for v2 API."""
        from litellm.types.proxy.guardrails.guardrail_hooks.pointguardai import (
            PointGuardAIGuardrailConfigModel,
        )

        model = PointGuardAIGuardrailConfigModel()

        # Check default values are None (v2 fields only)
        assert model.api_key is None
        assert model.org_code is None
        assert model.policy_config_name is None
        assert model.correlation_key is None
                
    def test_config_model_with_values(self):
        """Test config model with provided values for v2 API."""
        from litellm.types.proxy.guardrails.guardrail_hooks.pointguardai import (
            PointGuardAIGuardrailConfigModel,
        )

        model = PointGuardAIGuardrailConfigModel(
            api_key="test_key",
            org_code="test-org",
            policy_config_name="test-policy",
            correlation_key="test-correlation-123",
        )

        assert model.api_key == "test_key"
        assert model.org_code == "test-org"
        assert model.policy_config_name == "test-policy"
        assert model.correlation_key == "test-correlation-123"
                

class TestPointGuardAIGuardrailRegistry:
    """Tests for guardrail registry integration."""

    def test_pointguardai_in_supported_integrations(self):
        """Test that POINTGUARDAI is in SupportedGuardrailIntegrations enum."""
        from litellm.types.guardrails import SupportedGuardrailIntegrations

        assert hasattr(SupportedGuardrailIntegrations, "POINTGUARDAI")
        assert SupportedGuardrailIntegrations.POINTGUARDAI.value == "pointguard_ai"

    def test_initialize_guardrail_function_exists(self):
        """Test that initialize_guardrail function is properly exported."""
        from litellm.proxy.guardrails.guardrail_hooks.pointguardai import (
            guardrail_initializer_registry,
            initialize_guardrail,
        )

        assert initialize_guardrail is not None
        assert "pointguard_ai" in guardrail_initializer_registry

    def test_guardrail_class_registry_exists(self):
        """Test that guardrail_class_registry is properly exported."""
        from litellm.proxy.guardrails.guardrail_hooks.pointguardai import (
            guardrail_class_registry,
        )

        assert "pointguard_ai" in guardrail_class_registry
        assert guardrail_class_registry["pointguard_ai"] == PointGuardAIGuardrail

    def test_get_config_model_returns_correct_class(self):
        """Test that get_config_model returns the correct config model class."""
        config_model = PointGuardAIGuardrail.get_config_model()

        from litellm.types.proxy.guardrails.guardrail_hooks.pointguardai import (
            PointGuardAIGuardrailConfigModel,
        )

        assert config_model == PointGuardAIGuardrailConfigModel
