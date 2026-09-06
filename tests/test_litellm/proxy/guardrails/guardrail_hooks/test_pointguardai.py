"""
Unit tests for PointGuardAI guardrail integration.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

from litellm.exceptions import GuardrailRaisedException
from litellm.llms.openai.chat.guardrail_translation.handler import (
    OpenAIChatCompletionsHandler,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.pointguardai.pointguardai import (
    PointGuardAIGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail import (
    unified_guardrail as unified_module,
)
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    CallTypes,
    Delta,
    GenericGuardrailAPIInputs,
    ModelResponseStream,
    StreamingChoices,
)


class _RecordingCallbackManager:
    def __init__(self) -> None:
        self.callback: PointGuardAIGuardrail | None = None

    def add_litellm_callback(self, callback: PointGuardAIGuardrail) -> None:
        self.callback = callback


def _pointguard_guardrail(**kwargs: object) -> PointGuardAIGuardrail:
    return PointGuardAIGuardrail(
        async_handler=MagicMock(),
        **kwargs,  # pyright: ignore[reportArgumentType]  # test helper forwards validated constructor fixtures
    )


class TestPointGuardAIGuardrailInit:
    """Tests for PointGuardAIGuardrail initialization."""

    def test_init_with_required_params(self):
        """Test initialization with all required parameters."""
        guardrail = _pointguard_guardrail(
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
            _pointguard_guardrail(
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
            _pointguard_guardrail(
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
            _pointguard_guardrail(
                api_key="test_key",
                api_base="https://api.appsoc.com",
                org_code="test-org",
                policy_config_name="",
            )

        assert exc_info.value.status_code == 401
        assert "policy_config_name" in str(exc_info.value.detail)

    def test_init_with_default_api_base(self):
        """Test the production API base is used when no override is configured."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        assert guardrail.pointguardai_api_base == "https://api.appsoc.com"
        assert guardrail.input_endpoint == "https://api.appsoc.com/aisec-rdc-v2/api/v1/orgs/test-org/inspect/input"
        assert guardrail.output_endpoint == "https://api.appsoc.com/aisec-rdc-v2/api/v1/orgs/test-org/inspect/output"

    def test_init_with_custom_api_base(self):
        """Test initialization with custom API base URL."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://custom.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        assert guardrail.pointguardai_api_base == "https://custom.appsoc.com"
        assert guardrail.input_endpoint.startswith("https://custom.appsoc.com/")
        assert guardrail.output_endpoint.startswith("https://custom.appsoc.com/")

    def test_init_with_org_code_template_replacement(self):
        """Test that derived endpoints include the configured org code."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="my-org-123",
            policy_config_name="test-policy",
        )

        assert "my-org-123" in guardrail.input_endpoint
        assert "my-org-123" in guardrail.output_endpoint

    def test_init_headers_configuration(self):
        """Test that headers are correctly configured"""
        guardrail = _pointguard_guardrail(
            api_key="my_secret_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        assert guardrail.headers["X-appsoc-api-key"] == "my_secret_key"
        assert guardrail.headers["Content-Type"] == "application/json"

    def test_init_rejects_during_call_when_shared_strict_validation_is_disabled(self, monkeypatch):
        monkeypatch.setenv("LITELLM_STRICT_GUARDRAIL_MODES", "false")

        with pytest.raises(ValueError, match="during_call"):
            _pointguard_guardrail(
                api_key="test_key",
                api_base="https://api.appsoc.com",
                org_code="test-org",
                policy_config_name="test-policy",
                event_hook=[
                    GuardrailEventHooks.pre_call,
                    GuardrailEventHooks.during_call,
                    GuardrailEventHooks.post_call,
                ],
            )


class TestPointGuardAIGuardrailMessageTransformation:
    """Tests for message transformation to API format."""

    def test_transform_messages_with_supported_roles(self):
        """Test transformation of messages with supported roles."""
        guardrail = _pointguard_guardrail(
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
        """Swagger-compatible string roles should retain their semantics."""
        guardrail = _pointguard_guardrail(
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
        assert result[1]["role"] == "tool"
        assert result[1]["content"] == "Weather is sunny"

    def test_transform_messages_emits_only_swagger_message_fields(self):
        """PointGuard InspectMessage accepts only role and string content."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        messages = [
            {"role": "user", "content": "This is my message", "extra_field": "value"},
        ]

        result = guardrail.transform_messages(messages)

        assert result == [{"role": "user", "content": "This is my message"}]

    def test_transform_messages_does_not_send_tool_metadata(self):
        """Tool metadata must not leak into PointGuard's text-only payload."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )
        messages = [
            {
                "role": "tool",
                "content": "Weather is sunny",
                "tool_call_id": "call-123",
                "name": "get_weather",
            }
        ]

        result = guardrail.transform_messages(messages)

        assert result == [{"role": "tool", "content": "Weather is sunny"}]

    @pytest.mark.parametrize("role", [None, "", 123])
    def test_transform_messages_defaults_invalid_roles_to_user(self, role):
        """InspectMessage always requires a non-empty string role."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        result = guardrail.transform_messages([{"role": role, "content": "Hello"}])

        assert result == [{"role": "user", "content": "Hello"}]

    def test_transform_messages_flattens_only_multimodal_text(self):
        """PointGuard's text-only API must not receive image payloads or null content."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First text block"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,secret-image"},
                    },
                    {"type": "input_text", "text": "Second text block"},
                ],
            },
            {"role": "assistant", "content": None},
        ]

        result = guardrail.transform_messages(messages)

        assert result[0]["content"] == "First text block\nSecond text block"
        assert "secret-image" not in result[0]["content"]
        assert result[1]["content"] == ""


class TestPointGuardAIGuardrailRequestPreparation:
    """Tests for API request preparation."""

    @pytest.mark.asyncio
    async def test_prepare_request_with_input_only(self):
        """Test request preparation with input messages only (pre_call)."""
        guardrail = _pointguard_guardrail(
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
        assert result["policyName"] == "my-policy"
        assert "input" in result
        assert result["input"] == messages
        assert "output" not in result

    @pytest.mark.asyncio
    async def test_prepare_request_with_output_only(self):
        """Test request preparation with output only (post_call response)."""
        guardrail = _pointguard_guardrail(
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
        assert result["policyName"] == "my-policy"
        assert result["input"] == []
        assert "output" in result
        assert result["output"][0]["role"] == "assistant"
        assert result["output"][0]["content"] == "This is the response"

    @pytest.mark.asyncio
    async def test_prepare_request_with_both_input_output(self):
        """Test request preparation with both input and output."""
        guardrail = _pointguard_guardrail(
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
    async def test_prepare_request_returns_none_for_empty_data(self):
        """Test that None is returned when no messages or response provided."""
        guardrail = _pointguard_guardrail(
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

    @pytest.mark.asyncio
    async def test_prepare_request_fails_closed_when_policy_state_is_missing(self):
        """Invalid runtime configuration must not silently bypass inspection."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="my-policy",
        )
        guardrail.pointguardai_policy_config_name = ""

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.prepare_pointguard_ai_runtime_scanner_request(
                new_messages=[{"role": "user", "content": "Hello"}],
            )

        assert exc_info.value.status_code == 500
        assert "policy configuration" in str(exc_info.value.detail)


class TestPointGuardAIGuardrailResponseProcessing:
    """Tests for response processing and violation detection."""

    def test_check_sections_present_with_input(self):
        """Test detection of input section in response."""
        guardrail = _pointguard_guardrail(
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

        input_present, output_present = guardrail._check_sections_present(response_data, messages, None)

        assert input_present is True
        assert output_present is False

    def test_check_sections_present_with_output(self):
        """Test detection of output section in response."""
        guardrail = _pointguard_guardrail(
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

        input_present, output_present = guardrail._check_sections_present(response_data, [], "Hi there")

        assert input_present is False
        assert output_present is True

    def test_extract_status_flags_input_blocked(self):
        """Test extraction of input blocked flag."""
        guardrail = _pointguard_guardrail(
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
        guardrail = _pointguard_guardrail(
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
        guardrail = _pointguard_guardrail(
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
                        "dlpViolations": [
                            {
                                "name": "credit-card",
                                "dlpDataTypeId": "cc",
                                "action": "BLOCK",
                                "categories": [{"name": "pii"}],
                                "matchCount": 1,
                            }
                        ],
                        "aiViolations": [
                            {
                                "name": "prompt-injection",
                                "aiThreatCategoryId": "threat-1",
                                "type": "PROMPT_INJECTION",
                                "action": "BLOCK",
                            }
                        ],
                    }
                ],
            }
        }

        violations = guardrail._extract_violations(response_data, True, False)

        assert len(violations) == 2
        assert violations[0]["type"] == "DLP"
        assert violations[1]["type"] == "AI_THREAT"

    def test_create_violation_details(self):
        """Test creation of violation detail objects."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        violations = [
            {
                "type": "DLP",
                "name": "credit-card",
                "action": "BLOCK",
                "categories": [{"name": "credit_card"}, {"code": "ssn"}],
                "match_count": 2,
                "dlp_data_type_id": "cc",
            },
            {
                "type": "AI_THREAT",
                "name": "prompt-injection",
                "threat_type": "PROMPT_INJECTION",
                "action": "BLOCK",
                "ai_threat_category_id": "threat-1",
            },
        ]

        details = guardrail._create_violation_details(violations)

        assert len(details) == 2
        assert details[0]["type"] == "DLP"
        assert details[0]["name"] == "credit-card"
        assert details[0]["categories"] == ["credit_card", "ssn"]
        assert details[1]["type"] == "AI_THREAT"
        assert details[1]["threat_type"] == "PROMPT_INJECTION"

    def test_handle_modifications_input(self):
        """Test handling of input modifications."""
        guardrail = _pointguard_guardrail(
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

    def test_handle_modifications_prefers_output_when_both_are_modified(self):
        """Output redaction must not be lost when both response sections are modified."""
        guardrail = _pointguard_guardrail(
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
                        "role": "user",
                        "originalContent": "input secret",
                        "modifiedContent": "[REDACTED INPUT]",
                    }
                ],
            },
            "output": {
                "modified": True,
                "content": [
                    {
                        "role": "assistant",
                        "originalContent": "output secret",
                        "modifiedContent": "[REDACTED OUTPUT]",
                    }
                ],
            },
        }

        result = guardrail._handle_modifications(response_data, True, True)

        assert result == [
            {
                "role": "assistant",
                "originalContent": "output secret",
                "modifiedContent": "[REDACTED OUTPUT]",
                "index": 0,
            }
        ]


class TestPointGuardAIGuardrailAPICall:
    """Tests for API call with httpx client."""

    @pytest.mark.asyncio
    async def test_api_call_no_violations(self):
        """Test API call when no violations are detected."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        # Mock successful response with no violations
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
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
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        # Mock blocked response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {
                "blocked": True,
                "modified": False,
                "content": [
                    {
                        "aiViolations": [
                            {
                                "name": "prompt-injection",
                                "aiThreatCategoryId": "threat-1",
                                "type": "PROMPT_INJECTION",
                                "action": "BLOCK",
                            }
                        ]
                    }
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Bad content"}]

        with pytest.raises(GuardrailRaisedException) as exc_info:
            await guardrail.make_pointguard_api_request(
                request_data={},
                new_messages=messages,
                response_string=None,
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.blocked_content is True
        assert exc_info.value.guardrail_name == "POINTGUARDAI"
        assert str(exc_info.value) == "Content blocked by PointGuardAI policy"

    @pytest.mark.asyncio
    async def test_api_call_with_modified_content(self):
        """Test API call when content is modified."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        # Mock modified response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {
                "blocked": False,
                "modified": True,
                "content": [
                    {
                        "originalContent": "My email is test@example.com",
                        "modifiedContent": "My email is [REDACTED]",
                    }
                ],
            },
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
        guardrail = _pointguard_guardrail(
            api_key="my_secret_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {"blocked": False, "modified": False, "content": []},
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
        assert "X-appsoc-api-email" not in call_kwargs["headers"]

    @pytest.mark.asyncio
    async def test_api_call_non_200_success_is_invalid(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.text = ""
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.make_pointguard_api_request(
                request_data={},
                new_messages=messages,
                response_string=None,
            )

        assert exc_info.value.status_code == 502


class TestPointGuardAIGuardrailApplyGuardrail:
    """Tests for the unified apply_guardrail method."""

    @pytest.mark.parametrize(
        ("request_data", "expected"),
        [
            (
                {"input": "Hello from the Responses API"},
                [{"role": "user", "content": "Hello from the Responses API"}],
            ),
            (
                {"input": [{"role": "user", "content": "Structured input"}]},
                [{"role": "user", "content": "Structured input"}],
            ),
        ],
    )
    def test_get_input_messages_from_responses_api_request(self, request_data, expected):
        """Responses API input should be usable without a metadata copy."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        assert guardrail._get_input_messages_from_request_data(request_data) == expected

    def test_get_input_messages_uses_only_latest_conversation_message(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        request_data = {
            "messages": [
                {"role": "user", "content": "First input"},
                {"role": "assistant", "content": "First output"},
                {"role": "user", "content": "Second input"},
            ]
        }

        assert guardrail._get_input_messages_from_request_data(request_data) == [
            {"role": "user", "content": "Second input"}
        ]

    @pytest.mark.parametrize(
        ("skip_system", "expected"),
        [
            (
                False,
                [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "Second input"},
                ],
            ),
            (True, [{"role": "user", "content": "Second input"}]),
        ],
    )
    def test_get_input_messages_includes_system_prompt_when_enabled(self, skip_system, expected):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )
        guardrail.skip_system_message_in_guardrail = skip_system
        request_data = {
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "First input"},
                {"role": "assistant", "content": "First output"},
                {"role": "user", "content": "Second input"},
            ]
        }

        assert guardrail._get_input_messages_from_request_data(request_data) == expected

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_no_violations(self):
        """Test apply_guardrail for request input with no violations."""
        guardrail = _pointguard_guardrail(
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
            "policyName": "test-policy",
            "input": {"blocked": False, "modified": False, "content": []},
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        inputs = GenericGuardrailAPIInputs(
            texts=["Hello, world!"],
            structured_messages=[{"role": "user", "content": "Hello, world!"}],
        )

        request_data = {
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "metadata": {"existing": "value"},
        }

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
            logging_obj=None,
        )

        assert result == inputs  # No modifications
        assert "_pointguardai_input_messages" not in request_data["metadata"]

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_sends_system_and_current_turn_messages(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
        )
        guardrail.skip_system_message_in_guardrail = False
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {"blocked": False, "modified": False, "content": []},
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Previous input"},
            {"role": "assistant", "content": "Previous output"},
            {"role": "user", "content": "Current input part one"},
            {"role": "user", "content": "Current input part two"},
        ]

        await guardrail.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(
                texts=[
                    "System prompt",
                    "Previous input",
                    "Previous output",
                    "Current input part one",
                    "Current input part two",
                ],
                structured_messages=messages,  # pyright: ignore[reportArgumentType]  # fixture uses the equivalent message dictionary shape
            ),
            request_data={"messages": messages},
            input_type="request",
            logging_obj=None,
        )

        sent_body = json.loads(guardrail.async_handler.post.call_args.kwargs["data"])
        assert sent_body["input"] == [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Current input part one"},
            {"role": "user", "content": "Current input part two"},
        ]

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_scan_only_tool_results(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
        )
        guardrail.scan_only_tool_results = True
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {"blocked": False, "modified": False, "content": []},
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User prompt"},
            {"role": "assistant", "content": "Calling a tool"},
            {"role": "tool", "content": "Tool result"},
        ]

        await guardrail.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(
                texts=["System prompt", "User prompt", "Calling a tool", "Tool result"],
                structured_messages=messages,  # pyright: ignore[reportArgumentType]  # fixture uses the equivalent message dictionary shape
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Tool definition must not be inspected",
                        },
                    }
                ],
            ),
            request_data={"messages": messages},
            input_type="request",
            logging_obj=None,
        )

        sent_body = json.loads(guardrail.async_handler.post.call_args.kwargs["data"])
        assert sent_body["input"] == [{"role": "tool", "content": "Tool result"}]

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_fallback_includes_system_and_latest_input(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
        )
        guardrail.skip_system_message_in_guardrail = False
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {"blocked": False, "modified": False, "content": []},
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        await guardrail.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(
                texts=["First input", "First output", "Second input"],
                structured_messages=[
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "First input"},
                    {"role": "assistant", "content": "First output"},
                    {"role": "user", "content": "Second input"},
                ],
            ),
            request_data={},
            input_type="request",
            logging_obj=None,
        )

        sent_body = json.loads(guardrail.async_handler.post.call_args.kwargs["data"])
        assert sent_body["input"] == [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Second input"},
        ]

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_blocked(self):
        """Test apply_guardrail for request that gets blocked."""
        guardrail = _pointguard_guardrail(
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
            "policyName": "test-policy",
            "input": {
                "blocked": True,
                "modified": False,
                "content": [
                    {
                        "aiViolations": [
                            {
                                "name": "prompt-injection",
                                "aiThreatCategoryId": "threat-1",
                                "type": "PROMPT_INJECTION",
                                "action": "BLOCK",
                            }
                        ]
                    }
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        inputs = GenericGuardrailAPIInputs(
            texts=["Bad content"],
            structured_messages=[{"role": "user", "content": "Bad content"}],
        )

        with pytest.raises(GuardrailRaisedException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"messages": [{"role": "user", "content": "Bad content"}]},
                input_type="request",
                logging_obj=None,
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.blocked_content is True
        assert exc_info.value.guardrail_name == "pointguardai-guard"

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_modified(self):
        """Test apply_guardrail for request with content modification."""
        guardrail = _pointguard_guardrail(
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
            "policyName": "test-policy",
            "input": {
                "blocked": False,
                "modified": True,
                "content": [
                    {
                        "originalContent": "My SSN is 123-45-6789",
                        "modifiedContent": "My SSN is [REDACTED]",
                    }
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        messages = [
            {"role": "user", "content": "Previous input"},
            {"role": "assistant", "content": "Previous output"},
            {"role": "user", "content": "My SSN is 123-45-6789"},
            {"role": "user", "content": "Current clean input"},
        ]
        inputs = GenericGuardrailAPIInputs(
            texts=[
                "Previous input",
                "Previous output",
                "My SSN is 123-45-6789",
                "Current clean input",
            ],
            structured_messages=messages,  # pyright: ignore[reportArgumentType]  # fixture uses the equivalent message dictionary shape
        )

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"messages": messages},
            input_type="request",
            logging_obj=None,
        )

        # Content should be modified
        assert result["structured_messages"][0]["content"] == "Previous input"
        assert result["structured_messages"][2]["content"] == "My SSN is [REDACTED]"
        assert result["structured_messages"][3]["content"] == "Current clean input"

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_redacts_duplicate_responses_api_texts(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )
        duplicate_text = "Contact test@example.com"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {
                "blocked": False,
                "modified": True,
                "content": [
                    {
                        "originalContent": "test@example.com",
                        "modifiedContent": "[EMAIL_REDACTED]",
                    }
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        result = await guardrail.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(texts=[duplicate_text, duplicate_text]),
            request_data={
                "input": [
                    {"role": "user", "content": duplicate_text},
                    {"role": "user", "content": duplicate_text},
                ]
            },
            input_type="request",
            logging_obj=None,
        )

        assert result["texts"] == [
            "Contact [EMAIL_REDACTED]",
            "Contact [EMAIL_REDACTED]",
        ]

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_modifies_multimodal_text_safely(self):
        """Text blocks are modified without touching images or null message content."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {
                "blocked": False,
                "modified": True,
                "content": [
                    {
                        "originalContent": "test@example.com",
                        "modifiedContent": "[EMAIL_REDACTED]",
                    }
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        image_block = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,unchanged-image"},
        }
        messages = [
            {"role": "assistant", "content": None},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Contact test@example.com"},
                    image_block,
                ],
            },
        ]
        inputs = GenericGuardrailAPIInputs(
            texts=["Contact test@example.com"],
            structured_messages=messages,  # pyright: ignore[reportArgumentType]  # fixture covers mixed multimodal message dictionaries
            images=["https://example.com/unchanged.png"],
            model="test-model",
            stream_holdback_chars=[4],
        )

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"messages": messages},
            input_type="request",
            logging_obj=None,
        )

        assert result["structured_messages"][0]["content"] is None
        assert result["structured_messages"][1]["content"] == [
            {"type": "text", "text": "Contact [EMAIL_REDACTED]"},
            image_block,
        ]
        assert result["texts"] == ["Contact [EMAIL_REDACTED]"]
        assert result["images"] == ["https://example.com/unchanged.png"]
        assert result["model"] == "test-model"
        assert result["stream_holdback_chars"] == [4]

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_inspects_and_redacts_tool_definitions(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )
        original_tool = {
            "type": "function",
            "function": {
                "name": "send_contact",
                "description": "Send test@example.com",
                "parameters": {"type": "object"},
            },
        }
        modified_tool = {
            "type": "function",
            "function": {
                "name": "send_contact",
                "description": "Send [EMAIL_REDACTED]",
                "parameters": {"type": "object"},
            },
        }
        original_content = json.dumps(original_tool, sort_keys=True, separators=(",", ":"))
        modified_content = json.dumps(modified_tool, sort_keys=True, separators=(",", ":"))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {
                "blocked": False,
                "modified": True,
                "content": [
                    {"originalContent": "Use the contact tool"},
                    {
                        "role": "tool",
                        "originalContent": original_content,
                        "modifiedContent": modified_content,
                    },
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)
        inputs = GenericGuardrailAPIInputs(
            texts=["Use the contact tool"],
            structured_messages=[{"role": "user", "content": "Use the contact tool"}],
            tools=[original_tool],  # pyright: ignore[reportArgumentType]  # fixture uses the equivalent tool dictionary shape
        )

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"messages": [{"role": "user", "content": "Use the contact tool"}]},
            input_type="request",
            logging_obj=None,
        )

        sent_body = json.loads(guardrail.async_handler.post.call_args.kwargs["data"])
        assert sent_body["input"][-1] == {"role": "tool", "content": original_content}
        assert result["tools"] == [modified_tool]

    @pytest.mark.asyncio
    async def test_apply_guardrail_response_no_violations(self):
        """Test apply_guardrail for response with no violations."""
        guardrail = _pointguard_guardrail(
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
            "policyName": "test-policy",
            "input": {"blocked": False, "modified": False, "content": []},
            "output": {"blocked": False, "modified": False, "content": []},
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        inputs = GenericGuardrailAPIInputs(
            texts=["I'm doing well, thanks!"],
        )

        input_messages = [
            {"role": "user", "content": "First input"},
            {"role": "assistant", "content": "First output"},
            {"role": "user", "content": "How are you?"},
        ]
        request_data = {
            "messages": input_messages,
            "metadata": {"existing": "value"},
        }

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="response",
            logging_obj=None,
        )

        assert result == inputs
        sent_body = json.loads(guardrail.async_handler.post.call_args.kwargs["data"])
        assert sent_body["input"] == [{"role": "user", "content": "How are you?"}]
        assert "_pointguardai_input_messages" not in request_data["metadata"]

    @pytest.mark.asyncio
    async def test_streaming_output_modifications_are_emitted(self, monkeypatch):
        monkeypatch.setattr(
            unified_module,
            "endpoint_guardrail_translation_mappings",
            {CallTypes.acompletion: OpenAIChatCompletionsHandler},
        )
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
            event_hook=GuardrailEventHooks.post_call,
        )
        original = "Contact me at test@example.com"
        redacted = "Contact me at [EMAIL_REDACTED]"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {"blocked": False, "modified": False, "content": []},
            "output": {
                "blocked": False,
                "modified": True,
                "content": [{"originalContent": original, "modifiedContent": redacted}],
            },
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        async def response_stream():
            for content, finish_reason in (
                ("Contact me at test", None),
                ("@example.com", None),
                ("", "stop"),
            ):
                yield ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(content=content, role="assistant"),
                            finish_reason=finish_reason,
                        )
                    ]
                )

        emitted = []
        async for item in UnifiedLLMGuardrails().async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key", request_route="/v1/chat/completions"),
            response=response_stream(),
            request_data={
                "guardrail_to_apply": guardrail,
                "guardrails": ["pointguardai-guard"],
                "model": "test-model",
                "messages": [{"role": "user", "content": "Provide a contact address"}],
            },
        ):
            emitted.append(item)

        streamed = "".join(item.choices[0].delta.content or "" for item in emitted if item.choices)
        assert streamed == redacted
        guardrail.async_handler.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_apply_guardrail_response_ignores_input_modifications(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {
                "blocked": False,
                "modified": True,
                "content": [
                    {
                        "originalContent": "same text",
                        "modifiedContent": "modified input",
                    }
                ],
            },
            "output": {"blocked": False, "modified": False, "content": []},
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)
        inputs = GenericGuardrailAPIInputs(texts=["same text"])

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"messages": [{"role": "user", "content": "same text"}]},
            input_type="response",
            logging_obj=None,
        )

        assert result["texts"] == ["same text"]

    @pytest.mark.asyncio
    async def test_apply_guardrail_response_rejects_unmatched_output_modification(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            unreachable_fallback="fail_closed",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {"blocked": False, "modified": False, "content": []},
            "output": {
                "blocked": False,
                "modified": True,
                "content": [
                    {
                        "originalContent": "content not present in the response",
                        "modifiedContent": "replacement",
                    }
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=GenericGuardrailAPIInputs(texts=["actual model response"]),
                request_data={"messages": [{"role": "user", "content": "question"}]},
                input_type="response",
                logging_obj=None,
            )

        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_output_inspection_honors_message_role_filters(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )
        guardrail.skip_system_message_in_guardrail = True
        guardrail.skip_tool_message_in_guardrail = True
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {"blocked": False, "modified": False, "content": []},
            "output": {"blocked": False, "modified": False, "content": []},
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        await guardrail.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(texts=["answer"]),
            request_data={
                "messages": [
                    {"role": "system", "content": "system instructions"},
                    {"role": "user", "content": "question"},
                    {"role": "tool", "content": "tool result"},
                ]
            },
            input_type="response",
            logging_obj=None,
        )

        sent_body = json.loads(guardrail.async_handler.post.call_args.kwargs["data"])
        assert sent_body["input"] == [{"role": "user", "content": "question"}]

    @pytest.mark.asyncio
    async def test_apply_guardrail_response_blocked(self):
        """Test apply_guardrail for response that gets blocked."""
        guardrail = _pointguard_guardrail(
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
            "policyName": "test-policy",
            "input": {"blocked": False, "modified": False, "content": []},
            "output": {
                "blocked": True,
                "modified": False,
                "content": [
                    {
                        "aiViolations": [
                            {
                                "name": "policy-violation",
                                "aiThreatCategoryId": "threat-2",
                                "type": "SENSITIVE_OUTPUT",
                                "action": "BLOCK",
                            }
                        ]
                    }
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        inputs = GenericGuardrailAPIInputs(
            texts=["Sensitive response"],
        )

        with pytest.raises(GuardrailRaisedException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="response",
                logging_obj=None,
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.blocked_content is True

    @pytest.mark.asyncio
    async def test_apply_guardrail_response_modified(self):
        """Test apply_guardrail for response with content modification."""
        guardrail = _pointguard_guardrail(
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
            "policyName": "test-policy",
            "input": {"blocked": False, "modified": False, "content": []},
            "output": {
                "blocked": False,
                "modified": True,
                "content": [
                    {
                        "originalContent": "Contact me at test@example.com",
                        "modifiedContent": "Contact me at [EMAIL_REDACTED]",
                    }
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        inputs = GenericGuardrailAPIInputs(
            texts=["Contact me at test@example.com"],
            images=["https://example.com/unchanged.png"],
            model="test-model",
            stream_holdback_chars=[7],
        )

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="response",
            logging_obj=None,
        )

        # Content should be modified
        assert result["texts"][0] == "Contact me at [EMAIL_REDACTED]"
        assert result["images"] == ["https://example.com/unchanged.png"]
        assert result["model"] == "test-model"
        assert result["stream_holdback_chars"] == [7]
        sent_body = json.loads(guardrail.async_handler.post.call_args.kwargs["data"])
        assert sent_body["input"] == []
        assert sent_body["output"] == [
            {
                "role": "assistant",
                "content": "Contact me at test@example.com",
            }
        ]

    @pytest.mark.asyncio
    async def test_apply_guardrail_inspects_all_output_texts_and_preserves_positions(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {"blocked": False, "modified": False, "content": []},
            "output": {
                "blocked": False,
                "modified": True,
                "content": [
                    {
                        "originalContent": "First: first@example.com",
                        "modifiedContent": "First: [FIRST_EMAIL]",
                    },
                    {"originalContent": "Second response is clean"},
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)
        inputs = GenericGuardrailAPIInputs(
            texts=[
                "First: first@example.com",
                "Second response is clean",
            ],
            model="test-model",
        )

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"messages": [{"role": "user", "content": "List contacts"}]},
            input_type="response",
            logging_obj=None,
        )

        sent_body = json.loads(guardrail.async_handler.post.call_args.kwargs["data"])
        assert sent_body["output"] == [
            {"role": "assistant", "content": "First: first@example.com"},
            {"role": "assistant", "content": "Second response is clean"},
        ]
        assert result["texts"] == [
            "First: [FIRST_EMAIL]",
            "Second response is clean",
        ]
        assert result["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_apply_guardrail_inspects_and_redacts_tool_call_only_response(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )
        original_tool_call = {
            "id": "call-1",
            "type": "function",
            "index": 0,
            "function": {
                "name": "send_contact",
                "arguments": '{"email":"test@example.com"}',
            },
        }
        modified_tool_call = {
            "id": "call-1",
            "type": "function",
            "index": 0,
            "function": {
                "name": "send_contact",
                "arguments": '{"email":"[EMAIL_REDACTED]"}',
            },
        }
        original_content = json.dumps(original_tool_call, sort_keys=True, separators=(",", ":"))
        modified_content = json.dumps(modified_tool_call, sort_keys=True, separators=(",", ":"))
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {"blocked": False, "modified": False, "content": []},
            "output": {
                "blocked": False,
                "modified": True,
                "content": [
                    {
                        "role": "assistant",
                        "originalContent": original_content,
                        "modifiedContent": modified_content,
                    }
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        result = await guardrail.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(
                texts=[],
                tool_calls=[original_tool_call],  # pyright: ignore[reportArgumentType]  # fixture uses the equivalent tool-call dictionary shape
            ),
            request_data={"messages": [{"role": "user", "content": "Send the contact"}]},
            input_type="response",
            logging_obj=None,
        )

        sent_body = json.loads(guardrail.async_handler.post.call_args.kwargs["data"])
        assert sent_body["output"] == [{"role": "assistant", "content": original_content}]
        assert result["tool_calls"] == [modified_tool_call]

    @pytest.mark.asyncio
    async def test_apply_guardrail_extracts_messages_from_request_data(self):
        """Test that apply_guardrail extracts messages from request_data when not in inputs."""
        guardrail = _pointguard_guardrail(
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
            "policyName": "test-policy",
            "input": {"blocked": False, "modified": False, "content": []},
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
        guardrail = _pointguard_guardrail(
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

        error = httpx.HTTPStatusError("401 error", request=MagicMock(), response=mock_response)

        with pytest.raises(HTTPException) as exc_info:
            guardrail._handle_http_status_error(error)

        assert exc_info.value.status_code == 401
        assert "authentication failed" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_handle_network_timeout(self):
        """Test handling of timeout error."""
        guardrail = _pointguard_guardrail(
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
        guardrail = _pointguard_guardrail(
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

    @pytest.mark.asyncio
    async def test_explicit_fail_open_returns_all_inputs(self):
        """Explicit fail-open should preserve every field when PointGuard is unreachable."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
            unreachable_fallback="fail_open",
        )
        guardrail.async_handler.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        inputs = GenericGuardrailAPIInputs(
            texts=["Hello"],
            structured_messages=[{"role": "user", "content": "Hello"}],
            images=["https://example.com/image.png"],
        )

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"messages": inputs["structured_messages"]},
            input_type="request",
            logging_obj=MagicMock(
                litellm_call_id="call-123",
                litellm_trace_id="trace-456",
            ),
        )

        assert result == inputs
        assert result is not inputs

    @pytest.mark.asyncio
    async def test_explicit_fail_closed_blocks_on_connection_errors(self):
        """Explicit fail-closed should block when PointGuard is unavailable."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
            unreachable_fallback="fail_closed",
        )
        guardrail.async_handler.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=GenericGuardrailAPIInputs(
                    texts=["Hello"],
                    structured_messages=[{"role": "user", "content": "Hello"}],
                ),
                request_data={"messages": [{"role": "user", "content": "Hello"}]},
                input_type="request",
                logging_obj=None,
            )

        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_explicit_fail_open_returns_inputs_on_provider_500(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
            unreachable_fallback="fail_open",
        )
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "PointGuard internal error",
            request=MagicMock(),
            response=mock_response,
        )
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)
        inputs = GenericGuardrailAPIInputs(
            texts=["Hello"],
            structured_messages=[{"role": "user", "content": "Hello"}],
        )

        result = await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"messages": inputs["structured_messages"]},
            input_type="request",
            logging_obj=None,
        )

        assert result == inputs

    @pytest.mark.asyncio
    async def test_explicit_fail_closed_blocks_on_provider_500(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
            unreachable_fallback="fail_closed",
        )
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "PointGuard internal error",
            request=MagicMock(),
            response=mock_response,
        )
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=GenericGuardrailAPIInputs(
                    texts=["Hello"],
                    structured_messages=[{"role": "user", "content": "Hello"}],
                ),
                request_data={"messages": [{"role": "user", "content": "Hello"}]},
                input_type="request",
                logging_obj=None,
            )

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_malformed_success_response_fails_closed_despite_fail_open(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)
        inputs = GenericGuardrailAPIInputs(
            texts=["Hello"],
            structured_messages=[{"role": "user", "content": "Hello"}],
        )

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"messages": inputs["structured_messages"]},
                input_type="request",
                logging_obj=None,
            )

        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_malformed_success_response_follows_fail_closed(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            unreachable_fallback="fail_closed",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=GenericGuardrailAPIInputs(
                    texts=["Hello"],
                    structured_messages=[{"role": "user", "content": "Hello"}],
                ),
                request_data={"messages": [{"role": "user", "content": "Hello"}]},
                input_type="request",
                logging_obj=None,
            )

        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_output_success_response_requires_input_result(self):
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            unreachable_fallback="fail_closed",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "output": {"blocked": False, "modified": False, "content": []},
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=GenericGuardrailAPIInputs(texts=["model response"]),
                request_data={"messages": [{"role": "user", "content": "question"}]},
                input_type="response",
                logging_obj=None,
            )

        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_fail_open_does_not_bypass_policy_block(self):
        """Fail-open must apply only to unavailability, never policy decisions."""
        guardrail = _pointguard_guardrail(
            api_key="test_key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            guardrail_name="pointguardai-guard",
            unreachable_fallback="fail_open",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policyName": "test-policy",
            "input": {
                "blocked": True,
                "modified": False,
                "content": [
                    {
                        "aiViolations": [
                            {
                                "name": "prompt-injection",
                                "aiThreatCategoryId": "threat-1",
                                "type": "PROMPT_INJECTION",
                                "action": "BLOCK",
                            }
                        ]
                    }
                ],
            },
        }
        mock_response.raise_for_status = MagicMock()
        guardrail.async_handler.post = AsyncMock(return_value=mock_response)

        with pytest.raises(GuardrailRaisedException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=GenericGuardrailAPIInputs(
                    texts=["Bad content"],
                    structured_messages=[{"role": "user", "content": "Bad content"}],
                ),
                request_data={"messages": [{"role": "user", "content": "Bad content"}]},
                input_type="request",
                logging_obj=None,
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.blocked_content is True


class TestPointGuardAIGuardrailShouldRun:
    """Tests for should_run_guardrail method."""

    def test_should_run_guardrail_with_guardrail_in_metadata(self):
        """Test that guardrail runs when specified in metadata."""
        guardrail = _pointguard_guardrail(
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

        result = guardrail.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call)

        assert result is True

    def test_should_not_run_guardrail_when_not_in_metadata(self):
        """Test that guardrail doesn't run when not specified in metadata."""
        guardrail = _pointguard_guardrail(
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

        result = guardrail.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call)

        assert result is False

    def test_should_run_guardrail_with_default_on(self):
        """Test that guardrail runs when default_on is True."""
        guardrail = _pointguard_guardrail(
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

        result = guardrail.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call)

        assert result is True

    def test_should_run_guardrail_with_wrong_event_hook(self):
        """Test that guardrail doesn't run with mismatched event hook."""
        guardrail = _pointguard_guardrail(
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

        result = guardrail.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call)

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
        """Test that config model has expected fields for API."""
        from litellm.types.proxy.guardrails.guardrail_hooks.pointguardai import (
            PointGuardAIGuardrailConfigModel,
        )

        model = PointGuardAIGuardrailConfigModel()

        # Check default values are None
        assert model.api_key is None
        assert model.org_code is None
        assert model.policy_config_name is None
        assert model.unreachable_fallback == "fail_closed"

    def test_config_model_with_values(self):
        """Test config model with provided values"""
        from litellm.types.proxy.guardrails.guardrail_hooks.pointguardai import (
            PointGuardAIGuardrailConfigModel,
        )

        model = PointGuardAIGuardrailConfigModel(
            api_key="test_key",
            org_code="test-org",
            policy_config_name="test-policy",
            unreachable_fallback="fail_open",
        )

        assert model.api_key == "test_key"
        assert model.org_code == "test-org"
        assert model.policy_config_name == "test-policy"
        assert model.unreachable_fallback == "fail_open"


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

    def test_initializer_uses_pointguard_fail_closed_default(self):
        """PointGuard's YAML configuration should default to fail-closed."""
        from litellm.proxy.guardrails.guardrail_hooks.pointguardai import (
            initialize_guardrail,
        )
        from litellm.types.guardrails import LitellmParams

        params = LitellmParams(
            guardrail="pointguard_ai",
            mode="pre_call",
            api_key="test-key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

        callback_manager = _RecordingCallbackManager()
        initialized = initialize_guardrail(
            params,
            {"guardrail_name": "pointguardai-guard", "litellm_params": params},
            callback_manager=callback_manager,
        )

        assert initialized.unreachable_fallback == "fail_closed"
        assert callback_manager.callback is initialized

    def test_initializer_preserves_explicit_fail_open(self):
        """An explicit PointGuard fail-open setting should override its default."""
        from litellm.proxy.guardrails.guardrail_hooks.pointguardai import (
            initialize_guardrail,
        )
        from litellm.types.guardrails import LitellmParams

        params = LitellmParams(
            guardrail="pointguard_ai",
            mode="pre_call",
            api_key="test-key",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
            unreachable_fallback="fail_open",
        )

        callback_manager = _RecordingCallbackManager()
        initialized = initialize_guardrail(
            params,
            {"guardrail_name": "pointguardai-guard", "litellm_params": params},
            callback_manager=callback_manager,
        )

        assert initialized.unreachable_fallback == "fail_open"
        assert callback_manager.callback is initialized

    def test_initializer_resolves_pointguard_environment_references(self, monkeypatch):
        from litellm.proxy.guardrails.guardrail_hooks.pointguardai import (
            initialize_guardrail,
        )
        from litellm.types.guardrails import LitellmParams

        monkeypatch.setenv("POINTGUARDAI_TEST_ORG", "resolved-org")
        monkeypatch.setenv("POINTGUARDAI_TEST_POLICY", "resolved-policy")
        params = LitellmParams(
            guardrail="pointguard_ai",
            mode="pre_call",
            api_key="test-key",
            api_base="https://api.appsoc.com",
            org_code="os.environ/POINTGUARDAI_TEST_ORG",
            policy_config_name="os.environ/POINTGUARDAI_TEST_POLICY",
        )

        callback_manager = _RecordingCallbackManager()
        initialized = initialize_guardrail(
            params,
            {"guardrail_name": "pointguardai-guard", "litellm_params": params},
            callback_manager=callback_manager,
        )

        assert initialized.pointguardai_org_code == "resolved-org"
        assert initialized.pointguardai_policy_config_name == "resolved-policy"
        assert callback_manager.callback is initialized

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

    def test_supported_event_hooks(self):
        """Test that the integration advertises every documented execution mode."""
        assert PointGuardAIGuardrail.get_supported_event_hooks() == [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]
