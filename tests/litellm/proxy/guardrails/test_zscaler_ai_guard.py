"""
Unit tests for Zscaler AI Guard guardrail

Tests covering:
- super().__init__(**kwargs) passes event_hook correctly
- _resolve_metadata_value works for both pre-call and post-call metadata
- _prepare_headers correctly maps all kwargs
- resolve-and-execute-policy endpoint omits policyId
- Config parameters are passed from __init__.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from litellm.proxy.guardrails.guardrail_hooks.zscaler_ai_guard import ZscalerAIGuard


class TestZscalerAIGuardInit:
    """Tests for ZscalerAIGuard initialization"""

    def test_should_pass_event_hook_to_parent_class(self):
        """Test that super().__init__(**kwargs) passes event_hook to CustomGuardrail"""
        guardrail = ZscalerAIGuard(
            api_key="test_key",
            policy_id=100,
            event_hook="pre_call",
            guardrail_name="test-guardrail",
        )
        assert guardrail.event_hook == "pre_call"
        assert guardrail.guardrail_name == "test-guardrail"

    def test_should_pass_default_on_to_parent_class(self):
        """Test that default_on is passed to CustomGuardrail"""
        guardrail = ZscalerAIGuard(
            api_key="test_key",
            policy_id=100,
            default_on=True,
        )
        assert guardrail.default_on is True

    def test_should_use_config_send_user_api_key_alias_when_true(self):
        """Test that send_user_api_key_alias=True from config is used"""
        guardrail = ZscalerAIGuard(
            api_key="test_key",
            send_user_api_key_alias=True,
        )
        assert guardrail.send_user_api_key_alias is True

    def test_should_use_config_send_user_api_key_alias_when_false(self):
        """Test that send_user_api_key_alias=False from config is respected (not overridden by env)"""
        guardrail = ZscalerAIGuard(
            api_key="test_key",
            send_user_api_key_alias=False,
        )
        assert guardrail.send_user_api_key_alias is False

    def test_should_use_config_send_user_api_key_user_id(self):
        """Test that send_user_api_key_user_id from config is used"""
        guardrail = ZscalerAIGuard(
            api_key="test_key",
            send_user_api_key_user_id=True,
        )
        assert guardrail.send_user_api_key_user_id is True

    def test_should_use_config_send_user_api_key_team_id(self):
        """Test that send_user_api_key_team_id from config is used"""
        guardrail = ZscalerAIGuard(
            api_key="test_key",
            send_user_api_key_team_id=True,
        )
        assert guardrail.send_user_api_key_team_id is True


class TestResolveMetadataValue:
    """Tests for _resolve_metadata_value static method"""

    def test_should_resolve_from_metadata_during_pre_call(self):
        """Test that user_api_key_alias is resolved from metadata during pre-call"""
        request_data = {
            "metadata": {
                "user_api_key_alias": "test-alias-pre-call"
            }
        }
        result = ZscalerAIGuard._resolve_metadata_value(request_data, "user_api_key_alias")
        assert result == "test-alias-pre-call"

    def test_should_resolve_from_litellm_metadata_during_post_call(self):
        """Test that user_api_key_alias is resolved from litellm_metadata during post-call"""
        request_data = {
            "litellm_metadata": {
                "user_api_key_alias": "test-alias-post-call"
            }
        }
        result = ZscalerAIGuard._resolve_metadata_value(request_data, "user_api_key_alias")
        assert result == "test-alias-post-call"

    def test_should_resolve_user_api_key_key_alias_mapping(self):
        """Test key_alias -> user_api_key_key_alias mapping in litellm_metadata"""
        # transform_user_api_key_dict_to_metadata prefixes "key_alias" -> "user_api_key_key_alias"
        request_data = {
            "litellm_metadata": {
                "user_api_key_key_alias": "test-key-alias"
            }
        }
        result = ZscalerAIGuard._resolve_metadata_value(request_data, "user_api_key_alias")
        assert result == "test-key-alias"

    def test_should_prioritize_litellm_metadata_over_metadata(self):
        """Test that litellm_metadata takes precedence over metadata"""
        request_data = {
            "litellm_metadata": {
                "user_api_key_alias": "from-litellm-metadata"
            },
            "metadata": {
                "user_api_key_alias": "from-metadata"
            }
        }
        result = ZscalerAIGuard._resolve_metadata_value(request_data, "user_api_key_alias")
        assert result == "from-litellm-metadata"

    def test_should_return_none_when_key_not_found(self):
        """Test that None is returned when key is not found in either location"""
        request_data = {
            "metadata": {},
            "litellm_metadata": {}
        }
        result = ZscalerAIGuard._resolve_metadata_value(request_data, "user_api_key_alias")
        assert result is None

    def test_should_return_none_when_request_data_is_none(self):
        """Test that None is returned when request_data is None"""
        result = ZscalerAIGuard._resolve_metadata_value(None, "user_api_key_alias")
        assert result is None

    def test_should_strip_whitespace(self):
        """Test that whitespace is stripped from values"""
        request_data = {
            "metadata": {
                "user_api_key_alias": "  test-alias  "
            }
        }
        result = ZscalerAIGuard._resolve_metadata_value(request_data, "user_api_key_alias")
        assert result == "test-alias"


class TestPrepareHeaders:
    """Tests for _prepare_headers method"""

    def test_should_include_user_api_key_alias_header(self):
        """Test that user-api-key-alias header is included when send_user_api_key_alias is True"""
        guardrail = ZscalerAIGuard(
            api_key="test_key",
            send_user_api_key_alias=True,
        )
        headers = guardrail._prepare_headers("test_key", user_api_key_alias="test-alias")
        assert headers.get("user-api-key-alias") == "test-alias"

    def test_should_include_user_api_key_team_id_header(self):
        """Test that user-api-key-team-id header is included when send_user_api_key_team_id is True"""
        guardrail = ZscalerAIGuard(
            api_key="test_key",
            send_user_api_key_team_id=True,
        )
        headers = guardrail._prepare_headers("test_key", user_api_key_team_id="test-team-id")
        assert headers.get("user-api-key-team-id") == "test-team-id"

    def test_should_include_user_api_key_user_id_header(self):
        """Test that user-api-key-user-id header is included when send_user_api_key_user_id is True"""
        guardrail = ZscalerAIGuard(
            api_key="test_key",
            send_user_api_key_user_id=True,
        )
        headers = guardrail._prepare_headers("test_key", user_api_key_user_id="test-user-id")
        assert headers.get("user-api-key-user-id") == "test-user-id"

    def test_should_use_underscore_key_for_user_id_kwarg(self):
        """Test that _prepare_headers uses underscore key 'user_api_key_user_id' (not hyphenated)"""
        guardrail = ZscalerAIGuard(
            api_key="test_key",
            send_user_api_key_user_id=True,
        )
        # This test verifies the fix for the typo: kwargs.get("user-api-key-user-id") -> kwargs.get("user_api_key_user_id")
        headers = guardrail._prepare_headers("test_key", user_api_key_user_id="correct-user-id")
        assert headers.get("user-api-key-user-id") == "correct-user-id"


class TestMakeZscalerAIGuardAPICall:
    """Tests for make_zscaler_ai_guard_api_call method"""

    @pytest.mark.asyncio
    async def test_should_include_policy_id_when_greater_than_zero(self):
        """Test that policyId is included in request body when policy_id > 0"""
        from unittest.mock import AsyncMock, Mock, patch

        guardrail = ZscalerAIGuard(
            api_key="test_key",
            policy_id=100,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "statusCode": 200,
            "action": "ALLOW",
        }

        with patch.object(guardrail, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = mock_response

            await guardrail.make_zscaler_ai_guard_api_call(
                zscaler_ai_guard_url="http://example.com",
                api_key="test_key",
                policy_id=100,
                direction="IN",
                content="test content",
            )

            call_args = mock_send.call_args
            data = call_args[0][2]  # Third positional arg is data
            assert "policyId" in data
            assert data["policyId"] == 100

    @pytest.mark.asyncio
    async def test_should_omit_policy_id_when_zero_or_negative(self):
        """Test that policyId is omitted from request body when policy_id <= 0 (for resolve-and-execute-policy)"""
        from unittest.mock import AsyncMock, Mock, patch

        guardrail = ZscalerAIGuard(
            api_key="test_key",
            policy_id=-1,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "statusCode": 200,
            "action": "ALLOW",
        }

        with patch.object(guardrail, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = mock_response

            await guardrail.make_zscaler_ai_guard_api_call(
                zscaler_ai_guard_url="http://example.com",
                api_key="test_key",
                policy_id=-1,
                direction="OUT",
                content="test content",
            )

            call_args = mock_send.call_args
            data = call_args[0][2]  # Third positional arg is data
            assert "policyId" not in data

    @pytest.mark.asyncio
    async def test_should_omit_policy_id_when_none(self):
        """Test that policyId is omitted from request body when policy_id is None"""
        from unittest.mock import AsyncMock, Mock, patch

        guardrail = ZscalerAIGuard(
            api_key="test_key",
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "statusCode": 200,
            "action": "ALLOW",
        }

        with patch.object(guardrail, "_send_request", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = mock_response

            await guardrail.make_zscaler_ai_guard_api_call(
                zscaler_ai_guard_url="http://example.com",
                api_key="test_key",
                policy_id=None,
                direction="IN",
                content="test content",
            )

            call_args = mock_send.call_args
            data = call_args[0][2]
            assert "policyId" not in data
