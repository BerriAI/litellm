"""
Test guardrail status consistency across NOMA and Bedrock providers.

This test validates that both providers return consistent status values:
- "success": Guardrail completed successfully with no violations
- "blocked": Guardrail detected violations and intervened  
- "failure": Technical error or API failure

Tests cover:
1. Successful guardrail checks (no violations)
2. Blocked requests (content violations)
3. Technical failures (API errors)
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import json
from datetime import datetime

from litellm.proxy.guardrails.guardrail_hooks.noma.noma import NomaGuardrail
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import StandardLoggingGuardrailInformation
import httpx


class TestGuardrailStatusConsistency:
    """Test guardrail status consistency across providers."""

    @pytest.fixture
    def mock_user_auth(self):
        """Mock user authentication."""
        return UserAPIKeyAuth(
            user_id="test_user",
            team_id="test_team",
            api_key="test_key"
        )

    @pytest.fixture
    def mock_request_data(self):
        """Mock request data."""
        return {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "metadata": {}
        }

    def test_noma_success_status(self, mock_user_auth, mock_request_data):
        """Test NOMA returns 'success' status for allowed content."""
        # Test the status determination logic directly
        noma = NomaGuardrail(api_key="test_key")
        
        # Mock successful response (verdict: true)
        response_json = {
            "verdict": True,
            "originalResponse": {
                "prompt": {
                    "harmfulContent": {"result": False, "confidence": 0.1}
                }
            }
        }
        
        status = noma._determine_guardrail_status(response_json)
        assert status == "success"

    def test_noma_blocked_status(self, mock_user_auth, mock_request_data):
        """Test NOMA returns 'blocked' status for flagged content."""
        noma = NomaGuardrail(api_key="test_key")
        
        # Mock blocked response (verdict: false)
        response_json = {
            "verdict": False,
            "originalResponse": {
                "prompt": {
                    "harmfulContent": {"result": True, "confidence": 0.95}
                }
            }
        }
        
        status = noma._determine_guardrail_status(response_json)
        assert status == "blocked"

    def test_noma_failure_status(self, mock_user_auth, mock_request_data):
        """Test NOMA returns 'failure' status for invalid responses."""
        noma = NomaGuardrail(api_key="test_key")
        
        # Test various failure scenarios
        test_cases = [
            {},  # Empty response
            {"verdict": None},  # Null verdict
            "invalid",  # Non-dict response
            None,  # None response
        ]
        
        for test_case in test_cases:
            status = noma._determine_guardrail_status(test_case)
            assert status == "failure"

    def test_bedrock_success_status(self):
        """Test Bedrock returns 'success' status for allowed content."""
        bedrock = BedrockGuardrail(
            guardrailIdentifier="test-id",
            guardrailVersion="1.0"
        )
        
        # Mock successful HTTP response with no intervention
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "action": "NONE",
            "assessments": []
        }
        
        status = bedrock._get_bedrock_guardrail_response_status(mock_response)
        assert status == "success"

    def test_bedrock_blocked_status(self):
        """Test Bedrock returns 'blocked' status when guardrail intervenes."""
        bedrock = BedrockGuardrail(
            guardrailIdentifier="test-id", 
            guardrailVersion="1.0"
        )
        
        # Mock HTTP response with guardrail intervention
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "action": "GUARDRAIL_INTERVENED",
            "assessments": [
                {
                    "topicPolicy": {
                        "topics": [
                            {"action": "BLOCKED", "type": "VIOLENCE"}
                        ]
                    }
                }
            ]
        }
        
        status = bedrock._get_bedrock_guardrail_response_status(mock_response)
        assert status == "blocked"

    def test_bedrock_failure_status(self):
        """Test Bedrock returns 'failure' status for API errors."""
        bedrock = BedrockGuardrail(
            guardrailIdentifier="test-id",
            guardrailVersion="1.0"
        )
        
        # Test various failure scenarios
        failure_cases = [
            # HTTP error status codes
            Mock(status_code=400),
            Mock(status_code=500),
            Mock(status_code=403),
            # Exception during JSON parsing
            Mock(status_code=200, json=Mock(side_effect=Exception("JSON error"))),
            # Exception response in payload
            Mock(status_code=200, json=Mock(return_value={
                "Output": {"__type": "SomeException"}
            }))
        ]
        
        for mock_response in failure_cases:
            status = bedrock._get_bedrock_guardrail_response_status(mock_response)
            assert status == "failure"

    @pytest.mark.asyncio
    async def test_noma_logs_guardrail_information(self, mock_user_auth, mock_request_data):
        """Test NOMA consistently logs guardrail information."""
        noma = NomaGuardrail(api_key="test_key", monitor_mode=False)
        
        # Mock the API call to return a successful response
        with patch.object(noma, '_call_noma_api', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {
                "verdict": True,
                "originalResponse": {"prompt": {}}
            }
            
            # Mock the verdict check to not raise exceptions
            with patch.object(noma, '_check_verdict', new_callable=AsyncMock):
                with patch.object(noma, '_extract_user_message', new_callable=AsyncMock) as mock_extract:
                    mock_extract.return_value = "Hello, how are you?"
                    
                    # Process the message check
                    result = await noma._process_user_message_check(mock_request_data, mock_user_auth)
                    
                    # Verify guardrail information was logged
                    metadata = mock_request_data.get("metadata", {})
                    guardrail_info = metadata.get("standard_logging_guardrail_information")
                    
                    assert guardrail_info is not None
                    assert guardrail_info["guardrail_provider"] == "noma"
                    assert guardrail_info["guardrail_status"] == "success"
                    assert "start_time" in guardrail_info
                    assert "end_time" in guardrail_info
                    assert "duration" in guardrail_info

    def test_status_values_consistency(self):
        """Test that both providers use the same status value definitions."""
        from typing import get_args
        from litellm.types.utils import StandardLoggingGuardrailInformation
        
        # Get the allowed status values from the type definition
        status_field = StandardLoggingGuardrailInformation.__annotations__["guardrail_status"]
        allowed_statuses = get_args(status_field)
        
        # Verify we have exactly three status values
        assert len(allowed_statuses) == 3
        assert "success" in allowed_statuses
        assert "blocked" in allowed_statuses  
        assert "failure" in allowed_statuses

    @pytest.mark.asyncio
    async def test_noma_exception_handling_logs_failure(self, mock_user_auth, mock_request_data):
        """Test NOMA logs 'failure' status when exceptions occur."""
        noma = NomaGuardrail(api_key="test_key", block_failures=False)
        
        # Mock an exception in the pre-call hook
        with patch.object(noma, '_check_user_message', side_effect=Exception("API error")):
            # This should not raise due to block_failures=False
            result = await noma.async_pre_call_hook(mock_user_auth, None, mock_request_data, "completion")
            
            # Verify the failure was logged
            metadata = mock_request_data.get("metadata", {})
            guardrail_info = metadata.get("standard_logging_guardrail_information")
            
            assert guardrail_info is not None
            assert guardrail_info["guardrail_provider"] == "noma"
            assert guardrail_info["guardrail_status"] == "failure"

    def test_guardrail_status_troubleshooting_fields(self):
        """Test that all necessary fields are present for troubleshooting."""
        # Test data that simulates what should be logged
        guardrail_info = StandardLoggingGuardrailInformation(
            guardrail_name="test-guard",
            guardrail_provider="noma",
            guardrail_status="blocked",
            start_time=datetime.now().timestamp(),
            end_time=datetime.now().timestamp(),
            duration=0.5,
            guardrail_response={"verdict": False}
        )
        
        # Verify all troubleshooting fields are present
        required_fields = [
            "guardrail_name",      # Which guardrail was triggered
            "guardrail_provider",  # Which provider (noma/bedrock)
            "guardrail_status",    # What happened (success/blocked/failure)
            "start_time",          # When it started
            "end_time",            # When it ended
            "duration",            # How long it took
            "guardrail_response"   # Full response for debugging
        ]
        
        for field in required_fields:
            assert field in guardrail_info
            assert guardrail_info[field] is not None

    def test_status_fields_creation(self):
        """Test that status fields are created correctly based on guardrail status."""
        from litellm.litellm_core_utils.litellm_logging import _get_status_fields
        from litellm.types.utils import StandardLoggingGuardrailInformation
        
        # Test successful request with no guardrail issues
        status_fields = _get_status_fields(
            status="success",
            guardrail_information=None,
            error_str=None
        )
        assert status_fields["is_guardrail_failed"] == False
        assert status_fields["is_guardrail_intervened"] == False
        assert status_fields["is_llm_request_successful"] == True
        
        # Test guardrail intervention
        guardrail_info = {
            "guardrail_status": "blocked",
            "guardrail_provider": "noma"
        }
        status_fields = _get_status_fields(
            status="success",
            guardrail_information=guardrail_info,
            error_str=None
        )
        assert status_fields["is_guardrail_failed"] == False
        assert status_fields["is_guardrail_intervened"] == True
        assert status_fields["is_llm_request_successful"] == True
        
        # Test guardrail failure
        guardrail_info = {
            "guardrail_status": "failure",
            "guardrail_provider": "bedrock"
        }
        status_fields = _get_status_fields(
            status="failure",
            guardrail_information=guardrail_info,
            error_str="Guardrail API failed"
        )
        assert status_fields["is_guardrail_failed"] == True
        assert status_fields["is_guardrail_intervened"] == False
        assert status_fields["is_llm_request_successful"] == False

    def test_status_fields_with_error_string(self):
        """Test status field determination based on error strings."""
        from litellm.litellm_core_utils.litellm_logging import _get_status_fields
        
        # Test guardrail blocked error
        status_fields = _get_status_fields(
            status="failure",
            guardrail_information=None,
            error_str="Request blocked by guardrail policy"
        )
        assert status_fields["is_guardrail_intervened"] == True
        assert status_fields["is_guardrail_failed"] == False
        
        # Test guardrail failure error
        status_fields = _get_status_fields(
            status="failure", 
            guardrail_information=None,
            error_str="Guardrail service error: timeout"
        )
        assert status_fields["is_guardrail_failed"] == True
        assert status_fields["is_guardrail_intervened"] == False


if __name__ == "__main__":
    pytest.main([__file__])