import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.exceptions import ContentPolicyViolationError
from litellm.litellm_core_utils.exception_mapping_utils import exception_type


class TestAzureExceptionMapping:
    """Test Azure OpenAI exception mapping with provider-specific fields"""

    def test_azure_content_policy_violation_innererror_access(self):
        """Test that Azure content policy violation exceptions provide access to innererror details"""
        
        # Create a mock Azure OpenAI exception with body containing innererror
        mock_exception = Exception("The response was filtered due to the prompt triggering Azure OpenAI's content management policy")
        mock_exception.body = {
            "innererror": {
                "code": "ResponsibleAIPolicyViolation",
                "content_filter_result": {
                    "hate": {
                        "filtered": True,
                        "severity": "high"
                    },
                    "jailbreak": {
                        "filtered": False,
                        "detected": False
                    },
                    "self_harm": {
                        "filtered": False,
                        "severity": "safe"
                    },
                    "sexual": {
                        "filtered": False,
                        "severity": "safe"
                    },
                    "violence": {
                        "filtered": True,
                        "severity": "medium"
                    }
                }
            }
        }
        
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response
        
        # Test the exception mapping directly
        with pytest.raises(ContentPolicyViolationError) as exc_info:
            exception_type(
                model="azure/gpt-4",
                original_exception=mock_exception,
                custom_llm_provider="azure"
            )
        
        # Access the exception and verify provider_specific_fields
        e = exc_info.value
        assert e.provider_specific_fields is not None
        assert "innererror" in e.provider_specific_fields
        
        innererror = e.provider_specific_fields["innererror"]
        assert innererror["code"] == "ResponsibleAIPolicyViolation"
        assert "content_filter_result" in innererror
        
        content_filter_result = innererror["content_filter_result"]
        assert content_filter_result["hate"]["filtered"] is True
        assert content_filter_result["hate"]["severity"] == "high"
        assert content_filter_result["violence"]["filtered"] is True
        assert content_filter_result["violence"]["severity"] == "medium"
        assert content_filter_result["sexual"]["filtered"] is False
        assert content_filter_result["self_harm"]["filtered"] is False
        assert content_filter_result["jailbreak"]["filtered"] is False

    def test_azure_content_policy_violation_different_categories(self):
        """Test Azure content policy violation with different filtering categories"""
        
        # Mock exception with different content filter results  
        mock_exception = Exception("The response was filtered due to the prompt triggering Azure OpenAI's content management policy")
        mock_exception.body = {
            "innererror": {
                "code": "ResponsibleAIPolicyViolation",
                "content_filter_result": {
                    "hate": {
                        "filtered": False,
                        "severity": "safe"
                    },
                    "jailbreak": {
                        "filtered": True,
                        "detected": True
                    },
                    "self_harm": {
                        "filtered": True,
                        "severity": "high"
                    },
                    "sexual": {
                        "filtered": True,
                        "severity": "medium"
                    },
                    "violence": {
                        "filtered": False,
                        "severity": "safe"
                    }
                }
            }
        }
        
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response
        
        # Test the exception mapping directly with different violation type
        with pytest.raises(ContentPolicyViolationError) as exc_info:
            exception_type(
                model="azure/gpt-4",
                original_exception=mock_exception,
                custom_llm_provider="azure"
            )
        
        # Verify provider_specific_fields contains the expected innererror structure
        e = exc_info.value
        assert e.provider_specific_fields is not None
        print("got provider_specific_fields=", e.provider_specific_fields)
        innererror = e.provider_specific_fields["innererror"]
        content_filter_result = innererror["content_filter_result"]
        
        # Check different filter categories
        assert content_filter_result["sexual"]["filtered"] is True
        assert content_filter_result["sexual"]["severity"] == "medium"
        assert content_filter_result["self_harm"]["filtered"] is True
        assert content_filter_result["self_harm"]["severity"] == "high" 
        assert content_filter_result["jailbreak"]["filtered"] is True
        assert content_filter_result["jailbreak"]["detected"] is True
        assert content_filter_result["hate"]["filtered"] is False
        assert content_filter_result["violence"]["filtered"] is False

    def test_azure_content_policy_violation_missing_innererror(self):
        """Test Azure content policy violation when innererror is missing from response"""
        
        # Mock exception without body attribute
        mock_exception = Exception("The response was filtered due to the prompt triggering Azure OpenAI's content management policy")
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response
        # Note: no mock_exception.body attribute set
        
        # Test the exception mapping directly
        with pytest.raises(ContentPolicyViolationError) as exc_info:
            exception_type(
                model="azure/gpt-4",
                original_exception=mock_exception,
                custom_llm_provider="azure"
            )
        
        # Verify that even without innererror, the exception is still raised properly
        e = exc_info.value
        print("got exception=", e)
        # provider_specific_fields should still exist but innererror should be None
        assert e.provider_specific_fields is not None
        assert e.provider_specific_fields.get("innererror") is None

    def test_azure_content_policy_violation_non_dict_body(self):
        """Test Azure content policy violation when body is not a dictionary"""
        
        # Mock exception with non-dict body
        mock_exception = Exception("The response was filtered due to the prompt triggering Azure OpenAI's content management policy")
        mock_exception.body = "invalid body format"
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response
        
        # Test the exception mapping directly
        with pytest.raises(ContentPolicyViolationError) as exc_info:
            exception_type(
                model="azure/gpt-4",
                original_exception=mock_exception,
                custom_llm_provider="azure"
            )
        
        # Verify that with invalid body format, innererror should be None
        e = exc_info.value
        print("got exception=", e)
        print("exception fields=", vars(e))
        assert e.provider_specific_fields is not None
        assert e.provider_specific_fields.get("innererror") is None 

    def test_azure_images_content_policy_violation_preserves_nested_inner_error(self):
        """Azure Images endpoints return errors nested under body['error'] with inner_error.

        Ensure we:
        - Detect the violation via structured payload (code=content_policy_violation)
        - Preserve code/type/message
        - Surface inner_error + revised_prompt + content_filter_results
        """

        mock_exception = Exception("Bad request")  # does not include policy substrings
        mock_exception.body = {
            "error": {
                "code": "content_policy_violation",
                "inner_error": {
                    "code": "ResponsibleAIPolicyViolation",
                    "content_filter_results": {
                        "violence": {"filtered": True, "severity": "low"}
                    },
                    "revised_prompt": "revised",
                },
                "message": "Your request was rejected as a result of our safety system.",
                "type": "invalid_request_error",
            }
        }

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_exception.response = mock_response

        with pytest.raises(ContentPolicyViolationError) as exc_info:
            exception_type(
                model="azure/dall-e-3",
                original_exception=mock_exception,
                custom_llm_provider="azure",
            )

        e = exc_info.value

        # OpenAI-style error fields should be populated
        assert getattr(e, "code", None) == "content_policy_violation"
        assert getattr(e, "type", None) == "invalid_request_error"
        assert "safety system" in str(e)

        # Provider-specific nested details must be preserved
        assert e.provider_specific_fields is not None
        assert e.provider_specific_fields["inner_error"]["code"] == "ResponsibleAIPolicyViolation"
        assert e.provider_specific_fields["inner_error"]["revised_prompt"] == "revised"
        assert e.provider_specific_fields["inner_error"]["content_filter_results"]["violence"]["filtered"] is True