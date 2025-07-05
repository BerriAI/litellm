## Unit tests for Anthropic User ID validation

import pytest
from unittest.mock import MagicMock, patch
import litellm
from litellm.llms.anthropic.chat.transformation import AnthropicConfig, _valid_user_id
from litellm.llms.anthropic.completion.transformation import AnthropicTextConfig


class TestAnthropicUserIdValidation:
    """Test suite for Anthropic user ID validation functionality"""

    def test_valid_user_id_function_accepts_valid_ids(self):
        """Test that _valid_user_id accepts valid user IDs"""
        valid_ids = [
            "user123",
            "abc-def-123",
            "user_123",
            "UUID-1234-5678",
            "some-random-string",
            "user.name",
            "test_user_id",
            "valid-user-id-123",
        ]
        
        for user_id in valid_ids:
            assert _valid_user_id(user_id) is True, f"Valid user ID {user_id} should be accepted"

    def test_valid_user_id_function_rejects_emails(self):
        """Test that _valid_user_id rejects email addresses"""
        invalid_emails = [
            "user@example.com",
            "test.email@domain.org",
            "user+tag@example.co.uk",
            "firstname.lastname@company.com",
            "user123@test.io",
            "a@b.co",
            "very.long.email.address@very.long.domain.name.com",
        ]
        
        for email in invalid_emails:
            assert _valid_user_id(email) is False, f"Email {email} should be rejected"
        
        # These should NOT be rejected because they don't match the email pattern
        # (they don't have a dot after @)
        not_emails = [
            "admin@localhost",  # No dot after @
            "user@domain",      # No dot after @
        ]
        
        for not_email in not_emails:
            assert _valid_user_id(not_email) is True, f"'{not_email}' should be accepted (not a valid email pattern)"

    def test_valid_user_id_function_rejects_phone_numbers(self):
        """Test that _valid_user_id rejects phone numbers"""
        invalid_phones = [
            "+1234567890",
            "123-456-7890",
            "(123) 456-7890",
            "+1 (123) 456-7890",
            "1234567890",
            "+44 20 7946 0958",
            "+33 1 42 86 83 26",
            "555-123-4567",
            "123456789",  # 9 digits, should be detected as phone number
        ]
        
        for phone in invalid_phones:
            assert _valid_user_id(phone) is False, f"Phone number {phone} should be rejected"
        
        # These should NOT be rejected because they don't match the phone pattern
        # (they contain characters not allowed in phone numbers)
        not_phones = [
            "123.456.7890",  # Contains dots which aren't allowed
            "123abc4567",    # Contains letters
            "12345",         # Too short (less than 7 characters)
        ]
        
        for not_phone in not_phones:
            assert _valid_user_id(not_phone) is True, f"'{not_phone}' should be accepted (not a valid phone pattern)"

    def test_anthropic_chat_config_validates_user_param(self):
        """Test that AnthropicConfig validates the user parameter"""
        config = AnthropicConfig()
        
        # Test with valid user ID
        params = config.map_openai_params(
            non_default_params={"user": "valid_user_123"},
            optional_params={},
            model="claude-3-sonnet-20240229",
            drop_params=True
        )
        assert "metadata" in params
        assert params["metadata"]["user_id"] == "valid_user_123"
        
        # Test with invalid user ID (email)
        params = config.map_openai_params(
            non_default_params={"user": "user@example.com"},
            optional_params={},
            model="claude-3-sonnet-20240229",
            drop_params=True
        )
        assert "metadata" not in params, "Email should be rejected and metadata should not be set"

    def test_anthropic_chat_config_validates_litellm_metadata_user_id(self):
        """Test that AnthropicConfig validates user_id from litellm_params metadata"""
        config = AnthropicConfig()
        
        # Test with valid user ID in metadata
        result = config.transform_request(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={"metadata": {"user_id": "valid_user_123"}},
            headers={}
        )
        assert "metadata" in result
        assert result["metadata"]["user_id"] == "valid_user_123"
        
        # Test with invalid user ID (email) in metadata
        result = config.transform_request(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={"metadata": {"user_id": "user@example.com"}},
            headers={}
        )
        assert "metadata" not in result, "Email in metadata should be rejected"

    def test_anthropic_completion_config_validates_user_param(self):
        """Test that AnthropicTextConfig validates the user parameter"""
        config = AnthropicTextConfig()
        
        # Test with valid user ID
        params = config.map_openai_params(
            non_default_params={"user": "valid_user_123"},
            optional_params={},
            model="claude-2",
            drop_params=True
        )
        assert "metadata" in params
        assert params["metadata"]["user_id"] == "valid_user_123"
        
        # Test with invalid user ID (email)
        params = config.map_openai_params(
            non_default_params={"user": "user@example.com"},
            optional_params={},
            model="claude-2",
            drop_params=True
        )
        assert "metadata" not in params, "Email should be rejected and metadata should not be set"

    def test_anthropic_chat_config_handles_none_user(self):
        """Test that AnthropicConfig handles None user values gracefully"""
        config = AnthropicConfig()
        
        params = config.map_openai_params(
            non_default_params={"user": None},
            optional_params={},
            model="claude-3-sonnet-20240229",
            drop_params=True
        )
        assert "metadata" not in params

    def test_anthropic_completion_config_handles_none_user(self):
        """Test that AnthropicTextConfig handles None user values gracefully"""
        config = AnthropicTextConfig()
        
        params = config.map_openai_params(
            non_default_params={"user": None},
            optional_params={},
            model="claude-2",
            drop_params=True
        )
        assert "metadata" not in params

    def test_anthropic_chat_config_handles_non_string_user(self):
        """Test that AnthropicConfig handles non-string user values gracefully"""
        config = AnthropicConfig()
        
        params = config.map_openai_params(
            non_default_params={"user": 123},
            optional_params={},
            model="claude-3-sonnet-20240229",
            drop_params=True
        )
        assert "metadata" not in params

    def test_anthropic_completion_config_handles_non_string_user(self):
        """Test that AnthropicTextConfig handles non-string user values gracefully"""
        config = AnthropicTextConfig()
        
        params = config.map_openai_params(
            non_default_params={"user": 123},
            optional_params={},
            model="claude-2",
            drop_params=True
        )
        assert "metadata" not in params

    def test_anthropic_chat_config_preserves_existing_metadata(self):
        """Test that AnthropicConfig preserves existing metadata when adding user_id"""
        config = AnthropicConfig()
        
        existing_metadata = {"some_field": "some_value"}
        params = config.map_openai_params(
            non_default_params={"user": "valid_user_123"},
            optional_params={"metadata": existing_metadata},
            model="claude-3-sonnet-20240229",
            drop_params=True
        )
        
        # The user_id should be added to the existing metadata
        assert "metadata" in params
        assert params["metadata"]["user_id"] == "valid_user_123"

    def test_anthropic_completion_config_preserves_existing_metadata(self):
        """Test that AnthropicTextConfig preserves existing metadata when adding user_id"""
        config = AnthropicTextConfig()
        
        existing_metadata = {"some_field": "some_value"}
        params = config.map_openai_params(
            non_default_params={"user": "valid_user_123"},
            optional_params={"metadata": existing_metadata},
            model="claude-2",
            drop_params=True
        )
        
        # The user_id should be added to the existing metadata
        assert "metadata" in params
        assert params["metadata"]["user_id"] == "valid_user_123"

    @pytest.mark.parametrize(
        "user_id,expected_valid",
        [
            ("valid_user_123", True),
            ("user@example.com", False),
            ("+1234567890", False),
            ("123-456-7890", False),
            ("user.name", True),
            ("test_user", True),
            ("user+tag@example.com", False),
            ("(123) 456-7890", False),
            ("uuid-1234-5678", True),
            ("user@domain", True),  # Not a valid email (no dot after @)
            ("123456789", False),  # This is detected as a phone number
            ("12345", True),  # Too short to be a phone number
            ("abc123", True),  # Contains letters, not a phone number
            ("123.456.7890", True),  # Dots not allowed in phone pattern
        ],
    )
    def test_valid_user_id_parametrized(self, user_id, expected_valid):
        """Parametrized test for _valid_user_id function"""
        assert _valid_user_id(user_id) == expected_valid

    def test_integration_with_litellm_completion_chat(self):
        """Test integration with litellm completion call for chat models"""
        with patch('litellm.completion') as mock_completion:
            # Mock a successful response
            mock_completion.return_value = MagicMock()
            
            # Test with valid user ID - should pass through
            try:
                litellm.completion(
                    model="anthropic/claude-3-sonnet-20240229",
                    messages=[{"role": "user", "content": "Hello"}],
                    user="valid_user_123"
                )
                mock_completion.assert_called_once()
                
                # Get the call arguments
                call_args = mock_completion.call_args
                # The user param should be processed and converted to metadata
                assert call_args is not None
                
            except Exception as e:
                pytest.fail(f"Valid user ID should not cause an exception: {e}")

    def test_integration_with_litellm_completion_text(self):
        """Test integration with litellm completion call for text models"""
        with patch('litellm.completion') as mock_completion:
            # Mock a successful response
            mock_completion.return_value = MagicMock()
            
            # Test with valid user ID - should pass through
            try:
                litellm.completion(
                    model="anthropic_text/claude-2",
                    messages=[{"role": "user", "content": "Hello"}],
                    user="valid_user_123"
                )
                mock_completion.assert_called_once()
                
                # Get the call arguments
                call_args = mock_completion.call_args
                # The user param should be processed and converted to metadata
                assert call_args is not None
                
            except Exception as e:
                pytest.fail(f"Valid user ID should not cause an exception: {e}")

    def test_edge_cases_empty_strings(self):
        """Test edge cases with empty strings"""
        assert _valid_user_id("") is True, "Empty string should be considered valid"
        
        config = AnthropicConfig()
        params = config.map_openai_params(
            non_default_params={"user": ""},
            optional_params={},
            model="claude-3-sonnet-20240229",
            drop_params=True
        )
        assert "metadata" in params
        assert params["metadata"]["user_id"] == ""

    def test_edge_cases_whitespace_strings(self):
        """Test edge cases with whitespace strings"""
        # These should be considered valid (not phone numbers)
        valid_whitespace_cases = [
            "   ",    # Only 3 spaces, too short for phone number
            "\t",     # Only 1 tab, too short
            "\n",     # Only 1 newline, too short
        ]
        
        for whitespace in valid_whitespace_cases:
            assert _valid_user_id(whitespace) is True, f"Whitespace '{repr(whitespace)}' should be considered valid"
        
        # These should be rejected (match phone pattern - 7+ characters of allowed phone chars)
        invalid_whitespace_cases = [
            "  \t  \n  ",  # 9 characters, matches phone pattern
            "       ",     # 7 spaces, matches phone pattern
        ]
        
        for whitespace in invalid_whitespace_cases:
            assert _valid_user_id(whitespace) is False, f"Whitespace '{repr(whitespace)}' should be rejected (matches phone pattern)"

    def test_anthropic_error_prevention_email_in_user_param(self):
        """Test that emails in user param are rejected and don't cause BadRequestError"""
        config = AnthropicConfig()
        
        # This used to cause: metadata.user_id: user_id appears to contain an email address
        params = config.map_openai_params(
            non_default_params={"user": "test@example.com"},
            optional_params={},
            model="claude-3-sonnet-20240229",
            drop_params=True
        )
        
        # The email should be rejected and metadata should not be set
        assert "metadata" not in params, "Email should be rejected to prevent Anthropic BadRequestError"

    def test_anthropic_error_prevention_phone_in_user_param(self):
        """Test that phone numbers in user param are rejected and don't cause BadRequestError"""
        config = AnthropicConfig()
        
        # This used to cause: metadata.user_id: user_id appears to contain a phone number
        params = config.map_openai_params(
            non_default_params={"user": "+1234567890"},
            optional_params={},
            model="claude-3-sonnet-20240229",
            drop_params=True
        )
        
        # The phone number should be rejected and metadata should not be set
        assert "metadata" not in params, "Phone number should be rejected to prevent Anthropic BadRequestError"

    def test_anthropic_error_prevention_completion_email_in_user_param(self):
        """Test that emails in user param are rejected for completion endpoint"""
        config = AnthropicTextConfig()
        
        # This used to cause: metadata.user_id: user_id appears to contain an email address
        params = config.map_openai_params(
            non_default_params={"user": "test@example.com"},
            optional_params={},
            model="claude-2",
            drop_params=True
        )
        
        # The email should be rejected and metadata should not be set
        assert "metadata" not in params, "Email should be rejected to prevent Anthropic BadRequestError"

    def test_anthropic_error_prevention_completion_phone_in_user_param(self):
        """Test that phone numbers in user param are rejected for completion endpoint"""
        config = AnthropicTextConfig()
        
        # This used to cause: metadata.user_id: user_id appears to contain a phone number
        params = config.map_openai_params(
            non_default_params={"user": "+1234567890"},
            optional_params={},
            model="claude-2",
            drop_params=True
        )
        
        # The phone number should be rejected and metadata should not be set
        assert "metadata" not in params, "Phone number should be rejected to prevent Anthropic BadRequestError"

    def test_anthropic_error_prevention_email_in_metadata(self):
        """Test that emails in litellm metadata are rejected"""
        config = AnthropicConfig()
        
        # This used to cause: metadata.user_id: user_id appears to contain an email address
        result = config.transform_request(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={"metadata": {"user_id": "test@example.com"}},
            headers={}
        )
        
        # The email should be rejected and metadata should not be set
        assert "metadata" not in result, "Email in metadata should be rejected to prevent Anthropic BadRequestError"

    def test_anthropic_error_prevention_phone_in_metadata(self):
        """Test that phone numbers in litellm metadata are rejected"""
        config = AnthropicConfig()
        
        # This used to cause: metadata.user_id: user_id appears to contain a phone number
        result = config.transform_request(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={"metadata": {"user_id": "+1234567890"}},
            headers={}
        )
        
        # The phone number should be rejected and metadata should not be set
        assert "metadata" not in result, "Phone number in metadata should be rejected to prevent Anthropic BadRequestError" 