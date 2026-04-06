from litellm.proxy.management_endpoints.key_management_endpoints import (
    _mask_sensitive_fields_in_key_info,
)


class TestMaskSensitiveFieldsInKeyInfo:
    """Test the _mask_sensitive_fields_in_key_info function to ensure sensitive data is masked."""

    def test_mask_langfuse_credentials_in_metadata_logging(self):
        """Test that langfuse_public_key and langfuse_secret_key are masked in metadata.logging."""
        key_info = {
            "token": "hashed_token",
            "user_id": "user123",
            "metadata": {
                "logging": [
                    {
                        "callback_name": "langfuse",
                        "callback_type": "success_and_failure",
                        "callback_vars": {
                            "langfuse_public_key": "pk-lf-1234567890abcdef",
                            "langfuse_secret_key": "sk-lf-0987654321fedcba",
                            "langfuse_host": "https://langfuse.example.com",
                        },
                    }
                ]
            },
        }

        result = _mask_sensitive_fields_in_key_info(key_info)

        # Verify the sensitive keys are masked (format: first 4 chars + **** + last 4 chars)
        callback_vars = result["metadata"]["logging"][0]["callback_vars"]
        assert callback_vars["langfuse_public_key"] == "pk-l**************cdef"
        assert callback_vars["langfuse_secret_key"] == "sk-l**************dcba"
        # Non-sensitive fields should not be masked
        assert callback_vars["langfuse_host"] == "https://langfuse.example.com"

    def test_mask_callback_settings_callback_vars(self):
        """Test that callback_vars in callback_settings are masked."""
        key_info = {
            "token": "hashed_token",
            "metadata": {
                "callback_settings": {
                    "success_callback": ["langfuse"],
                    "callback_vars": {
                        "langfuse_public_key": "pk-lf-test-key-1234",
                        "langfuse_secret_key": "sk-lf-test-secret-5678",
                    },
                }
            },
        }

        result = _mask_sensitive_fields_in_key_info(key_info)

        callback_vars = result["metadata"]["callback_settings"]["callback_vars"]
        assert "****" in callback_vars["langfuse_public_key"]
        assert "****" in callback_vars["langfuse_secret_key"]

    def test_no_metadata(self):
        """Test that key_info without metadata is returned unchanged."""
        key_info = {
            "token": "hashed_token",
            "user_id": "user123",
        }

        result = _mask_sensitive_fields_in_key_info(key_info)

        assert result["token"] == "hashed_token"
        assert result["user_id"] == "user123"
        assert "metadata" not in result

    def test_empty_metadata(self):
        """Test that key_info with empty metadata is handled correctly."""
        key_info = {
            "token": "hashed_token",
            "metadata": {},
        }

        result = _mask_sensitive_fields_in_key_info(key_info)

        assert result["metadata"] == {}

    def test_metadata_without_sensitive_data(self):
        """Test that metadata without sensitive fields is not modified."""
        key_info = {
            "token": "hashed_token",
            "metadata": {
                "team": "engineering",
                "app": "my-app",
                "tags": ["production", "api"],
            },
        }

        result = _mask_sensitive_fields_in_key_info(key_info)

        assert result["metadata"]["team"] == "engineering"
        assert result["metadata"]["app"] == "my-app"
        assert result["metadata"]["tags"] == ["production", "api"]

    def test_mask_other_sensitive_fields(self):
        """Test that other sensitive field patterns are also masked."""
        key_info = {
            "token": "hashed_token",
            "metadata": {
                "api_key": "sk-1234567890abcdef",
                "password": "super_secret_password",
                "token": "bearer_token_123",
                "secret": "my_secret_value",
                "normal_field": "this_should_remain",
            },
        }

        result = _mask_sensitive_fields_in_key_info(key_info)

        metadata = result["metadata"]
        assert "****" in metadata["api_key"]
        assert "****" in metadata["password"]
        assert "****" in metadata["token"]
        assert "****" in metadata["secret"]
        assert metadata["normal_field"] == "this_should_remain"

    def test_original_key_info_not_modified(self):
        """Test that the original key_info dict is not modified (deep copy works)."""
        key_info = {
            "token": "hashed_token",
            "metadata": {
                "logging": [
                    {
                        "callback_vars": {
                            "langfuse_secret_key": "sk-lf-original-secret",
                        }
                    }
                ]
            },
        }

        result = _mask_sensitive_fields_in_key_info(key_info)

        # Verify original is unchanged
        assert (
            key_info["metadata"]["logging"][0]["callback_vars"]["langfuse_secret_key"]
            == "sk-lf-original-secret"
        )
        # Verify result is masked
        assert (
            result["metadata"]["logging"][0]["callback_vars"]["langfuse_secret_key"]
            != "sk-lf-original-secret"
        )

    def test_none_key_info(self):
        """Test that None key_info is handled gracefully."""
        result = _mask_sensitive_fields_in_key_info(None)
        assert result is None

    def test_non_dict_key_info(self):
        """Test that non-dict key_info is handled gracefully."""
        result = _mask_sensitive_fields_in_key_info("not a dict")
        assert result == "not a dict"
