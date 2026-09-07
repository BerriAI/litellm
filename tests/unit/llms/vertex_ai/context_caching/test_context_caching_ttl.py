import pytest
from litellm.llms.vertex_ai.context_caching.transformation import (
    _normalize_ttl_to_seconds,
    extract_ttl_from_cached_messages,
    transform_openai_messages_to_gemini_context_caching,
)


class TestTTLNormalization:
    @pytest.mark.parametrize(
        "ttl, expected",
        [
            ("3600s", "3600s"),
            ("1s", "1s"),
            ("1.5s", "1.5s"),
            ("0.1s", "0.1s"),
            ("123.456s", "123.456s"),
            ("1.3333333333333333s", "1.333333333s"),
            ("5m", "300s"),
            ("90m", "5400s"),
            ("1h", "3600s"),
            ("0.5h", "1800s"),
            ("48h", "172800s"),
            ("61320000h", "220752000000s"),
        ],
    )
    def test_normalizes_supported_units_to_seconds(self, ttl, expected):
        assert _normalize_ttl_to_seconds(ttl) == expected

    @pytest.mark.parametrize(
        "ttl",
        [
            "3600",
            "s",
            "-1s",
            "0s",
            "0m",
            "0h",
            "5d",
            "abc.s",
            "",
            "3600.s",
            "3600 s",
            "3600ss",
            "1 h",
            "0.0000000001s",
            "251700000000s",
            "69920000h",
            "9" * 400 + "h",
            None,
            123,
        ],
    )
    def test_rejects_unparseable_ttl(self, ttl):
        assert _normalize_ttl_to_seconds(ttl) is None


class TestTTLExtraction:
    """Test TTL extraction from cached messages"""

    @pytest.mark.parametrize("ttl, expected", [("1h", "3600s"), ("5m", "300s")])
    def test_extract_ttl_normalizes_anthropic_units(self, ttl, expected):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "cached",
                        "cache_control": {"type": "ephemeral", "ttl": ttl},
                    }
                ],
            }
        ]

        assert extract_ttl_from_cached_messages(messages) == expected

    def test_extract_ttl_from_single_message(self):
        """Test extracting TTL from a single cached message"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is cached content",
                        "cache_control": {"type": "ephemeral", "ttl": "3600s"},
                    }
                ],
            }
        ]

        ttl = extract_ttl_from_cached_messages(messages)
        assert ttl == "3600s"

    def test_extract_ttl_from_multiple_messages(self):
        """Test extracting TTL from multiple cached messages (should return first valid one)"""
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "System message",
                        "cache_control": {"type": "ephemeral", "ttl": "7200s"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "User message",
                        "cache_control": {"type": "ephemeral", "ttl": "3600s"},
                    }
                ],
            },
        ]

        ttl = extract_ttl_from_cached_messages(messages)
        assert ttl == "7200s"  # Should return the first valid TTL found

    def test_extract_ttl_no_cache_control(self):
        """Test extracting TTL from messages without cache_control"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Regular message without cache control"}
                ],
            }
        ]

        ttl = extract_ttl_from_cached_messages(messages)
        assert ttl is None

    def test_extract_ttl_invalid_format(self):
        """Test extracting TTL with invalid format"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Cached content with invalid TTL",
                        "cache_control": {"type": "ephemeral", "ttl": "invalid"},
                    }
                ],
            }
        ]

        ttl = extract_ttl_from_cached_messages(messages)
        assert ttl is None

    def test_extract_ttl_missing_ttl_field(self):
        """Test extracting TTL when ttl field is missing"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Cached content without TTL field",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]

        ttl = extract_ttl_from_cached_messages(messages)
        assert ttl is None

    def test_extract_ttl_mixed_valid_invalid(self):
        """Test extracting TTL when some messages have valid TTL and others don't"""
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "System message with invalid TTL",
                        "cache_control": {"type": "ephemeral", "ttl": "invalid"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "User message with valid TTL",
                        "cache_control": {"type": "ephemeral", "ttl": "3600s"},
                    }
                ],
            },
        ]

        ttl = extract_ttl_from_cached_messages(messages)
        assert ttl == "3600s"  # Should return the first valid TTL found

    def test_extract_ttl_string_content(self):
        """Test extracting TTL when message content is a string (not a list)"""
        messages = [{"role": "user", "content": "String content"}]

        ttl = extract_ttl_from_cached_messages(messages)
        assert ttl is None


class TestTransformationWithTTL:
    """Test the complete transformation with TTL support"""

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    def test_transform_with_valid_ttl(self, custom_llm_provider):
        """Test transformation includes TTL when provided"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Cached content",
                        "cache_control": {"type": "ephemeral", "ttl": "3600s"},
                    }
                ],
            }
        ]

        vertex_location = "test_location"
        vertex_project = "test_project"

        result = transform_openai_messages_to_gemini_context_caching(
            model="gemini-2.5-pro",
            messages=messages,
            cache_key="test-cache-key",
            custom_llm_provider=custom_llm_provider,
            vertex_location="test_location",
            vertex_project="test_project",
        )

        assert "ttl" in result
        assert result["ttl"] == "3600s"

        if custom_llm_provider == "gemini":
            assert result["model"] == "models/gemini-2.5-pro"
        else:
            assert (
                result["model"]
                == f"projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/gemini-2.5-pro"
            )

        assert result["displayName"] == "test-cache-key"

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    def test_transform_without_ttl(self, custom_llm_provider):
        """Test transformation without TTL"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Cached content",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]

        vertex_location = "test_location"
        vertex_project = "test_project"

        result = transform_openai_messages_to_gemini_context_caching(
            model="gemini-2.5-pro",
            messages=messages,
            cache_key="test-cache-key",
            custom_llm_provider=custom_llm_provider,
            vertex_location=vertex_location,
            vertex_project=vertex_project,
        )

        assert "ttl" not in result

        if custom_llm_provider == "gemini":
            assert result["model"] == "models/gemini-2.5-pro"
        else:
            assert (
                result["model"]
                == f"projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/gemini-2.5-pro"
            )

        assert result["displayName"] == "test-cache-key"

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    def test_transform_with_invalid_ttl(self, custom_llm_provider):
        """Test transformation with invalid TTL (should be ignored)"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Cached content",
                        "cache_control": {"type": "ephemeral", "ttl": "invalid"},
                    }
                ],
            }
        ]
        vertex_location = "test_location"
        vertex_project = "test_project"

        result = transform_openai_messages_to_gemini_context_caching(
            model="gemini-2.5-pro",
            messages=messages,
            cache_key="test-cache-key",
            custom_llm_provider=custom_llm_provider,
            vertex_location=vertex_location,
            vertex_project=vertex_project,
        )

        assert "ttl" not in result

        if custom_llm_provider == "gemini":
            assert result["model"] == "models/gemini-2.5-pro"
        else:
            assert (
                result["model"]
                == f"projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/gemini-2.5-pro"
            )

        assert result["displayName"] == "test-cache-key"

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    def test_transform_with_system_message_and_ttl(self, custom_llm_provider):
        """Test transformation with system message and TTL"""
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "System instruction",
                        "cache_control": {"type": "ephemeral", "ttl": "7200s"},
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "User message"}]},
        ]

        vertex_location = "test_location"
        vertex_project = "test_project"

        result = transform_openai_messages_to_gemini_context_caching(
            model="gemini-2.5-pro",
            messages=messages,
            cache_key="test-cache-key",
            custom_llm_provider=custom_llm_provider,
            vertex_location=vertex_location,
            vertex_project=vertex_project,
        )

        assert "ttl" in result
        assert result["ttl"] == "7200s"
        assert "systemInstruction" in result

        if custom_llm_provider == "gemini":
            assert result["model"] == "models/gemini-2.5-pro"
        else:
            assert (
                result["model"]
                == f"projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/gemini-2.5-pro"
            )

        assert result["displayName"] == "test-cache-key"


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_ttl_extraction_empty_messages(self):
        """Test TTL extraction with empty message list"""
        messages = []
        ttl = extract_ttl_from_cached_messages(messages)
        assert ttl is None

    def test_ttl_extraction_none_content(self):
        """Test TTL extraction when content is None"""
        messages = [{"role": "user", "content": None}]
        ttl = extract_ttl_from_cached_messages(messages)
        assert ttl is None

    def test_ttl_extraction_empty_content_list(self):
        """Test TTL extraction when content list is empty"""
        messages = [{"role": "user", "content": []}]
        ttl = extract_ttl_from_cached_messages(messages)
        assert ttl is None

    def test_ttl_validation_type_conversion(self):
        """Test TTL validation handles type conversion properly"""
        # Test that numeric TTL gets converted to string
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Cached content",
                        "cache_control": {"type": "ephemeral", "ttl": "3600s"},
                    }
                ],
            }
        ]

        ttl = extract_ttl_from_cached_messages(messages)
        assert isinstance(ttl, str)
        assert ttl == "3600s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
