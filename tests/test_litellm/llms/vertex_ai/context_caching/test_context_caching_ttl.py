import pytest
from litellm.llms.vertex_ai.context_caching.transformation import (
    extract_ttl_from_cached_messages,
    get_gemini_context_caching_min_tokens,
    _is_valid_ttl_format,
    _normalize_ttl_to_seconds,
    transform_openai_messages_to_gemini_context_caching,
)


class TestGeminiContextCachingMinTokens:
    """Per-model floor for explicit Gemini context cache creation."""

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("gemini-1.5-pro", 32768),
            ("gemini-1.5-flash", 32768),
            ("vertex_ai/gemini-1.5-pro-001", 32768),
            ("gemini-2.5-flash", 2048),
            ("gemini-2.5-pro", 2048),
            ("gemini/gemini-2.5-pro", 2048),
            ("vertex_ai/gemini-2.5-flash", 2048),
            ("gemini-3.5-flash", 4096),
            ("gemini-3.1-pro-preview", 4096),
            ("gemini/gemini-3.5-flash", 4096),
            ("gemini-unknown-future-model", 32768),
        ],
    )
    def test_min_tokens_by_model(self, model, expected):
        assert get_gemini_context_caching_min_tokens(model) == expected


class TestTTLValidation:
    """Test TTL format validation"""

    def test_valid_ttl_formats(self):
        """Test various valid TTL formats"""
        valid_ttls = ["3600s", "1s", "7200s", "1.5s", "0.1s", "86400s", "123.456s"]

        for ttl in valid_ttls:
            assert _is_valid_ttl_format(ttl), f"TTL {ttl} should be valid"

    def test_invalid_ttl_formats(self):
        """Test various invalid TTL formats"""
        invalid_ttls = [
            "3600",  # missing 's'
            "s",  # missing number
            "-1s",  # negative number
            "0s",  # zero
            "3600m",  # wrong unit
            "abc.s",  # invalid number
            "",  # empty string
            "3600.s",  # invalid decimal
            "3600 s",  # space
            "3600ss",  # extra 's'
            None,  # None
            123,  # not a string
        ]

        for ttl in invalid_ttls:
            assert not _is_valid_ttl_format(ttl), f"TTL {ttl} should be invalid"


class TestTTLNormalization:
    """Normalization of anthropic-style TTL units into Gemini's seconds format."""

    @pytest.mark.parametrize(
        "ttl, expected",
        [
            ("3600s", "3600s"),
            ("1.5s", "1.5s"),
            ("5m", "300s"),
            ("90m", "5400s"),
            ("1h", "3600s"),
            ("2h", "7200s"),
            ("0.5h", "1800s"),
        ],
    )
    def test_normalizes_units_to_seconds(self, ttl, expected):
        assert _normalize_ttl_to_seconds(ttl) == expected

    @pytest.mark.parametrize(
        "ttl",
        ["invalid", "", "0m", "0h", "-1h", "5d", "1 h", "m", None, 123, 3600],
    )
    def test_rejects_unparseable_ttl(self, ttl):
        assert _normalize_ttl_to_seconds(ttl) is None

    def test_extract_ttl_normalizes_anthropic_hour_unit(self):
        """Claude Code / Anthropic send "1h"; Gemini must receive "3600s"."""
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "cached",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            }
        ]

        assert extract_ttl_from_cached_messages(messages) == "3600s"

    def test_extract_ttl_normalizes_anthropic_minute_unit(self):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "cached",
                        "cache_control": {"type": "ephemeral", "ttl": "5m"},
                    }
                ],
            }
        ]

        assert extract_ttl_from_cached_messages(messages) == "300s"


class TestTTLExtraction:
    """Test TTL extraction from cached messages"""

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
                "content": [{"type": "text", "text": "Regular message without cache control"}],
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

    @pytest.mark.parametrize("custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"])
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

    @pytest.mark.parametrize("custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"])
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

    @pytest.mark.parametrize("custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"])
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

    @pytest.mark.parametrize("custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"])
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
        assert "system_instruction" in result

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

    def test_cache_control_preserved_for_object_content_items(self):
        """Test that cache_control is preserved when content items are real Pydantic models."""
        from pydantic import BaseModel, Field
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        class MockContentBlock:
            def __init__(self):
                self.type = "text"
                self.text = "hello"
                self.cache_control = {"type": "ephemeral"}

        class RealPydanticV2Block(BaseModel):
            type: str = "text"
            text: str = "hello v2"
            cache_control: dict = Field(default_factory=lambda: {"type": "ephemeral"})

        class MockBlockWithNoneCacheControl:
            def __init__(self):
                self.type = "text"
                self.text = "hello none"
                self.cache_control = None

        content = [
            MockContentBlock(),
            RealPydanticV2Block(),
            MockBlockWithNoneCacheControl(),
        ]
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(content)
        assert result == [
            {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "hello v2", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "hello none"},
        ]

    def test_is_cached_message_for_object_message_and_content_item(self):
        """Test is_cached_message on custom objects / models."""
        from litellm.utils import is_cached_message

        # Test message level cache_control object
        class MockCacheControl:
            def __init__(self):
                self.type = "ephemeral"

        class MockMessageLevelObj:
            def __init__(self):
                self.role = "system"
                self.content = "hello"
                self.cache_control = MockCacheControl()

        msg = MockMessageLevelObj()
        assert is_cached_message(msg) is True

        # Test content level cache_control object
        class MockContentItem:
            def __init__(self):
                self.type = "text"
                self.text = "hello"
                self.cache_control = MockCacheControl()

        class MockContentLevelObj:
            def __init__(self):
                self.role = "system"
                self.content = [MockContentItem()]

        msg = MockContentLevelObj()
        assert is_cached_message(msg) is True

    def test_extract_ttl_from_cached_messages_for_object_models(self):
        """Test extract_ttl_from_cached_messages with object-based messages and content items."""

        class MockCacheControl:
            def __init__(self):
                self.type = "ephemeral"
                self.ttl = "3600s"

        class MockContentItem:
            def __init__(self):
                self.type = "text"
                self.text = "hello"
                self.cache_control = MockCacheControl()

        class MockMessageObj:
            def __init__(self):
                self.role = "system"
                self.content = [MockContentItem()]

        messages = [MockMessageObj()]
        ttl = extract_ttl_from_cached_messages(messages)
        assert ttl == "3600s"

    def test_extract_ttl_from_cached_messages_with_message_level_object_cache_control(self):
        """Test extract_ttl_from_cached_messages with message-level object cache_control."""

        class MockCacheControl:
            def __init__(self):
                self.type = "ephemeral"
                self.ttl = "7200s"

        class MockMessageObj:
            def __init__(self):
                self.role = "system"
                self.content = "hello"
                self.cache_control = MockCacheControl()

        messages = [MockMessageObj()]
        ttl = extract_ttl_from_cached_messages(messages)
        assert ttl == "7200s"

    def test_is_cached_message_for_dict_message_with_dict_content_items(self):
        """Test is_cached_message with dict message and dict content list items."""
        from litellm.utils import is_cached_message

        # Dictionary message without content should return False
        assert is_cached_message({"role": "user"}) is False

        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
            ],
        }
        assert is_cached_message(msg) is True

    def test_normalize_responses_api_object_to_dict_pydantic_v1(self):
        """Test _normalize_responses_api_object_to_dict with Pydantic v1 dict fallback."""
        from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig

        class MockPydanticV1Model:
            def dict(self):
                return {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}

        item = MockPydanticV1Model()
        res = LiteLLMCompletionResponsesConfig._normalize_responses_api_object_to_dict(item)
        assert res == {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
