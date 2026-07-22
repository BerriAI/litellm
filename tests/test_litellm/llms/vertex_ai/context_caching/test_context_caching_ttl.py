import pytest
from litellm.llms.vertex_ai.context_caching.transformation import (
    extract_ttl_from_cached_messages,
    _is_valid_ttl_format,
    transform_openai_messages_to_gemini_context_caching,
)


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


class TestToolChoiceContextCaching:
    """Tests for tool_choice handling in context caching.

    Regression tests for https://github.com/BerriAI/litellm/issues/29088

    Before the fix, `tool_choice` was never popped from `optional_params` during
    cache creation.  It then reached `_transform_request_body()` where the guard
    ``can_send_cache_incompatible_fields`` silently dropped it because
    ``cached_content`` was already set.  Result: `toolConfig` appeared in neither
    the cached-content request nor the final API call.

    The fix mirrors the existing `tools` handling: pop `tool_choice` from
    `optional_params` at cache-creation time, include it in the cache key, and
    set it on the `CachedContentRequestBody` as `toolConfig`.
    """

    def _make_cached_messages(self) -> list:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "stable prefix",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]

    def test_tool_choice_popped_from_optional_params(self) -> None:
        """tool_choice must be removed from optional_params so the downstream
        guard in transformation.py cannot silently drop it."""
        from unittest.mock import MagicMock, patch
        from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import (
            ContextCachingEndpoints,
        )

        endpoint = ContextCachingEndpoints()
        tool_choice = {"mode": "ANY", "allowedFunctionNames": ["my_func"]}
        optional_params = {
            "tools": [{"function_declarations": [{"name": "my_func"}]}],
            "tool_choice": tool_choice,
        }

        with (
            patch.object(endpoint, "_get_token_and_url_context_caching", return_value=("token", "http://url")),
            patch.object(endpoint, "check_cache", return_value=None),
            patch("litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.is_prompt_caching_valid_prompt", return_value=True),
            patch("litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.HTTPHandler") as mock_http,
        ):
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"name": "cached/123", "model": "gemini-1.5-pro"}
            mock_http.return_value.post.return_value = mock_resp

            endpoint.check_and_create_cache(
                messages=self._make_cached_messages(),
                optional_params=optional_params,
                api_key="test-key",
                api_base=None,
                model="gemini-1.5-pro",
                client=None,
                timeout=None,
                logging_obj=MagicMock(),
                custom_llm_provider="gemini",
                vertex_project=None,
                vertex_location=None,
                vertex_auth_header=None,
            )

        assert "tool_choice" not in optional_params, (
            "tool_choice must be popped from optional_params so the downstream "
            "can_send_cache_incompatible_fields guard cannot drop it silently"
        )

    def test_tool_choice_included_in_cached_content_body(self) -> None:
        """toolConfig must appear in the body sent to the cache-creation endpoint."""
        from unittest.mock import MagicMock, call, patch
        from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import (
            ContextCachingEndpoints,
        )

        endpoint = ContextCachingEndpoints()
        tool_choice = {"mode": "ANY", "allowedFunctionNames": ["my_func"]}
        optional_params = {
            "tools": [{"function_declarations": [{"name": "my_func"}]}],
            "tool_choice": tool_choice,
        }

        with (
            patch.object(endpoint, "_get_token_and_url_context_caching", return_value=("token", "http://url")),
            patch.object(endpoint, "check_cache", return_value=None),
            patch("litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.is_prompt_caching_valid_prompt", return_value=True),
            patch("litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.HTTPHandler") as mock_http,
        ):
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"name": "cached/123", "model": "gemini-1.5-pro"}
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_http.return_value = mock_client

            endpoint.check_and_create_cache(
                messages=self._make_cached_messages(),
                optional_params=optional_params,
                api_key="test-key",
                api_base=None,
                model="gemini-1.5-pro",
                client=None,
                timeout=None,
                logging_obj=MagicMock(),
                custom_llm_provider="gemini",
                vertex_project=None,
                vertex_location=None,
                vertex_auth_header=None,
            )

        post_calls = mock_client.post.call_args_list
        assert len(post_calls) == 1, "exactly one POST to the cache endpoint expected"
        sent_body = post_calls[0].kwargs.get("json") or post_calls[0].args[1]
        assert "toolConfig" in sent_body, (
            "toolConfig must be present in the cached-content request body"
        )
        assert sent_body["toolConfig"] == tool_choice


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
