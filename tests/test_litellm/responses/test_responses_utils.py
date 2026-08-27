import base64
from unittest.mock import MagicMock, patch

import pytest


import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.utils import ResponseAPILoggingUtils, ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIOptionalRequestParams
from litellm.types.utils import Usage


class TestResponsesAPIRequestUtils:
    def test_get_optional_params_responses_api(self):
        """Test that optional parameters are correctly processed for responses API"""
        # Setup
        model = "gpt-4o"
        config = OpenAIResponsesAPIConfig()
        optional_params = ResponsesAPIOptionalRequestParams(
            {
                "temperature": 0.7,
                "max_output_tokens": 100,
                "prompt": {"id": "pmpt_123"},
            }
        )

        # Execute
        result = ResponsesAPIRequestUtils.get_optional_params_responses_api(
            model=model,
            responses_api_provider_config=config,
            response_api_optional_params=optional_params,
        )

        # Assert
        assert result == optional_params
        assert "temperature" in result
        assert result["temperature"] == 0.7
        assert "max_output_tokens" in result
        assert result["max_output_tokens"] == 100
        assert "prompt" in result
        assert result["prompt"] == {"id": "pmpt_123"}

    def test_get_optional_params_responses_api_unsupported_param(self):
        """Test that unsupported parameters raise an error"""
        # Setup
        model = "gpt-4o"
        config = OpenAIResponsesAPIConfig()
        optional_params = ResponsesAPIOptionalRequestParams({"temperature": 0.7, "unsupported_param": "value"})

        # Execute and Assert
        with pytest.raises(litellm.UnsupportedParamsError) as excinfo:
            ResponsesAPIRequestUtils.get_optional_params_responses_api(
                model=model,
                responses_api_provider_config=config,
                response_api_optional_params=optional_params,
            )

        assert "unsupported_param" in str(excinfo.value)
        assert model in str(excinfo.value)

    def test_get_optional_params_responses_api_request_level_drop_params(self, monkeypatch):
        """Request-level drop_params must reach both _check_valid_arg and map_openai_params"""
        monkeypatch.setattr(litellm, "drop_params", False)
        config = MagicMock(spec=OpenAIResponsesAPIConfig)
        config.get_supported_openai_params.return_value = ["temperature"]
        config.custom_llm_provider = "openai"
        config.map_openai_params.return_value = {"temperature": 0.7}

        result = ResponsesAPIRequestUtils.get_optional_params_responses_api(
            model="gpt-4o",
            responses_api_provider_config=config,
            response_api_optional_params=ResponsesAPIOptionalRequestParams(
                {"temperature": 0.7, "service_tier": "priority"}
            ),
            drop_params=True,
        )

        assert config.map_openai_params.call_args.kwargs["drop_params"] is True
        assert result == {"temperature": 0.7}

    @pytest.mark.parametrize("request_drop_params", [None, False])
    def test_get_optional_params_responses_api_still_raises_without_drop(self, monkeypatch, request_drop_params):
        """Absent or False request-level drop_params must not suppress the unsupported-param error"""
        monkeypatch.setattr(litellm, "drop_params", False)
        config = OpenAIResponsesAPIConfig()

        with pytest.raises(litellm.UnsupportedParamsError):
            ResponsesAPIRequestUtils.get_optional_params_responses_api(
                model="gpt-4o",
                responses_api_provider_config=config,
                response_api_optional_params=ResponsesAPIOptionalRequestParams(
                    {"temperature": 0.7, "unsupported_param": "value"}
                ),
                drop_params=request_drop_params,
            )

    def test_get_requested_response_api_optional_param(self):
        """Test filtering parameters to only include those in ResponsesAPIOptionalRequestParams"""
        # Setup
        params = {
            "temperature": 0.7,
            "max_output_tokens": 100,
            "prompt": {"id": "pmpt_456"},
            "invalid_param": "value",
            "model": "gpt-4o",  # This is not in ResponsesAPIOptionalRequestParams
        }

        # Execute
        result = ResponsesAPIRequestUtils.get_requested_response_api_optional_param(params)

        # Assert
        assert "temperature" in result
        assert "max_output_tokens" in result
        assert "invalid_param" not in result
        assert "model" not in result
        assert result["temperature"] == 0.7
        assert result["max_output_tokens"] == 100
        assert result["prompt"] == {"id": "pmpt_456"}

    def test_decode_previous_response_id_to_original_previous_response_id(self):
        """Test decoding a LiteLLM encoded previous_response_id to the original previous_response_id"""
        # Setup
        test_provider = "openai"
        test_model_id = "gpt-4o"
        original_response_id = "resp_abc123"

        # Use the helper method to build an encoded response ID
        encoded_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            custom_llm_provider=test_provider,
            model_id=test_model_id,
            response_id=original_response_id,
        )

        # Execute
        result = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(encoded_id)

        # Assert
        assert result == original_response_id

        # Test with a non-encoded ID
        plain_id = "resp_xyz789"
        result_plain = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(plain_id)
        assert result_plain == plain_id

    def test_update_responses_api_response_id_with_model_id_handles_dict(self):
        """Ensure _update_responses_api_response_id_with_model_id works with dict input"""
        responses_api_response = {"id": "resp_abc123"}
        litellm_metadata = {"model_info": {"id": "gpt-4o"}}
        updated = ResponsesAPIRequestUtils._update_responses_api_response_id_with_model_id(
            responses_api_response=responses_api_response,
            custom_llm_provider="openai",
            litellm_metadata=litellm_metadata,
        )
        assert updated["id"] != "resp_abc123"
        decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(updated["id"])
        assert decoded.get("response_id") == "resp_abc123"
        assert decoded.get("model_id") == "gpt-4o"
        assert decoded.get("custom_llm_provider") == "openai"

    def test_update_responses_api_response_id_with_model_id_is_idempotent_for_litellm_ids(self):
        raw = "resp_" + "a" * 48
        litellm_metadata = {"model_info": {"id": "model-123"}}

        once = ResponsesAPIRequestUtils._update_responses_api_response_id_with_model_id(
            {"id": raw},
            custom_llm_provider="openai",
            litellm_metadata=litellm_metadata,
        )
        twice = ResponsesAPIRequestUtils._update_responses_api_response_id_with_model_id(
            {"id": once["id"]},
            custom_llm_provider="openai",
            litellm_metadata=litellm_metadata,
        )

        assert twice == once
        assert ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(twice["id"]) == raw
        assert ResponsesAPIRequestUtils._decode_responses_api_response_id(once["id"]).get("response_id") == raw

    def test_build_decode_container_id_omits_none_model_id(self):
        """model_id=None must not round-trip as the truthy string 'None'."""
        encoded = ResponsesAPIRequestUtils._build_container_id(
            custom_llm_provider="azure",
            model_id=None,
            container_id="cntr_upstream_abc",
        )
        assert "None" not in base64.b64decode(encoded.replace("cntr_", "").encode("utf-8")).decode("utf-8")
        decoded = ResponsesAPIRequestUtils._decode_container_id(encoded)
        assert decoded.get("custom_llm_provider") == "azure"
        assert decoded.get("model_id") is None
        assert decoded.get("response_id") == "cntr_upstream_abc"

    def test_decode_container_id_legacy_literal_none_model_id(self):
        """IDs encoded before the None fix should decode without a bogus model_id."""
        legacy_inner = "litellm:custom_llm_provider:azure;model_id:None;container_id:cntr_x"
        legacy_id = "cntr_" + base64.b64encode(legacy_inner.encode("utf-8")).decode("utf-8")
        decoded = ResponsesAPIRequestUtils._decode_container_id(legacy_id)
        assert decoded.get("model_id") is None
        assert decoded.get("custom_llm_provider") == "azure"
        assert decoded.get("response_id") == "cntr_x"


class TestResponseAPILoggingUtils:
    def test_is_response_api_usage_true(self):
        """Test identification of Response API usage format"""
        # Setup
        usage = {"input_tokens": 10, "output_tokens": 20}

        # Execute
        result = ResponseAPILoggingUtils._is_response_api_usage(usage)

        # Assert
        assert result is True

    def test_is_response_api_usage_false(self):
        """Test identification of non-Response API usage format"""
        # Setup
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

        # Execute
        result = ResponseAPILoggingUtils._is_response_api_usage(usage)

        # Assert
        assert result is False

    def test_transform_response_api_usage_to_chat_usage(self):
        """Test transformation from Response API usage to Chat usage format"""
        # Setup
        usage = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 5},
        }

        # Execute
        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(usage)

        # Assert
        assert isinstance(result, Usage)
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30
        assert result.prompt_tokens_details and result.prompt_tokens_details.cached_tokens == 2

    def test_transform_response_api_usage_with_none_values(self):
        """Test transformation handles None values properly"""
        # Setup
        usage = {
            "input_tokens": 0,  # Changed from None to 0
            "output_tokens": 20,
            "total_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 5},
        }

        # Execute
        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(usage)

        # Assert
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 20
        assert result.total_tokens == 20

    def test_transform_response_api_usage_calculates_total_from_input_and_output_tokens_if_available(
        self,
    ):
        """Test transformation calculates total_tokens when it's None and input / output tokens are present"""
        # Setup
        usage = {
            "input_tokens": 15,
            "output_tokens": 25,
            "total_tokens": None,
        }

        # Execute
        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(usage)

        # Assert
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 25
        assert result.total_tokens == 40  # 15 + 25

    def test_transform_response_api_usage_with_image_tokens(self):
        """Test transformation handles image_tokens from image generation responses.

        Note: _transform_response_api_usage_to_chat_usage() is used by multiple
        endpoints including /images/generations and Response API (/responses),
        both of which use the input_tokens/output_tokens format.

        This tests the fix for image generation responses that include image_tokens
        in both input_tokens_details and output_tokens_details.

        Example from gpt-image-1.5:
        - input: text prompt with 13 tokens
        - output: generated image with 272 image tokens + 100 text tokens
        """
        # Setup - simulating image generation usage from OpenAI
        usage = {
            "input_tokens": 13,
            "output_tokens": 372,
            "total_tokens": 385,
            "input_tokens_details": {
                "image_tokens": 0,
                "text_tokens": 13,
            },
            "output_tokens_details": {
                "image_tokens": 272,
                "text_tokens": 100,
            },
        }

        # Execute
        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(usage)

        # Assert - verify basic token counts
        assert isinstance(result, Usage)
        assert result.prompt_tokens == 13
        assert result.completion_tokens == 372
        assert result.total_tokens == 385

        # Assert - verify prompt_tokens_details includes image_tokens and text_tokens
        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.image_tokens == 0
        assert result.prompt_tokens_details.text_tokens == 13

        # Assert - verify completion_tokens_details includes image_tokens and text_tokens
        assert result.completion_tokens_details is not None
        assert result.completion_tokens_details.image_tokens == 272
        assert result.completion_tokens_details.text_tokens == 100

    def test_transform_response_api_usage_maps_cache_write_tokens(self):
        """Responses API (/v1/responses) cache-write tokens must survive the usage transform.

        gpt-5.6 returns usage.input_tokens_details.cache_write_tokens (an extra field
        not typed on InputTokensDetails). Before the fix the transform rebuilt the token
        details and dropped it, leaving the cache-creation metric empty (LIT-4633).
        """
        usage = {
            "input_tokens": 10062,
            "output_tokens": 16,
            "total_tokens": 10078,
            "input_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 10059,
            },
        }

        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(usage)

        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.cache_write_tokens == 10059
        assert result.prompt_tokens_details.cache_creation_tokens == 10059
        assert result.prompt_tokens_details.cached_tokens == 0

    def test_transform_response_api_usage_mixed_details(self):
        """Test transformation handles mixed token details (cached + image + audio)."""
        # Setup - hypothetical usage with mixed token types
        usage = {
            "input_tokens": 100,
            "output_tokens": 200,
            "total_tokens": 300,
            "input_tokens_details": {
                "cached_tokens": 50,
                "audio_tokens": 10,
                "image_tokens": 20,
                "text_tokens": 20,
            },
            "output_tokens_details": {
                "reasoning_tokens": 30,
                "image_tokens": 100,
                "text_tokens": 50,
                "audio_tokens": 20,
            },
        }

        # Execute
        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(usage)

        # Assert - all token detail types should be preserved
        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.cached_tokens == 50
        assert result.prompt_tokens_details.audio_tokens == 10
        assert result.prompt_tokens_details.image_tokens == 20
        assert result.prompt_tokens_details.text_tokens == 20

        assert result.completion_tokens_details is not None
        assert result.completion_tokens_details.reasoning_tokens == 30
        assert result.completion_tokens_details.image_tokens == 100
        assert result.completion_tokens_details.text_tokens == 50
        assert result.completion_tokens_details.audio_tokens == 20

    def test_transform_response_api_usage_with_realtime_keys(self):
        """Realtime input_token_details / output_token_details normalize for Usage."""
        usage = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_token_details": {
                "text_tokens": 8,
                "audio_tokens": 2,
                "cached_tokens": 0,
            },
            "output_token_details": {
                "text_tokens": 12,
                "audio_tokens": 8,
            },
        }

        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(usage)

        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.text_tokens == 8
        assert result.prompt_tokens_details.audio_tokens == 2

        assert result.completion_tokens_details is not None
        assert result.completion_tokens_details.text_tokens == 12
        assert result.completion_tokens_details.audio_tokens == 8

    def test_transform_response_api_usage_tokens_details_keep_values(self):
        """Keeps input_tokens_details / output_tokens_details when singular keys are also present."""
        usage = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_tokens_details": {"text_tokens": 10},
            "output_tokens_details": {"text_tokens": 20},
            "input_token_details": {"text_tokens": 1, "audio_tokens": 99},
            "output_token_details": {"text_tokens": 2, "audio_tokens": 98},
        }

        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(usage)

        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.text_tokens == 10
        assert result.prompt_tokens_details.audio_tokens is None

        assert result.completion_tokens_details is not None
        assert result.completion_tokens_details.text_tokens == 20
        assert result.completion_tokens_details.audio_tokens is None

    def test_transform_response_api_usage_carries_extra_provider_fields(self):
        """Non-standard usage fields (e.g. xAI tool details) must survive chat normalization."""
        details = {"web_search_calls": 2, "x_search_calls": 0}
        usage = ResponseAPIUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            server_side_tool_usage_details=details,
        )

        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(usage)

        assert isinstance(result, Usage)
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 20
        assert getattr(result, "server_side_tool_usage_details") == details

    def test_transform_response_api_usage_ignores_chat_shaped_extras(self):
        """Gemini image usage carries chat-shaped keys as extras; they must not collide with explicit kwargs."""
        usage = ResponseAPIUsage(
            input_tokens=35,
            output_tokens=1716,
            total_tokens=1751,
            prompt_tokens=35,
            prompt_tokens_details={"image_tokens": 5, "text_tokens": 30},
            completion_tokens=1716,
            completion_tokens_details={"image_tokens": 1120, "text_tokens": 596},
            server_side_tool_usage_details={"web_search_calls": 1},
        )

        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(usage)

        assert result.prompt_tokens == 35
        assert result.completion_tokens == 1716
        assert getattr(result, "server_side_tool_usage_details") == {"web_search_calls": 1}

    def test_transform_already_chat_usage_passthrough_keeps_tool_details(self):
        """Re-running the bridge on an already-converted chat Usage must not drop fields."""
        details = {"web_search_calls": 2, "x_search_calls": 0}
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details={"web_search_requests": 2},
        )
        setattr(usage, "server_side_tool_usage_details", details)

        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(usage)

        assert result is usage
        assert getattr(result, "server_side_tool_usage_details") == details
        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.web_search_requests == 2

    def test_transform_chat_shaped_usage_dict_keeps_tool_details(self):
        """Streaming chat bridge dumps already-converted Usage as a prompt_tokens dict."""
        details = {
            "web_search_calls": 3,
            "x_search_calls": 0,
            "code_interpreter_calls": 0,
            "file_search_calls": 0,
            "mcp_calls": 0,
            "document_search_calls": 0,
            "image_generation_calls": 0,
        }
        usage = {
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_tokens": 60,
            "prompt_tokens_details": {"web_search_requests": 3, "cached_tokens": 8},
            "completion_tokens_details": {"reasoning_tokens": 4},
            "server_side_tool_usage_details": details,
        }

        result = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(usage)

        assert isinstance(result, Usage)
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 10
        assert result.total_tokens == 60
        assert getattr(result, "server_side_tool_usage_details") == details
        assert result.prompt_tokens_details is not None
        assert result.prompt_tokens_details.web_search_requests == 3
        assert result.prompt_tokens_details.cached_tokens == 8
        assert result.completion_tokens_details is not None
        assert result.completion_tokens_details.reasoning_tokens == 4


class TestResponsesAPIProviderSpecificParams:
    """
    Tests for fix #19782: provider-specific params (aws_*, vertex_*) should work
    without explicitly passing custom_llm_provider.
    """

    def test_provider_specific_params_no_crash_with_bedrock(self):
        """Test that processing aws_* params with bedrock provider doesn't crash."""
        params = {
            "temperature": 0.7,
            "custom_llm_provider": "bedrock",
            "kwargs": {"aws_region_name": "eu-central-1"},
        }

        # Should not raise any exception
        result = ResponsesAPIRequestUtils.get_requested_response_api_optional_param(params)
        assert "temperature" in result

    def test_provider_specific_params_no_crash_with_openai(self):
        """Test that processing aws_* params with openai provider doesn't crash."""
        params = {
            "temperature": 0.7,
            "custom_llm_provider": "openai",
            "kwargs": {"aws_region_name": "eu-central-1"},
        }

        # Should not raise any exception
        result = ResponsesAPIRequestUtils.get_requested_response_api_optional_param(params)
        assert "temperature" in result

    def test_provider_specific_params_no_crash_with_vertex_ai(self):
        """Test that processing vertex_* params with vertex_ai provider doesn't crash."""
        params = {
            "temperature": 0.7,
            "custom_llm_provider": "vertex_ai",
            "kwargs": {"vertex_project": "my-project"},
        }

        # Should not raise any exception
        result = ResponsesAPIRequestUtils.get_requested_response_api_optional_param(params)
        assert "temperature" in result


def test_responses_extra_body_forwarded_to_completion_transformation_handler():
    """
    Regression test: extra_body must be forwarded to response_api_handler
    when responses_api_provider_config is None (completion transformation path).

    Before the fix, extra_body was a named parameter of responses() but was
    not passed to litellm_completion_transformation_handler.response_api_handler(),
    so it was silently dropped.
    """
    with (
        patch(
            "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config",
            return_value=None,
        ),
        patch(
            "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler",
        ) as mock_handler,
    ):
        mock_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/gpt-4o",
            input="Hello",
            extra_body={"custom_key": "custom_value"},
        )

        mock_handler.assert_called_once()
        call_kwargs = mock_handler.call_args
        # extra_body can be a positional or keyword arg; check both
        assert call_kwargs.kwargs.get("extra_body") == {"custom_key": "custom_value"}


def test_responses_maps_reasoning_effort_from_litellm_params_to_reasoning():
    """
    Test that when reasoning_effort is passed in kwargs (e.g. from proxy litellm_params)
    and reasoning is None, it is mapped to reasoning before the request.

    Supports per-model reasoning_effort/summary config in proxy for clients like Open WebUI
    that cannot set extra_body.
    """
    with (
        patch(
            "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config",
            return_value=None,
        ),
        patch(
            "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler",
        ) as mock_handler,
    ):
        mock_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/gpt-4o",
            input="Hello",
            reasoning_effort={"effort": "high", "summary": "detailed"},
        )

        mock_handler.assert_called_once()
        call_kwargs = mock_handler.call_args
        responses_api_request = call_kwargs.kwargs.get("responses_api_request", {})
        assert "reasoning" in responses_api_request
        assert responses_api_request["reasoning"] == {
            "effort": "high",
            "summary": "detailed",
        }


class TestMergePromptManagementInputReshape:
    """Chat-shaped text parts produced by prompt management hooks become input_text parts (#37509)."""

    EXPLICIT = {"mode": "explicit"}

    def _run_cache_hook(self, client_input, points, model="openai/gpt-5.6"):
        from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook

        _, merged, _ = AnthropicCacheControlHook().get_chat_completion_prompt(
            model=model,
            messages=client_input,
            non_default_params={"cache_control_injection_points": points},
            prompt_id=None,
            prompt_variables=None,
            dynamic_callback_params={},
        )
        return merged

    def test_string_system_item_becomes_input_text_with_marker(self):
        original_input = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"}]
        merged = self._run_cache_hook(list(original_input), [{"location": "message", "role": "system"}])

        result = ResponsesAPIRequestUtils.merge_prompt_management_input(
            original_input=original_input, client_input=list(original_input), merged_input=merged
        )

        assert result[0]["content"] == [
            {"type": "input_text", "text": "You are terse.", "prompt_cache_breakpoint": self.EXPLICIT}
        ]
        assert result[1] == {"role": "user", "content": "hi"}

    def test_reshape_returns_copies_and_leaves_hook_output_untouched(self):
        user_part = {"type": "text", "text": "follow-up"}
        user_message = {"role": "user", "content": [user_part]}
        merged = [user_message]

        result = ResponsesAPIRequestUtils.merge_prompt_management_input(
            original_input="ignored", client_input=[], merged_input=merged
        )

        assert result == [{"role": "user", "content": [{"type": "input_text", "text": "follow-up"}]}]
        assert user_part == {"type": "text", "text": "follow-up"}
        assert user_message == {"role": "user", "content": [user_part]}
        assert result[0] is not user_message

    def test_reshape_keeps_non_message_items_when_hook_returns_client_objects(self):
        user_message = {"role": "user", "content": [{"type": "text", "text": "question"}]}
        reference = {"type": "item_reference", "id": "msg_123"}
        original_input = [reference, user_message]

        result = ResponsesAPIRequestUtils.merge_prompt_management_input(
            original_input=original_input, client_input=[user_message], merged_input=[user_message]
        )

        assert result == [reference, {"role": "user", "content": [{"type": "input_text", "text": "question"}]}]
        assert result[0] is reference
        assert user_message["content"] == [{"type": "text", "text": "question"}]

    def test_assistant_text_parts_are_left_alone(self):
        merged = [
            {"role": "assistant", "content": [{"type": "text", "text": "earlier answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "follow-up"}]},
        ]

        result = ResponsesAPIRequestUtils.merge_prompt_management_input(
            original_input="ignored", client_input=[], merged_input=merged
        )

        assert result[0]["content"] == [{"type": "text", "text": "earlier answer"}]
        assert result[1]["content"] == [{"type": "input_text", "text": "follow-up"}]

    def test_parts_already_in_responses_shape_are_unchanged(self):
        merged = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "a", "prompt_cache_breakpoint": self.EXPLICIT},
                    {"type": "input_image", "image_url": "https://example.com/a.png"},
                ],
            }
        ]

        result = ResponsesAPIRequestUtils.merge_prompt_management_input(
            original_input="ignored", client_input=[], merged_input=merged
        )

        assert result == merged


class TestResponsesInputToChatMessages:
    def test_none_input_returns_empty_list(self):
        assert ResponsesAPIRequestUtils.responses_input_to_chat_messages(None) == []

    def test_str_input_becomes_user_message(self):
        assert ResponsesAPIRequestUtils.responses_input_to_chat_messages("hi") == [
            {"role": "user", "content": "hi"}
        ]

    def test_list_input_keeps_only_role_items(self):
        reasoning_item = {"type": "reasoning", "id": "rs_1", "summary": []}
        user_message = {"role": "user", "content": "hi"}
        assert ResponsesAPIRequestUtils.responses_input_to_chat_messages(
            [reasoning_item, user_message, "stray"]
        ) == [user_message]
