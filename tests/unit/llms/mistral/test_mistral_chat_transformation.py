from typing import Final, List, cast
from unittest.mock import MagicMock, patch

import pytest

from litellm.litellm_core_utils.prompt_templates.common_utils import TOOL_RESULT_IMAGE_BOUNDARY
from litellm.llms.mistral.chat.transformation import (
    MistralChatResponseIterator,
    MistralConfig,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse


@pytest.mark.asyncio
async def test_mistral_chat_transformation():
    mistral_config = MistralConfig()
    result = mistral_config.transform_messages(
        **{
            "messages": [
                {
                    "content": [
                        {"type": "text", "text": "Here is a representation of text"},
                        {
                            "type": "image_url",
                            "image_url": "https://images.pexels.com/photos/13268478/pexels-photo-13268478.jpeg",
                        },
                    ],
                    "role": "user",
                }
            ],
            "model": "mistral-medium-latest",
            "is_async": True,
        }
    )


class TestMistralReasoningSupport:
    """Test suite for Mistral Magistral reasoning functionality."""

    def test_get_supported_openai_params_magistral_model(self):
        """Test that magistral models support reasoning parameters."""
        mistral_config = MistralConfig()

        # Test magistral model supports reasoning parameters
        supported_params = mistral_config.get_supported_openai_params(
            "mistral/magistral-medium-2506"
        )
        assert "reasoning_effort" in supported_params
        assert "thinking" in supported_params

        supported_params_reasoning = mistral_config.get_supported_openai_params(
            "mistral/mistral-medium-latest"
        )
        assert "reasoning_effort" in supported_params_reasoning
        assert "thinking" not in supported_params_reasoning

        supported_params_normal = mistral_config.get_supported_openai_params(
            "mistral/mistral-large-latest"
        )
        assert "reasoning_effort" not in supported_params_normal
        assert "thinking" not in supported_params_normal

    @pytest.mark.parametrize(
        ("model", "requested", "sent"),
        [
            ("mistral-medium-latest", "high", "high"),
            ("mistral-medium-latest", "none", "none"),
            ("mistral-medium-latest", "low", "high"),
            ("mistral-medium-latest", "medium", "high"),
            ("mistral-medium-latest", "xhigh", "high"),
            ("mistral-small-latest", "medium", "high"),
            ("mistral-vibe-cli-latest", "medium", "high"),
            ("magistral-small-latest", "high", "high"),
            ("magistral-small-latest", "none", "none"),
            ("magistral-small-latest", "low", "high"),
            ("magistral-small-latest", "medium", "high"),
            ("magistral-medium-latest", "minimal", "high"),
            ("magistral-medium-latest", "none", "none"),
            ("zai-glm-5", "none", "none"),
            ("zai-glm-5", "minimal", "low"),
            ("zai-glm-5", "medium", "high"),
            ("zai-glm-5", "xhigh", "max"),
            ("zai-glm-5-2", "medium", "medium"),
            ("zai-glm-5-2", "xhigh", "xhigh"),
            ("zai-glm-5-2", "max", "max"),
        ],
    )
    def test_reasoning_effort_is_sent_as_a_level_the_model_accepts(self, local_model_cost_map, model, requested, sent):
        import litellm

        optional_params = litellm.get_optional_params(
            model=model,
            custom_llm_provider="mistral",
            reasoning_effort=requested,
        )
        assert optional_params["reasoning_effort"] == sent

    def test_reasoning_effort_is_forwarded_verbatim_when_the_map_declares_no_levels(
        self, local_model_cost_map, monkeypatch
    ):
        import litellm

        monkeypatch.setitem(
            litellm.model_cost,
            "mistral/undeclared-reasoner",
            {"litellm_provider": "mistral", "mode": "chat", "supports_reasoning": True},
        )
        optional_params = litellm.get_optional_params(
            model="undeclared-reasoner",
            custom_llm_provider="mistral",
            reasoning_effort="medium",
        )
        assert optional_params["reasoning_effort"] == "medium"

    def test_reasoning_effort_stays_unsupported_for_non_reasoning_models(self):
        import litellm

        with pytest.raises(litellm.UnsupportedParamsError):
            litellm.get_optional_params(
                model="codestral-latest",
                custom_llm_provider="mistral",
                reasoning_effort="high",
            )

        dropped = litellm.get_optional_params(
            model="codestral-latest",
            custom_llm_provider="mistral",
            reasoning_effort="high",
            drop_params=True,
        )
        assert "reasoning_effort" not in dropped

    def test_client_metadata_stripped_from_request(self):
        mistral_config = MistralConfig()

        request = mistral_config.transform_request(
            model="mistral-medium-latest",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"client_metadata": {"originator": "codex_cli_rs"}, "temperature": 0.2},
            litellm_params={},
            headers={},
        )

        assert "client_metadata" not in request
        assert request["temperature"] == 0.2

    @pytest.mark.parametrize(
        ("thinking", "sent"),
        [
            ({"type": "enabled", "budget_tokens": 1024}, "high"),
            ({"type": "enabled", "budget_tokens": 100}, "high"),
            ({"type": "disabled"}, "none"),
            ({"type": "adaptive"}, "high"),
            ({"type": "adaptive", "display": "summarized"}, "high"),
        ],
    )
    def test_magistral_sends_thinking_as_a_reasoning_effort_the_model_accepts(
        self, local_model_cost_map, caplog, thinking, sent
    ):
        import litellm

        with caplog.at_level("WARNING", logger="LiteLLM"):
            optional_params: Final = litellm.get_optional_params(
                model="magistral-small-latest",
                custom_llm_provider="mistral",
                thinking=thinking,
            )
        assert optional_params["reasoning_effort"] == sent
        assert "thinking" not in optional_params
        assert "Ignoring thinking.budget_tokens=" not in caplog.text

    def test_magistral_prefers_reasoning_effort_over_thinking(self, local_model_cost_map):
        import litellm

        optional_params: Final = litellm.get_optional_params(
            model="magistral-small-latest",
            custom_llm_provider="mistral",
            reasoning_effort="none",
            thinking={"type": "enabled", "budget_tokens": 4096},
        )
        assert optional_params["reasoning_effort"] == "none"

    @pytest.mark.parametrize(
        ("reasoning_effort", "sent"),
        [
            ({"effort": "medium", "summary": "detailed"}, "high"),
            ({"effort": "none", "summary": "auto"}, "none"),
        ],
    )
    def test_magistral_sends_only_the_level_of_a_reasoning_effort_with_a_summary(
        self, local_model_cost_map, reasoning_effort, sent
    ):
        import litellm

        optional_params: Final = litellm.get_optional_params(
            model="magistral-small-latest",
            custom_llm_provider="mistral",
            reasoning_effort=reasoning_effort,
        )
        assert optional_params["reasoning_effort"] == sent

    @pytest.mark.parametrize("model", ["magistral-small-latest", "zai-glm-5-2"])
    @pytest.mark.parametrize("drop_params", [False, True])
    @pytest.mark.parametrize(
        ("reasoning_effort", "warned"),
        [
            (5, True),
            ({"effort": 5}, True),
            ("", True),
            ("HIGH", True),
            ("default", True),
            ({"effort": "default"}, True),
            ({"effort": "auto"}, True),
            ({"summary": "detailed"}, False),
            ({"effort": None, "summary": "auto"}, False),
        ],
    )
    def test_reasoning_effort_without_a_level_is_sent_as_medium(
        self, local_model_cost_map, caplog, reasoning_effort, warned, drop_params, model
    ):
        import litellm

        medium: Final = litellm.get_optional_params(
            model=model, custom_llm_provider="mistral", reasoning_effort="medium"
        )
        with caplog.at_level("WARNING", logger="LiteLLM"):
            optional_params: Final = litellm.get_optional_params(
                model=model,
                custom_llm_provider="mistral",
                reasoning_effort=reasoning_effort,
                drop_params=drop_params,
            )
        assert optional_params["reasoning_effort"] == medium["reasoning_effort"]
        assert ("Ignoring reasoning_effort=" in caplog.text) is warned

    @pytest.mark.parametrize("model", ["magistral-small-latest", "my-magistral-finetune"])
    @pytest.mark.parametrize("drop_params", [False, True])
    @pytest.mark.parametrize("budget_tokens", ["1024", True, None])
    def test_magistral_non_numeric_thinking_budget_is_read_as_left_out_with_a_warning(
        self, local_model_cost_map, caplog, budget_tokens, drop_params, model
    ):
        import litellm

        left_out: Final = litellm.get_optional_params(
            model=model, custom_llm_provider="mistral", thinking={"type": "enabled"}
        )
        with caplog.at_level("WARNING", logger="LiteLLM"):
            optional_params: Final = litellm.get_optional_params(
                model=model,
                custom_llm_provider="mistral",
                thinking={"type": "enabled", "budget_tokens": budget_tokens},
                drop_params=drop_params,
            )
        assert optional_params["reasoning_effort"] == left_out["reasoning_effort"]
        assert "thinking" not in optional_params
        assert "Ignoring thinking.budget_tokens=" in caplog.text

    def test_magistral_thinking_without_a_type_sends_no_reasoning_effort(self, local_model_cost_map):
        import litellm

        optional_params: Final = litellm.get_optional_params(
            model="magistral-small-latest",
            custom_llm_provider="mistral",
            thinking={"budget_tokens": 4096},
        )
        assert "reasoning_effort" not in optional_params
        assert "thinking" not in optional_params

    @pytest.mark.parametrize(
        "reasoning", [{"reasoning_effort": "medium"}, {"thinking": {"type": "enabled", "budget_tokens": 1024}}]
    )
    def test_magistral_request_carries_reasoning_effort_and_the_callers_messages_only(
        self, local_model_cost_map, reasoning
    ):
        import litellm

        optional_params: Final = litellm.get_optional_params(
            model="magistral-small-latest", custom_llm_provider="mistral", **reasoning
        )

        request: Final = MistralConfig().transform_request(
            model="magistral-small-latest",
            messages=[{"role": "user", "content": "What is 17*23?"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert request["reasoning_effort"] == "high"
        assert request["messages"] == [{"role": "user", "content": "What is 17*23?"}]

    def test_case_insensitive_magistral_detection(self):
        """Test that magistral model detection is case-insensitive."""
        mistral_config = MistralConfig()

        # Test various case combinations
        models_to_test = [
            "mistral/Magistral-medium-2506",
            "mistral/MAGISTRAL-MEDIUM-2506",
            "mistral/magistral-SMALL-2506",
            "MaGiStRaL-medium-2506",
        ]

        for model in models_to_test:
            supported_params = mistral_config.get_supported_openai_params(model)
            assert "reasoning_effort" in supported_params, f"Failed for model: {model}"


def test_mistral_streaming_chunk_preserves_thinking_blocks():
    """Ensure streaming chunks keep magistral reasoning content."""
    iterator = MistralChatResponseIterator(
        streaming_response=iter([]), sync_stream=True, json_mode=False
    )

    streamed_chunk = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 123456,
        "model": "magistral-medium-2509",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": "Working it out."}],
                        },
                        {"type": "text", "text": " Hello"},
                    ],
                },
                "finish_reason": None,
            }
        ],
    }

    parsed_chunk = iterator.chunk_parser(streamed_chunk)

    delta = parsed_chunk.choices[0].delta
    assert delta.thinking_blocks is not None
    assert delta.thinking_blocks[0]["thinking"] == "Working it out."
    assert delta.thinking_blocks[0]["signature"] == "mistral"
    assert delta.reasoning_content == "Working it out."
    assert delta.content == " Hello"


class TestMistralNameHandling:
    """Test suite for Mistral name handling in messages."""

    def test_handle_name_in_message_tool_role_empty_name_removes_name(self):
        """Test that empty name is removed for tool messages."""
        # Test with empty string
        tool_message = {"role": "tool", "content": "Function result", "name": ""}
        result = MistralConfig._handle_name_in_message(tool_message)
        assert "name" not in result
        assert result["role"] == "tool"
        assert result["content"] == "Function result"

    def test_handle_name_in_message_tool_role_valid_name_keeps_name(self):
        """Test that valid name is kept for tool messages."""
        # Test with normal function name
        tool_message = {
            "role": "tool",
            "content": "Function result",
            "name": "get_weather",
        }
        result = MistralConfig._handle_name_in_message(tool_message)
        assert "name" in result
        assert result["name"] == "get_weather"
        assert result["role"] == "tool"
        assert result["content"] == "Function result"

    def test_handle_name_in_message_no_name_field(self):
        """Test that messages without name field are unchanged."""
        # Test with user role
        user_message = {"role": "user", "content": "Hello"}
        result = MistralConfig._handle_name_in_message(user_message)
        assert "name" not in result
        assert result["role"] == "user"
        assert result["content"] == "Hello"


class TestMistralParallelToolCalls:
    """Test suite for Mistral parallel tool calls functionality."""

    def test_get_supported_openai_params_includes_parallel_tool_calls(self):
        """Test that parallel_tool_calls is in supported parameters."""
        mistral_config = MistralConfig()
        supported_params = mistral_config.get_supported_openai_params(
            "mistral/mistral-large-latest"
        )
        assert "parallel_tool_calls" in supported_params

    def test_transform_request_preserves_parallel_tool_calls(self):
        """Test that transform_request preserves parallel_tool_calls parameter."""
        mistral_config = MistralConfig()

        messages = [{"role": "user", "content": "What's the weather like?"}]
        optional_params = {"parallel_tool_calls": True}

        result = mistral_config.transform_request(
            model="mistral/mistral-large-latest",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result.get("parallel_tool_calls") is True
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"


class TestMistralThinkingContentHandling:
    """Test suite for Mistral thinking content response handling functionality."""

    def test_transform_response_with_thinking_content(self):
        """Test that Mistral responses with thinking content are correctly transformed."""
        import json
        from unittest.mock import Mock

        import litellm

        # Raw response from Mistral with thinking content
        raw_response_data = {
            "id": "12a18e1439f24f95b9812a016e0af235",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": [
                                    {
                                        "type": "text",
                                        "text": "Well, the capital of France is a well-known fact. It's Paris. But just to be sure, I recall that Paris is indeed the capital city of France. I don't need to look it up because it's a common knowledge fact. But if I were unsure, I would double-check using a reliable source or a knowledge base. Since I'm confident about this, I can provide the answer directly.",
                                    }
                                ],
                            },
                            {"type": "text", "text": "The capital of France is Paris."},
                        ],
                        "refusal": None,
                        "role": "assistant",
                        "annotations": None,
                        "audio": None,
                        "function_call": None,
                        "tool_calls": None,
                    },
                }
            ],
            "created": 1754654178,
            "model": "magistral-medium-2507",
            "object": "chat.completion",
            "service_tier": None,
            "system_fingerprint": None,
            "usage": {
                "completion_tokens": 93,
                "prompt_tokens": 11,
                "total_tokens": 104,
                "completion_tokens_details": None,
                "prompt_tokens_details": None,
            },
        }

        # Mock httpx response
        mock_response = Mock()
        mock_response.json.return_value = raw_response_data
        mock_response.headers = {}
        mock_response.text = json.dumps(raw_response_data)

        # Mock logging object with proper attributes
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}

        # Test the transformation
        mistral_config = MistralConfig()
        model_response = litellm.ModelResponse()

        # Test transform_response method
        final_response = mistral_config.transform_response(
            model="mistral/magistral-medium-2507",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging_obj,
            request_data={},
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        # Verify the response structure
        assert final_response is not None
        assert len(final_response.choices) == 1
        choice = final_response.choices[0]

        # Verify message content
        message = choice.message
        assert message.role == "assistant"

        # The content should be processed - either as text or as thinking blocks
        # Content could be the text part or the full content list
        content_str = str(message.content) if message.content else ""

        # Verify the actual text content is preserved somewhere
        assert "The capital of France is Paris." in content_str or (
            hasattr(message, "thinking_blocks") and message.thinking_blocks
        )

        # Verify usage information
        assert final_response.usage.completion_tokens == 93
        assert final_response.usage.prompt_tokens == 11
        assert final_response.usage.total_tokens == 104

        # Verify model and metadata
        assert final_response.id == "12a18e1439f24f95b9812a016e0af235"
        assert final_response.created == 1754654178


class TestMistralEmptyContentHandling:
    """Test suite for Mistral empty content response handling functionality."""

    def test_handle_empty_content_response_converts_empty_string_to_none(self):
        """Test that empty string content is converted to None."""
        response_data = {
            "choices": [
                {
                    "message": {"content": "", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ]
        }

        result = MistralConfig._handle_empty_content_response(response_data)

        assert result["choices"][0]["message"]["content"] is None

    def test_handle_empty_content_response_preserves_actual_content(self):
        """Test that actual content is preserved unchanged."""
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": "Hello, how can I help you?",
                        "role": "assistant",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        result = MistralConfig._handle_empty_content_response(response_data)

        assert (
            result["choices"][0]["message"]["content"] == "Hello, how can I help you?"
        )

    def test_handle_empty_content_response_handles_multiple_choices(self):
        """Test that only the first choice is processed for empty content."""
        response_data = {
            "choices": [
                {
                    "message": {"content": "", "role": "assistant"},
                    "finish_reason": "stop",
                },
                {
                    "message": {"content": "", "role": "assistant"},
                    "finish_reason": "stop",
                },
            ]
        }

        result = MistralConfig._handle_empty_content_response(response_data)

        # Only first choice should be converted to None
        assert result["choices"][0]["message"]["content"] is None
        # Second choice should remain as empty string
        assert result["choices"][1]["message"]["content"] is None

    def test_is_empty_assistant_message(self):
        """Test that is_empty_assistant_message returns True for empty assistant message."""
        message = {"role": "assistant", "content": ""}
        assert MistralConfig._is_empty_assistant_message(message) is True

    def test_is_empty_assistant_message_with_content(self):
        """Test that is_empty_assistant_message returns False for assistant message with content."""
        message = {"role": "assistant", "content": "Hello"}
        assert MistralConfig._is_empty_assistant_message(message) is False


class TestMistralFileHandling:
    """Test suite for Mistral file handling functionality."""

    def test_handle_file_message_with_file_id(self):
        """Test that file messages with file_id are handled correctly."""
        mistral_config = MistralConfig()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please review this file."},
                    {"type": "file", "file": {"file_id": "file-12345"}},
                ],
            }
        ]
        casted_message = cast(list[AllMessageValues], messages)
        result = mistral_config._handle_message_with_file(casted_message)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        # Check that content is transformed correctly
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2
        # Check that file type is preserved
        assert result[0]["content"][1]["type"] == "file"
        # Check that file_id is modified to match Mistral's expected format
        assert result[0]["content"][1]["file_id"] == "file-12345"  # type: ignore

    def test_handle_file_message_without_file_id(self):
        """Test that file messages without file_id are ignored."""
        mistral_config = MistralConfig()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Please review this file."}],
            }
        ]
        casted_message = cast(list[AllMessageValues], messages)
        result = mistral_config._handle_message_with_file(casted_message)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 1  # Only text part remains

    def test_handle_message_with_file_multiple_files(self):
        """Test that multiple file messages are handled correctly."""
        mistral_config = MistralConfig()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please review these files."},
                    {"type": "file", "file": {"file_id": "file-12345"}},
                    {"type": "file", "file": {"file_id": "file-67890"}},
                ],
            }
        ]
        casted_message = cast(list[AllMessageValues], messages)
        result = mistral_config._handle_message_with_file(casted_message)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        # Check that content is transformed correctly
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 3  # Text + 2 files
        # Check that file types are preserved
        assert result[0]["content"][1]["type"] == "file"
        assert result[0]["content"][2]["type"] == "file"
        # Check that file_ids are modified to match Mistral's expected format
        assert result[0]["content"][1]["file_id"] == "file-12345"  # type: ignore
        assert result[0]["content"][2]["file_id"] == "file-67890"  # type: ignore


class TestMistralStripsOutputOnlyFields:
    """Mistral rejects unknown input fields with a 422 ``extra_forbidden``.

    LiteLLM attaches ``reasoning_content`` / ``thinking_blocks`` to assistant
    responses, so replaying an assistant turn verbatim must not forward them.
    Regression for https://github.com/BerriAI/litellm/issues/30835.
    """

    def test_assistant_reasoning_content_is_dropped(self):
        messages = cast(
            List[AllMessageValues],
            [
                {"role": "user", "content": "Question?"},
                {
                    "role": "assistant",
                    "content": "Follow-up",
                    "reasoning_content": "Some internal reasoning text.",
                    "thinking_blocks": [{"type": "thinking", "thinking": "step", "signature": "mistral"}],
                },
            ],
        )

        result = cast(
            List[AllMessageValues],
            MistralConfig().transform_messages(messages=messages, model="mistral-medium-3-5"),
        )

        assistant_message = result[-1]
        assert "reasoning_content" not in assistant_message
        assert "thinking_blocks" not in assistant_message
        assert assistant_message["content"] == "Follow-up"
        assert assistant_message["role"] == "assistant"

    def test_tool_search_reply_drops_litellm_response_fields(self):
        tool_search_reply = {
            "role": "assistant",
            "content": "",
            "reasoning_items": [{"type": "reasoning", "id": "rs_1", "summary": []}],
            "provider_specific_fields": {"tool_search_items": [{"type": "tool_search_call", "id": "ts_1"}]},
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    "provider_specific_fields": {"namespace": "weather"},
                }
            ],
        }
        messages = cast(
            List[AllMessageValues],
            [
                {"role": "user", "content": "Weather in Paris?"},
                tool_search_reply,
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ],
        )

        result = cast(
            List[AllMessageValues],
            MistralConfig().transform_messages(messages=messages, model="mistral-medium-3-5"),
        )

        assert result[1] == {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        }

    def test_non_assistant_messages_are_untouched(self):
        messages = cast(
            List[AllMessageValues],
            [{"role": "user", "content": "Question?", "reasoning_content": "noise"}],
        )

        result = cast(
            List[AllMessageValues],
            MistralConfig().transform_messages(messages=messages, model="mistral-medium-3-5"),
        )

        assert result[0].get("reasoning_content") == "noise"

    def test_reasoning_content_dropped_when_image_present(self):
        """The image branch returns early, so stripping must run before it."""
        messages = cast(
            List[AllMessageValues],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/cat.png"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "A cat.",
                    "reasoning_content": "leaked reasoning",
                },
            ],
        )

        with patch.object(
            MistralConfig,
            "_transform_messages_sync",
            side_effect=lambda transformed, model: transformed,
        ):
            result = cast(
                List[AllMessageValues],
                MistralConfig().transform_messages(messages=messages, model="mistral-medium-3-5", is_async=False),
            )

        assert "reasoning_content" not in result[-1]


def test_mistral_transform_request_hoists_tool_message_image():
    """Images inside role:"tool" messages must be moved to a following user
    message (Mistral rejects/ignores non-text tool content), including when
    Mistral's own _transform_messages override takes its image handling path."""
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    messages: List[AllMessageValues] = cast(
        List[AllMessageValues],
        [
            {"role": "user", "content": "read the screenshot"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "image_url", "image_url": {"url": data_uri}}],
            },
        ],
    )

    request = MistralConfig().transform_request(
        model="mistral-medium-2508", messages=messages, optional_params={}, litellm_params={}, headers={}
    )

    result = request["messages"]
    assert [m.get("role") for m in result] == ["user", "assistant", "tool", "user"]
    tool_message = result[2]
    assert tool_message.get("tool_call_id") == "call_1"
    assert isinstance(tool_message.get("content"), str)
    assert result[3].get("content") == [
        {"type": "text", "text": TOOL_RESULT_IMAGE_BOUNDARY},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]
