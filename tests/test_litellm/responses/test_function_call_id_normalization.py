import pytest

from litellm.responses.utils import ResponsesAPIRequestUtils


class TestNormalizeCallIdForProvider:
    def test_strips_thought_signature_for_non_gemini_target(self):
        """Gemini thought signature should be stripped when replaying to a non-Gemini model"""
        call_id = "call_23299__thought__AY89a1abc123"
        result = ResponsesAPIRequestUtils._normalize_call_id_for_provider(
            call_id=call_id,
            model="gpt-4.1",
            custom_llm_provider="openai",
        )
        assert "__thought__" not in result
        assert result == "call_23299"

    def test_preserves_thought_signature_for_gemini_target(self):
        """Gemini thought signature must be preserved when target is also Gemini"""
        call_id = "call_23299__thought__AY89a1abc123"
        result = ResponsesAPIRequestUtils._normalize_call_id_for_provider(
            call_id=call_id,
            model="gemini-3.5-flash",
            custom_llm_provider="gemini",
        )
        assert result == call_id

    def test_no_thought_signature_is_unaffected(self):
        """A call_id with no thought signature should pass through unchanged"""
        call_id = "call_plainid123"
        result = ResponsesAPIRequestUtils._normalize_call_id_for_provider(
            call_id=call_id,
            model="gpt-4.1",
            custom_llm_provider="openai",
        )
        assert result == call_id


class TestNormalizeFunctionCallItemIdForProvider:
    @pytest.mark.parametrize(
        "raw_id,expected",
        [
            ("call_abc123", "fc_abc123"),
            ("tooluse_abc123", "fc_abc123"),
            ("toolu_vrtx_abc123", "fc_abc123"),
        ],
    )
    def test_rewrites_foreign_prefixes_to_fc_for_openai(self, raw_id, expected):
        result = ResponsesAPIRequestUtils._normalize_function_call_item_id_for_provider(
            item_id=raw_id,
            model="gpt-4.1",
            custom_llm_provider="openai",
        )
        assert result == expected

    def test_leaves_prefix_unchanged_for_non_openai_target(self):
        """Only rewrite to fc_ when the target provider is openai"""
        raw_id = "call_abc123"
        result = ResponsesAPIRequestUtils._normalize_function_call_item_id_for_provider(
            item_id=raw_id,
            model="claude-sonnet-4",
            custom_llm_provider="anthropic",
        )
        assert result == raw_id

    def test_strips_thought_signature_and_rewrites_prefix_together(self):
        """Gemini id with embedded thought signature replayed to OpenAI: both fixes apply"""
        raw_id = "call_23299__thought__AY89a1abc123"
        result = ResponsesAPIRequestUtils._normalize_function_call_item_id_for_provider(
            item_id=raw_id,
            model="gpt-4.1",
            custom_llm_provider="openai",
        )
        assert result == "fc_23299"

    def test_already_fc_prefixed_id_is_unchanged(self):
        raw_id = "fc_already_correct"
        result = ResponsesAPIRequestUtils._normalize_function_call_item_id_for_provider(
            item_id=raw_id,
            model="gpt-4.1",
            custom_llm_provider="openai",
        )
        assert result == raw_id


class TestNormalizeFunctionCallIdsInInput:
    def test_non_list_input_passes_through_unchanged(self):
        assert (
            ResponsesAPIRequestUtils._normalize_function_call_ids_in_input(
                request_input="not a list",
                model="gpt-4.1",
                custom_llm_provider="openai",
            )
            == "not a list"
        )

    def test_normalizes_function_call_and_function_call_output_items(self):
        """A Gemini function_call replayed with its matching function_call_output to OpenAI"""
        gemini_call_id = "call_23299__thought__AY89a1abc123"
        request_input = [
            {"role": "user", "content": "What is the weather in Tokyo?"},
            {
                "type": "function_call",
                "id": gemini_call_id,
                "call_id": gemini_call_id,
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}',
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": gemini_call_id,
                "output": '{"temperature": "18C"}',
            },
        ]

        result = ResponsesAPIRequestUtils._normalize_function_call_ids_in_input(
            request_input=request_input,
            model="gpt-4.1",
            custom_llm_provider="openai",
        )

        function_call_item = result[1]
        function_call_output_item = result[2]

        assert function_call_item["id"] == "fc_23299"
        assert function_call_item["call_id"] == "call_23299"
        assert function_call_output_item["call_id"] == "call_23299"
        # non function-call items must be left alone
        assert result[0] == {"role": "user", "content": "What is the weather in Tokyo?"}

    def test_items_missing_ids_are_skipped_without_error(self):
        request_input = [
            {"type": "function_call", "name": "get_weather", "arguments": "{}"},
            {"type": "function_call_output", "output": "done"},
        ]
        result = ResponsesAPIRequestUtils._normalize_function_call_ids_in_input(
            request_input=request_input,
            model="gpt-4.1",
            custom_llm_provider="openai",
        )
        assert result == request_input

    def test_non_dict_item_in_list_passes_through_unchanged(self):
        """A non-dict item in the input list must be preserved as-is"""
        request_input = [
            {"role": "user", "content": "hello"},
            "some non-dict item",
            {
                "type": "function_call",
                "id": "call_abc",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": "{}",
            },
        ]
        result = ResponsesAPIRequestUtils._normalize_function_call_ids_in_input(
            request_input=request_input,
            model="gpt-4.1",
            custom_llm_provider="openai",
        )
        assert result[1] == "some non-dict item"


class TestNormalizeFunctionCallIdsIntegration:
    def test_gemini_function_call_replayed_to_openai_via_responses_api(self):
        """
        Integration-style check for the exact repro in the linked issue:
        a Gemini function_call item with an embedded thought signature, replayed
        to an OpenAI-compatible model via /v1/responses, must arrive at the
        chat-completions bridge with a clean fc_-prefixed id and no thought signature.
        """
        gemini_call_id = "call_23299__thought__AY89a1abc123"
        request_input = [
            {
                "type": "function_call",
                "id": gemini_call_id,
                "call_id": gemini_call_id,
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}',
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": gemini_call_id,
                "output": '{"temperature": "18C"}',
            },
        ]

        normalized = ResponsesAPIRequestUtils._normalize_function_call_ids_in_input(
            request_input=request_input,
            model="gpt-4.1",
            custom_llm_provider="openai",
        )

        assert normalized[0]["id"] == "fc_23299"
        assert normalized[0]["call_id"] == "call_23299"
        assert normalized[1]["call_id"] == "call_23299"
        assert "__thought__" not in normalized[0]["id"]
        assert "__thought__" not in normalized[0]["call_id"]
        assert "__thought__" not in normalized[1]["call_id"]
