"""Tests for the shared format mapping between Responses API and Chat Completions."""

import pytest

from litellm.responses.format_mapping import (
    FINISH_REASON_TO_STATUS,
    STATUS_TO_FINISH_REASON,
    chat_usage_to_response_api_usage,
    finish_reason_to_status,
    normalize_provider_specific_fields,
    response_api_usage_to_chat_usage,
    response_format_to_text_format,
    status_to_finish_reason,
    text_format_to_response_format,
)
from litellm.types.llms.openai import ResponseAPIUsage
from litellm.types.utils import ModelResponse, Usage


class TestStatusToFinishReason:
    def test_completed(self):
        assert status_to_finish_reason("completed") == "stop"

    def test_incomplete(self):
        assert status_to_finish_reason("incomplete") == "length"

    def test_failed(self):
        assert status_to_finish_reason("failed") == "stop"

    def test_cancelled(self):
        assert status_to_finish_reason("cancelled") == "stop"

    def test_none(self):
        assert status_to_finish_reason(None) == "stop"

    def test_empty_string(self):
        assert status_to_finish_reason("") == "stop"

    def test_unknown(self):
        assert status_to_finish_reason("some_future_status") == "stop"


class TestFinishReasonToStatus:
    def test_stop(self):
        assert finish_reason_to_status("stop") == "completed"

    def test_tool_calls(self):
        assert finish_reason_to_status("tool_calls") == "completed"

    def test_function_call(self):
        assert finish_reason_to_status("function_call") == "completed"

    def test_length(self):
        assert finish_reason_to_status("length") == "incomplete"

    def test_content_filter(self):
        assert finish_reason_to_status("content_filter") == "incomplete"

    def test_none(self):
        assert finish_reason_to_status(None) == "completed"

    def test_unknown(self):
        assert finish_reason_to_status("some_future_reason") == "completed"


class TestDictsAreConsistent:
    """Verify the two dicts agree on the mappings they share."""

    def test_completed_roundtrips(self):
        assert FINISH_REASON_TO_STATUS[STATUS_TO_FINISH_REASON["completed"]] == "completed"

    def test_incomplete_roundtrips(self):
        assert FINISH_REASON_TO_STATUS[STATUS_TO_FINISH_REASON["incomplete"]] == "incomplete"

    def test_stop_roundtrips(self):
        assert STATUS_TO_FINISH_REASON[FINISH_REASON_TO_STATUS["stop"]] == "stop"

    def test_length_roundtrips(self):
        assert STATUS_TO_FINISH_REASON[FINISH_REASON_TO_STATUS["length"]] == "length"


class TestResponseFormatToTextFormat:
    def test_json_schema(self):
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "my_schema",
                "schema": {"type": "object"},
                "strict": True,
            },
        }
        result = response_format_to_text_format(response_format)
        assert result == {
            "format": {
                "type": "json_schema",
                "name": "my_schema",
                "schema": {"type": "object"},
                "strict": True,
            }
        }

    def test_json_object(self):
        result = response_format_to_text_format({"type": "json_object"})
        assert result == {"format": {"type": "json_object"}}

    def test_text(self):
        result = response_format_to_text_format({"type": "text"})
        assert result == {"format": {"type": "text"}}

    def test_none(self):
        assert response_format_to_text_format(None) is None

    def test_empty_dict(self):
        assert response_format_to_text_format({}) is None

    def test_unknown_type(self):
        assert response_format_to_text_format({"type": "unknown"}) is None

    def test_json_schema_defaults(self):
        result = response_format_to_text_format({"type": "json_schema"})
        assert result["format"]["name"] == "response_schema"
        assert result["format"]["schema"] == {}
        assert result["format"]["strict"] is False


class TestTextFormatToResponseFormat:
    def test_json_schema(self):
        text_param = {
            "format": {
                "type": "json_schema",
                "name": "my_schema",
                "schema": {"type": "object"},
                "strict": True,
            }
        }
        result = text_format_to_response_format(text_param)
        assert result == {
            "type": "json_schema",
            "json_schema": {
                "name": "my_schema",
                "schema": {"type": "object"},
                "strict": True,
            },
        }

    def test_json_object(self):
        result = text_format_to_response_format({"format": {"type": "json_object"}})
        assert result == {"type": "json_object"}

    def test_text_returns_none(self):
        """text is the implicit default in CC, so returns None."""
        result = text_format_to_response_format({"format": {"type": "text"}})
        assert result is None

    def test_none(self):
        assert text_format_to_response_format(None) is None

    def test_empty_dict(self):
        assert text_format_to_response_format({}) is None

    def test_json_schema_defaults(self):
        result = text_format_to_response_format(
            {"format": {"type": "json_schema"}}
        )
        assert result["json_schema"]["name"] == "response_schema"
        assert result["json_schema"]["schema"] == {}
        assert result["json_schema"]["strict"] is False


class TestFormatRoundtrip:
    """Verify json_schema and json_object survive a roundtrip."""

    def test_json_schema_cc_to_responses_to_cc(self):
        original = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {"type": "object", "properties": {"x": {"type": "int"}}},
                "strict": True,
            },
        }
        responses_fmt = response_format_to_text_format(original)
        back = text_format_to_response_format(responses_fmt)
        assert back == original

    def test_json_object_roundtrip(self):
        original = {"type": "json_object"}
        responses_fmt = response_format_to_text_format(original)
        back = text_format_to_response_format(responses_fmt)
        assert back == original


class TestNormalizeProviderSpecificFields:
    def test_dict_with_field(self):
        obj = {"provider_specific_fields": {"key": "value"}}
        assert normalize_provider_specific_fields(obj) == {"key": "value"}

    def test_dict_without_field(self):
        assert normalize_provider_specific_fields({"other": 1}) is None

    def test_dict_with_none_field(self):
        assert normalize_provider_specific_fields({"provider_specific_fields": None}) is None

    def test_dict_with_empty_dict(self):
        assert normalize_provider_specific_fields({"provider_specific_fields": {}}) is None

    def test_object_with_attr(self):
        class Obj:
            provider_specific_fields = {"key": "value"}
        assert normalize_provider_specific_fields(Obj()) == {"key": "value"}

    def test_object_without_attr(self):
        class Obj:
            pass
        assert normalize_provider_specific_fields(Obj()) is None

    def test_object_with_none_attr(self):
        class Obj:
            provider_specific_fields = None
        assert normalize_provider_specific_fields(Obj()) is None

    def test_none_input(self):
        assert normalize_provider_specific_fields(None) is None


class TestResponseApiUsageToChatUsage:
    def test_none(self):
        result = response_api_usage_to_chat_usage(None)
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0

    def test_basic(self):
        usage = ResponseAPIUsage(input_tokens=10, output_tokens=20, total_tokens=30)
        result = response_api_usage_to_chat_usage(usage)
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30

    def test_from_dict(self):
        result = response_api_usage_to_chat_usage(
            {"input_tokens": 5, "output_tokens": 15, "total_tokens": 20}
        )
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 15

    def test_dict_computes_total_tokens(self):
        result = response_api_usage_to_chat_usage(
            {"input_tokens": 5, "output_tokens": 15}
        )
        assert result.total_tokens == 20

    def test_preserves_cost(self):
        usage = ResponseAPIUsage(input_tokens=10, output_tokens=20, total_tokens=30)
        usage.cost = 0.05  # type: ignore
        result = response_api_usage_to_chat_usage(usage)
        assert getattr(result, "cost") == 0.05


class TestChatUsageToResponseApiUsage:
    def test_from_usage(self):
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        result = chat_usage_to_response_api_usage(usage)
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        assert result.total_tokens == 30

    def test_from_model_response(self):
        resp = ModelResponse()
        resp.usage = Usage(prompt_tokens=5, completion_tokens=15, total_tokens=20)
        result = chat_usage_to_response_api_usage(resp)
        assert result.input_tokens == 5
        assert result.output_tokens == 15

    def test_none_usage_on_model_response(self):
        resp = ModelResponse()
        resp.usage = None  # type: ignore
        result = chat_usage_to_response_api_usage(resp)
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_preserves_cost(self):
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        usage.cost = 0.05  # type: ignore
        result = chat_usage_to_response_api_usage(usage)
        assert getattr(result, "cost") == 0.05


class TestUsageRoundtrip:
    def test_basic_roundtrip_responses_to_cc_to_responses(self):
        original = ResponseAPIUsage(input_tokens=10, output_tokens=20, total_tokens=30)
        cc = response_api_usage_to_chat_usage(original)
        back = chat_usage_to_response_api_usage(cc)
        assert back.input_tokens == 10
        assert back.output_tokens == 20
        assert back.total_tokens == 30

    def test_basic_roundtrip_cc_to_responses_to_cc(self):
        original = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        resp = chat_usage_to_response_api_usage(original)
        back = response_api_usage_to_chat_usage(resp)
        assert back.prompt_tokens == 10
        assert back.completion_tokens == 20
        assert back.total_tokens == 30
