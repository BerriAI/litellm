import json
import os
import unittest
from collections.abc import Mapping
from litellm._uuid import uuid
from datetime import datetime, timezone
from typing import Final
from unittest.mock import MagicMock, patch

from litellm.integrations.deepeval.api import Endpoints, HttpMethods
from litellm.integrations.deepeval.deepeval import DeepEvalLogger
from litellm.integrations.deepeval.types import SpanApiType, TraceSpanApiStatus

_CHAT_TOOL_CALL: Final = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Boston"}'},
}
_RESPONSES_FUNCTION_CALL: Final = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "get_weather",
    "arguments": '{"city": "Boston"}',
}
_EXPECTED_TOOL_CALL: Final = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Boston"}'},
}


def _chat_response(content: str | None, tool_calls: tuple[Mapping[str, object], ...]) -> Mapping[str, object]:
    return {
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        "choices": [{"message": {"role": "assistant", "content": content, "tool_calls": list(tool_calls)}}],
    }


def _responses_message(*texts: str) -> Mapping[str, object]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []} for text in texts],
    }


class TestDeepEvalLogger(unittest.TestCase):
    @patch.dict(os.environ, {"CONFIDENT_API_KEY": "test-api-key"})
    def setUp(self):
        # Mock the Api class before initializing DeepEvalLogger
        self.api_patcher = patch("litellm.integrations.deepeval.deepeval.Api")
        self.mock_api_class = self.api_patcher.start()
        self.mock_api_instance = MagicMock()
        self.mock_api_class.return_value = self.mock_api_instance

        self.logger = DeepEvalLogger()
        self.start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        self.end_time = datetime(2023, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
        self.mock_response_obj = {"id": "resp_123"}
        self.trace_id = str(uuid.uuid4())
        self.span_id = str(uuid.uuid4())
        self.model = "gpt-3.5-turbo"
        self.input_str = "Hello, world!"

    def tearDown(self):
        self.api_patcher.stop()

    def _common_assertions(
        self, expected_status: TraceSpanApiStatus, expected_output: str
    ):
        self.mock_api_instance.send_request.assert_called_once()
        call_args = self.mock_api_instance.send_request.call_args

        self.assertEqual(call_args.kwargs["method"], HttpMethods.POST)
        self.assertEqual(call_args.kwargs["endpoint"], Endpoints.TRACING_ENDPOINT)

        body = call_args.kwargs["body"]

        self.assertIsInstance(body, dict)
        self.assertEqual(body["uuid"], self.trace_id)
        self.assertIn("startTime", body)
        self.assertIn("endTime", body)

        self.assertIsInstance(body["llmSpans"], list)
        self.assertEqual(len(body["llmSpans"]), 1)

        llm_span = body["llmSpans"][0]
        self.assertEqual(llm_span["uuid"], self.span_id)

        expected_name = (
            "litellm_success_callback"
            if expected_status == TraceSpanApiStatus.SUCCESS
            else "litellm_failure_callback"
        )
        self.assertEqual(llm_span["name"], expected_name)

        self.assertEqual(llm_span["status"], expected_status.value)
        self.assertEqual(llm_span["type"], SpanApiType.LLM.value)
        self.assertEqual(llm_span["traceUuid"], self.trace_id)
        self.assertIn("startTime", llm_span)
        self.assertIn("endTime", llm_span)
        self.assertEqual(llm_span["input"], self.input_str)
        self.assertEqual(llm_span["output"], expected_output)
        self.assertEqual(llm_span["model"], self.model)

        return llm_span

    def test_log_success_event(self):
        kwargs = {
            "input": self.input_str,
            "standard_logging_object": {
                "id": self.span_id,
                "trace_id": self.trace_id,
                "model": self.model,
                "response": {
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                    "choices": [{"message": {"content": "This is a success."}}],
                },
            },
        }

        self.logger.log_success_event(
            kwargs, self.mock_response_obj, self.start_time, self.end_time
        )

        llm_span = self._common_assertions(
            TraceSpanApiStatus.SUCCESS, "This is a success."
        )
        self.assertEqual(llm_span["inputTokenCount"], 10)
        self.assertEqual(llm_span["outputTokenCount"], 20)

    def _sent_span_output(self, response: Mapping[str, object]) -> object:
        """Log ``response`` and return the span output exactly as it goes over the wire (JSON)."""
        kwargs: Final = {
            "input": self.input_str,
            "standard_logging_object": {
                "id": self.span_id,
                "trace_id": self.trace_id,
                "model": self.model,
                "response": response,
            },
        }
        self.logger.log_success_event(kwargs, self.mock_response_obj, self.start_time, self.end_time)
        body: Final = self.mock_api_instance.send_request.call_args.kwargs["body"]
        return json.loads(json.dumps(body))["llmSpans"][0].get("output")

    def test_chat_tool_call_only_records_tool_call(self):
        output: Final = self._sent_span_output(_chat_response(content=None, tool_calls=(_CHAT_TOOL_CALL,)))

        self.assertEqual(output, {"role": "assistant", "content": None, "tool_calls": [_EXPECTED_TOOL_CALL]})

    def test_chat_tool_call_with_empty_string_content_records_tool_call(self):
        output: Final = self._sent_span_output(_chat_response(content="", tool_calls=(_CHAT_TOOL_CALL,)))

        self.assertEqual(output, {"role": "assistant", "content": None, "tool_calls": [_EXPECTED_TOOL_CALL]})

    def test_chat_text_and_tool_call_records_both(self):
        output: Final = self._sent_span_output(
            _chat_response(content="Checking the weather.", tool_calls=(_CHAT_TOOL_CALL,))
        )

        self.assertEqual(
            output,
            {"role": "assistant", "content": "Checking the weather.", "tool_calls": [_EXPECTED_TOOL_CALL]},
        )

    def test_responses_text_joins_parts_and_separates_messages(self):
        output: Final = self._sent_span_output(
            {
                "usage": {},
                "output": [
                    {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking"}]},
                    _responses_message("Let me check. "),
                    _responses_message("Hello ", "from responses"),
                ],
            }
        )

        self.assertEqual(output, "Let me check. \n\nHello from responses")

    def test_responses_function_call_only_records_tool_call(self):
        output: Final = self._sent_span_output({"usage": {}, "output": [_RESPONSES_FUNCTION_CALL]})

        self.assertEqual(output, {"role": "assistant", "content": None, "tool_calls": [_EXPECTED_TOOL_CALL]})

    def test_responses_text_and_function_call_records_both(self):
        output: Final = self._sent_span_output(
            {"usage": {}, "output": [_responses_message("Checking the weather."), _RESPONSES_FUNCTION_CALL]}
        )

        self.assertEqual(
            output,
            {"role": "assistant", "content": "Checking the weather.", "tool_calls": [_EXPECTED_TOOL_CALL]},
        )

    def test_responses_reasoning_only_records_no_output(self):
        output: Final = self._sent_span_output(
            {"usage": {}, "output": [{"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "secret"}]}]}
        )

        self.assertEqual(output, "NO_OUTPUT")

    def test_turn_off_message_logging_redacts_tool_arguments(self):
        self.logger.turn_off_message_logging = True
        chat_output: Final = self._sent_span_output(_chat_response(content=None, tool_calls=(_CHAT_TOOL_CALL,)))
        responses_output: Final = self._sent_span_output({"usage": {}, "output": [_RESPONSES_FUNCTION_CALL]})
        redacted_tool_call: Final = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "redacted-by-litellm"},
        }

        self.assertEqual(chat_output, {"role": "assistant", "content": None, "tool_calls": [redacted_tool_call]})
        self.assertEqual(responses_output, {"role": "assistant", "content": None, "tool_calls": [redacted_tool_call]})

    def test_turn_off_message_logging_redacts_text(self):
        self.logger.turn_off_message_logging = True
        chat_text: Final = self._sent_span_output(_chat_response(content="secret answer", tool_calls=()))
        responses_text: Final = self._sent_span_output({"usage": {}, "output": [_responses_message("secret answer")]})
        chat_mixed: Final = self._sent_span_output(
            _chat_response(content="secret answer", tool_calls=(_CHAT_TOOL_CALL,))
        )
        responses_mixed: Final = self._sent_span_output(
            {"usage": {}, "output": [_responses_message("secret answer"), _RESPONSES_FUNCTION_CALL]}
        )

        self.assertEqual(chat_text, "redacted-by-litellm")
        self.assertEqual(responses_text, "redacted-by-litellm")
        for mixed in (chat_mixed, responses_mixed):
            self.assertEqual(mixed["content"], "redacted-by-litellm")
            self.assertEqual(mixed["tool_calls"][0]["function"]["arguments"], "redacted-by-litellm")

    def test_empty_choices_records_no_output(self):
        output: Final = self._sent_span_output({"usage": {}, "choices": []})

        self.assertEqual(output, "NO_OUTPUT")

    def test_log_failure_event(self):
        error_message = "This is an error."
        kwargs = {
            "input": self.input_str,
            "standard_logging_object": {
                "id": self.span_id,
                "trace_id": self.trace_id,
                "model": self.model,
                "error_string": error_message,
                "response": {},
            },
        }

        self.logger.log_failure_event(
            kwargs, self.mock_response_obj, self.start_time, self.end_time
        )

        llm_span = self._common_assertions(TraceSpanApiStatus.ERRORED, error_message)
        self.assertIsNone(llm_span.get("inputTokenCount"))
        self.assertIsNone(llm_span.get("outputTokenCount"))


if __name__ == "__main__":
    unittest.main()
