import os
import unittest
from litellm._uuid import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from litellm.integrations.deepeval.api import Endpoints, HttpMethods
from litellm.integrations.deepeval.deepeval import DeepEvalLogger
from litellm.integrations.deepeval.types import SpanApiType, TraceSpanApiStatus


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

    def _log_success_with_response(self, response: dict) -> dict:
        kwargs = {
            "input": self.input_str,
            "standard_logging_object": {
                "id": self.span_id,
                "trace_id": self.trace_id,
                "model": self.model,
                "response": response,
            },
        }
        self.logger.log_success_event(kwargs, self.mock_response_obj, self.start_time, self.end_time)
        return self.mock_api_instance.send_request.call_args.kwargs["body"]["llmSpans"][0]

    def test_log_success_event_tool_call_only_records_tool_calls(self):
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Boston"}'},
            }
        ]
        llm_span = self._log_success_with_response(
            {
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
                "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": tool_calls}}],
            }
        )

        self.assertEqual(llm_span["output"], tool_calls)

    def test_log_success_event_tool_call_with_empty_content_string(self):
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Boston"}'},
            }
        ]
        llm_span = self._log_success_with_response(
            {
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
                "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}}],
            }
        )

        self.assertEqual(llm_span["output"], tool_calls)

    def test_log_success_event_responses_api_records_output_text(self):
        llm_span = self._log_success_with_response(
            {
                "usage": {"prompt_tokens": 11, "completion_tokens": 5},
                "output": [
                    {"type": "reasoning", "id": "rs_1", "summary": []},
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Hello ", "annotations": []},
                            {"type": "output_text", "text": "from responses", "annotations": []},
                        ],
                    },
                ],
            }
        )

        self.assertEqual(llm_span["output"], "Hello from responses")

    def test_log_success_event_responses_api_function_call_records_output_items(self):
        output_items = [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "Boston"}',
            }
        ]
        llm_span = self._log_success_with_response({"usage": {}, "output": output_items})

        self.assertEqual(llm_span["output"], output_items)

    def test_log_success_event_empty_choices_does_not_raise(self):
        llm_span = self._log_success_with_response({"usage": {}, "choices": []})

        self.assertEqual(llm_span["output"], "NO_OUTPUT")

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
