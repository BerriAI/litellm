import json
from litellm._uuid import uuid
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.ollama.completion.transformation import (
    OllamaConfig,
    OllamaTextCompletionResponseIterator,
)
from litellm.types.utils import Message, ModelResponse, ModelResponseStream
from litellm.utils import get_optional_params


class TestOllamaConfig:
    def test_transform_response_standard(self):
        config = OllamaConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "Hello, I am an AI assistant",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        assert result.choices[0]["message"].content == "Hello, I am an AI assistant"
        assert result.choices[0]["finish_reason"] == "stop"
        assert result.model == "ollama/llama2"
        assert result.created is not None
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    @patch("uuid.uuid4")
    def test_transform_response_json_function_call(self, mock_uuid4):
        mock_uuid4.return_value = "test-uuid"

        config = OllamaConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": json.dumps(
                {"name": "get_weather", "arguments": {"location": "San Francisco"}}
            )
        }

        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={"format": "json"},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        assert result.choices[0]["message"].content is None
        assert result.choices[0]["finish_reason"] == "tool_calls"
        assert len(result.choices[0]["message"].tool_calls) == 1
        assert result.choices[0]["message"].tool_calls[0]["id"].startswith("call_")
        assert (
            result.choices[0]["message"].tool_calls[0]["function"]["name"]
            == "get_weather"
        )
        assert json.loads(
            result.choices[0]["message"].tool_calls[0]["function"]["arguments"]
        ) == {"location": "San Francisco"}

    def test_transform_response_regular_json(self):
        config = OllamaConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": json.dumps(
                {"result": "success", "data": {"temperature": 72, "unit": "F"}}
            )
        }

        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={"format": "json"},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        expected_content = json.dumps(
            {"result": "success", "data": {"temperature": 72, "unit": "F"}}
        )
        assert result.choices[0]["message"].content == expected_content
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_with_thinking_tags(self):
        config = OllamaConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<think>I need to think about this problem step by step</think>Here is my answer",
            "prompt_eval_count": 15,
            "eval_count": 8,
        }

        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        assert (
            result.choices[0]["message"].reasoning_content
            == "I need to think about this problem step by step"
        )
        assert result.choices[0]["message"].content == "Here is my answer"
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_with_thinking_tags_alternative(self):
        config = OllamaConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<thinking>Let me analyze this carefully</thinking>The solution is X",
        }

        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        assert (
            result.choices[0]["message"].reasoning_content
            == "Let me analyze this carefully"
        )
        assert result.choices[0]["message"].content == "The solution is X"
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_with_multiline_thinking_tags(self):
        config = OllamaConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<think>\nThis is a complex problem.\nI need to break it down:\n1. First step\n2. Second step\n</think>Based on my analysis, the answer is Y",
        }

        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        expected_reasoning = "\nThis is a complex problem.\nI need to break it down:\n1. First step\n2. Second step\n"
        assert result.choices[0]["message"].reasoning_content == expected_reasoning
        assert (
            result.choices[0]["message"].content
            == "Based on my analysis, the answer is Y"
        )
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_thinking_only(self):
        config = OllamaConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<think>Just internal thoughts, no response</think>",
        }

        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        assert (
            result.choices[0]["message"].reasoning_content
            == "Just internal thoughts, no response"
        )
        assert result.choices[0]["message"].content == ""
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_json_mode_with_thinking_tags(self):
        config = OllamaConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "<think>Planning my JSON response</think>This is not valid JSON",
        }

        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={"format": "json"},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        assert (
            result.choices[0]["message"].reasoning_content
            == "Planning my JSON response"
        )
        assert result.choices[0]["message"].content == "This is not valid JSON"
        assert result.choices[0]["finish_reason"] == "stop"

    def test_transform_response_no_thinking_tags(self):
        config = OllamaConfig()

        raw_response = MagicMock()
        raw_response.json.return_value = {
            "response": "Regular response without any thinking tags",
        }

        model_response = ModelResponse(
            id="test_id",
            choices=[{"message": Message(content="")}],
        )

        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]

        result = config.transform_response(
            model="llama2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=mock_encoding,
        )

        assert result.choices[0]["message"].reasoning_content is None
        assert (
            result.choices[0]["message"].content
            == "Regular response without any thinking tags"
        )
        assert result.choices[0]["finish_reason"] == "stop"


class TestOllamaTextCompletionResponseIterator:
    def test_chunk_parser_with_thinking_field(self):
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]), sync_stream=True, json_mode=False
        )

        chunk_with_thinking = {
            "model": "gpt-oss:20b",
            "created_at": "2025-08-06T14:34:31.5276077Z",
            "response": "",
            "thinking": "User",
            "done": False,
        }

        result = iterator.chunk_parser(chunk_with_thinking)

        assert isinstance(result, ModelResponseStream)
        assert result.choices and result.choices[0].delta is not None
        assert getattr(result.choices[0].delta, "reasoning_content") == "User"

    def test_chunk_parser_normal_response(self):
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]), sync_stream=True, json_mode=False
        )

        normal_chunk = {
            "model": "llama2",
            "created_at": "2025-08-06T14:34:31.5276077Z",
            "response": "Hello world",
            "done": False,
        }

        result = iterator.chunk_parser(normal_chunk)

        assert isinstance(result, ModelResponseStream)
        assert result.choices and result.choices[0].delta is not None
        assert result.choices[0].delta.content == "Hello world"
        assert getattr(result.choices[0].delta, "reasoning_content", None) is None

    def test_chunk_parser_empty_response_without_thinking(self):
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]), sync_stream=True, json_mode=False
        )

        empty_response_chunk = {
            "model": "qwen3:4b",
            "created_at": "2025-10-16T11:27:14.82881Z",
            "response": "",
            "done": False,
        }

        result = iterator.chunk_parser(empty_response_chunk)

        assert isinstance(result, ModelResponseStream)
        assert result.choices and result.choices[0].delta is not None
        assert result.choices[0].delta.content == None
        assert getattr(result.choices[0].delta, "reasoning_content", None) == ""

    def test_chunk_parser_done_chunk(self):
        iterator = OllamaTextCompletionResponseIterator(
            streaming_response=iter([]), sync_stream=True, json_mode=False
        )

        done_chunk = {
            "model": "llama2",
            "created_at": "2025-08-06T14:34:31.5276077Z",
            "response": "",
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        result = iterator.chunk_parser(done_chunk)

        assert result["text"] == ""
        assert result["is_finished"] is True
        assert result["finish_reason"] == "stop"
        assert result["usage"] is not None
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15


class TestOllamaFakeStreamActivation:
    def _tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def test_tools_and_stream_activate_fake_stream(self):
        optional_params = get_optional_params(
            model="llama2",
            custom_llm_provider="ollama",
            tools=self._tools(),
            stream=True,
            drop_params=True,
        )

        assert optional_params.get("fake_stream") is True
        assert optional_params.get("format") == "json"
        assert "functions_unsupported_model" in optional_params

    def test_tools_without_stream_does_not_activate_fake_stream(self):
        optional_params = get_optional_params(
            model="llama2",
            custom_llm_provider="ollama",
            tools=self._tools(),
            stream=False,
            drop_params=True,
        )

        assert "fake_stream" not in optional_params

    def test_stream_without_tools_does_not_activate_fake_stream(self):
        optional_params = get_optional_params(
            model="llama2",
            custom_llm_provider="ollama",
            stream=True,
        )

        assert "fake_stream" not in optional_params


class TestOllamaFakeStreamToolCalls:
    def test_tools_stream_true_reconstructs_tool_calls_via_fake_stream(self):
        tool_call_json = {
            "name": "get_current_weather",
            "arguments": {"location": "San Francisco"},
        }
        mock_ollama_response = {
            "model": "llama2",
            "response": json.dumps(tool_call_json),
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 42,
            "eval_count": 16,
        }

        mock_client = MagicMock(spec=HTTPHandler)
        mock_client.post.return_value = httpx.Response(
            status_code=200,
            content=json.dumps(mock_ollama_response).encode(),
            request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
        )

        response = litellm.completion(
            model="ollama/llama2",
            api_base="http://127.0.0.1:11434",
            messages=[
                {
                    "role": "user",
                    "content": "What is the weather in San Francisco?",
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get current weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            stream=True,
            drop_params=True,
            client=mock_client,
        )

        reassembled_content = ""
        tool_calls_seen = []
        finish_reasons = []
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                reassembled_content += delta.content
            if getattr(delta, "tool_calls", None):
                tool_calls_seen.extend(delta.tool_calls)
            if chunk.choices[0].finish_reason:
                finish_reasons.append(chunk.choices[0].finish_reason)

        assert mock_client.post.call_count == 1
        request_body = json.loads(mock_client.post.call_args.kwargs["data"])

        assert request_body.get("stream") is False
        assert "tools" not in request_body

        assert tool_calls_seen, "expected delta.tool_calls to be populated"
        assert tool_calls_seen[0]["function"]["name"] == "get_current_weather"
        assert json.loads(tool_calls_seen[0]["function"]["arguments"]) == {
            "location": "San Francisco"
        }

        assert finish_reasons == ["tool_calls"]
        assert json.dumps(tool_call_json) not in reassembled_content


async def test_ollama_async_completion_inlines_remote_images_off_the_event_loop(
    async_only_image_fetch,
):
    image_url = f"https://img.example/{uuid.uuid4()}.png"
    captured = {}

    def handle(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "llava",
                "response": "Green",
                "done": True,
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))

    response = await litellm.acompletion(
        model="ollama/llava",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colour is this?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        api_base="http://ollama.example:11434",
        client=client,
    )

    assert response.choices[0].message.content == "Green"
    assert async_only_image_fetch.fetched == [image_url]
    assert captured["body"]["images"] == [async_only_image_fetch.base64_png]