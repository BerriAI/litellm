import os
import sys
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import pytest

import litellm
from litellm.types.llms.openai import (
    ChatCompletionAnnotationURLCitation,
)
from base_audio_transcription_unit_tests import BaseLLMAudioTranscriptionTest


def test_openai_chat_completion_streaming_handler_reasoning_content():
    from litellm.llms.openai.chat.gpt_transformation import (
        OpenAIChatCompletionStreamingHandler,
    )
    from unittest.mock import MagicMock

    streaming_handler = OpenAIChatCompletionStreamingHandler(
        streaming_response=MagicMock(),
        sync_stream=True,
    )
    response = streaming_handler.chunk_parser(
        chunk={
            "id": "e89b6501-8ac2-464c-9550-7cd3daf94350",
            "object": "chat.completion.chunk",
            "created": 1741037890,
            "model": "deepseek-reasoner",
            "system_fingerprint": "fp_5417b77867_prod0225",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": None, "reasoning_content": "."},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
    )

    assert response.choices[0].delta.reasoning_content == "."


def validate_response_url_citation(url_citation: ChatCompletionAnnotationURLCitation):
    assert "end_index" in url_citation
    assert "start_index" in url_citation
    assert "url" in url_citation


class TestOpenAIGPT4OAudioTranscription(BaseLLMAudioTranscriptionTest):
    def get_base_audio_transcription_call_args(self) -> dict:
        return {
            "model": "openai/gpt-4o-transcribe",
            # "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.OPENAI


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4o"])
async def test_openai_pdf_url(model):
    from litellm.utils import return_raw_request, CallTypes

    request = return_raw_request(
        CallTypes.completion,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the first page of the PDF?"},
                        {
                            "type": "file",
                            "file": {"file_id": "https://arxiv.org/pdf/2303.08774"},
                        },
                    ],
                }
            ],
        },
    )
    print("request: ", request)

    assert (
        "file_data" in request["raw_request_body"]["messages"][0]["content"][1]["file"]
    )


def test_responses_gpt54_with_xhigh_reasoning():
    """
    Ensure chat->responses bridge sends the correct request payload for
    openai/responses/gpt-5.4 with reasoning_effort="xhigh".
    """
    with patch("litellm.responses") as mock_responses:
        # Stop execution right after request generation to avoid external API calls.
        mock_responses.side_effect = RuntimeError("stop_after_request_build")

        with pytest.raises(Exception):
            litellm.completion(
                model="openai/responses/gpt-5.4",
                messages=[{"role": "user", "content": "What is 2+2?"}],
                reasoning_effort="xhigh",
                max_tokens=100,
            )

        mock_responses.assert_called_once()
        request_body = mock_responses.call_args.kwargs

        # The responses prefix should be stripped before routing.
        assert request_body["model"] == "gpt-5.4"
        # chat-completions reasoning_effort must map to Responses API reasoning.
        assert request_body["reasoning"] == {"effort": "xhigh"}
