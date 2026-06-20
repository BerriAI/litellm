import os
import sys
import json
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import transcription
from litellm.llms.fireworks_ai.chat.transformation import FireworksAIConfig
from base_audio_transcription_unit_tests import BaseLLMAudioTranscriptionTest

fireworks = FireworksAIConfig()


def test_map_openai_params_tool_choice():
    # Test case 1: tool_choice is "required"
    result = fireworks.map_openai_params(
        {"tool_choice": "required"}, {}, "some_model", drop_params=False
    )
    assert result == {"tool_choice": "any"}

    # Test case 2: tool_choice is "auto"
    result = fireworks.map_openai_params(
        {"tool_choice": "auto"}, {}, "some_model", drop_params=False
    )
    assert result == {"tool_choice": "auto"}

    # Test case 3: tool_choice is not present
    result = fireworks.map_openai_params(
        {"some_other_param": "value"}, {}, "some_model", drop_params=False
    )
    assert result == {}

    # Test case 4: tool_choice is None
    result = fireworks.map_openai_params(
        {"tool_choice": None}, {}, "some_model", drop_params=False
    )
    assert result == {"tool_choice": None}


def test_map_response_format():
    """
    json_schema response_format is passed through to Fireworks unchanged.

    Fireworks accepts the OpenAI strict json_schema shape natively. The earlier
    downgrade to {type: json_object, schema: ...} silently dropped `strict` and
    `name`, producing a request that Fireworks treats as "any valid JSON" per
    its docs, disabling grammar-guided decoding.

    Ref: https://docs.fireworks.ai/structured-responses/structured-response-formatting
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "properties": {"result": {"type": "boolean"}},
                "required": ["result"],
                "type": "object",
            },
            "name": "BooleanResponse",
            "strict": True,
        },
    }
    result = fireworks.map_openai_params(
        {"response_format": response_format}, {}, "some_model", drop_params=False
    )
    assert result == {"response_format": response_format}


_AUDIO_FILE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "gettysburg.wav"
)


class TestFireworksAIAudioTranscription(BaseLLMAudioTranscriptionTest):
    def get_base_audio_transcription_call_args(self) -> dict:
        return {
            "model": "fireworks_ai/whisper-v3",
            "api_base": "https://audio-prod.api.fireworks.ai/v1",
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.FIREWORKS_AI

    def test_audio_transcription(self):
        from unittest.mock import MagicMock

        from openai.types.audio import Transcription

        audio_file = open(_AUDIO_FILE_PATH, "rb")
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = Transcription(
            text="four score and seven years ago"
        )

        transcript = transcription(
            **self.get_base_audio_transcription_call_args(),
            file=audio_file,
            api_key="fw-test-key",
            client=mock_client,
        )

        assert transcript.text == "four score and seven years ago"
        sent = mock_client.audio.transcriptions.create.call_args.kwargs
        assert sent["model"] == "whisper-v3"
        assert sent["file"] is audio_file

    @pytest.mark.asyncio
    async def test_audio_transcription_async(self):
        from unittest.mock import AsyncMock, MagicMock

        from openai.types.audio import Transcription

        audio_file = open(_AUDIO_FILE_PATH, "rb")
        raw_response = MagicMock()
        raw_response.headers = {}
        raw_response.parse.return_value = Transcription(
            text="four score and seven years ago"
        )
        mock_client = MagicMock()
        mock_client.audio.transcriptions.with_raw_response.create = AsyncMock(
            return_value=raw_response
        )

        transcript = await litellm.atranscription(
            **self.get_base_audio_transcription_call_args(),
            file=audio_file,
            api_key="fw-test-key",
            client=mock_client,
        )

        assert transcript.text == "four score and seven years ago"
        sent = (
            mock_client.audio.transcriptions.with_raw_response.create.call_args.kwargs
        )
        assert sent["model"] == "whisper-v3"
        assert sent["file"] is audio_file


@pytest.mark.parametrize(
    "disable_add_transform_inline_image_block",
    [True, False],
)
def test_document_inlining_example(disable_add_transform_inline_image_block):
    """
    Fireworks document inlining has been removed from the platform. LiteLLM
    must not append ``#transform=inline`` regardless of the legacy disable flag.
    """
    from unittest.mock import patch

    from litellm import completion
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    pdf_url = "https://storage.googleapis.com/fireworks-public/test/sample_resume.pdf"

    with patch.object(client, "post") as mock_post:
        try:
            completion(
                model="fireworks_ai/accounts/fireworks/models/deepseek-v3p1",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": pdf_url},
                            },
                            {
                                "type": "text",
                                "text": "What are the candidate's BA and MBA GPAs?",
                            },
                        ],
                    }
                ],
                disable_add_transform_inline_image_block=disable_add_transform_inline_image_block,
                client=client,
            )
        except Exception:
            pass

        mock_post.assert_called_once()
        json_data = json.loads(mock_post.call_args.kwargs["data"])
        sent_url = json_data["messages"][0]["content"][0]["image_url"]["url"]
        assert sent_url == pdf_url
        assert "#transform=inline" not in sent_url


@pytest.mark.parametrize(
    "content, expected_url",
    [
        (
            {"image_url": "http://example.com/image.png"},
            "http://example.com/image.png",
        ),
        (
            {"image_url": {"url": "http://example.com/image.png"}},
            {"url": "http://example.com/image.png"},
        ),
        (
            {"image_url": "data:image/png;base64,iVBORw0KGgo="},
            "data:image/png;base64,iVBORw0KGgo=",
        ),
        (
            {"image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ=="}},
            {"url": "data:image/jpeg;base64,/9j/4AAQ=="},
        ),
        (
            {"image_url": "Data:image/png;base64,iVBORw0KGgo="},
            "Data:image/png;base64,iVBORw0KGgo=",
        ),
    ],
)
def test_transform_inline_no_longer_added(content, expected_url):
    image_block = {"type": "image_url", **content}
    messages = [{"role": "user", "content": [image_block]}]

    result = litellm.FireworksAIConfig()._transform_messages_helper(
        messages=messages,
        model="accounts/fireworks/models/minimax-m3",
        litellm_params={},
    )
    result_image_block = result[0]["content"][0]
    if isinstance(expected_url, str):
        assert result_image_block["image_url"] == expected_url
    else:
        assert result_image_block["image_url"]["url"] == expected_url["url"]


@pytest.mark.parametrize(
    "is_disabled",
    [True, False],
)
def test_global_disable_flag_no_longer_adds_transform_inline(is_disabled):
    url = "http://example.com/image.png"
    litellm.disable_add_transform_inline_image_block = is_disabled
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": url}],
        }
    ]
    result = litellm.FireworksAIConfig()._transform_messages_helper(
        messages=messages,
        model="accounts/fireworks/models/minimax-m3",
        litellm_params={},
    )
    assert result[0]["content"][0]["image_url"] == url
    litellm.disable_add_transform_inline_image_block = False  # Reset for other tests


def test_global_disable_flag_with_transform_messages_helper(monkeypatch):
    from unittest.mock import patch
    from litellm import completion
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    monkeypatch.setattr(litellm, "disable_add_transform_inline_image_block", True)

    with patch.object(
        client,
        "post",
    ) as mock_post:
        try:
            completion(
                model="fireworks_ai/accounts/fireworks/models/deepseek-v3p1",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png"
                                },
                            },
                        ],
                    }
                ],
                client=client,
            )
        except Exception:
            pass

        mock_post.assert_called_once()
        json_data = json.loads(mock_post.call_args.kwargs["data"])
        assert (
            "#transform=inline"
            not in json_data["messages"][0]["content"][1]["image_url"]["url"]
        )
