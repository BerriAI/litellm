import asyncio
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm import text_completion
from litellm.types.utils import ModelResponse, TextCompletionResponse, Usage
from litellm.utils import LiteLLMResponseObjectHandler


def _mock_text_completion_post(*args: object, **kwargs: object) -> MagicMock:
    return MagicMock(
        status_code=200,
        headers={"Content-Type": "application/json"},
        parse=MagicMock(
            return_value=MagicMock(
                model_dump=MagicMock(
                    return_value={
                        "id": "cmpl-7a59383dd4234092b9e5d652a7ab8143",
                        "object": "text_completion",
                        "created": 1718824735,
                        "model": "Sao10K/L3-70B-Euryale-v2.1",
                        "choices": [
                            {
                                "index": 0,
                                "text": ") might be faster than then answering, and the added time it takes for the",
                                "logprobs": None,
                                "finish_reason": "length",
                                "stop_reason": None,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 2,
                            "total_tokens": 18,
                            "completion_tokens": 16,
                        },
                    }
                )
            )
        ),
    )


def test_async_text_completion_together_ai():
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="my-fake-key")

    async def run_call():
        with patch.object(client.completions.with_raw_response, "create", side_effect=mock_post) as mock_call:
            response = await litellm.atext_completion(
                model="together_ai/Qwen/Qwen2-1.5B-Instruct",
                prompt="good morning",
                max_tokens=10,
                client=client,
            )
        return response, mock_call.call_args.kwargs

    response, sent = asyncio.run(run_call())
    assert sent["model"] == "Qwen/Qwen2-1.5B-Instruct"
    assert sent["prompt"] == "good morning"
    assert sent["max_tokens"] == 10
    assert response.choices[0].text == ") might be faster than then answering, and the added time it takes for the"
    assert response.usage.total_tokens == 18


@pytest.mark.parametrize("provider", ["openai", "hosted_vllm"])
def test_completion_vllm(provider):
    """
    Asserts a text completion call for vllm actually goes to the text completion endpoint
    """
    from openai import OpenAI

    client = OpenAI(api_key="my-fake-key")

    with patch.object(
        client.completions.with_raw_response, "create", side_effect=mock_post
    ) as mock_call:
        response = text_completion(
            model="{provider}/gemini-2.5-flash-lite".format(provider=provider),
            prompt="ping",
            client=client,
            hello="world",
        )
        print("raw response", response)

        assert response.usage.prompt_tokens == 2

        mock_call.assert_called_once()

        assert "hello" in mock_call.call_args.kwargs["extra_body"]


def test_convert_chat_to_text_completion():
    """Test converting chat completion to text completion"""
    chat_response = ModelResponse(
        id="chat123",
        created=1234567890,
        model="gpt-3.5-turbo",
        choices=[
            {
                "index": 0,
                "message": {"content": "Hello, world!"},
                "finish_reason": "stop",
            }
        ],
        usage={"total_tokens": 10, "completion_tokens": 10},
        _hidden_params={"api_key": "test"},
    )

    text_completion = TextCompletionResponse()
    result = LiteLLMResponseObjectHandler.convert_chat_to_text_completion(
        response=chat_response, text_completion_response=text_completion
    )

    assert isinstance(result, TextCompletionResponse)
    assert result.id == "chat123"
    assert result.object == "text_completion"
    assert result.created == 1234567890
    assert result.model == "gpt-3.5-turbo"
    assert result.choices[0].text == "Hello, world!"
    assert result.choices[0].finish_reason == "stop"
    assert result.usage == Usage(
        completion_tokens=10,
        prompt_tokens=0,
        total_tokens=10,
        completion_tokens_details=None,
        prompt_tokens_details=None,
    )


def test_convert_provider_response_logprobs_non_huggingface():
    """Test converting provider logprobs for non-huggingface provider"""
    response = ModelResponse(id="test123", _hidden_params={})

    result = LiteLLMResponseObjectHandler._convert_provider_response_logprobs_to_text_completion_logprobs(
        response=response, custom_llm_provider="openai"
    )

    assert result is None


def test_convert_chat_to_text_completion_multiple_choices():
    """Test converting chat completion to text completion with multiple choices"""
    chat_response = ModelResponse(
        id="chat456",
        created=1234567890,
        model="gpt-3.5-turbo",
        choices=[
            {
                "index": 0,
                "message": {"content": "First response"},
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "message": {"content": "Second response"},
                "finish_reason": "length",
            },
        ],
        usage={"total_tokens": 20},
        _hidden_params={"api_key": "test"},
    )

    text_completion = TextCompletionResponse()
    result = LiteLLMResponseObjectHandler.convert_chat_to_text_completion(
        response=chat_response, text_completion_response=text_completion
    )

    assert isinstance(result, TextCompletionResponse)
    assert result.id == "chat456"
    assert result.object == "text_completion"
    assert len(result.choices) == 2
    assert result.choices[0].text == "First response"
    assert result.choices[0].finish_reason == "stop"
    assert result.choices[1].text == "Second response"
    assert result.choices[1].finish_reason == "length"
    assert result.usage == Usage(
        completion_tokens=0,
        prompt_tokens=0,
        total_tokens=20,
        completion_tokens_details=None,
        prompt_tokens_details=None,
    )


def test_unit_test_text_completion_object():
    openai_object = {
        "id": "cmpl-99y7B2svVoRWe1xd7UFRmeGjZrFSh",
        "choices": [
            {
                "finish_reason": "length",
                "index": 0,
                "logprobs": {
                    "text_offset": [101],
                    "token_logprobs": [-0.00023488728],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.00023488728,
                            "1": -8.375235,
                            "zero": -14.101797,
                            "__": -14.554922,
                            "00": -14.98461,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 1,
                "logprobs": {
                    "text_offset": [116],
                    "token_logprobs": [-0.013745008],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.013745008,
                            "1": -4.294995,
                            "00": -12.287183,
                            "2": -12.771558,
                            "3": -14.013745,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 2,
                "logprobs": {
                    "text_offset": [108],
                    "token_logprobs": [-3.655073e-5],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -3.655073e-5,
                            "1": -10.656286,
                            "__": -11.789099,
                            "false": -12.984411,
                            "00": -14.039099,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 3,
                "logprobs": {
                    "text_offset": [106],
                    "token_logprobs": [-0.1345946],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.1345946,
                            "1": -2.0720947,
                            "2": -12.798657,
                            "false": -13.970532,
                            "00": -14.27522,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 4,
                "logprobs": {
                    "text_offset": [95],
                    "token_logprobs": [-0.10491652],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.10491652,
                            "1": -2.3236666,
                            "2": -7.0111666,
                            "3": -7.987729,
                            "4": -9.050229,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 5,
                "logprobs": {
                    "text_offset": [121],
                    "token_logprobs": [-0.00026300468],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.00026300468,
                            "1": -8.250263,
                            "zero": -14.976826,
                            " ": -15.461201,
                            "000": -15.773701,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 6,
                "logprobs": {
                    "text_offset": [146],
                    "token_logprobs": [-5.085517e-5],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -5.085517e-5,
                            "1": -9.937551,
                            "000": -13.929738,
                            "__": -14.968801,
                            "zero": -15.070363,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 7,
                "logprobs": {
                    "text_offset": [100],
                    "token_logprobs": [-0.13875218],
                    "tokens": ["1"],
                    "top_logprobs": [
                        {
                            "1": -0.13875218,
                            "0": -2.0450022,
                            "2": -9.7559395,
                            "3": -11.1465645,
                            "4": -11.5528145,
                        }
                    ],
                },
                "text": "1",
            },
            {
                "finish_reason": "length",
                "index": 8,
                "logprobs": {
                    "text_offset": [143],
                    "token_logprobs": [-0.0005573204],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0005573204,
                            "1": -7.6099324,
                            "3": -10.070869,
                            "2": -11.617744,
                            " ": -12.859932,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 9,
                "logprobs": {
                    "text_offset": [143],
                    "token_logprobs": [-0.0018747397],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0018747397,
                            "1": -6.29875,
                            "3": -11.2675,
                            "4": -11.634687,
                            "2": -11.822187,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 10,
                "logprobs": {
                    "text_offset": [110],
                    "token_logprobs": [-0.003476763],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.003476763,
                            "1": -5.6909766,
                            "__": -10.526915,
                            "None": -10.925352,
                            "False": -11.88629,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 11,
                "logprobs": {
                    "text_offset": [106],
                    "token_logprobs": [-0.00032962486],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.00032962486,
                            "1": -8.03158,
                            "__": -13.445642,
                            "2": -13.828455,
                            "zero": -15.453455,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 12,
                "logprobs": {
                    "text_offset": [143],
                    "token_logprobs": [-9.984788e-5],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -9.984788e-5,
                            "1": -9.21885,
                            " ": -14.836038,
                            "zero": -16.265724,
                            "00": -16.578224,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 13,
                "logprobs": {
                    "text_offset": [106],
                    "token_logprobs": [-0.0010039895],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0010039895,
                            "1": -6.907254,
                            "2": -13.743192,
                            "false": -15.227567,
                            "3": -15.297879,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 14,
                "logprobs": {
                    "text_offset": [106],
                    "token_logprobs": [-0.0005681643],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0005681643,
                            "1": -7.5005684,
                            "__": -11.836506,
                            "zero": -13.242756,
                            "file": -13.445881,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 15,
                "logprobs": {
                    "text_offset": [146],
                    "token_logprobs": [-3.9769227e-5],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -3.9769227e-5,
                            "1": -10.15629,
                            "000": -15.078165,
                            "00": -15.664103,
                            "zero": -16.015665,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 16,
                "logprobs": {
                    "text_offset": [143],
                    "token_logprobs": [-0.0006509595],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0006509595,
                            "1": -7.344401,
                            "2": -13.352214,
                            " ": -13.852214,
                            "3": -14.680339,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 17,
                "logprobs": {
                    "text_offset": [103],
                    "token_logprobs": [-0.0093299495],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0093299495,
                            "1": -4.681205,
                            "2": -11.173392,
                            "3": -13.439017,
                            "00": -14.673392,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 18,
                "logprobs": {
                    "text_offset": [130],
                    "token_logprobs": [-0.00024382756],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.00024382756,
                            "1": -8.328369,
                            " ": -13.640869,
                            "zero": -14.859619,
                            "null": -16.51587,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 19,
                "logprobs": {
                    "text_offset": [107],
                    "token_logprobs": [-0.0006452414],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0006452414,
                            "1": -7.36002,
                            "00": -12.328771,
                            "000": -12.961583,
                            "2": -14.211583,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 20,
                "logprobs": {
                    "text_offset": [143],
                    "token_logprobs": [-0.0012751155],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0012751155,
                            "1": -6.67315,
                            "__": -11.970025,
                            "<|endoftext|>": -14.907525,
                            "3": -14.930963,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 21,
                "logprobs": {
                    "text_offset": [107],
                    "token_logprobs": [-7.1954215e-5],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -7.1954215e-5,
                            "1": -9.640697,
                            "00": -13.500072,
                            "000": -13.523509,
                            "__": -13.945384,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 22,
                "logprobs": {
                    "text_offset": [108],
                    "token_logprobs": [-0.0032367748],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0032367748,
                            "1": -5.737612,
                            "<|endoftext|>": -13.940737,
                            "2": -14.167299,
                            "00": -14.292299,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 23,
                "logprobs": {
                    "text_offset": [117],
                    "token_logprobs": [-0.00018673266],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.00018673266,
                            "1": -8.593937,
                            "zero": -15.179874,
                            "null": -15.515812,
                            "None": -15.851749,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 24,
                "logprobs": {
                    "text_offset": [104],
                    "token_logprobs": [-0.0010223285],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0010223285,
                            "1": -6.8916473,
                            "__": -13.05571,
                            "00": -14.071335,
                            "zero": -14.235397,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 25,
                "logprobs": {
                    "text_offset": [108],
                    "token_logprobs": [-0.0038979414],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0038979414,
                            "1": -5.550773,
                            "2": -13.160148,
                            "00": -14.144523,
                            "3": -14.41796,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 26,
                "logprobs": {
                    "text_offset": [143],
                    "token_logprobs": [-0.00074721366],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.00074721366,
                            "1": -7.219497,
                            "3": -11.430435,
                            "2": -13.367935,
                            " ": -13.735123,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 27,
                "logprobs": {
                    "text_offset": [146],
                    "token_logprobs": [-8.566264e-5],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -8.566264e-5,
                            "1": -9.375086,
                            "000": -15.359461,
                            "__": -15.671961,
                            "00": -15.679773,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 28,
                "logprobs": {
                    "text_offset": [119],
                    "token_logprobs": [-0.000274683],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.000274683,
                            "1": -8.2034,
                            "00": -14.898712,
                            "2": -15.633087,
                            "__": -16.844025,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 29,
                "logprobs": {
                    "text_offset": [143],
                    "token_logprobs": [-0.014869375],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.014869375,
                            "1": -4.217994,
                            "2": -11.63987,
                            "3": -11.944557,
                            "5": -12.26487,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 30,
                "logprobs": {
                    "text_offset": [110],
                    "token_logprobs": [-0.010907865],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.010907865,
                            "1": -4.5265326,
                            "2": -11.440596,
                            "<|endoftext|>": -12.456221,
                            "file": -13.049971,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 31,
                "logprobs": {
                    "text_offset": [143],
                    "token_logprobs": [-0.00070528337],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.00070528337,
                            "1": -7.2663302,
                            "6": -13.141331,
                            "2": -13.797581,
                            "3": -13.836643,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 32,
                "logprobs": {
                    "text_offset": [143],
                    "token_logprobs": [-0.0004983439],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0004983439,
                            "1": -7.6098733,
                            "3": -14.211436,
                            "2": -14.336436,
                            " ": -15.117686,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 33,
                "logprobs": {
                    "text_offset": [110],
                    "token_logprobs": [-3.6908343e-5],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -3.6908343e-5,
                            "1": -10.250037,
                            "00": -14.2266,
                            "__": -14.7266,
                            "000": -16.164099,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 34,
                "logprobs": {
                    "text_offset": [104],
                    "token_logprobs": [-0.003917157],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.003917157,
                            "1": -5.550792,
                            "2": -11.355479,
                            "00": -12.777354,
                            "3": -13.652354,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 35,
                "logprobs": {
                    "text_offset": [146],
                    "token_logprobs": [-5.0139948e-5],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -5.0139948e-5,
                            "1": -9.921926,
                            "000": -14.851613,
                            "00": -15.414113,
                            "zero": -15.687551,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 36,
                "logprobs": {
                    "text_offset": [143],
                    "token_logprobs": [-0.0005143099],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0005143099,
                            "1": -7.5786395,
                            " ": -14.406764,
                            "00": -14.570827,
                            "999": -14.633327,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 37,
                "logprobs": {
                    "text_offset": [103],
                    "token_logprobs": [-0.00013691289],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.00013691289,
                            "1": -8.968887,
                            "__": -12.547012,
                            "zero": -13.57045,
                            "00": -13.8517,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 38,
                "logprobs": {
                    "text_offset": [103],
                    "token_logprobs": [-0.00032569113],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.00032569113,
                            "1": -8.047201,
                            "2": -13.570639,
                            "zero": -14.023764,
                            "false": -14.726889,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 39,
                "logprobs": {
                    "text_offset": [113],
                    "token_logprobs": [-3.7146747e-5],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -3.7146747e-5,
                            "1": -10.203162,
                            "zero": -18.437536,
                            "2": -20.117224,
                            " zero": -20.210974,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 40,
                "logprobs": {
                    "text_offset": [110],
                    "token_logprobs": [-7.4695905e-5],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -7.4695905e-5,
                            "1": -9.515699,
                            "00": -14.836012,
                            "__": -16.093824,
                            "file": -16.468824,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 41,
                "logprobs": {
                    "text_offset": [111],
                    "token_logprobs": [-0.02289473],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.02289473,
                            "1": -3.7885196,
                            "2": -12.499457,
                            "3": -14.546332,
                            "00": -15.66352,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 42,
                "logprobs": {
                    "text_offset": [108],
                    "token_logprobs": [-0.0011367622],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0011367622,
                            "1": -6.782387,
                            "2": -13.493324,
                            "00": -15.071449,
                            "zero": -15.727699,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 43,
                "logprobs": {
                    "text_offset": [115],
                    "token_logprobs": [-0.0006384541],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0006384541,
                            "1": -7.3600135,
                            "00": -14.0397005,
                            "2": -14.4303255,
                            "000": -15.563138,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 44,
                "logprobs": {
                    "text_offset": [143],
                    "token_logprobs": [-0.0007382771],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0007382771,
                            "1": -7.219488,
                            "4": -13.516363,
                            "2": -13.555426,
                            "3": -13.602301,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 45,
                "logprobs": {
                    "text_offset": [143],
                    "token_logprobs": [-0.0014242834],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0014242834,
                            "1": -6.5639243,
                            "2": -12.493611,
                            "__": -12.712361,
                            "3": -12.884236,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 46,
                "logprobs": {
                    "text_offset": [111],
                    "token_logprobs": [-0.00017088225],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.00017088225,
                            "1": -8.765796,
                            "zero": -12.695483,
                            "__": -12.804858,
                            "time": -12.882983,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 47,
                "logprobs": {
                    "text_offset": [146],
                    "token_logprobs": [-0.000107238506],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.000107238506,
                            "1": -9.171982,
                            "000": -13.648544,
                            "__": -14.531357,
                            "zero": -14.586044,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 48,
                "logprobs": {
                    "text_offset": [106],
                    "token_logprobs": [-0.0028172398],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0028172398,
                            "1": -5.877817,
                            "00": -12.16688,
                            "2": -12.487192,
                            "000": -14.182505,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 49,
                "logprobs": {
                    "text_offset": [104],
                    "token_logprobs": [-0.00043460296],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.00043460296,
                            "1": -7.7816844,
                            "00": -13.570747,
                            "2": -13.60981,
                            "__": -13.789497,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 50,
                "logprobs": {
                    "text_offset": [143],
                    "token_logprobs": [-0.0046973573],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0046973573,
                            "1": -5.3640723,
                            "null": -14.082823,
                            " ": -14.707823,
                            "2": -14.746885,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 51,
                "logprobs": {
                    "text_offset": [100],
                    "token_logprobs": [-0.2487161],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.2487161,
                            "1": -1.5143411,
                            "2": -9.037779,
                            "3": -10.100279,
                            "4": -10.756529,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 52,
                "logprobs": {
                    "text_offset": [108],
                    "token_logprobs": [-0.0011751055],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0011751055,
                            "1": -6.751175,
                            " ": -13.73555,
                            "2": -15.258987,
                            "3": -15.399612,
                        }
                    ],
                },
                "text": "0",
            },
            {
                "finish_reason": "length",
                "index": 53,
                "logprobs": {
                    "text_offset": [143],
                    "token_logprobs": [-0.0012339224],
                    "tokens": ["0"],
                    "top_logprobs": [
                        {
                            "0": -0.0012339224,
                            "1": -6.719984,
                            "6": -11.430922,
                            "3": -12.165297,
                            "2": -12.696547,
                        }
                    ],
                },
                "text": "0",
            },
        ],
        "created": 1712163061,
        "model": "ft:babbage-002:ai-r-d-zapai:v3-fields-used:84jb9rtr",
        "object": "text_completion",
        "system_fingerprint": None,
        "usage": {"completion_tokens": 54, "prompt_tokens": 1877, "total_tokens": 1931},
    }

    text_completion_obj = TextCompletionResponse(**openai_object)

    ## WRITE UNIT TESTS FOR TEXT_COMPLETION_OBJECT
    assert text_completion_obj.id == "cmpl-99y7B2svVoRWe1xd7UFRmeGjZrFSh"
    assert text_completion_obj.object == "text_completion"
    assert text_completion_obj.created == 1712163061
    assert (
        text_completion_obj.model
        == "ft:babbage-002:ai-r-d-zapai:v3-fields-used:84jb9rtr"
    )
    assert text_completion_obj.system_fingerprint == None
    assert len(text_completion_obj.choices) == len(openai_object["choices"])

    # TEST FIRST CHOICE #
    first_text_completion_obj = text_completion_obj.choices[0]
    assert first_text_completion_obj.index == 0
    assert first_text_completion_obj.logprobs.text_offset == [101]
    assert first_text_completion_obj.logprobs.tokens == ["0"]
    assert first_text_completion_obj.logprobs.token_logprobs == [-0.00023488728]
    assert len(first_text_completion_obj.logprobs.top_logprobs) == len(
        openai_object["choices"][0]["logprobs"]["top_logprobs"]
    )
    assert first_text_completion_obj.text == "0"
    assert first_text_completion_obj.finish_reason == "length"

    # TEST SECOND CHOICE #
    second_text_completion_obj = text_completion_obj.choices[1]
    assert second_text_completion_obj.index == 1
    assert second_text_completion_obj.logprobs.text_offset == [116]
    assert second_text_completion_obj.logprobs.tokens == ["0"]
    assert second_text_completion_obj.logprobs.token_logprobs == [-0.013745008]
    assert len(second_text_completion_obj.logprobs.top_logprobs) == len(
        openai_object["choices"][0]["logprobs"]["top_logprobs"]
    )
    assert second_text_completion_obj.text == "0"
    assert second_text_completion_obj.finish_reason == "length"

    # TEST LAST CHOICE #
    last_text_completion_obj = text_completion_obj.choices[-1]
    assert last_text_completion_obj.index == 53
    assert last_text_completion_obj.logprobs.text_offset == [143]
    assert last_text_completion_obj.logprobs.tokens == ["0"]
    assert last_text_completion_obj.logprobs.token_logprobs == [-0.0012339224]
    assert len(last_text_completion_obj.logprobs.top_logprobs) == len(
        openai_object["choices"][0]["logprobs"]["top_logprobs"]
    )
    assert last_text_completion_obj.text == "0"
    assert last_text_completion_obj.finish_reason == "length"

    assert text_completion_obj.usage.completion_tokens == 54
    assert text_completion_obj.usage.prompt_tokens == 1877
    assert text_completion_obj.usage.total_tokens == 1931


def mock_post(*args, **kwargs):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.parse.return_value.model_dump.return_value = {
        "id": "cmpl-7a59383dd4234092b9e5d652a7ab8143",
        "object": "text_completion",
        "created": 1718824735,
        "model": "Sao10K/L3-70B-Euryale-v2.1",
        "choices": [
            {
                "index": 0,
                "text": ") might be faster than then answering, and the added time it takes for the",
                "logprobs": None,
                "finish_reason": "length",
                "stop_reason": None,
            }
        ],
        "usage": {"prompt_tokens": 2, "total_tokens": 18, "completion_tokens": 16},
    }
    return mock_response
