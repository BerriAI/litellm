import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from unittest.mock import MagicMock, patch

import litellm


@pytest.fixture
def openai_api_response():
    mock_response_data = {
        "id": "chatcmpl-B0W3vmiM78Xkgx7kI7dr7PC949DMS",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "",
                    "refusal": None,
                    "role": "assistant",
                    "audio": None,
                    "function_call": None,
                    "tool_calls": None,
                },
            }
        ],
        "created": 1739462947,
        "model": "gpt-4o-mini-2024-07-18",
        "object": "chat.completion",
        "service_tier": "default",
        "system_fingerprint": "fp_bd83329f63",
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": 121,
            "total_tokens": 122,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
        },
    }

    return mock_response_data


def test_completion_missing_role(openai_api_response):
    from openai import OpenAI

    from litellm.types.utils import ModelResponse

    client = OpenAI(api_key="test_api_key")

    mock_raw_response = MagicMock()
    mock_raw_response.headers = {
        "x-request-id": "123",
        "openai-organization": "org-123",
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "99",
    }
    mock_raw_response.parse.return_value = ModelResponse(**openai_api_response)

    print(f"openai_api_response: {openai_api_response}")

    with patch.object(
        client.chat.completions.with_raw_response, "create", mock_raw_response
    ) as mock_create:
        litellm.completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Hey"},
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_m0vFJjQmTH1McvaHBPR2YFwY",
                            "function": {
                                "arguments": '{"input": "dksjsdkjdhskdjshdskhjkhlk"}',
                                "name": "tool_name",
                            },
                            "type": "function",
                            "index": 0,
                        },
                        {
                            "id": "call_Vw6RaqV2n5aaANXEdp5pYxo2",
                            "function": {
                                "arguments": '{"input": "jkljlkjlkjlkjlk"}',
                                "name": "tool_name",
                            },
                            "type": "function",
                            "index": 1,
                        },
                        {
                            "id": "call_hBIKwldUEGlNh6NlSXil62K4",
                            "function": {
                                "arguments": '{"input": "jkjlkjlkjlkj;lj"}',
                                "name": "tool_name",
                            },
                            "type": "function",
                            "index": 2,
                        },
                    ],
                },
            ],
            client=client,
        )

        mock_create.assert_called_once()


@pytest.mark.parametrize(
    "model",
    [
        "gemini/gemini-1.5-flash",
        "bedrock/claude-3-5-sonnet",
        "anthropic/claude-3-5-sonnet",
    ],
)
def test_signed_s3_url_with_format(model):
    from litellm import completion
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    args = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://litellm-logo-aws-marketplace.s3.us-west-2.amazonaws.com/berriai-logo-github.png?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJGMEQCIHlAy6QneghdEo4Dp4rw%2BHhdInKX4MU3T0hZT1qV3AD%2FAiBGY%2FtfxmBJkj%2BK6%2FxAgek6L3tpOcq6su1mBrj87El%2FCirLAwghEAEaDDg4ODYwMjIyMzQyOCIMzds7lsxAFHHCRHmkKqgDgnsJBaEmmwXBWqzyMMe3BUKsCqfvrYupFGxBREP%2BaEz%2ByLSKiTM3xWzaRz6vrP9T4HSJ97B9wQ3dhUBT22XzdOFsaq49wZapwy9hoPNrMyZ77DIa0MlEbg0uudGOaMAw4NbVEqoERQuZmIMMbNHCeoJsZxKCttRZlTDzU%2FeNNy96ltb%2FuIkX5b3OOYdUaKj%2FUjmPz%2FEufY%2Bn%2FFHawunSYXJwL4pYuBF1IKRtPjqamaYscH%2FrzD7fubGUMqk6hvyGEo%2BLqnVyruQEmVFqAnXyWlpHGqeWazEC7xcsC2lhLO%2FKUouyVML%2FxyYtL4CuKp52qtLWWauAFGnyBZnCHtSL58KLaMTSh7inhoFFIKDN2hymrJ4D9%2Bxv%2FMOzefH5X%2B0pcdJUwyxcwgL3myggRmIYq1L6IL4I%2F54BIU%2FMctJcRXQ8NhQNP2PsaCsXYHHVMXRZxps9v8t9Ciorb0PAaLr0DIGVgEqejSjwbzNTctQf59Rj0GhZ0A6A3nFaq3nL4UvO51aPP6aelN6RnLwHh8fF80iPWII7Oj9PWn9bkON%2F7%2B5k42oPFR0KDTD0yaO%2BBjrlAouRvkyHZnCuLuJdEeqc8%2Fwm4W8SbMiYDzIEPPe2wFR2sH4%2FDlnJRqia9Or00d4N%2BOefBkPv%2Bcdt68r%2FwjeWOrulczzLGjJE%2FGw1Lb9dtGtmupGm2XKOW3geJwXkk1qcr7u5zwy6DNamLJbitB026JFKorRnPajhe5axEDv%2BRu6l1f0eailIrCwZ2iytA94Ni8LTha2GbZvX7fFHcmtyNlgJPpMcELdkOEGTCNBldGck5MFHG27xrVrlR%2F7HZIkKYlImNmsOIjuK7acDiangvVdB6GlmVbzNUKtJ7YJhS2ivwvdDIf8XuaFAkhjRNpewDl0GzPvojK%2BDTizZydyJL%2B20pVkSXptyPwrrHEeiOFWwhszW2iTZij4rlRAoZW6NEdfkWsXrGMbxJTZa3E5URejJbg%2B4QgGtjLrgJhRC1pJGP02GX7VMxVWZzomfC2Hn7WaF44wgcuqjE4HGJfpA2ZLBxde52g%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA45ZGR4NCKIUOODV3%2F20250305%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20250305T235823Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=71a900a9467eaf3811553500aaf509a10a9e743a8133cfb6a78dcbcbc6da4a05",
                            "format": "image/jpeg",
                        },
                    },
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ],
    }
    with patch.object(client, "post", new=MagicMock()) as mock_client:
        try:
            response = completion(**args, client=client)
            print(response)
        except Exception as e:
            print(e)

        print(mock_client.call_args.kwargs)

        mock_client.assert_called()

        print(mock_client.call_args.kwargs)

        json_str = json.dumps(mock_client.call_args.kwargs["json"])
        assert "image/jpeg" in json_str
        assert "image/png" not in json_str
