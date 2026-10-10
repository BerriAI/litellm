import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.bedrock.chat.invoke_transformations.amazon_mistral_transformation import (
    AmazonMistralConfig,
)
from litellm.types.utils import ModelResponse


def test_mistral_get_outputText():
    # Set initial model response with arbitrary finish reason
    model_response = ModelResponse()
    model_response.choices[0].finish_reason = "None"

    # Models like pixtral will return a completion with the openai format.
    mock_json_with_choices = {
        "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}]
    }

    outputText = AmazonMistralConfig.get_outputText(
        completion_response=mock_json_with_choices, model_response=model_response
    )

    assert outputText == "Hello!"
    assert model_response.choices[0].finish_reason == "stop"

    # Other models might return a completion behind "outputs"
    mock_json_with_output = {"outputs": [{"text": "Hi!", "stop_reason": "finish"}]}

    outputText = AmazonMistralConfig.get_outputText(
        completion_response=mock_json_with_output, model_response=model_response
    )

    assert outputText == "Hi!"
    assert model_response.choices[0].finish_reason == "finish"


@pytest.mark.respx(assert_all_called=True)
def test_mistral_invoke_outputs_become_message_content(respx_mock: respx.Router) -> None:
    route: Final = respx_mock.post(
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/mistral.mistral-7b-instruct-v0%3A2/invoke"
    ).mock(
        return_value=httpx.Response(
            200, json={"outputs": [{"text": " The sky is a canvas of blue.", "stop_reason": "length"}]}
        )
    )

    response: Final = litellm.completion(
        model="bedrock/mistral.mistral-7b-instruct-v0:2",
        messages=[{"role": "user", "content": "Write a short poem about the sky"}],
        max_tokens=10,
        temperature=0.1,
        aws_access_key_id="AKIAMISTRALKEY",
        aws_secret_access_key="mistral-secret",
        aws_region_name="us-west-2",
        client=HTTPHandler(),
    )

    assert json.loads(route.calls[0].request.content) == {
        "prompt": "<s>[INST] Write a short poem about the sky [/INST]\n",
        "max_tokens": 10,
        "temperature": 0.1,
    }
    assert response.choices[0].message.content == " The sky is a canvas of blue."
    assert response.choices[0].finish_reason == "length"
