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
