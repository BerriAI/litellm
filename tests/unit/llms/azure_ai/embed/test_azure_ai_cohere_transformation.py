from typing import Final

from litellm.llms.azure_ai.embed.cohere_transformation import AzureAICohereConfig
from litellm.types.utils import EmbeddingResponse, Usage


def test_azure_ai_cohere_request_splits_images_from_text_and_keeps_order():
    config: Final = AzureAICohereConfig()
    image_request, text_request, image_indexes = config.transform_request(
        input=[
            "describe this image",
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB",
        ],
        optional_params={},
        model="Cohere-embed-v3-multilingual",
    )

    assert image_request == {
        "input": [{"image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"}]
    }
    assert text_request["input"] == ["describe this image"]
    assert image_indexes == [1]


def test_azure_ai_cohere_response_maps_model_group_and_usage_headers():
    config: Final = AzureAICohereConfig()
    response: Final = EmbeddingResponse(
        model="",
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        data=[],
        hidden_params={
            "additional_headers": {
                "llm_provider-num_tokens": "7",
                "llm_provider-azureml-model-group": "offer-cohere-embed-multili-paygo",
            }
        },
    )

    result: Final = config.transform_response(response)

    assert result.model == "Cohere-embed-v3-multilingual"
    assert result.usage.prompt_tokens == 7
