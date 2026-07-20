from litellm.llms.azure.azure import AzureChatCompletion


def test_azure_image_generation_v1_url_and_model_payload():
    azure_chat = AzureChatCompletion()
    url = azure_chat.create_azure_base_url(
        azure_client_params={
            "azure_endpoint": "https://my-resource.openai.azure.com",
            "api_version": "v1",
        },
        model="image-deployment",
        base_model="gpt-image-1",
    )
    assert url == "https://my-resource.openai.azure.com/openai/v1/images/generations?api-version=preview"

    data = {"model": "gpt-image-1", "prompt": "x"}
    prepared = azure_chat._prepare_image_generation_data_for_url(
        data=data,
        model="image-deployment",
        api_base=url,
    )
    assert prepared == {"model": "image-deployment", "prompt": "x"}
    assert data == {"model": "gpt-image-1", "prompt": "x"}
