from litellm.llms.azure.image_edit.transformation import AzureImageEditConfig
from litellm.types.router import GenericLiteLLMParams


def test_azure_deployment_image_edit_form_data_strips_model():
    url = (
        "https://example.openai.azure.com/openai/deployments/my-dep/"
        "images/edits?api-version=2025-02-01-preview"
    )
    data = {"model": "my-dep", "prompt": "x", "n": 1}
    out = AzureImageEditConfig.azure_deployment_image_edit_form_data(data, url)
    assert "model" not in out
    assert out == {"prompt": "x", "n": 1}


def test_azure_deployment_image_edit_form_data_keeps_model_non_deployment_url():
    url = "https://api.openai.com/v1/images/edits"
    data = {"model": "gpt-image-1", "prompt": "x"}
    out = AzureImageEditConfig.azure_deployment_image_edit_form_data(data, url)
    assert out == data


def test_azure_finalize_image_edit_strips_model_after_openai_transform():
    """OpenAI transform still includes model; finalize uses the real request URL."""
    config = AzureImageEditConfig()
    model = "gpt-image-2-dep"
    prompt = "add a hat"
    image = b"fake_png_bytes"
    litellm_params = GenericLiteLLMParams(
        api_base="https://example.openai.azure.com",
        api_version="2025-02-01-preview",
    )
    data, files = config.transform_image_edit_request(
        model=model,
        prompt=prompt,
        image=image,
        image_edit_optional_request_params={"n": 1},
        litellm_params=litellm_params,
        headers={},
    )
    assert data.get("model") == model
    resolved = config.get_complete_url(
        model=model,
        api_base=litellm_params.api_base,
        litellm_params=litellm_params.model_dump(exclude_none=True),
    )
    data_out = config.finalize_image_edit_request_data(data, resolved)
    assert "model" not in data_out
    assert data_out.get("prompt") == prompt
    assert data_out.get("n") == 1
    assert len(files) >= 1
