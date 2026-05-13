from unittest.mock import patch

from litellm.llms.azure.image_edit.transformation import AzureImageEditConfig
from litellm.types.router import GenericLiteLLMParams


def test_validate_environment_uses_api_key_header_for_subscription_key():
    """
    Azure OpenAI / API Management gateways authenticate via the ``api-key``
    header. Using ``Authorization: Bearer <subscription-key>`` is the OpenAI
    direct convention and is rejected by Azure with
    ``401 "Access denied due to missing subscription key"``.

    Regression guard for the previous unconditional Bearer header.
    """
    config = AzureImageEditConfig()
    headers = config.validate_environment(
        headers={},
        model="gpt-image-1",
        api_key="my-azure-subscription-key",
        litellm_params={},
    )
    assert headers.get("api-key") == "my-azure-subscription-key"
    assert "Authorization" not in headers


def test_validate_environment_prefers_litellm_params_api_key():
    config = AzureImageEditConfig()
    headers = config.validate_environment(
        headers={},
        model="gpt-image-1",
        api_key=None,
        litellm_params={"api_key": "from-params"},
    )
    assert headers.get("api-key") == "from-params"
    assert "Authorization" not in headers


def test_validate_environment_litellm_params_api_key_beats_positional_arg():
    """
    Precedence pin: when both ``api_key`` (positional) and
    ``litellm_params["api_key"]`` are set with different values, the
    ``litellm_params`` value wins.

    This matches the convention used by every other Azure
    ``validate_environment`` (videos, vector_stores, responses, ...), where
    ``litellm_params.api_key`` is the source of truth and the positional
    kwarg only fills in when ``litellm_params`` lacks a key. In production
    the only caller (``llm_http_handler.image_edit``) sources both values
    from the same ``litellm_params.api_key``, so this only matters for
    direct callers.
    """
    config = AzureImageEditConfig()
    headers = config.validate_environment(
        headers={},
        model="gpt-image-1",
        api_key="from-positional",
        litellm_params={"api_key": "from-params"},
    )
    assert headers.get("api-key") == "from-params"
    assert "Authorization" not in headers


def test_validate_environment_falls_back_to_aad_bearer_when_no_api_key():
    """
    When neither ``api_key`` nor any AZURE_*_API_KEY env var is available, the
    base helper resolves an AAD token and falls back to
    ``Authorization: Bearer <token>``. This mirrors the behavior of every
    other Azure provider class (videos, vector_stores, responses, ...).
    """
    config = AzureImageEditConfig()
    with (
        patch(
            "litellm.llms.azure.common_utils.get_azure_ad_token",
            return_value="fake-aad-token",
        ),
        patch(
            "litellm.llms.azure.common_utils.get_secret_str",
            return_value=None,
        ),
        patch("litellm.api_key", None),
        patch("litellm.azure_key", None),
    ):
        headers = config.validate_environment(
            headers={},
            model="gpt-image-1",
            api_key=None,
            litellm_params={},
        )
    assert headers.get("Authorization") == "Bearer fake-aad-token"
    assert "api-key" not in headers


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
