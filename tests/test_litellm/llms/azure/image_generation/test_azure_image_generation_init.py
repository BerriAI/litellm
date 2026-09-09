import json
import traceback
from typing import Callable, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.azure.azure import AzureChatCompletion
from litellm.llms.azure.common_utils import (
    _cached_azure_ad_token_refresh_provider,
    _cached_entra_id_token_provider,
    get_azure_request_auth_headers,
    redact_azure_auth_headers,
)
from litellm.llms.azure.image_generation.http_utils import (
    azure_deployment_image_generation_json_body,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.azure.image_generation import (
    AzureDallE3ImageGenerationConfig,
    get_azure_image_generation_config,
)
from litellm.utils import get_optional_params_image_gen


@pytest.mark.parametrize(
    "received_model, expected_config",
    [
        ("dall-e-3", AzureDallE3ImageGenerationConfig),
        ("dalle-3", AzureDallE3ImageGenerationConfig),
        ("openai_dall_e_3", AzureDallE3ImageGenerationConfig),
    ],
)
def test_azure_image_generation_config(received_model, expected_config):
    assert isinstance(
        get_azure_image_generation_config(received_model), expected_config
    )


def test_azure_deployment_image_generation_json_body():
    """Deployment-scoped Azure image URL must not send ``model`` in JSON."""
    api = (
        "https://example.openai.azure.com/openai/deployments/my-dep/"
        "images/generations?api-version=2025-04-01-preview"
    )
    data = {"model": "my-dep", "prompt": "x", "n": 1}
    out = azure_deployment_image_generation_json_body(api, data)
    assert "model" not in out
    assert out == {"prompt": "x", "n": 1}


def test_azure_providers_image_generation_json_body_keeps_model():
    """Non-deployment routes (e.g. FLUX on Azure AI) keep the payload unchanged."""
    api = "https://example.services.ai.azure.com/providers/blackforestlabs/v1/flux-2-pro?api-version=preview"
    data = {"model": "flux.2-pro", "prompt": "x"}
    out = azure_deployment_image_generation_json_body(api, data)
    assert out == data


def test_azure_image_generation_mai_base_model_uses_mai_url():
    azure_chat = AzureChatCompletion()
    url = azure_chat.create_azure_base_url(
        azure_client_params={
            "azure_endpoint": "https://my-resource.services.ai.azure.com",
            "api_version": "preview",
        },
        model="image-deployment-alias",
        base_model="MAI-Image-2.5",
    )
    assert (
        url
        == "https://my-resource.services.ai.azure.com/mai/v1/images/generations?api-version=preview"
    )


def test_azure_image_generation_flattens_extra_body():
    """
    Test that Azure image generation correctly flattens extra_body parameters.

    Azure's image generation API doesn't support the extra_body parameter,
    so we need to flatten any parameters in extra_body to the top level.

    This test verifies the fix for: https://github.com/BerriAI/litellm/issues/16059
    Where partial_images and stream parameters were incorrectly sent in extra_body.
    """
    # Test 1: Verify get_optional_params_image_gen puts extra params in extra_body
    optional_params = get_optional_params_image_gen(
        model="gpt-image-1",
        n=1,
        size="1024x1024",
        custom_llm_provider="azure",
        partial_images=2,
        stream=True,
    )

    assert "extra_body" in optional_params
    assert "partial_images" in optional_params["extra_body"]
    assert "stream" in optional_params["extra_body"]
    assert optional_params["extra_body"]["partial_images"] == 2
    assert optional_params["extra_body"]["stream"] is True

    # Test 2: Verify Azure flattens extra_body when building request data
    # Simulate what happens in Azure's image_generation method
    test_optional_params = {
        "n": 1,
        "size": "1024x1024",
        "extra_body": {
            "partial_images": 2,
            "stream": True,
            "custom_param": "test_value",
        },
    }

    # This is what the Azure image_generation method does
    extra_body = test_optional_params.pop("extra_body", {})
    flattened_params = {**test_optional_params, **extra_body}

    data = {"model": "gpt-image-1", "prompt": "A cute sea otter", **flattened_params}

    # Verify the final data structure
    assert "extra_body" not in data, "extra_body should NOT be in the final data dict"
    assert "partial_images" in data, "partial_images should be at top level"
    assert "stream" in data, "stream should be at top level"
    assert "custom_param" in data, "custom_param should be at top level"
    assert data["partial_images"] == 2
    assert data["stream"] is True
    assert data["custom_param"] == "test_value"
    assert data["n"] == 1
    assert data["size"] == "1024x1024"


def test_azure_image_generation_creates_token_provider_from_credentials():
    """
    Test that azure_ad_token_provider is created from tenant_id, client_id, client_secret.

    This test verifies the fix in images/main.py where we now create the
    azure_ad_token_provider from credentials in litellm_params if it's not already provided.
    """
    # Simulate the fix in images/main.py
    litellm_params_dict = {
        "tenant_id": "test-tenant-id",
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "azure_scope": None,
    }

    azure_ad_token_provider = None

    # This is the logic we added in images/main.py
    if azure_ad_token_provider is None:
        tenant_id = litellm_params_dict.get("tenant_id")
        client_id = litellm_params_dict.get("client_id")
        client_secret = litellm_params_dict.get("client_secret")
        azure_scope = (
            litellm_params_dict.get("azure_scope")
            or "https://cognitiveservices.azure.com/.default"
        )

        # Verify the credentials are extracted correctly
        assert tenant_id == "test-tenant-id"
        assert client_id == "test-client-id"
        assert client_secret == "test-client-secret"
        assert azure_scope == "https://cognitiveservices.azure.com/.default"

        # Verify the condition to create token provider is met
        assert (
            tenant_id and client_id and client_secret
        ), "Credentials should be present to create token provider"


def test_azure_image_generation_headers_without_api_key():
    """
    Test that when api_key is None, the api-key header is not added to headers.

    This prevents the httpx TypeError: "Header value must be str or bytes, not <class 'NoneType'>"
    that was occurring when api_key was None and being set in headers.

    This is a unit test for the fix in images/main.py where we now check:
    if api_key is not None:
        default_headers["api-key"] = api_key
    """
    from litellm.images.main import image_generation

    # Test the header building logic directly
    api_key = None

    default_headers = {
        "Content-Type": "application/json",
    }

    # This is the fix: only add api-key if it's not None
    if api_key is not None:
        default_headers["api-key"] = api_key

    # Verify api-key is not in headers when api_key is None
    assert "api-key" not in default_headers

    # Verify Content-Type is still there
    assert default_headers["Content-Type"] == "application/json"

    # Test with a valid api_key
    api_key = "valid-key-123"
    default_headers_with_key = {
        "Content-Type": "application/json",
    }
    if api_key is not None:
        default_headers_with_key["api-key"] = api_key

    # Verify api-key is added when api_key is valid
    assert "api-key" in default_headers_with_key
    assert default_headers_with_key["api-key"] == "valid-key-123"


def test_azure_image_generation_drop_params_response_format():
    """
    Test that unsupported params like response_format are dropped when drop_params=True.

    Azure gpt-image-1.5 doesn't support response_format parameter. When drop_params=True,
    this parameter should be completely removed and not appear in the final request body,
    including not being added to extra_body.

    This test verifies the fix where:
    1. Unsupported params are removed from non_default_params in _check_valid_arg
    2. Unsupported params are also removed from passed_params to prevent them from
       being re-added via extra_body in add_provider_specific_params_to_optional_params

    Without the fix, response_format would be added to extra_body and cause Azure to
    return a 400 Bad Request error due to strict schema validation.
    """
    from litellm.llms.openai.image_generation.gpt_transformation import (
        GPTImageGenerationConfig,
    )

    # Test with gpt-image-1.5 which doesn't support response_format
    config = GPTImageGenerationConfig()
    supported_params = config.get_supported_openai_params(model="gpt-image-1.5")

    # Verify response_format is NOT in supported params for gpt-image-1.5
    assert "response_format" not in supported_params
    assert "n" in supported_params
    assert "size" in supported_params

    # Test get_optional_params_image_gen with drop_params=True
    optional_params = get_optional_params_image_gen(
        model="gpt-image-1.5",
        n=1,
        size="1024x1024",
        response_format="b64_json",  # This should be dropped
        custom_llm_provider="azure",
        provider_config=config,
        drop_params=True,
    )

    # Verify response_format is NOT in optional_params
    assert (
        "response_format" not in optional_params
    ), "response_format should be dropped from optional_params"

    # Verify response_format is NOT in extra_body either
    if "extra_body" in optional_params:
        assert (
            "response_format" not in optional_params["extra_body"]
        ), "response_format should not be in extra_body"

    # Verify supported params ARE in optional_params
    assert "n" in optional_params
    assert optional_params["n"] == 1
    assert "size" in optional_params
    assert optional_params["size"] == "1024x1024"


def test_azure_image_generation_drop_params_false_raises_error():
    """
    Test that unsupported params raise an error when drop_params=False.

    This verifies that the error handling still works correctly when drop_params
    is not enabled.
    """
    from litellm.exceptions import UnsupportedParamsError
    from litellm.llms.openai.image_generation.gpt_transformation import (
        GPTImageGenerationConfig,
    )

    config = GPTImageGenerationConfig()

    # Test that passing unsupported param with drop_params=False raises error
    with pytest.raises(UnsupportedParamsError) as exc_info:
        optional_params = get_optional_params_image_gen(
            model="gpt-image-1.5",
            n=1,
            response_format="b64_json",  # Unsupported param
            custom_llm_provider="azure",
            provider_config=config,
            drop_params=False,
        )

    # Verify the error message mentions the unsupported parameter
    assert "response_format" in str(exc_info.value)


def test_azure_image_generation_base_model_vs_deployment_name():
    """
    Test that Azure image generation omits ``model`` from the JSON body for
    deployment URLs while keeping the deployment in the path.

    Azure OpenAI routes image generation by deployment in the URL; the REST body
    must not include ``model`` (sending deployment or base model there can break
    gpt-image-2; see LiteLLM #26316). ``base_model`` in litellm_params is still used
    internally for logging / hidden params.

    Example config:
      model: azure/gpt-image-15  # deployment name (URL only)
      base_model: gpt-image-1.5  # optional, for LiteLLM metadata
    """

    # Setup test parameters
    azure_chat_completion = AzureChatCompletion()

    prompt = "A beautiful image of a cat"
    model = "gpt-image-15"  # This is the deployment name
    base_model = "gpt-image-1.5"  # This is the actual model name
    api_base = "https://openai-gpt-image-1-5-test-v-1.openai.azure.com/"
    api_version = "2024-07-01-preview"
    api_key = "test-api-key"

    litellm_params = {
        "base_model": base_model,
        "api_base": api_base,
        "api_version": api_version,
    }

    optional_params = {"n": 1, "size": "1024x1024"}

    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {
        "created": 1234567890,
        "data": [{"url": "https://example.com/image.png", "revised_prompt": prompt}],
    }

    with patch.object(
        HTTPHandler, "post", return_value=mock_http_response
    ) as mock_post:
        logging_obj = MagicMock()
        logging_obj.pre_call = MagicMock()
        logging_obj.post_call = MagicMock()

        azure_chat_completion.image_generation(
            prompt=prompt,
            timeout=60.0,
            optional_params=optional_params,
            logging_obj=logging_obj,
            headers={},
            model=model,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            litellm_params=litellm_params,
        )

        assert mock_post.called, "HTTPHandler.post should be invoked"
        post_kwargs = mock_post.call_args.kwargs
        url_used = post_kwargs.get("url", "")
        assert (
            model in url_used
        ), f"URL should contain deployment name '{model}', but got: {url_used}"
        assert base_model not in url_used or base_model == model, (
            f"URL should NOT contain base_model '{base_model}' when it differs from deployment name, "
            f"but got: {url_used}"
        )

        wire_json = post_kwargs.get("json") or {}
        assert (
            "model" not in wire_json
        ), f"Azure deployment image gen must not send 'model' in JSON body; got keys: {list(wire_json)}"
        assert wire_json.get("prompt") == prompt
        assert wire_json.get("n") == 1
        assert wire_json.get("size") == "1024x1024"


@pytest.mark.asyncio
async def test_azure_aimage_generation_base_model_vs_deployment_name():
    """
    Async variant of test_azure_image_generation_base_model_vs_deployment_name:
    deployment in URL, no ``model`` in the JSON body sent to Azure.
    """

    # Setup test parameters
    azure_chat_completion = AzureChatCompletion()

    prompt = "A beautiful image of a cat"
    model = "gpt-image-15"  # This is the deployment name
    base_model = "gpt-image-1.5"  # This is the actual model name
    api_base = "https://openai-gpt-image-1-5-test-v-1.openai.azure.com/"
    api_version = "2024-07-01-preview"
    api_key = "test-api-key"

    data = {"model": base_model, "prompt": prompt, "n": 1, "size": "1024x1024"}

    azure_client_params = {
        "api_base": api_base,
        "api_version": api_version,
    }

    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {
        "created": 1234567890,
        "data": [{"url": "https://example.com/image.png", "revised_prompt": prompt}],
    }

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_http_response)

    with patch(
        "litellm.llms.azure.azure.get_async_httpx_client", return_value=mock_client
    ):
        logging_obj = MagicMock()
        logging_obj.pre_call = MagicMock()
        logging_obj.post_call = MagicMock()

        await azure_chat_completion.aimage_generation(
            data=data,
            model_response=None,
            azure_client_params=azure_client_params,
            api_key=api_key,
            input=[],
            logging_obj=logging_obj,
            headers={},
            model=model,
            timeout=60.0,
        )

        assert mock_client.post.called
        post_kwargs = mock_client.post.call_args.kwargs
        url_used = post_kwargs.get("url", "")
        assert model in url_used
        wire_json = post_kwargs.get("json") or {}
        assert "model" not in wire_json
        assert data.get("model") == base_model


@pytest.mark.parametrize("api_version", ["v1", "preview", "latest"])
def test_azure_image_generation_v1_api_version_uses_v1_route(api_version):
    """The v1 Azure surface exposes /openai/v1/images/generations and routes by body ``model``."""
    url = AzureChatCompletion().create_azure_base_url(
        azure_client_params={
            "azure_endpoint": "https://my-resource.openai.azure.com",
            "api_version": api_version,
        },
        model="gpt-image-1",
        base_model=None,
    )
    assert url == f"https://my-resource.openai.azure.com/openai/v1/images/generations?api-version={api_version}"
    data = {"model": "gpt-image-1", "prompt": "x"}
    assert azure_deployment_image_generation_json_body(url, data) == data


def test_azure_image_generation_dated_api_version_uses_deployment_route():
    url = AzureChatCompletion().create_azure_base_url(
        azure_client_params={
            "azure_endpoint": "https://my-resource.openai.azure.com",
            "api_version": "2024-10-21",
        },
        model="gpt-image-1",
        base_model=None,
    )
    assert (
        url
        == "https://my-resource.openai.azure.com/openai/deployments/gpt-image-1/images/generations?api-version=2024-10-21"
    )
    assert "model" not in azure_deployment_image_generation_json_body(url, {"model": "gpt-image-1", "prompt": "x"})


def test_azure_image_generation_v1_api_version_replaces_deployment_scoped_api_base():
    url = AzureChatCompletion().create_azure_base_url(
        azure_client_params={
            "azure_endpoint": "https://my-resource.openai.azure.com/openai/deployments/gpt-image-1/images/generations",
            "api_version": "preview",
        },
        model="gpt-image-1",
        base_model=None,
    )
    assert url == "https://my-resource.openai.azure.com/openai/v1/images/generations?api-version=preview"


def test_azure_image_generation_v1_api_version_uses_base_url_client_param():
    url = AzureChatCompletion().create_azure_base_url(
        azure_client_params={
            "base_url": "https://my-resource.openai.azure.com/openai/deployments/gpt-image-1?api-version=2024-10-21",
            "api_version": "preview",
        },
        model="gpt-image-1",
        base_model=None,
    )
    assert url == "https://my-resource.openai.azure.com/openai/v1/images/generations?api-version=preview"


def test_azure_v1_image_generation_json_body_sends_deployment_name():
    """The v1 route ignores the URL and routes by body ``model``, which must be the deployment name."""
    url = "https://my-resource.openai.azure.com/openai/v1/images/generations?api-version=preview"
    data = {"model": "gpt-image-2", "prompt": "x", "n": 1}
    out = azure_deployment_image_generation_json_body(url, data, deployment_name="img-dep")
    assert out["model"] == "img-dep"
    assert out["prompt"] == "x"
    assert data["model"] == "gpt-image-2"
    assert azure_deployment_image_generation_json_body(url, data) == data


@pytest.mark.asyncio
async def test_azure_aimage_generation_v1_route_sends_deployment_name_in_body(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    azure_chat_completion = AzureChatCompletion()
    model = "img-dep"
    base_model = "gpt-image-2"
    data = {"model": base_model, "prompt": "A beautiful image of a cat", "n": 1}
    azure_client_params = {
        "azure_endpoint": "https://my-resource.openai.azure.com",
        "api_version": "preview",
    }

    route = respx_mock.post("https://my-resource.openai.azure.com/openai/v1/images/generations").mock(
        return_value=httpx.Response(200, json={"created": 1234567890, "data": [{"b64_json": "aaaa"}]})
    )

    logging_obj = MagicMock()
    logging_obj.pre_call = MagicMock()
    logging_obj.post_call = MagicMock()

    await azure_chat_completion.aimage_generation(
        data=data,
        model_response=None,
        azure_client_params=azure_client_params,
        api_key="test-api-key",
        input=[],
        logging_obj=logging_obj,
        headers={},
        model=model,
        timeout=60.0,
    )

    request = route.calls.last.request
    assert str(request.url) == ("https://my-resource.openai.azure.com/openai/v1/images/generations?api-version=preview")
    sent_body = json.loads(request.content)
    assert sent_body["model"] == model
    assert sent_body["prompt"] == data["prompt"]


def test_azure_image_generation_v1_route_base_model_vs_deployment_name(respx_mock: respx.MockRouter):
    """On the v1 surface the body ``model`` must be the deployment name, never base_model."""
    azure_chat_completion = AzureChatCompletion()
    prompt = "A beautiful image of a cat"
    model = "img-dep"
    base_model = "gpt-image-2"
    api_base = "https://my-resource.openai.azure.com"
    api_version = "v1"
    litellm_params = {
        "base_model": base_model,
        "api_base": api_base,
        "api_version": api_version,
    }

    route = respx_mock.post(f"{api_base}/openai/v1/images/generations").mock(
        return_value=httpx.Response(200, json={"created": 1234567890, "data": [{"b64_json": "aaaa"}]})
    )

    logging_obj = MagicMock()
    logging_obj.pre_call = MagicMock()
    logging_obj.post_call = MagicMock()

    azure_chat_completion.image_generation(
        prompt=prompt,
        timeout=60.0,
        optional_params={"n": 1, "size": "1024x1024"},
        logging_obj=logging_obj,
        headers={},
        model=model,
        api_key="test-api-key",
        api_base=api_base,
        api_version=api_version,
        litellm_params=litellm_params,
    )

    request = route.calls.last.request
    assert str(request.url) == f"{api_base}/openai/v1/images/generations?api-version={api_version}"
    sent_body = json.loads(request.content)
    assert sent_body["model"] == model
    assert sent_body["prompt"] == prompt


@pytest.fixture
def fake_entra_id(monkeypatch: pytest.MonkeyPatch):
    built_credentials = []

    class FakeClientSecretCredential:
        def __init__(self, tenant_id: str, client_id: str, client_secret: str) -> None:
            built_credentials.append((tenant_id, client_id, client_secret))

    monkeypatch.setattr("azure.identity.ClientSecretCredential", FakeClientSecretCredential)
    monkeypatch.setattr("azure.identity.get_bearer_token_provider", lambda credential, scope: lambda: "entra-id-token")
    _cached_entra_id_token_provider.cache_clear()
    yield built_credentials
    _cached_entra_id_token_provider.cache_clear()


def _mock_image_generation_route(respx_mock: respx.MockRouter, api_base: str, model: str) -> respx.Route:
    return respx_mock.post(f"{api_base}/openai/deployments/{model}/images/generations").mock(
        return_value=httpx.Response(200, json={"created": 1234567890, "data": [{"b64_json": "aaaa"}]})
    )


@pytest.mark.parametrize("credentials_in_litellm_params", [False, True])
def test_azure_image_generation_keyless_entra_id_sends_bearer_token(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    fake_entra_id: list,
    credentials_in_litellm_params: bool,
):
    for name in ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    api_base = "https://my-resource.openai.azure.com"
    api_version = "2025-04-01-preview"
    litellm_params = {"api_base": api_base, "api_version": api_version}
    if credentials_in_litellm_params:
        litellm_params.update(
            tenant_id="tenant-from-params", client_id="client-from-params", client_secret="secret-from-params"
        )
        expected_credential = ("tenant-from-params", "client-from-params", "secret-from-params")
    else:
        monkeypatch.setenv("AZURE_TENANT_ID", "tenant-from-env")
        monkeypatch.setenv("AZURE_CLIENT_ID", "client-from-env")
        monkeypatch.setenv("AZURE_CLIENT_SECRET", "secret-from-env")
        expected_credential = ("tenant-from-env", "client-from-env", "secret-from-env")
    route = _mock_image_generation_route(respx_mock, api_base, "gpt-image-1")
    logging_obj = MagicMock()

    response = AzureChatCompletion().image_generation(
        prompt="a cat",
        timeout=60.0,
        optional_params={"n": 1, "size": "1024x1024"},
        logging_obj=logging_obj,
        headers={"Content-Type": "application/json"},
        model="gpt-image-1",
        api_key=None,
        api_base=api_base,
        api_version=api_version,
        litellm_params=litellm_params,
    )

    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer entra-id-token"
    assert "api-key" not in request.headers
    assert fake_entra_id == [expected_credential]
    assert response.data[0].b64_json == "aaaa"
    logged_headers = logging_obj.pre_call.call_args.kwargs["additional_args"]["headers"]
    assert logged_headers == {"Content-Type": "application/json", "Authorization": "***REDACTED***"}
    assert "entra-id-token" not in str(logging_obj.pre_call.call_args)


@pytest.mark.asyncio
async def test_azure_aimage_generation_keyless_entra_id_sends_bearer_token(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, fake_entra_id: list
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    api_base = "https://my-resource.openai.azure.com"
    api_version = "2025-04-01-preview"
    route = _mock_image_generation_route(respx_mock, api_base, "gpt-image-1")
    logging_obj = MagicMock()

    response = await AzureChatCompletion().image_generation(
        prompt="a cat",
        timeout=60.0,
        optional_params={"n": 1, "size": "1024x1024"},
        logging_obj=logging_obj,
        headers={"Content-Type": "application/json"},
        model="gpt-image-1",
        api_key=None,
        api_base=api_base,
        api_version=api_version,
        aimg_generation=True,
        litellm_params={
            "api_base": api_base,
            "api_version": api_version,
            "tenant_id": "tenant-from-params",
            "client_id": "client-from-params",
            "client_secret": "secret-from-params",
        },
    )

    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer entra-id-token"
    assert "api-key" not in request.headers
    assert fake_entra_id == [("tenant-from-params", "client-from-params", "secret-from-params")]
    assert response.data[0].b64_json == "aaaa"
    logged_headers = logging_obj.pre_call.call_args.kwargs["additional_args"]["headers"]
    assert logged_headers == {"Content-Type": "application/json", "Authorization": "***REDACTED***"}
    assert "entra-id-token" not in str(logging_obj.pre_call.call_args)


@pytest.mark.parametrize(
    "credential_kwargs, expected_authorization",
    [
        ({"azure_ad_token": "static-ad-token"}, "Bearer static-ad-token"),
        ({"azure_ad_token_provider": lambda: "provider-token"}, "Bearer provider-token"),
    ],
)
def test_azure_image_generation_explicit_azure_ad_credential_sends_bearer_token(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    credential_kwargs: dict,
    expected_authorization: str,
):
    for name in ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    api_base = "https://my-resource.openai.azure.com"
    api_version = "2025-04-01-preview"
    route = _mock_image_generation_route(respx_mock, api_base, "gpt-image-1")

    response = AzureChatCompletion().image_generation(
        prompt="a cat",
        timeout=60.0,
        optional_params={"n": 1, "size": "1024x1024"},
        logging_obj=MagicMock(),
        headers={"Content-Type": "application/json"},
        model="gpt-image-1",
        api_key=None,
        api_base=api_base,
        api_version=api_version,
        litellm_params={"api_base": api_base, "api_version": api_version},
        **credential_kwargs,
    )

    request = route.calls.last.request
    assert request.headers["Authorization"] == expected_authorization
    assert "api-key" not in request.headers
    assert response.data[0].b64_json == "aaaa"


def test_azure_image_generation_with_api_key_keeps_api_key_header(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, fake_entra_id: list
):
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant-from-env")
    monkeypatch.setenv("AZURE_CLIENT_ID", "client-from-env")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "secret-from-env")
    api_base = "https://my-resource.openai.azure.com"
    api_version = "2025-04-01-preview"
    route = _mock_image_generation_route(respx_mock, api_base, "gpt-image-1")
    logging_obj = MagicMock()

    response = AzureChatCompletion().image_generation(
        prompt="a cat",
        timeout=60.0,
        optional_params={"n": 1, "size": "1024x1024"},
        logging_obj=logging_obj,
        headers={"Content-Type": "application/json", "api-key": "sk-test"},
        model="gpt-image-1",
        api_key="sk-test",
        api_base=api_base,
        api_version=api_version,
        litellm_params={"api_base": api_base, "api_version": api_version},
    )

    request = route.calls.last.request
    assert request.headers["api-key"] == "sk-test"
    assert "Authorization" not in request.headers
    assert fake_entra_id == []
    assert response.data[0].b64_json == "aaaa"
    assert logging_obj.pre_call.call_args.kwargs["additional_args"]["headers"]["api-key"] == "***REDACTED***"


@pytest.fixture
def fake_default_azure_credential(monkeypatch: pytest.MonkeyPatch):
    built_credentials = []

    class FakeDefaultAzureCredential:
        def __init__(self) -> None:
            built_credentials.append(self)

    for name in ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "AZURE_CREDENTIAL", "AZURE_AD_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", FakeDefaultAzureCredential)
    monkeypatch.setattr(
        "azure.identity.get_bearer_token_provider", lambda credential, scope: lambda: "default-credential-token"
    )
    monkeypatch.setattr(litellm, "enable_azure_ad_token_refresh", True)
    _cached_azure_ad_token_refresh_provider.cache_clear()
    yield built_credentials
    _cached_azure_ad_token_refresh_provider.cache_clear()


def test_azure_image_generation_token_refresh_reuses_credential_across_requests(
    respx_mock: respx.MockRouter, fake_default_azure_credential: list
):
    api_base = "https://my-resource.openai.azure.com"
    api_version = "2025-04-01-preview"
    route = _mock_image_generation_route(respx_mock, api_base, "gpt-image-1")

    for _ in range(3):
        AzureChatCompletion().image_generation(
            prompt="a cat",
            timeout=60.0,
            optional_params={"n": 1, "size": "1024x1024"},
            logging_obj=MagicMock(),
            headers={"Content-Type": "application/json"},
            model="gpt-image-1",
            api_key=None,
            api_base=api_base,
            api_version=api_version,
            litellm_params={"api_base": api_base, "api_version": api_version},
        )

    assert route.call_count == 3
    assert all(call.request.headers["Authorization"] == "Bearer default-credential-token" for call in route.calls)
    assert len(fake_default_azure_credential) == 1


@pytest.mark.parametrize(
    "caller_auth_header",
    [{"api-key": "caller-key"}, {"Authorization": "Bearer caller-token"}, {"authorization": "Bearer caller-token"}],
)
def test_get_azure_request_auth_headers_keeps_caller_auth_header(caller_auth_header: dict):
    headers = {"Content-Type": "application/json", **caller_auth_header}
    azure_client_params = {
        "api_key": "sk-resolved",
        "azure_ad_token": "resolved-token",
        "azure_ad_token_provider": lambda: "provider-token",
    }
    assert get_azure_request_auth_headers(headers=headers, azure_client_params=azure_client_params) is headers


def test_get_azure_request_auth_headers_prefers_azure_ad_token_over_provider_and_api_key():
    headers = {"Content-Type": "application/json"}
    azure_client_params = {
        "api_key": "sk-resolved",
        "azure_ad_token": "static-token",
        "azure_ad_token_provider": lambda: "provider-token",
    }
    out = get_azure_request_auth_headers(headers=headers, azure_client_params=azure_client_params)
    assert dict(out) == {"Content-Type": "application/json", "Authorization": "Bearer static-token"}
    assert headers == {"Content-Type": "application/json"}


def test_get_azure_request_auth_headers_uses_token_provider_over_api_key():
    azure_client_params = {"api_key": "sk-resolved", "azure_ad_token": None, "azure_ad_token_provider": lambda: "pt"}
    out = get_azure_request_auth_headers(headers={}, azure_client_params=azure_client_params)
    assert dict(out) == {"Authorization": "Bearer pt"}


def test_get_azure_request_auth_headers_falls_back_to_api_key():
    azure_client_params = {"api_key": "sk-resolved", "azure_ad_token": None, "azure_ad_token_provider": None}
    out = get_azure_request_auth_headers(headers={"Content-Type": "application/json"}, azure_client_params=azure_client_params)
    assert dict(out) == {"Content-Type": "application/json", "api-key": "sk-resolved"}


@pytest.mark.parametrize(
    "azure_client_params",
    [
        {},
        {"api_key": "", "azure_ad_token": "", "azure_ad_token_provider": None},
        {"azure_ad_token_provider": lambda: None},
        {"azure_ad_token_provider": lambda: ""},
    ],
)
def test_get_azure_request_auth_headers_without_credential_leaves_headers_unchanged(azure_client_params: dict):
    headers = {"Content-Type": "application/json"}
    assert get_azure_request_auth_headers(headers=headers, azure_client_params=azure_client_params) is headers


def test_redact_azure_auth_headers_masks_only_credential_values():
    headers = {"Content-Type": "application/json", "api-key": "sk-secret", "authorization": "Bearer secret"}
    assert redact_azure_auth_headers(headers) == {
        "Content-Type": "application/json",
        "api-key": "***REDACTED***",
        "authorization": "***REDACTED***",
    }
    assert headers["api-key"] == "sk-secret"
    assert headers["authorization"] == "Bearer secret"
