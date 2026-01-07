import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io

from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm.types.router import LiteLLM_Params

def test_get_llm_provider():
    _, response, _, _ = litellm.get_llm_provider(model="anthropic.claude-v2:1")

    assert response == "bedrock"


# test_get_llm_provider()


def test_get_llm_provider_fireworks():  # tests finetuned fireworks models - https://github.com/BerriAI/litellm/issues/4923
    model, custom_llm_provider, _, _ = litellm.get_llm_provider(
        model="fireworks_ai/accounts/my-test-1234"
    )

    assert custom_llm_provider == "fireworks_ai"
    assert model == "accounts/my-test-1234"


def test_get_llm_provider_catch_all():
    _, response, _, _ = litellm.get_llm_provider(model="*")
    assert response == "openai"


def test_get_llm_provider_gpt_instruct():
    _, response, _, _ = litellm.get_llm_provider(model="gpt-3.5-turbo-instruct-0914")

    assert response == "text-completion-openai"


def test_get_llm_provider_mistral_custom_api_base():
    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="mistral/mistral-large-fr",
        api_base="https://mistral-large-fr-ishaan.francecentral.inference.ai.azure.com/v1",
    )
    assert custom_llm_provider == "mistral"
    assert model == "mistral-large-fr"
    assert (
        api_base
        == "https://mistral-large-fr-ishaan.francecentral.inference.ai.azure.com/v1"
    )


def test_get_llm_provider_deepseek_custom_api_base():
    os.environ["DEEPSEEK_API_BASE"] = "MY-FAKE-BASE"
    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="deepseek/deep-chat",
    )
    assert custom_llm_provider == "deepseek"
    assert model == "deep-chat"
    assert api_base == "MY-FAKE-BASE"

    os.environ.pop("DEEPSEEK_API_BASE")


def test_get_llm_provider_vertex_ai_image_models():
    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="imagegeneration@006", custom_llm_provider=None
    )
    assert custom_llm_provider == "vertex_ai"


def test_get_llm_provider_ai21_chat():
    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="jamba-1.5-large",
    )
    assert custom_llm_provider == "ai21_chat"
    assert model == "jamba-1.5-large"
    assert api_base == "https://api.ai21.com/studio/v1"


def test_get_llm_provider_ai21_chat_test2():
    """
    if user prefix with ai21/ but calls jamba-1.5-large then it should be ai21_chat provider
    """
    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="ai21/jamba-1.5-large",
    )

    print("model=", model)
    print("custom_llm_provider=", custom_llm_provider)
    print("api_base=", api_base)
    assert custom_llm_provider == "ai21_chat"
    assert model == "jamba-1.5-large"
    assert api_base == "https://api.ai21.com/studio/v1"


def test_get_llm_provider_cohere_chat_test2():
    """
    if user prefix with cohere/ but calls command-r-plus then it should be cohere_chat provider
    """
    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="cohere/command-r-plus",
    )

    print("model=", model)
    print("custom_llm_provider=", custom_llm_provider)
    print("api_base=", api_base)
    assert custom_llm_provider == "cohere_chat"
    assert model == "command-r-plus"


def test_get_llm_provider_azure_o1():

    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="azure/o1-mini",
    )
    assert custom_llm_provider == "azure"
    assert model == "o1-mini"


def test_default_api_base():
    from litellm.litellm_core_utils.get_llm_provider_logic import (
        _get_openai_compatible_provider_info,
    )

    # Patch environment variable to remove API base if it's set
    with patch.dict(os.environ, {}, clear=True):
        for provider in litellm.openai_compatible_providers:
            # Get the API base for the given provider
            if provider == "github_copilot":
                continue
            # Skip ragflow as it requires specific model format: ragflow/chat/{id}/{model} or ragflow/agent/{id}/{model}
            if provider == "ragflow":
                continue
            _, _, _, api_base = _get_openai_compatible_provider_info(
                model=f"{provider}/*", api_base=None, api_key=None, dynamic_api_key=None
            )
            if api_base is None:
                continue

            for other_provider in litellm.provider_list:
                if other_provider != provider and provider != "{}_chat".format(
                    other_provider.value
                ):
                    if provider == "codestral" and other_provider == "mistral":
                        continue
                    elif provider == "github" and other_provider == "azure":
                        continue
                    assert other_provider.value not in api_base.replace("/openai", "")


def test_hosted_vllm_default_api_key():
    from litellm.litellm_core_utils.get_llm_provider_logic import (
        _get_openai_compatible_provider_info,
    )

    _, _, dynamic_api_key, _ = _get_openai_compatible_provider_info(
        model="hosted_vllm/llama-3.1-70b-instruct",
        api_base=None,
        api_key=None,
        dynamic_api_key=None,
    )
    assert dynamic_api_key == "fake-api-key"


def test_get_llm_provider_jina_ai():
    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="jina_ai/jina-embeddings-v3",
    )
    assert custom_llm_provider == "jina_ai"
    assert api_base == "https://api.jina.ai/v1"
    assert model == "jina-embeddings-v3"


def test_get_llm_provider_hosted_vllm():
    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="hosted_vllm/llama-3.1-70b-instruct",
    )
    assert custom_llm_provider == "hosted_vllm"
    assert model == "llama-3.1-70b-instruct"
    assert dynamic_api_key == "fake-api-key"


def test_get_llm_provider_llamafile():
    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="llamafile/mistralai/mistral-7b-instruct-v0.2",
    )
    assert custom_llm_provider == "llamafile"
    assert model == "mistralai/mistral-7b-instruct-v0.2"
    assert dynamic_api_key == "fake-api-key"
    assert api_base == "http://127.0.0.1:8080/v1"


def test_get_llm_provider_watson_text():
    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="watsonx_text/watson-text-to-speech",
    )
    assert custom_llm_provider == "watsonx_text"
    assert model == "watson-text-to-speech"


def test_azure_global_standard_get_llm_provider():
    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="azure_ai/gpt-4o-global-standard",
        api_base="https://my-deployment-francecentral.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview",
        api_key="fake-api-key",
    )
    assert custom_llm_provider == "azure_ai"


def test_nova_bedrock_converse():
    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="amazon.nova-micro-v1:0",
    )
    assert custom_llm_provider == "bedrock"
    assert model == "amazon.nova-micro-v1:0"


def test_bedrock_invoke_anthropic():
    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        model="bedrock/invoke/anthropic.claude-3-5-sonnet-20240620-v1:0",
    )
    assert custom_llm_provider == "bedrock"
    assert model == "invoke/anthropic.claude-3-5-sonnet-20240620-v1:0"


@pytest.mark.parametrize("model", ["xai/grok-2-vision-latest", "grok-2-vision-latest"])
def test_xai_api_base(model):
    args = {
        "model": model,
        "custom_llm_provider": "xai",
        "api_base": None,
        "api_key": "xai-my-specialkey",
        "litellm_params": None,
    }
    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(
        **args
    )
    assert custom_llm_provider == "xai"
    assert model == "grok-2-vision-latest"
    assert api_base == "https://api.x.ai/v1"
    assert dynamic_api_key == "xai-my-specialkey"

# -------- Tests for force_use_litellm_proxy ---------

def test_get_litellm_proxy_custom_llm_provider():
    """
    Tests force_use_litellm_proxy uses LITELLM_PROXY_API_BASE and LITELLM_PROXY_API_KEY from env.
    """
    test_model = "gpt-3.5-turbo"
    expected_api_base = "http://localhost:8000"
    expected_api_key = "test_proxy_key"

    with patch.dict(os.environ, {
        "LITELLM_PROXY_API_BASE": expected_api_base,
        "LITELLM_PROXY_API_KEY": expected_api_key
    }, clear=True):
        model, provider, key, base = litellm.LiteLLMProxyChatConfig().litellm_proxy_get_custom_llm_provider_info(model=test_model)

    assert model == test_model
    assert provider == "litellm_proxy"
    assert key == expected_api_key
    assert base == expected_api_base

def test_get_litellm_proxy_with_args_override_env_vars():
    """
    Tests force_use_litellm_proxy uses api_base and api_key args over environment variables.
    """
    test_model = "gpt-4"
    arg_api_base = "http://custom-proxy.com"
    arg_api_key = "custom_key_from_arg"
    
    env_api_base = "http://env-proxy.com"
    env_api_key = "env_key"

    with patch.dict(os.environ, {
        "LITELLM_PROXY_API_BASE": env_api_base,
        "LITELLM_PROXY_API_KEY": env_api_key
    }, clear=True):
        model, provider, key, base = litellm.LiteLLMProxyChatConfig().litellm_proxy_get_custom_llm_provider_info(
            model=test_model,
            api_base=arg_api_base,
            api_key=arg_api_key
        )

    assert model == test_model
    assert provider == "litellm_proxy"
    assert key == arg_api_key
    assert base == arg_api_base

def test_get_litellm_proxy_model_prefix_stripping():
    """
    Tests force_use_litellm_proxy strips 'litellm_proxy/' prefix from model name.
    """
    original_model = "litellm_proxy/claude-2"
    expected_model = "claude-2"
    expected_api_base = "http://localhost:4000"
    expected_api_key = "proxy_secret_key"

    with patch.dict(os.environ, {
        "LITELLM_PROXY_API_BASE": expected_api_base,
        "LITELLM_PROXY_API_KEY": expected_api_key
    }, clear=True):
        model, provider, key, base = litellm.LiteLLMProxyChatConfig().litellm_proxy_get_custom_llm_provider_info(model=original_model)

    assert model == expected_model
    assert provider == "litellm_proxy"
    assert key == expected_api_key
    assert base == expected_api_base

# -------- Tests for get_llm_provider triggering use_litellm_proxy ---------

def test_get_llm_provider_LITELLM_PROXY_ALWAYS_true():
    """
    Tests get_llm_provider uses litellm_proxy when USE_LITELLM_PROXY is "True".
    """
    test_model_input = "openai/gpt-4"
    expected_model_output = "openai/gpt-4"
    proxy_api_base = "http://my-global-proxy.com"
    proxy_api_key = "global_proxy_key"

    with patch.dict(os.environ, {
        "USE_LITELLM_PROXY": "True",
        "LITELLM_PROXY_API_BASE": proxy_api_base,
        "LITELLM_PROXY_API_KEY": proxy_api_key
    }, clear=True):
        model, provider, key, base = litellm.get_llm_provider(model=test_model_input)
    
    print("get_llm_provider", model, provider, key, base)

    assert model == expected_model_output
    assert provider == "litellm_proxy"
    assert key == proxy_api_key
    assert base == proxy_api_base

def test_get_llm_provider_LITELLM_PROXY_ALWAYS_true_model_prefix():
    """
    Tests get_llm_provider with USE_LITELLM_PROXY="True" and model prefix "litellm_proxy/".
    """
    test_model_input = "litellm_proxy/gpt-4-turbo"
    expected_model_output = "gpt-4-turbo"
    proxy_api_base = "http://another-proxy.net"
    proxy_api_key = "another_key"

    with patch.dict(os.environ, {
        "USE_LITELLM_PROXY": "True",
        "LITELLM_PROXY_API_BASE": proxy_api_base,
        "LITELLM_PROXY_API_KEY": proxy_api_key
    }, clear=True):
        model, provider, key, base = litellm.get_llm_provider(model=test_model_input)

    assert model == expected_model_output
    assert provider == "litellm_proxy"
    assert key == proxy_api_key
    assert base == proxy_api_base


def test_get_llm_provider_use_proxy_arg_true():
    """
    Tests get_llm_provider uses litellm_proxy when use_proxy=True argument is passed.
    """
    test_model_input = "mistral/mistral-large"
    expected_model_output = "mistral/mistral-large" # force_use_litellm_proxy keep the model name
    proxy_api_base = "http://my-arg-proxy.com"
    proxy_api_key = "arg_proxy_key"
    
    # Ensure LITELLM_PROXY_ALWAYS is not set or False
    with patch.dict(os.environ, {
        "LITELLM_PROXY_API_BASE": proxy_api_base,
        "LITELLM_PROXY_API_KEY": proxy_api_key
    }, clear=True): # clear=True removes LITELLM_PROXY_ALWAYS if it was set by other tests
        model, provider, key, base = litellm.get_llm_provider(
            model=test_model_input, 
            litellm_params=LiteLLM_Params(use_litellm_proxy=True, model=test_model_input)
        )

    assert model == expected_model_output
    assert provider == "litellm_proxy"
    assert key == proxy_api_key
    assert base == proxy_api_base

def test_get_llm_provider_use_proxy_arg_true_with_direct_args():
    """
    Tests get_llm_provider with use_proxy=True and explicit api_base/api_key args.
    These args should be passed to force_use_litellm_proxy and override env vars.
    """
    test_model_input = "anthropic/claude-3-opus"
    expected_model_output = "anthropic/claude-3-opus"
    
    arg_api_base = "http://specific-proxy-endpoint.org"
    arg_api_key = "specific_key_for_call"

    # Set some env vars to ensure they are overridden
    env_proxy_api_base = "http://env-default-proxy.com"
    env_proxy_api_key = "env_default_key"

    with patch.dict(os.environ, {
        "LITELLM_PROXY_API_BASE": env_proxy_api_base,
        "LITELLM_PROXY_API_KEY": env_proxy_api_key
    }, clear=True):
        model, provider, key, base = litellm.get_llm_provider(
            model=test_model_input, 
            api_base=arg_api_base,
            api_key=arg_api_key,
            litellm_params=LiteLLM_Params(use_litellm_proxy=True, model=test_model_input)
        )

    assert model == expected_model_output
    assert provider == "litellm_proxy"
    assert key == arg_api_key  # Should use the argument key
    assert base == arg_api_base # Should use the argument base
