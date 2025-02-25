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
