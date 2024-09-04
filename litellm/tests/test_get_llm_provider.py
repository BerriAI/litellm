import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

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
