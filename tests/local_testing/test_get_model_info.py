# What is this?
## Unit testing for the 'get_model_info()' function
import os
import sys
import traceback
import json



from typing import List, Dict, Any

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
import pytest

import litellm
from litellm import get_model_info
from unittest.mock import AsyncMock, MagicMock, patch


def test_get_model_info_simple_model_name():
    """
    tests if model name given, and model exists in model info - the object is returned
    """
    model = "claude-3-opus-20240229"
    litellm.get_model_info(model)


def test_get_model_info_custom_llm_with_model_name():
    """
    Tests if {custom_llm_provider}/{model_name} name given, and model exists in model info, the object is returned
    """
    model = "anthropic/claude-3-opus-20240229"
    litellm.get_model_info(model)


def test_get_model_info_custom_llm_with_same_name_vllm(monkeypatch):
    """
    Tests if {custom_llm_provider}/{model_name} name given, and model exists in model info, the object is returned
    """
    model = "command-r-plus"
    provider = "openai"  # vllm is openai-compatible
    litellm.register_model(
        {
            "openai/command-r-plus": {
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
        }
    )
    model_info = litellm.get_model_info(model, custom_llm_provider=provider)
    print("model_info", model_info)
    assert model_info["input_cost_per_token"] == 0.0


def test_get_model_info_shows_correct_supports_vision():
    info = litellm.get_model_info("gemini/gemini-1.5-flash")
    print("info", info)
    assert info["supports_vision"] is True


def test_get_model_info_shows_assistant_prefill():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    info = litellm.get_model_info("deepseek/deepseek-chat")
    print("info", info)
    assert info.get("supports_assistant_prefill") is True


def test_get_model_info_shows_supports_prompt_caching():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    info = litellm.get_model_info("deepseek/deepseek-chat")
    print("info", info)
    assert info.get("supports_prompt_caching") is True


def test_get_model_info_finetuned_models():
    info = litellm.get_model_info("ft:gpt-3.5-turbo:my-org:custom_suffix:id")
    print("info", info)
    assert info["input_cost_per_token"] == 0.000003


def test_get_model_info_gemini_pro():
    info = litellm.get_model_info("gemini-1.5-pro-002")
    print("info", info)
    assert info["key"] == "gemini-1.5-pro-002"


def test_get_model_info_ollama_chat():
    from litellm.llms.ollama.completion.transformation import OllamaConfig

    with patch.object(
        litellm.module_level_client,
        "post",
        return_value=MagicMock(
            json=lambda: {
                "model_info": {"llama.context_length": 32768},
                "template": "tools",
            }
        ),
    ) as mock_client:
        info = OllamaConfig().get_model_info("unknown-model")
        assert info["supports_function_calling"] is True

        info = get_model_info("ollama/unknown-model")
        print("info", info)
        assert info["supports_function_calling"] is True

        mock_client.assert_called()

        print(mock_client.call_args.kwargs)

        assert mock_client.call_args.kwargs["json"]["name"] == "unknown-model"




def test_get_model_info_bedrock_region():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    args = {
        "model": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "custom_llm_provider": "bedrock",
    }
    litellm.model_cost.pop("us.anthropic.claude-3-5-sonnet-20241022-v2:0", None)
    info = litellm.get_model_info(**args)
    print("info", info)
    assert info["key"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert info["litellm_provider"] == "bedrock"


@pytest.mark.parametrize(
    "model",
    [
        "ft:gpt-3.5-turbo:my-org:custom_suffix:id",
        "ft:gpt-4-0613:my-org:custom_suffix:id",
        "ft:davinci-002:my-org:custom_suffix:id",
        "ft:gpt-4-0613:my-org:custom_suffix:id",
        "ft:babbage-002:my-org:custom_suffix:id",
        "gpt-35-turbo",
        "ada",
    ],
)
def test_get_model_info_completion_cost_unit_tests(model):
    info = litellm.get_model_info(model)
    print("info", info)


def test_get_model_info_ft_model_with_provider_prefix():
    args = {
        "model": "openai/ft:gpt-3.5-turbo:my-org:custom_suffix:id",
        "custom_llm_provider": "openai",
    }
    info = litellm.get_model_info(**args)
    print("info", info)
    assert info["key"] == "ft:gpt-3.5-turbo"



def _enforce_bedrock_converse_models(
    model_cost: List[Dict[str, Any]], whitelist_models: List[str]
):
    """
    Assert all new bedrock chat models are added as `bedrock_converse` unless explicitly whitelisted.
    """
    # Check for unwhitelisted models
    for model, info in litellm.model_cost.items():
        if (
            info["litellm_provider"] == "bedrock"
            and info["mode"] == "chat"
            and model not in whitelist_models
        ):
            raise AssertionError(
                f"New bedrock chat model detected: {model}. Please set `litellm_provider='bedrock_converse'` for this model."
            )


def test_model_info_bedrock_converse(monkeypatch):
    """
    Assert all new bedrock chat models are added as `bedrock_converse` unless explicitly whitelisted.

    This ensures they are automatically routed to the converse endpoint.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    try:
        # Load whitelist models from file
        with open("whitelisted_bedrock_models.txt", "r") as file:
            whitelist_models = [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        pytest.skip("whitelisted_bedrock_models.txt not found")

    _enforce_bedrock_converse_models(
        model_cost=litellm.model_cost, whitelist_models=whitelist_models
    )


@pytest.mark.flaky(retries=6, delay=2)
def test_model_info_bedrock_converse_enforcement(monkeypatch):
    """
    Test the enforcement of the whitelist by adding a fake model and ensuring the test fails.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")

    # Add a fake unwhitelisted model
    litellm.model_cost["fake.bedrock-chat-model"] = {
        "litellm_provider": "bedrock",
        "mode": "chat",
    }

    try:
        # Load whitelist models from file
        with open("whitelisted_bedrock_models.txt", "r") as file:
            whitelist_models = [line.strip() for line in file.readlines()]

        # Check for unwhitelisted models
        with pytest.raises(AssertionError):
            _enforce_bedrock_converse_models(
                model_cost=litellm.model_cost, whitelist_models=whitelist_models
            )
    except FileNotFoundError as e:
        pytest.skip("whitelisted_bedrock_models.txt not found")


def test_get_model_info_custom_provider():
    # Custom provider example copied from https://docs.litellm.ai/docs/providers/custom_llm_server:
    import litellm
    from litellm import CustomLLM, completion, get_llm_provider

    class MyCustomLLM(CustomLLM):
        def completion(self, *args, **kwargs) -> litellm.ModelResponse:
            return litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello world"}],
                mock_response="Hi!",
            )  # type: ignore

    my_custom_llm = MyCustomLLM()

    litellm.custom_provider_map = [  # 👈 KEY STEP - REGISTER HANDLER
        {"provider": "my-custom-llm", "custom_handler": my_custom_llm}
    ]

    resp = completion(
        model="my-custom-llm/my-fake-model",
        messages=[{"role": "user", "content": "Hello world!"}],
    )

    assert resp.choices[0].message.content == "Hi!"

    # Register model info
    model_info = {"my-custom-llm/my-fake-model": {"max_tokens": 2048}}
    litellm.register_model(model_info)

    # Get registered model info
    from litellm import get_model_info

    get_model_info(
        model="my-custom-llm/my-fake-model"
    )  # 💥 "Exception: This model isn't mapped yet." in v1.56.10


def test_get_model_info_custom_model_router():
    from litellm import Router
    from litellm import get_model_info

    litellm._turn_on_debug()

    router = Router(
        model_list=[
            {
                "model_name": "ma-summary",
                "litellm_params": {
                    "api_base": "http://ma-mix-llm-serving.cicero.svc.cluster.local/v1",
                    "input_cost_per_token": 1,
                    "output_cost_per_token": 1,
                    "model": "openai/meta-llama/Meta-Llama-3-8B-Instruct",
                },
                "model_info": {
                    "id": "c20d603e-1166-4e0f-aa65-ed9c476ad4ca",
                }
            }
        ]
    )
    info = get_model_info("c20d603e-1166-4e0f-aa65-ed9c476ad4ca")
    print("info", info)
    assert info is not None


def test_get_model_info_bedrock_models():
    """
    Check for drift in base model info for bedrock models and regional model info for bedrock models.
    """
    from litellm.llms.bedrock.common_utils import BedrockModelInfo

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    for k, v in litellm.model_cost.items():
        if v["litellm_provider"] == "bedrock":
            k = k.replace("*/", "")
            potential_commitments = [
                "1-month-commitment",
                "3-month-commitment",
                "6-month-commitment",
            ]
            if any(commitment in k for commitment in potential_commitments):
                for commitment in potential_commitments:
                    k = k.replace(f"{commitment}/", "")
            base_model = BedrockModelInfo.get_base_model(k)
            # get_base_model() returns model id without "bedrock/" prefix; cost map keys use "bedrock/<model>"
            base_model_key = (
                base_model
                if base_model in litellm.model_cost
                else f"bedrock/{base_model}"
            )
            if base_model_key not in litellm.model_cost:
                continue
            base_model_info = litellm.model_cost[base_model_key]
            for base_model_key, base_model_value in base_model_info.items():
                if "invoke/" in k:
                    continue
                if base_model_key.startswith("supports_"):
                    assert (
                        base_model_key in v
                    ), f"{base_model_key} is not in model cost map for {k}"
                    assert (
                        v[base_model_key] == base_model_value
                    ), f"{base_model_key} is not equal to {base_model_value} for model {k}"


def test_get_model_info_huggingface_models(monkeypatch):
    from litellm import Router
    from litellm.types.router import ModelGroupInfo

    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_abc123")

    router = Router(
        model_list=[
            {
                "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
                "litellm_params": {
                    "model": "huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
                    "api_base": "https://router.huggingface.co/hf-inference/models/meta-llama/Meta-Llama-3-8B-Instruct",
                    "api_key": os.environ["HUGGINGFACE_API_KEY"],
                },
            }
        ]
    )
    info = litellm.get_model_info("huggingface/meta-llama/Meta-Llama-3-8B-Instruct")
    print("info", info)
    assert info is not None

    ModelGroupInfo(
        model_group="meta-llama/Meta-Llama-3-8B-Instruct",
        providers=["huggingface"],
        **info,
    )


@pytest.mark.parametrize(
    "model, provider",
    [
        ("bedrock/us-east-2/us.anthropic.claude-3-haiku-20240307-v1:0", None),
        (
            "bedrock/us-east-2/us.anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock",
        ),
    ],
)
def test_get_model_info_cost_calculator_bedrock_region_cris_stripped(model, provider):
    """
    ensure cross region inferencing model is used correctly
    Relevant Issue: https://github.com/BerriAI/litellm/issues/8115
    """
    info = get_model_info(model=model, custom_llm_provider=provider)
    print("info", info)
    assert info["key"] == "us.anthropic.claude-3-haiku-20240307-v1:0"
    assert info["litellm_provider"] == "bedrock"


def test_get_model_info_case_insensitive_lookup(monkeypatch):
    """
    Test that model info lookup is case-insensitive.

    This ensures that users can use lowercase model names even when the model cost
    map has mixed-case keys (e.g., "Qwen/Qwen3-Next-80B-A3B-Thinking").

    Related Slack discussion: Users were getting "does not support parameters: ['tools']"
    errors when using lowercase model names like "qwen/qwen3-next-80b-a3b-thinking"
    because the lookup was case-sensitive.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")

    # Register a test model with mixed-case name
    litellm.register_model(
        {
            "together_ai/Qwen/Qwen3-Next-80B-A3B-Thinking": {
                "input_cost_per_token": 0.0001,
                "output_cost_per_token": 0.0002,
                "litellm_provider": "together_ai",
                "supports_function_calling": True,
            }
        }
    )

    # Test 1: Exact case should work
    info = litellm.get_model_info(
        model="Qwen/Qwen3-Next-80B-A3B-Thinking", custom_llm_provider="together_ai"
    )
    assert info is not None
    assert info["supports_function_calling"] is True

    # Test 2: Lowercase should also work (case-insensitive lookup)
    info_lower = litellm.get_model_info(
        model="qwen/qwen3-next-80b-a3b-thinking", custom_llm_provider="together_ai"
    )
    assert info_lower is not None
    assert info_lower["supports_function_calling"] is True

    # Test 3: Mixed case should also work
    info_mixed = litellm.get_model_info(
        model="QWEN/qwen3-NEXT-80b-a3b-thinking", custom_llm_provider="together_ai"
    )
    assert info_mixed is not None
    assert info_mixed["supports_function_calling"] is True


def test_get_model_info_case_insensitive_supports_function_calling(monkeypatch):
    """
    Test that supports_function_calling check works with case-insensitive model lookup.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")

    # Register a model with mixed-case name that supports function calling
    litellm.register_model(
        {
            "test_provider/TestModel-ABC": {
                "input_cost_per_token": 0.0001,
                "output_cost_per_token": 0.0002,
                "litellm_provider": "test_provider",
                "supports_function_calling": True,
            }
        }
    )

    # Test that supports_function_calling works with lowercase model name
    from litellm.utils import supports_function_calling

    # Exact case
    assert (
        supports_function_calling("TestModel-ABC", custom_llm_provider="test_provider")
        is True
    )

    # Lowercase (should now work with case-insensitive lookup)
    assert (
        supports_function_calling("testmodel-abc", custom_llm_provider="test_provider")
        is True
    )
