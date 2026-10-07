import os

import pytest

import litellm


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
