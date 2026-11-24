from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Final, Literal

import pytest
from e2e_config import SAGEMAKER_NOVA2_LITE_OPT_IN_ENV, SAGEMAKER_NOVA_OPT_IN_ENV, unique_marker
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = [pytest.mark.e2e, pytest.mark.provider_live]

CAT_IMAGE: Final = Path(__file__).parent / "fixtures" / "cat.jpg"


def _deployment(proxy: ProxyClient, resources: ResourceManager, endpoint_env: str, label: str) -> tuple[str, str]:
    endpoint: Final = os.environ[endpoint_env]
    model_name: Final = f"sagemaker_nova/{endpoint}"
    model: Final = f"e2e-{label}-{unique_marker()}"
    model_id: Final = proxy.create_model(model, LiteLLMParamsBody(model=model_name))
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _cat_image_data_uri() -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(CAT_IMAGE.read_bytes()).decode('ascii')}"


@pytest.mark.sagemaker_nova
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.CHAT_COMPLETIONS,
        providers=(Provider.SAGEMAKER,),
        mode=Mode.NONSTREAM,
    )
)
class TestSagemakerNova:
    @pytest.mark.covers("llm.chat_completions.sagemaker.vision.nonstream.works")
    def test_multimodal_image_input_is_accepted(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _deployment(proxy, resources, SAGEMAKER_NOVA_OPT_IN_ENV, "nova-image")
        response: Final = sdk.openai(key).chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What animal is in this image? Answer in one word."},
                        {
                            "type": "image_url",
                            "image_url": {"url": _cat_image_data_uri()},
                        },
                    ],
                }
            ],
            max_tokens=64,
        )
        answer: Final = response.choices[0].message.content or ""
        assert "cat" in answer.lower(), f"{model}: the image did not reach the model, it answered {answer!r}"

    @pytest.mark.covers("llm.chat_completions.sagemaker.top_k.nonstream.works")
    def test_nova_specific_top_k_parameter_is_accepted(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _deployment(proxy, resources, SAGEMAKER_NOVA_OPT_IN_ENV, "nova-top-k")
        response: Final = sdk.openai(key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=32,
            extra_body={"top_k": 40},
        )
        assert response.choices[0].message.content

    @pytest.mark.covers("llm.chat_completions.sagemaker.logprobs.nonstream.works")
    def test_logprobs_returns_token_details(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _deployment(proxy, resources, SAGEMAKER_NOVA_OPT_IN_ENV, "nova-logprobs")
        response: Final = sdk.openai(key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the word hello."}],
            max_tokens=8,
            logprobs=True,
            top_logprobs=3,
        )
        logprobs: Final = response.choices[0].logprobs
        assert logprobs and logprobs.content, f"{model}: no logprobs content in {response}"
        assert len(logprobs.content[0].top_logprobs) == 3, f"{model}: top_logprobs=3 not honored: {logprobs.content[0]}"


@pytest.mark.sagemaker_nova2_lite
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.CHAT_COMPLETIONS,
        providers=(Provider.SAGEMAKER,),
        capabilities=(Capability.REASONING,),
        mode=Mode.NONSTREAM,
    )
)
class TestSagemakerNova2Lite:
    @pytest.mark.parametrize("reasoning_effort", ("low", "high"))
    @pytest.mark.covers("llm.chat_completions.sagemaker.thinking.nonstream.works")
    def test_reasoning_effort_is_accepted(
        self,
        reasoning_effort: Literal["low", "high"],
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
    ) -> None:
        model, key = _deployment(proxy, resources, SAGEMAKER_NOVA2_LITE_OPT_IN_ENV, "nova2-reasoning")
        response: Final = sdk.openai(key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            max_tokens=32,
            reasoning_effort=reasoning_effort,
        )
        assert response.choices[0].message.content
