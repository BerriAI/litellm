from __future__ import annotations

import os
import base64
import struct
import zlib
from typing import Final, Literal

import pytest
from e2e_config import unique_marker
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = [pytest.mark.e2e, pytest.mark.provider_live]


def _deployment(
    proxy: ProxyClient, resources: ResourceManager, endpoint_env: str, label: str
) -> tuple[str, str]:
    endpoint: Final = os.environ[endpoint_env]
    model_name: Final = f"sagemaker_nova/{endpoint}"
    model: Final = f"e2e-{label}-{unique_marker()}"
    model_id: Final = proxy.create_model(model, LiteLLMParamsBody(model=model_name))
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _test_png_data_uri() -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        encoded: Final = kind + data
        return struct.pack(">I", len(data)) + encoded + struct.pack(">I", zlib.crc32(encoded))

    red: Final = b"\x00" + bytes((255, 0, 0, 255)) * 4
    mixed: Final = b"\x00" + bytes((255, 0, 0, 255)) + bytes((0, 0, 255, 255)) * 2 + bytes((255, 0, 0, 255))
    image: Final = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(red + mixed * 2 + red))
        + chunk(b"IEND", b"")
    )
    return f"data:image/png;base64,{base64.b64encode(image).decode('ascii')}"


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
        deployment: Final = _deployment(proxy, resources, "SAGEMAKER_NOVA_ENDPOINT", "nova-image")
        model, key = deployment
        response: Final = sdk.openai(key).chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colors are in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": _test_png_data_uri()},
                        },
                    ],
                }
            ],
            max_tokens=64,
        )
        assert response.choices[0].message.content

    @pytest.mark.covers("llm.chat_completions.sagemaker.top_k.nonstream.works")
    def test_nova_specific_top_k_parameter_is_accepted(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        deployment: Final = _deployment(proxy, resources, "SAGEMAKER_NOVA_ENDPOINT", "nova-top-k")
        model, key = deployment
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
        deployment: Final = _deployment(proxy, resources, "SAGEMAKER_NOVA_ENDPOINT", "nova-logprobs")
        model, key = deployment
        response: Final = sdk.openai(key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the word hello."}],
            max_tokens=8,
            logprobs=True,
            top_logprobs=3,
        )
        assert response.choices[0].logprobs is not None
        assert response.choices[0].logprobs.content


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
        deployment: Final = _deployment(proxy, resources, "SAGEMAKER_NOVA2_LITE_ENDPOINT", "nova2-reasoning")
        model, key = deployment
        response: Final = sdk.openai(key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            max_tokens=32,
            reasoning_effort=reasoning_effort,
        )
        assert response.choices[0].message.content
