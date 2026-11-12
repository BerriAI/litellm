from typing import Final

import httpx
import pytest
import respx

from litellm.llms.vllm.common_utils import HostedVLLMModelInfo


@respx.mock
def test_hosted_vllm_get_models_lists_the_server_ids_sending_the_key_as_a_bearer_token() -> None:
    respx.get("http://vllm-sdk.test:8000/v1/models", headers={"Authorization": "Bearer vllm-sdk-key"}).mock(
        return_value=httpx.Response(200, json={"data": [{"id": "qwen2.5"}, {"id": "meta-llama/Llama-3.1-8B-Instruct"}]})
    )

    listed: Final = HostedVLLMModelInfo().get_models(api_key="vllm-sdk-key", api_base="http://vllm-sdk.test:8000/v1")

    assert listed == ["qwen2.5", "meta-llama/Llama-3.1-8B-Instruct"]


def test_hosted_vllm_get_models_without_a_server_names_the_setting_that_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)

    with pytest.raises(ValueError, match="HOSTED_VLLM_API_BASE"):
        HostedVLLMModelInfo().get_models()
