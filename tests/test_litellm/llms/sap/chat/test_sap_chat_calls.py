import httpx
from unittest.mock import patch, PropertyMock

import pytest

mock_response = {
    "request_id": "e86a0b4e-53e3-97dc-a5f7-82e451376b23",
    "intermediate_results": {
        "templating": [{"content": "Say hello", "role": "user"}],
        "llm": {
            "id": "chatcmpl-CUB63bLTYnfO2CQR0r0rArkrbe8CH",
            "object": "chat.completion",
            "created": 1761308531,
            "model": "gpt-4o-2024-08-06",
            "system_fingerprint": "fp_4a331a0222",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from SAP!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 7, "prompt_tokens": 3, "total_tokens": 10},
        },
    },
    "final_result": {
        "id": "chatcmpl-CUB63bLTYnfO2CQR0r0rArkrbe8CH",
        "object": "chat.completion",
        "created": 1761308531,
        "model": "gpt-4o-2024-08-06",
        "system_fingerprint": "fp_4a331a0222",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from SAP!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 7, "prompt_tokens": 3, "total_tokens": 10},
    },
}
mock_stream_response = [
    b'data: {"request_id": "a07127d3-cb74-9427-a4dc-ef9bf424fb43", "intermediate_results": {"templating": [{"content": "Hi", "role": "user"}]}, "final_result": {"id": \'\', "object": \'\', "created": 0, "model": \'\', "system_fingerprint": null, "choices": [{"index": 0, "delta": {"content": ""}}]}}\n\n',
    b'data: {"request_id": "a07127d3-cb74-9427-a4dc-ef9bf424fb43", "intermediate_results": {"llm": {"id": "chatcmpl-HelloMsg", "object": "chat.completion.chunk", "created": 1761319270, "model": "gpt-4o-2024-08-06", "system_fingerprint": "fp_HelloMsg", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello "}}]}}, "final_result": {"id": "chatcmpl-HelloMsg", "object": "chat.completion.chunk", "created": 1761319270, "model": "gpt-4o-2024-08-06", "system_fingerprint": "fp_HelloMsg", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello "}}]}}\n\n',
    b'data: {"request_id": "a07127d3-cb74-9427-a4dc-ef9bf424fb43", "intermediate_results": {"llm": {"id": "chatcmpl-CUDtFmLex96SxakzBIzhLq2h8Axmk", "object": "chat.completion.chunk", "created": 1761319269, "model": "gpt-4o-2024-08-06", "system_fingerprint": "fp_4a331a0222", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "from SAP!"}, "finish_reason": "stop"}]}}, "final_result": {"id": "chatcmpl-CUDtFmLex96SxakzBIzhLq2h8Axmk", "object": "chat.completion.chunk", "created": 1761319269, "model": "gpt-4o-2024-08-06", "system_fingerprint": "fp_4a331a0222", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "from SAP!"}, "finish_reason": "stop"}]}}\n\n',
    b"data: [DONE]\n\n",
]


@pytest.fixture
def sap_api_response():
    return mock_response


@pytest.fixture
def sap_api_stream_response():
    return mock_response


@pytest.fixture
def fake_token_creator():
    return lambda: "Bearer FAKE_TOKEN", "https://api.ai.mock-sap.com", "fake-group"


@pytest.fixture
def fake_deployment_url():
    return "https://api.ai.mock-sap.com/v2/inference/deployments/mockid"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_sap_chat(
    respx_mock,
    sap_api_response,
    fake_token_creator,
    fake_deployment_url,
    sync_mode,
):
    import litellm

    litellm.disable_aiohttp_transport = True
    with patch(
        "litellm.llms.sap.chat.transformation.GenAIHubOrchestrationConfig.deployment_url",
        new_callable=PropertyMock,
        return_value=fake_deployment_url,
    ), patch(
        "litellm.llms.sap.chat.transformation.get_token_creator",
        return_value=fake_token_creator,
    ):
        model = "sap/gpt-4o"
        messages = [{"role": "user", "content": "Hello"}]
        respx_mock.post(f"{fake_deployment_url}/v2/completion").respond(
            json=sap_api_response
        )

        if sync_mode:
            response = litellm.completion(model=model, messages=messages)
        else:
            response = await litellm.acompletion(model=model, messages=messages)

        assert response.choices[0].message.content == "Hello from SAP!"
        assert response.model.startswith("gpt-4o")
        assert response.usage.total_tokens == 10


@pytest.mark.asyncio
async def test_sap_streaming(
    respx_mock,
    sap_api_stream_response,
    fake_token_creator,
    fake_deployment_url,
):
    import litellm

    litellm.disable_aiohttp_transport = True
    with patch(
        "litellm.llms.sap.chat.transformation.GenAIHubOrchestrationConfig.deployment_url",
        new_callable=PropertyMock,
        return_value=fake_deployment_url,
    ), patch(
        "litellm.llms.sap.chat.transformation.get_token_creator",
        return_value=fake_token_creator,
    ):
        model = "sap/gpt-4o"
        messages = [{"role": "user", "content": "Hello"}]

        respx_mock.post(f"{fake_deployment_url}/v2/completion").mock(
            return_value=httpx.Response(
                200,
                content=mock_stream_response,
                headers={"Content-Type": "text/event-stream"},
            )
        )

        stream = litellm.completion(model=model, messages=messages, stream=True)

        full = ""
        for chunk in stream:
            delta = getattr(chunk.choices[0].delta, "content", None) or ""
            full += delta

        assert full == "Hello from SAP!"
