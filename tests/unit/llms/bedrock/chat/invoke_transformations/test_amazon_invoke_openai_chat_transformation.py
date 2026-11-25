import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.bedrock.common_utils import get_bedrock_chat_config
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

OPENAI_CHAT_BODY_MODELS: Final = (
    "qwen.qwen3-32b-v1:0",
    "qwen.qwen3-coder-480b-a35b-v1:0",
    "mistral.mistral-large-3-675b-instruct",
    "mistral.voxtral-mini-3b-2507",
)
PROMPT_BODY_MODELS: Final = (
    "invoke/mistral.mistral-7b-instruct-v0:2",
    "invoke/mistral.mixtral-8x7b-instruct-v0:1",
    "invoke/mistral.mistral-large-2407-v1:0",
    "invoke/us.mistral.pixtral-large-2502-v1:0",
    "qwen3/arn:aws:bedrock:us-east-1:123456789012:imported-model/abc123",
)
MESSAGES: Final = ({"role": "user", "content": "Say hi in one word"},)


@pytest.fixture
def fake_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)


def _chat_completion(model_id: str) -> Mapping[str, object]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1795635566,
        "model": model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 20},
    }


def _handle_recording(requests: list[httpx.Request], model_id: str):
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_chat_completion(model_id))

    return handle


@pytest.mark.parametrize("model_id", OPENAI_CHAT_BODY_MODELS)
def test_invoke_sends_the_openai_chat_body_and_reads_the_chat_completion(fake_aws_env: None, model_id: str) -> None:
    requests: Final[list[httpx.Request]] = []
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_handle_recording(requests, model_id))))

    response: Final = litellm.completion(
        model=f"bedrock/invoke/{model_id}",
        messages=list(MESSAGES),
        max_tokens=16,
        aws_region_name="us-east-1",
        client=client,
    )

    assert requests[0].url.raw_path.endswith(b"/invoke")
    assert json.loads(requests[0].content) == {"messages": list(MESSAGES), "max_completion_tokens": 16}
    assert response.choices[0].message.content == "Hello!"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (17, 3)


@pytest.mark.asyncio
async def test_async_invoke_sends_the_openai_chat_body(fake_aws_env: None) -> None:
    requests: Final[list[httpx.Request]] = []
    model_id: Final = "mistral.mistral-large-3-675b-instruct"
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_handle_recording(requests, model_id)))

    response: Final = await litellm.acompletion(
        model=f"bedrock/invoke/{model_id}",
        messages=list(MESSAGES),
        temperature=0.2,
        aws_region_name="us-east-1",
        client=client,
    )

    assert json.loads(requests[0].content) == {"messages": list(MESSAGES), "temperature": 0.2}
    assert response.choices[0].message.content == "Hello!"


@pytest.mark.parametrize("model", PROMPT_BODY_MODELS)
def test_models_that_only_take_the_prompt_body_keep_it(model: str) -> None:
    body: Final = get_bedrock_chat_config(model).transform_request(
        model=model, messages=list(MESSAGES), optional_params={}, litellm_params={}, headers={}
    )

    assert "prompt" in body
    assert "messages" not in body


@pytest.mark.parametrize("model_id", ["qwen.qwen3-32b-v1:0", "mistral.mistral-large-3-675b-instruct"])
def test_logit_bias_is_refused_or_dropped_before_reaching_aws(fake_aws_env: None, model_id: str) -> None:
    requests: Final[list[httpx.Request]] = []
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_handle_recording(requests, model_id))))
    request: Final = {
        "model": f"bedrock/invoke/{model_id}",
        "messages": list(MESSAGES),
        "logit_bias": {"1": 1},
        "aws_region_name": "us-east-1",
        "client": client,
    }

    with pytest.raises(litellm.UnsupportedParamsError, match="logit_bias"):
        litellm.completion(**request)
    assert requests == []

    litellm.completion(**request, drop_params=True)

    assert "logit_bias" not in json.loads(requests[0].content)
