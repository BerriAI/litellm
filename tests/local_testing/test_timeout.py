#### What this tests ####
#    This tests the timeout decorator

from typing import Final

import httpx
import openai
import pytest

import litellm
from tests.fake_openai_endpoint import FAKE_OPENAI_API_BASE

_ESSAY_MESSAGES: Final = [{"role": "user", "content": "hello, write a 20 pg essay"}]
_TIMEOUT_SECONDS: Final = 0.5


def _slow_openai_deployment(model_name: str) -> dict:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": "openai/slow-endpoint",
            "api_base": FAKE_OPENAI_API_BASE,
            "api_key": "fake-key",
        },
    }


def _slow_azure_deployment(model_name: str) -> dict:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": "azure/slow-endpoint",
            "api_base": FAKE_OPENAI_API_BASE,
            "api_key": "fake-key",
            "api_version": "2024-10-21",
        },
    }


@pytest.mark.parametrize(
    "model, provider",
    [
        ("gpt-3.5-turbo", "openai"),
        ("azure/gpt-4.1-mini", "azure"),
    ],
)
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_httpx_timeout(model, provider, sync_mode):
    """
    Test if setting httpx.timeout works for completion calls
    """
    timeout_val = httpx.Timeout(10.0, connect=60.0)

    messages = [{"role": "user", "content": "Hey, how's it going?"}]

    if sync_mode:
        response = litellm.completion(
            model=model, messages=messages, timeout=timeout_val
        )
    else:
        response = await litellm.acompletion(
            model=model, messages=messages, timeout=timeout_val
        )

    print(f"response: {response}")


def test_timeout():
    litellm.set_verbose = False
    with pytest.raises(openai.APITimeoutError):
        litellm.completion(
            model="openai/slow-endpoint",
            messages=_ESSAY_MESSAGES,
            api_base=FAKE_OPENAI_API_BASE,
            api_key="fake-key",
            timeout=_TIMEOUT_SECONDS,
        )


def test_bedrock_timeout():
    litellm.set_verbose = True
    with pytest.raises(openai.APITimeoutError):
        litellm.completion(
            model="bedrock/converse/slow-endpoint",
            messages=_ESSAY_MESSAGES,
            api_base=FAKE_OPENAI_API_BASE,
            aws_access_key_id="fake-access-key",
            aws_secret_access_key="fake-secret-key",
            aws_region_name="us-east-1",
            timeout=_TIMEOUT_SECONDS,
        )


@pytest.mark.asyncio
async def test_hanging_request_azure():
    litellm.set_verbose = True
    router = litellm.Router(
        model_list=[_slow_azure_deployment("azure-gpt"), _slow_openai_deployment("openai-gpt")],
        num_retries=0,
    )
    with pytest.raises(openai.APITimeoutError):
        await router.acompletion(
            model="azure-gpt",
            messages=[{"role": "user", "content": "what color is red"}],
            timeout=_TIMEOUT_SECONDS,
        )


def test_hanging_request_openai():
    litellm.set_verbose = True
    router = litellm.Router(
        model_list=[_slow_azure_deployment("azure-gpt"), _slow_openai_deployment("openai-gpt")],
        num_retries=0,
    )
    with pytest.raises(openai.APITimeoutError):
        router.completion(
            model="openai-gpt",
            messages=[{"role": "user", "content": "what color is red"}],
            timeout=_TIMEOUT_SECONDS,
        )


def test_timeout_streaming():
    litellm.set_verbose = False
    with pytest.raises(openai.APITimeoutError):
        response = litellm.completion(
            model="openai/slow-endpoint",
            messages=_ESSAY_MESSAGES,
            api_base=FAKE_OPENAI_API_BASE,
            api_key="fake-key",
            timeout=_TIMEOUT_SECONDS,
            stream=True,
        )
        for chunk in response:
            print(chunk)


@pytest.mark.skip(reason="local test")
def test_timeout_ollama():
    # this Will Raise a timeout
    import litellm

    litellm.set_verbose = True
    try:
        litellm.request_timeout = 0.1
        litellm.set_verbose = True
        response = litellm.completion(
            model="ollama/phi",
            messages=[{"role": "user", "content": "hello, what llm are u"}],
            max_tokens=1,
            api_base="https://test-ollama-endpoint.onrender.com",
        )
        # Add any assertions here to check the response
        litellm.request_timeout = None
        print(response)
    except openai.APITimeoutError as e:
        print("got a timeout error! Passed ! ")
        pass


# test_timeout_ollama()


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_anthropic_timeout(streaming, sync_mode):
    litellm.set_verbose = False
    request: Final = {
        "model": "anthropic/slow-endpoint",
        "messages": _ESSAY_MESSAGES,
        "api_base": FAKE_OPENAI_API_BASE,
        "api_key": "fake-key",
        "timeout": _TIMEOUT_SECONDS,
        "stream": streaming,
    }
    with pytest.raises(openai.APITimeoutError):
        if sync_mode:
            response = litellm.completion(**request)
            if isinstance(response, litellm.CustomStreamWrapper):
                for _ in response:
                    pass
        else:
            response = await litellm.acompletion(**request)
            if isinstance(response, litellm.CustomStreamWrapper):
                async for _ in response:
                    pass
