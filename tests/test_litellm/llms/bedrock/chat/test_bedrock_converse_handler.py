"""Focused tests for ``BedrockConverseLLM.completion``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from botocore.credentials import Credentials

from litellm.llms.bedrock.chat.converse_handler import BedrockConverseLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import ModelResponse

RESOLVED_CREDENTIALS = Credentials(
    access_key="AKIARESOLVED",
    secret_key="resolved-secret",
    token="resolved-token",
)

CONVERSE_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
}


def _completion_kwargs(**overrides):
    kwargs = {
        "model": "bedrock/us-east-1/anthropic.claude-sonnet-4-5-v1:0",
        "messages": [{"role": "user", "content": "hi"}],
        "api_base": None,
        "custom_prompt_dict": {},
        "model_response": ModelResponse(),
        "encoding": None,
        "logging_obj": MagicMock(),
        "optional_params": {"maxTokens": 16},
        "acompletion": False,
        "timeout": 30.0,
        "litellm_params": {"rust": True},
        "extra_headers": None,
        "client": None,
        "api_key": None,
    }
    kwargs.update(overrides)
    return kwargs


def _run(*, credentials: Credentials | None = RESOLVED_CREDENTIALS, **overrides):
    with patch.object(BedrockConverseLLM, "get_credentials", return_value=credentials):
        return BedrockConverseLLM().completion(**_completion_kwargs(**overrides))


async def _drive_async_completion(*, skip_pre_call_logging: bool, logging_obj):
    """Run the real `async_completion` with a stubbed transport."""
    import httpx as _httpx

    client = MagicMock()

    async def post(**_kwargs):
        return _httpx.Response(
            200,
            json=CONVERSE_RESPONSE,
            request=_httpx.Request("POST", "https://bedrock-runtime.us-west-2.amazonaws.com"),
        )

    client.post = post
    client.__class__ = AsyncHTTPHandler

    return await BedrockConverseLLM().async_completion(
        model="anthropic.claude-sonnet-4-5-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        api_base="https://bedrock-runtime.us-west-2.amazonaws.com/model/m/converse",
        model_response=ModelResponse(),
        timeout=30.0,
        encoding=None,
        logging_obj=logging_obj,
        stream=None,
        optional_params={"maxTokens": 16},
        litellm_params={"aws_region_name": "us-west-2"},
        credentials=RESOLVED_CREDENTIALS,
        headers={},
        client=client,
        skip_pre_call_logging=skip_pre_call_logging,
    )


@pytest.mark.asyncio
async def test_async_completion_honors_the_pre_call_suppression():
    logging_obj = MagicMock()
    await _drive_async_completion(skip_pre_call_logging=True, logging_obj=logging_obj)
    assert logging_obj.pre_call.call_count == 0


@pytest.mark.asyncio
async def test_async_completion_logs_pre_call_by_default():
    """The suppression must be opt-in, so every existing caller keeps its log."""
    logging_obj = MagicMock()
    await _drive_async_completion(skip_pre_call_logging=False, logging_obj=logging_obj)
    assert logging_obj.pre_call.call_count == 1


def _sync_client_returning_converse_response():
    client = MagicMock()
    client.post.side_effect = lambda **_kwargs: httpx.Response(
        200,
        json=CONVERSE_RESPONSE,
        request=httpx.Request("POST", "https://bedrock-runtime.us-west-2.amazonaws.com"),
    )
    client.__class__ = HTTPHandler
    return client


def test_the_sync_python_path_still_logs_pre_call_without_the_opt_in():
    """The suppression must not swallow the log on a request the gate declined,
    so a deployment with no `rust` flag keeps exactly the log it always had."""
    logging_obj = MagicMock()
    response = _run(
        logging_obj=logging_obj,
        litellm_params={},
        client=_sync_client_returning_converse_response(),
    )

    assert response.choices[0].message.content == "hi"
    assert logging_obj.pre_call.call_count == 1


def test_bearer_token_auth_serves_when_boto3_resolves_no_sigv4_credentials(monkeypatch):
    """With only `AWS_BEARER_TOKEN_BEDROCK` configured boto3 resolves no
    credentials at all. Preparing the Rust handoff must not dereference that
    None: the bearer token signs the request on its own."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-bearer-token")
    client = _sync_client_returning_converse_response()

    response = _run(credentials=None, litellm_params={}, client=client)

    assert response.choices[0].message.content == "hi"
    sent_headers = client.post.call_args.kwargs["headers"]
    assert sent_headers["Authorization"] == "Bearer bedrock-bearer-token"


@pytest.mark.parametrize("configured_through", ["env_var", "api_key"])
def test_bearer_token_auth_never_runs_the_sigv4_credential_chain(monkeypatch, configured_through):
    """The deployment's AWS profile does not exist, so resolving SigV4 credentials
    raises; a bearer-token deployment must still serve the request, since the
    bearer token alone signs it."""
    if configured_through == "env_var":
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-bearer-token")
    else:
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    client = _sync_client_returning_converse_response()

    response = BedrockConverseLLM().completion(
        **_completion_kwargs(
            optional_params={"maxTokens": 16, "aws_profile_name": "litellm-no-such-aws-profile"},
            litellm_params={},
            client=client,
            api_key="bedrock-bearer-token" if configured_through == "api_key" else None,
        )
    )

    assert response.choices[0].message.content == "hi"
    assert client.post.call_args.kwargs["headers"]["Authorization"] == "Bearer bedrock-bearer-token"
