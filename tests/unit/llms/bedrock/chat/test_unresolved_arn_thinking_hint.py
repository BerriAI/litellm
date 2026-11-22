import asyncio
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.bedrock.chat.unresolved_arn_thinking_hint import UNRESOLVED_ARN_THINKING_HINT
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

_PROFILE_ARN: Final = "bedrock/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
_NAMED_MODEL: Final = "bedrock/converse/us.anthropic.claude-opus-4-8"
_ENABLED_THINKING_REJECTION: Final = (
    r'{"message":"The model returned the following errors: \"thinking.type.enabled\" is not supported for this model. '
    r'Use \"thinking.type.adaptive\" and \"output_config.effort\" to control thinking behavior."}'
)
_MALFORMED_THINKING: Final = (
    '{"message":"Malformed input request: #/thinking: thinking.type.enabled requires budget_tokens, '
    'please reformat your input and try again."}'
)
_OTHER_UNSUPPORTED_FIELD: Final = (
    r'{"message":"The model returned the following errors: \"top_k\" is not supported for this model."}'
)


def _bedrock_answering(body: str) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(400, content=body, request=request))


def _send(model: str, body: str, *, asynchronous: bool, stream: bool, **deployment: str) -> object:
    request: Final = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "adaptive"},
        "max_tokens": 16000,
        "stream": stream,
        "aws_access_key_id": "AKIAPROFILEKEY",
        "aws_secret_access_key": "profile-secret",
        "aws_region_name": "us-east-1",
        **deployment,
    }
    if asynchronous:
        return asyncio.run(litellm.acompletion(**request, client=AsyncHTTPHandler(transport=_bedrock_answering(body))))
    return litellm.completion(**request, client=HTTPHandler(client=httpx.Client(transport=_bedrock_answering(body))))


def _raised(model: str, body: str, *, asynchronous: bool, stream: bool, **deployment: str) -> litellm.BadRequestError:
    with pytest.raises(litellm.BadRequestError) as raised:
        _send(model, body, asynchronous=asynchronous, stream=stream, **deployment)
    return raised.value


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_profile_arn_without_base_model_names_base_model_when_bedrock_rejects_enabled_thinking(
    asynchronous: bool, stream: bool
) -> None:
    error: Final = _raised(_PROFILE_ARN, _ENABLED_THINKING_REJECTION, asynchronous=asynchronous, stream=stream)

    assert error.status_code == 400
    assert _ENABLED_THINKING_REJECTION in error.message
    assert error.message.endswith(f"\n\n{UNRESOLVED_ARN_THINKING_HINT}")
    assert "Set `base_model` on the deployment" in error.message


@pytest.mark.parametrize(
    ("model", "body", "deployment"),
    [
        (_PROFILE_ARN, _MALFORMED_THINKING, {}),
        (_PROFILE_ARN, _OTHER_UNSUPPORTED_FIELD, {}),
        (_NAMED_MODEL, _ENABLED_THINKING_REJECTION, {}),
        (_PROFILE_ARN, _ENABLED_THINKING_REJECTION, {"base_model": "us.anthropic.claude-opus-4-8"}),
    ],
    ids=["other_thinking_400", "other_unsupported_field", "named_model", "profile_arn_with_base_model"],
)
def test_bedrock_400_keeps_its_message_without_the_hint(model: str, body: str, deployment: dict[str, str]) -> None:
    error: Final = _raised(model, body, asynchronous=False, stream=False, **deployment)

    assert error.status_code == 400
    assert error.message.endswith(body)
    assert UNRESOLVED_ARN_THINKING_HINT not in error.message
