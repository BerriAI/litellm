"""Bedrock Converse retries once, re-signed, when the provider rejects extra toolSpec fields.

Bedrock validates some Claude models through an Anthropic-compatible validator that
accepts a narrower ``toolSpec`` than the Converse API documents and rejects the surplus
members by presence. Retrying without those members only works if the retry is signed
again, because SigV4 commits to a hash of the body. See BerriAI/litellm#33193.
"""

import httpx
import pytest

from litellm.llms.base_llm.base_utils import parse_rejected_tool_fields
from litellm.llms.bedrock.chat.converse_handler import BedrockConverseLLM
from litellm.llms.bedrock.common_utils import BedrockError, drop_bedrock_rejected_tool_fields

_STRICT_REJECTION = (
    '{"message":"The model returned the following errors: '
    'tools.0.custom.strict: Extra inputs are not permitted"}'
)

_REQUEST_DATA = {
    "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    "toolConfig": {
        "tools": [
            {
                "toolSpec": {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "inputSchema": {"json": {"type": "object", "properties": {}}},
                    "strict": False,
                }
            }
        ]
    },
}


def _credentials():
    from botocore.credentials import Credentials

    return Credentials(access_key="AKIAEXAMPLE", secret_key="secret", token=None)


def _retry_kwargs():
    return {
        "request_data": _REQUEST_DATA,
        "data": "original-body",
        "headers": {"Authorization": "signature-over-original"},
        "credentials": _credentials(),
        "aws_region_name": "us-east-1",
        "caller_headers": {"Content-Type": "application/json"},
        "endpoint_url": "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse",
        "api_key": None,
    }


@pytest.mark.parametrize(
    "error_text, expected",
    [
        ("tools.0.custom.strict: Extra inputs are not permitted", {0: frozenset({"strict"})}),
        ("tools[0].strict: Extra inputs are not permitted", {0: frozenset({"strict"})}),
        (
            "tools.0.custom.strict: Extra inputs are not permitted, "
            "tools.2.custom.defer_loading: Extra inputs are not permitted",
            {0: frozenset({"strict"}), 2: frozenset({"defer_loading"})},
        ),
        ("tools.0.custom.input_schema.type: Input should be 'object'", {}),
        ("ThrottlingException: rate exceeded", {}),
        ("", {}),
    ],
)
def test_parse_rejected_tool_fields(error_text: str, expected: dict) -> None:
    """Both provider spellings parse; anything that is not an extra-inputs rejection is ignored."""
    assert dict(parse_rejected_tool_fields(error_text)) == expected


def test_drop_bedrock_rejected_tool_fields_removes_only_the_named_field() -> None:
    result = drop_bedrock_rejected_tool_fields(_REQUEST_DATA, _STRICT_REJECTION)
    assert result is not None
    tool_spec = result["toolConfig"]["tools"][0]["toolSpec"]
    assert "strict" not in tool_spec
    assert tool_spec["name"] == "get_weather"
    assert tool_spec["inputSchema"] == {"json": {"type": "object", "properties": {}}}


def test_drop_bedrock_rejected_tool_fields_does_not_mutate_the_original() -> None:
    """The caller still needs the original payload to raise its untouched error on failure."""
    drop_bedrock_rejected_tool_fields(_REQUEST_DATA, _STRICT_REJECTION)
    assert _REQUEST_DATA["toolConfig"]["tools"][0]["toolSpec"]["strict"] is False


@pytest.mark.parametrize(
    "error_text",
    [
        "ThrottlingException: rate exceeded",
        "tools.9.custom.strict: Extra inputs are not permitted",
        "tools.0.custom.nonexistent_field: Extra inputs are not permitted",
    ],
)
def test_drop_bedrock_rejected_tool_fields_returns_none_when_nothing_applies(error_text: str) -> None:
    """Unrelated errors, out-of-range indices and fields the request never carried are all no-ops."""
    assert drop_bedrock_rejected_tool_fields(_REQUEST_DATA, error_text) is None


def _http_status_error(body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse-stream")
    return httpx.HTTPStatusError("400", request=request, response=httpx.Response(400, text=request and body))


@pytest.mark.parametrize(
    "raised",
    [
        BedrockError(status_code=400, message=_STRICT_REJECTION),
        _http_status_error(_STRICT_REJECTION),
    ],
    ids=["non-streaming raises BedrockError", "streaming raises HTTPStatusError"],
)
def test_sync_retry_resends_without_the_rejected_field_and_resigns(raised: Exception) -> None:
    """Both error shapes Converse can raise trigger the retry, and the retry is signed afresh."""
    attempts: list[tuple[str, dict]] = []

    def send(body: str, headers: dict) -> str:
        attempts.append((body, headers))
        if len(attempts) == 1:
            raise raised
        return "ok"

    result = BedrockConverseLLM()._send_retrying_rejected_tool_fields(send=send, **_retry_kwargs())

    assert result == "ok"
    assert len(attempts) == 2

    first_body, first_headers = attempts[0]
    retry_body, retry_headers = attempts[1]
    assert first_body == "original-body"
    assert '"strict"' not in retry_body
    assert '"get_weather"' in retry_body
    assert retry_headers["Authorization"] != first_headers["Authorization"]
    assert retry_headers["Authorization"].startswith("AWS4-HMAC-SHA256")


def test_sync_retry_leaves_unrelated_errors_alone() -> None:
    """A failure that is not an extra-tool-field rejection is sent once and raises as-is."""
    attempts: list[str] = []

    def send(body: str, headers: dict) -> str:
        attempts.append(body)
        raise BedrockError(status_code=429, message="ThrottlingException: rate exceeded")

    with pytest.raises(BedrockError) as excinfo:
        BedrockConverseLLM()._send_retrying_rejected_tool_fields(send=send, **_retry_kwargs())

    assert excinfo.value.status_code == 429
    assert len(attempts) == 1


def test_sync_retry_is_single_shot() -> None:
    """A second rejection surfaces instead of looping."""
    attempts: list[str] = []

    def send(body: str, headers: dict) -> str:
        attempts.append(body)
        raise BedrockError(status_code=400, message=_STRICT_REJECTION)

    with pytest.raises(BedrockError):
        BedrockConverseLLM()._send_retrying_rejected_tool_fields(send=send, **_retry_kwargs())

    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_async_retry_resends_without_the_rejected_field_and_resigns() -> None:
    attempts: list[tuple[str, dict]] = []

    async def send(body: str, headers: dict) -> str:
        attempts.append((body, headers))
        if len(attempts) == 1:
            raise BedrockError(status_code=400, message=_STRICT_REJECTION)
        return "ok"

    result = await BedrockConverseLLM()._asend_retrying_rejected_tool_fields(send=send, **_retry_kwargs())

    assert result == "ok"
    assert len(attempts) == 2
    assert '"strict"' not in attempts[1][0]
    assert attempts[1][1]["Authorization"].startswith("AWS4-HMAC-SHA256")


@pytest.mark.asyncio
async def test_async_retry_leaves_unrelated_errors_alone() -> None:
    attempts: list[str] = []

    async def send(body: str, headers: dict) -> str:
        attempts.append(body)
        raise BedrockError(status_code=500, message="InternalServerException")

    with pytest.raises(BedrockError) as excinfo:
        await BedrockConverseLLM()._asend_retrying_rejected_tool_fields(send=send, **_retry_kwargs())

    assert excinfo.value.status_code == 500
    assert len(attempts) == 1
