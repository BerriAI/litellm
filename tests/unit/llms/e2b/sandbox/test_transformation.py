import json

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.e2b.sandbox.transformation import E2BSandboxConfig


def _client_answering(body: bytes) -> AsyncHTTPHandler:
    return AsyncHTTPHandler(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)))


def _lines(*messages: object) -> list[str]:
    return [json.dumps(message) for message in messages]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("created", "expected_domain", "expected_tokens"),
    [
        (
            {"sandboxID": "sbx_1", "domain": "eu.e2b.app", "envdAccessToken": "envd", "trafficAccessToken": "traffic"},
            "eu.e2b.app",
            ("envd", "traffic"),
        ),
        ({"sandboxID": "sbx_1"}, "e2b.app", (None, None)),
        ({"sandboxID": "sbx_1", "domain": None, "envdAccessToken": "envd"}, "e2b.app", ("envd", None)),
        ({"sandboxID": "sbx_1", "domain": ""}, "e2b.app", (None, None)),
    ],
)
async def test_acreate_sandbox_builds_the_handle_from_the_create_response(
    created: dict[str, object], expected_domain: str, expected_tokens: tuple[str | None, str | None]
):
    handle = await E2BSandboxConfig().acreate_sandbox(
        api_key="e2b_key", client=_client_answering(json.dumps(created).encode())
    )

    assert (handle.id, handle.provider, handle.domain) == ("sbx_1", "e2b", expected_domain)
    assert handle._hidden_params == {
        "envd_access_token": expected_tokens[0],
        "traffic_access_token": expected_tokens[1],
        "api_key": "e2b_key",
        "api_base": "https://api.e2b.app",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b'["secret-token"]', b'"secret-token"', b"7", b"null"])
async def test_acreate_sandbox_rejects_a_create_response_that_is_not_an_object_without_echoing_it(body: bytes):
    with pytest.raises(ValidationError) as exc_info:
        await E2BSandboxConfig().acreate_sandbox(api_key="e2b_key", client=_client_answering(body))

    assert "secret-token" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_acreate_sandbox_requires_a_sandbox_id():
    with pytest.raises(KeyError, match="sandboxID"):
        await E2BSandboxConfig().acreate_sandbox(api_key="e2b_key", client=_client_answering(b'{"domain": "e2b.app"}'))


@pytest.mark.asyncio
@pytest.mark.parametrize("created", [{"sandboxID": 5}, {"sandboxID": "sbx_1", "domain": 5}])
async def test_acreate_sandbox_rejects_non_string_handle_fields(created: dict[str, object]):
    with pytest.raises(ValidationError, match="ContainerHandle"):
        await E2BSandboxConfig().acreate_sandbox(
            api_key="e2b_key", client=_client_answering(json.dumps(created).encode())
        )


def test_parse_lines_maps_every_message_type_onto_the_result():
    result = E2BSandboxConfig._parse_lines(
        [
            *_lines(
                {"type": "stdout", "text": "1\n", "timestamp": 1},
                {"type": "stderr", "text": "warn\n"},
                {"type": "stdout"},
                {"type": "result", "png": "BASE64", "is_main_result": True},
                {"type": "error", "name": "ValueError", "value": "bad", "traceback": "tb", "ignored": 1},
                {"type": "number_of_executions", "execution_count": 3},
                {"type": "stdout", "text": "2\n"},
                None,
            ),
            "",
            "not json",
        ]
    )

    assert result.model_dump() == {
        "stdout": "1\n2\n",
        "stderr": "warn\n",
        "results": [{"png": "BASE64", "is_main_result": True}],
        "error": {"name": "ValueError", "value": "bad", "traceback": "tb"},
        "execution_count": 3,
        "object": "code_execution",
    }


def test_parse_lines_rejects_an_execution_count_that_is_not_a_number():
    with pytest.raises(ValidationError, match="execution_count"):
        E2BSandboxConfig._parse_lines(_lines({"type": "number_of_executions", "execution_count": "many"}))


@pytest.mark.parametrize(
    "message",
    [
        ["secret-output"],
        "secret-output",
        7,
        False,
        {"type": "stdout", "text": ["secret-output"]},
        {"type": "stderr", "text": {"secret-output": 1}},
    ],
)
def test_parse_lines_rejects_malformed_messages_without_echoing_them(message: object):
    with pytest.raises(ValidationError) as exc_info:
        E2BSandboxConfig._parse_lines(_lines({"type": "stdout", "text": "ok"}, message))

    assert "secret-output" not in str(exc_info.value)
