import asyncio
import base64
import binascii
import json
import logging
import struct
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final, cast  # noqa: TID251  # narrows the untyped Messages entries to their call shape

import pytest
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ProfileNotFound
from pydantic import TypeAdapter

import litellm
from litellm.anthropic_beta_headers_manager import reload_beta_headers_config
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils import model_cost_table
from litellm.llms.anthropic.messages.native import handler as python_messages
from litellm.llms.bedrock.base_aws_llm import AwsAuthError
from tests.test_litellm_rust.messages.data import MESSAGES, MESSAGES_EVENTS, MESSAGES_RESPONSE, RATE_LIMITED
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordedRequest, RecordingServer, ResponseSpec, relaxed
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

BEDROCK_MODEL: Final = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
AWS: Final = {
    "aws_access_key_id": "test-access-key",
    "aws_secret_access_key": "test-secret-key",
    "aws_region_name": "us-east-1",
}
GUARDRAIL_TRACE: Final[dict[str, object]] = {
    "guardrail": {
        "input": {"gr-1": {"invocationMetrics": {"usage": {"contentPolicyUnits": 1, "topicPolicyUnits": 1}}}},
        "outputs": [],
    }
}


@pytest.fixture
def server(server: RecordingServer) -> Iterator[RecordingServer]:
    yield from relaxed(server)


@pytest.fixture(autouse=True)
def no_ambient_bedrock_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_BEDROCK_RUNTIME_ENDPOINT", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)


_OBJECT: Final = TypeAdapter(dict[str, object])
_RUST_ENTRY: Final = cast(  # cast-ok: the public entry's call shape, which dispatch sends to Rust
    Callable[..., Awaitable[object]], litellm.anthropic.messages.acreate
)
_PYTHON_ENTRY: Final = cast(  # cast-ok: the Python handler's call shape, which no dispatch reroutes
    Callable[..., Awaitable[object]], python_messages.anthropic_messages
)


async def _rust(server: RecordingServer, **kwargs: object) -> object:
    return await _RUST_ENTRY(**_arguments(server, kwargs))


async def _python(server: RecordingServer, **kwargs: object) -> object:
    return await _PYTHON_ENTRY(**_arguments(server, kwargs))


def _arguments(server: RecordingServer, kwargs: Mapping[str, object]) -> dict[str, object]:
    return {
        "model": BEDROCK_MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 64,
        "api_base": server.base_url,
        "litellm_call_id": str(uuid.uuid4()),
        **AWS,
        **kwargs,
    }


def _signed_headers(request: RecordedRequest) -> str:
    authorization: Final = request.headers["authorization"]
    assert authorization.startswith("AWS4-HMAC-SHA256 ")
    return authorization.split("SignedHeaders=", 1)[1].split(",", 1)[0]


def _public(response: object) -> object:
    return {name: value for name, value in _OBJECT.validate_python(response).items() if name != "_hidden_params"}


@pytest.mark.asyncio
async def test_bedrock_invoke_sends_and_answers_what_python_does(server: RecordingServer) -> None:
    server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)

    python_response: Final = await _python(server)
    rust_response: Final = await _rust(server)

    python_sent, rust_sent = server.requests
    assert rust_sent.path == python_sent.path
    assert rust_sent.body == python_sent.body
    assert _signed_headers(rust_sent) == _signed_headers(python_sent)
    assert _public(rust_response) == _public(python_response)


@pytest.mark.asyncio
async def test_bedrock_invoke_upstream_failure_raises_the_bedrock_exception_python_raises(
    server: RecordingServer,
) -> None:
    server.default_response = ResponseSpec(body=RATE_LIMITED, status=429)

    with pytest.raises(litellm.RateLimitError) as python_raised:
        await _python(server)
    with pytest.raises(litellm.RateLimitError) as rust_raised:
        await _rust(server)

    assert str(rust_raised.value) == str(python_raised.value)
    assert rust_raised.value.llm_provider == python_raised.value.llm_provider == "bedrock"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped
    assert rust_raised.value.status_code == python_raised.value.status_code == 429


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("thread", "error_class"),
    (
        pytest.param({"type": "create"}, litellm.BadRequestError, id="create"),
        pytest.param({"type": "continue", "previous_message_id": "msg_1"}, litellm.NotFoundError, id="continue"),
    ),
)
async def test_bedrock_invoke_refuses_a_thread_as_python_does(
    server: RecordingServer, thread: Mapping[str, str], error_class: type[Exception]
) -> None:
    with pytest.raises(error_class) as python_raised:
        await _python(server, thread=dict(thread))
    with pytest.raises(error_class) as rust_raised:
        await _rust(server, thread=dict(thread))

    assert str(rust_raised.value) == str(python_raised.value)
    assert server.requests == []


@pytest.mark.asyncio
async def test_bedrock_invoke_drops_a_thread_under_drop_params_as_python_does(server: RecordingServer) -> None:
    server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)

    await _python(server, thread={"type": "create"}, drop_params=True)
    await _rust(server, thread={"type": "create"}, drop_params=True)

    python_sent, rust_sent = server.requests
    assert "thread" not in _OBJECT.validate_python(rust_sent.body)
    assert rust_sent.body == python_sent.body


class _GuardrailBilling(CustomLogger):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger's untyped init
        self.billed: list[object] = []

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.billed.append(_OBJECT.validate_python(kwargs.get("standard_logging_object")).get("guardrail_information"))

    async def wait_for(self, count: int) -> None:
        async with asyncio.timeout(5):
            while len(self.billed) < count:
                await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_bedrock_invoke_bills_the_guardrail_trace_as_python_does(server: RecordingServer) -> None:
    server.default_response = ResponseSpec(body={**MESSAGES_RESPONSE, "amazon-bedrock-trace": GUARDRAIL_TRACE})
    guardrail: Final = {"guardrailConfig": {"guardrailIdentifier": "gr-1", "guardrailVersion": "1"}}
    billing: Final = _GuardrailBilling()

    with rebound(litellm, "callbacks", [billing]):
        await _python(server, **guardrail)
        await billing.wait_for(1)
        await _rust(server, **guardrail)
        await billing.wait_for(2)

    python_sent, rust_sent = server.requests
    python_billed, rust_billed = billing.billed
    assert python_billed
    assert rust_billed == python_billed
    assert {name: value for name, value in rust_sent.headers.items() if name.startswith("x-amzn-bedrock-")} == {
        name: value for name, value in python_sent.headers.items() if name.startswith("x-amzn-bedrock-")
    }


def _aws_header(name: str, value: str) -> bytes:
    return bytes((len(name),)) + name.encode() + b"\x07" + struct.pack("!H", len(value)) + value.encode()


def _wire_frame(message_type: str, kind: str, payload: bytes) -> bytes:
    headers: Final = (
        _aws_header(":exception-type" if message_type == "exception" else ":event-type", kind)
        + _aws_header(":content-type", "application/json")
        + _aws_header(":message-type", message_type)
    )
    prelude: Final = struct.pack("!II", 12 + len(headers) + len(payload) + 4, len(headers))
    prelude_crc: Final = struct.pack("!I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message: Final = prelude + prelude_crc + headers + payload
    return message + struct.pack("!I", binascii.crc32(message) & 0xFFFFFFFF)


def _chunk_frame(text: bytes) -> bytes:
    return _wire_frame("event", "chunk", json.dumps({"bytes": base64.b64encode(text).decode()}).encode())


def _aws_frame(event: Mapping[str, object]) -> bytes:
    return _chunk_frame(json.dumps(event).encode())


def _event_stream(final: Mapping[str, object]) -> bytes:
    *head, (_, stop) = MESSAGES_EVENTS
    events: Final = (*(data for _, data in head), {**_OBJECT.validate_python(stop), **final})
    return b"".join(_aws_frame(event) for event in events)


def _sse_events(chunks: Sequence[bytes]) -> tuple[object, ...]:
    frames: Final = b"".join(chunks).decode().split("\n\n")
    return tuple(json.loads(frame.split("data: ", 1)[1]) for frame in frames if frame)


async def _drain(response: object) -> tuple[bytes, ...]:
    stream: Final = cast(AsyncIterator[bytes], response)  # cast-ok: a streamed Messages answer yields SSE bytes
    return tuple([chunk async for chunk in stream])


@pytest.mark.asyncio
async def test_bedrock_invoke_stream_bills_the_guardrail_trace_of_message_stop_as_python_does(
    server: RecordingServer,
) -> None:
    server.default_response = ResponseSpec(
        body=_event_stream({"amazon-bedrock-trace": GUARDRAIL_TRACE}),
        headers={"content-type": "application/vnd.amazon.eventstream"},
    )
    guardrail: Final = {"guardrailConfig": {"guardrailIdentifier": "gr-1", "guardrailVersion": "1"}}
    billing: Final = _GuardrailBilling()

    with rebound(litellm, "callbacks", [billing]):
        python_chunks: Final = await _drain(await _python(server, stream=True, **guardrail))
        await billing.wait_for(1)
        rust_chunks: Final = await _drain(await _rust(server, stream=True, **guardrail))
        await billing.wait_for(2)

    python_billed, rust_billed = billing.billed
    assert python_billed
    assert rust_billed == python_billed
    assert _sse_events(rust_chunks)[-1] == _sse_events(python_chunks)[-1]


@pytest.fixture
def no_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ROLE_NAME", "AWS_WEB_IDENTITY_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_aws_credentials")
@pytest.mark.parametrize(
    ("credentials", "raised"),
    (
        pytest.param({}, NoCredentialsError, id="no-credentials"),
        pytest.param(
            {"aws_web_identity_token": "os.environ/AWS_WEB_IDENTITY_TOKEN"}, AwsAuthError, id="environment-reference"
        ),
        pytest.param(
            {"aws_web_identity_token": "bedrock-web-identity-token-not-set"}, AwsAuthError, id="unresolved-token"
        ),
    ),
)
async def test_bedrock_invoke_credential_failure_raises_what_python_raises(
    server: RecordingServer, credentials: Mapping[str, str], raised: type[Exception]
) -> None:
    web_identity: Final = (
        {"aws_role_name": "arn:aws:iam::123456789012:role/bedrock", "aws_session_name": "litellm-session"}
        if credentials
        else {}
    )
    unsigned: Final = {"aws_access_key_id": None, "aws_secret_access_key": None, **credentials, **web_identity}

    with pytest.raises(raised) as python_raised:
        await _python(server, **unsigned)
    with pytest.raises(raised) as rust_raised:
        await _rust(server, **unsigned)

    assert type(rust_raised.value) is type(python_raised.value)
    assert str(rust_raised.value) == str(python_raised.value)
    assert getattr(rust_raised.value, "status_code", None) == getattr(python_raised.value, "status_code", None)
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_aws_credentials")
async def test_bedrock_invoke_access_key_without_its_secret_raises_what_python_raises(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    unsigned: Final = {"aws_access_key_id": None, "aws_secret_access_key": None}

    with pytest.raises(PartialCredentialsError) as python_raised:
        await _python(server, **unsigned)
    with pytest.raises(PartialCredentialsError) as rust_raised:
        await _rust(server, **unsigned)

    assert str(rust_raised.value) == str(python_raised.value)
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment",
    (
        pytest.param(
            {"model_id": "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"},
            id="model-id-arn",
        ),
        pytest.param(
            {
                "model": "bedrock/invoke/arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
            },
            id="invoke-arn",
        ),
        pytest.param(
            {"model": "bedrock/invoke/us-west-2/anthropic.claude-3-5-sonnet-20240620-v1:0", "aws_region_name": None},
            id="invoke-region-path",
        ),
        pytest.param(
            {
                "model": "bedrock/invoke/us-west-2/arn:aws:bedrock:eu-west-1:123456789012:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "aws_region_name": None,
            },
            id="invoke-region-path-arn",
        ),
        pytest.param({"stream": False, "drop_params": True, "temperature": 0.2}, id="sampling"),
    ),
)
async def test_bedrock_invoke_sends_the_url_and_body_python_sends(
    server: RecordingServer, deployment: Mapping[str, object]
) -> None:
    server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)

    await _python(server, **deployment)
    await _rust(server, **deployment)

    python_sent, rust_sent = server.requests
    assert rust_sent.path == python_sent.path
    assert rust_sent.body == python_sent.body


def _signing(request: RecordedRequest) -> tuple[object, ...]:
    authorization: Final = request.headers["authorization"]
    return (
        request.path,
        authorization.split(", Signature=", 1)[0].split("/", 2)[2],
        request.headers.get("x-amz-security-token"),
        "x-amz-date" in request.headers,
        "date" in request.headers,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller_headers",
    (
        pytest.param(
            {
                "x-amz-date": "20200101T000000Z",
                "x-amz-security-token": "caller-token",
                "date": "Wed, 01 Jan 2020 00:00:00 GMT",
            },
            id="all",
        ),
        pytest.param({"x-amz-security-token": "caller-token"}, id="security-token"),
        pytest.param({"x-amz-date": "20200101T000000Z"}, id="amz-date"),
        pytest.param({"Date": "Wed, 01 Jan 2020 00:00:00 GMT"}, id="date"),
    ),
)
@pytest.mark.parametrize("session_token", (None, "deployment-session"), ids=("static", "session"))
async def test_bedrock_invoke_signs_caller_amz_headers_as_python_does(
    server: RecordingServer, caller_headers: Mapping[str, str], session_token: str | None
) -> None:
    server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    deployment: Final = {
        "model": "bedrock/invoke/us-west-2/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "aws_region_name": None,
        "aws_session_token": session_token,
        "extra_headers": dict(caller_headers),
    }

    await _python(server, **deployment)
    await _rust(server, **deployment)

    python_sent, rust_sent = server.requests
    assert _signing(rust_sent) == _signing(python_sent)


@pytest.mark.asyncio
async def test_bedrock_invoke_refuses_an_invalid_guardrail_config_as_python_does(server: RecordingServer) -> None:
    invalid: Final = {"guardrailConfig": {"guardrailVersion": "1"}}

    with pytest.raises(litellm.BadRequestError) as python_raised:
        await _python(server, **invalid)
    with pytest.raises(litellm.BadRequestError) as rust_raised:
        await _rust(server, **invalid)

    assert str(rust_raised.value) == str(python_raised.value)
    assert type(rust_raised.value) is type(python_raised.value)
    assert getattr(rust_raised.value, "status_code", None) == getattr(python_raised.value, "status_code", None)
    assert server.requests == []


@pytest.mark.asyncio
async def test_bedrock_invoke_refuses_a_web_search_tool_as_python_does(server: RecordingServer) -> None:
    web_search: Final = {"tools": [{"type": "web_search_20250305", "name": "web_search"}]}

    with pytest.raises(litellm.BadRequestError) as python_raised:
        await _python(server, **web_search)
    with pytest.raises(litellm.BadRequestError) as rust_raised:
        await _rust(server, **web_search)

    assert type(rust_raised.value) is type(python_raised.value)
    assert str(rust_raised.value) == str(python_raised.value)
    assert rust_raised.value.llm_provider == python_raised.value.llm_provider  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the public error's provider
    assert rust_raised.value.model == python_raised.value.model  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the public error's model
    assert server.requests == []
    assert getattr(rust_raised.value, "litellm_debug_info", None) == getattr(
        python_raised.value, "litellm_debug_info", None
    )


_EVENTS: Final = tuple(data for _, data in MESSAGES_EVENTS)


class _Recorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger's untyped init
        self.logged: list[tuple[str, dict[str, object]]] = []

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.logged.append(("success", _OBJECT.validate_python(kwargs.get("standard_logging_object"))))

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.logged.append(("failure", _OBJECT.validate_python(kwargs.get("standard_logging_object"))))

    async def wait_for(self, count: int) -> None:
        async with asyncio.timeout(5):
            while len(self.logged) < count:
                await asyncio.sleep(0.01)


def _billed(logged: tuple[str, Mapping[str, object]]) -> tuple[str, object, object]:
    kind, record = logged
    response: Final = record.get("response")
    return (
        kind,
        {name: value for name, value in _OBJECT.validate_python(response).items() if name != "created"}
        if isinstance(response, dict)
        else response,
        record.get("error_str"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    (
        pytest.param(_EVENTS, id="complete"),
        pytest.param(_EVENTS[:5], id="no-message-stop"),
        pytest.param(_EVENTS[:3], id="no-message-delta"),
        pytest.param(
            (*_EVENTS[:2], {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hé"}}),
            id="non-ascii",
        ),
    ),
)
async def test_bedrock_invoke_stream_relays_the_frames_and_bills_what_python_does(
    server: RecordingServer, events: Sequence[Mapping[str, object]]
) -> None:
    server.default_response = ResponseSpec(body=b"".join(_aws_frame(event) for event in events))
    recorder: Final = _Recorder()

    with rebound(litellm, "callbacks", [recorder]):
        python_chunks: Final = await _drain(await _python(server, stream=True))
        await recorder.wait_for(1)
        rust_chunks: Final = await _drain(await _rust(server, stream=True))
        await recorder.wait_for(2)

    assert rust_chunks == python_chunks
    python_logged, rust_logged = recorder.logged
    assert _billed(rust_logged) == _billed(python_logged)


_STREAM_HEAD: Final = b"".join(_aws_frame(event) for event in _EVENTS[:3])


async def _failed_stream(
    entry: Callable[..., Awaitable[object]], server: RecordingServer
) -> tuple[tuple[bytes, ...], Exception]:
    chunks: Final[list[bytes]] = []
    try:
        async for chunk in cast(
            AsyncIterator[bytes], await entry(server, stream=True)
        ):  # cast-ok: a streamed Messages answer yields SSE bytes
            chunks.append(chunk)
    except Exception as error:  # noqa: BLE001  # each lane's failure class is what the test compares
        return tuple(chunks), error
    raise AssertionError("the stream ended without a failure")


def _raised(error: BaseException | None) -> tuple[object, ...]:
    if error is None:
        return (None,)
    return (
        type(error),
        str(error),
        getattr(error, "status_code", None),
        getattr(error, "litellm_debug_info", None),
        *_raised(error.__cause__),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    (
        pytest.param(
            ResponseSpec(body=_STREAM_HEAD + _wire_frame("event", "internalServerException", b'{"message": "boom"}')),
            id="error-event-mid-stream",
        ),
        pytest.param(
            ResponseSpec(body=_wire_frame("exception", "throttlingException", b'{"message": "slow"}')),
            id="exception-frame-first",
        ),
        pytest.param(ResponseSpec(body=b""), id="empty-body"),
        pytest.param(ResponseSpec(body=_wire_frame("event", "somethingNew", b'{"message": "x"}')), id="unknown-event"),
        pytest.param(ResponseSpec(body=MESSAGES_RESPONSE), id="json-body"),
        pytest.param(ResponseSpec(body=_STREAM_HEAD + _chunk_frame(b'{"type": ')), id="invalid-json-chunk"),
        pytest.param(ResponseSpec(body=_STREAM_HEAD + _chunk_frame(b"[1]")), id="list-chunk"),
        pytest.param(ResponseSpec(body=_STREAM_HEAD + _STREAM_HEAD, body_chunks=2, cut_after=1), id="cut-body"),
    ),
)
async def test_bedrock_invoke_stream_failure_raises_and_logs_what_python_does(
    server: RecordingServer, response: ResponseSpec
) -> None:
    server.default_response = response
    recorder: Final = _Recorder()

    with rebound(litellm, "callbacks", [recorder]):
        python_chunks, python_error = await _failed_stream(_python, server)
        await recorder.wait_for(1)
        rust_chunks, rust_error = await _failed_stream(_rust, server)
        await recorder.wait_for(2)

    assert rust_chunks == python_chunks
    assert _raised(rust_error) == _raised(python_error)
    python_logged, rust_logged = recorder.logged
    assert _billed(rust_logged) == _billed(python_logged)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tools",
    (
        pytest.param(
            [
                {
                    "name": "a",
                    "description": "A",
                    "input_schema": {"type": "object"},
                    "custom": {"defer_loading": True},
                },
                {
                    "name": "b",
                    "description": "B",
                    "input_schema": {"type": "object"},
                    "custom": {"defer_loading": "no"},
                },
            ],
            id="custom-field",
        ),
        pytest.param(
            [
                {
                    "type": "custom",
                    "name": "n",
                    "input_schema": {"type": "custom", "properties": {"o": {"type": "custom"}}},
                }
            ],
            id="custom-schema-type",
        ),
        pytest.param(
            [{"name": " ", "input_schema": {"type": "object"}}, {"name": "", "input_schema": {"type": "object"}}],
            id="unnamed",
        ),
    ),
)
async def test_bedrock_invoke_logs_the_tools_python_shaped(
    server: RecordingServer, tools: Sequence[Mapping[str, object]]
) -> None:
    server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    recorder: Final = _Recorder()
    python_tools: Final = json.loads(json.dumps(tools))
    rust_tools: Final = json.loads(json.dumps(tools))

    with rebound(litellm, "callbacks", [recorder]):
        await _python(server, tools=python_tools)
        await recorder.wait_for(1)
        await _rust(server, tools=rust_tools)
        await recorder.wait_for(2)

    (_, python_logged), (_, rust_logged) = recorder.logged
    assert rust_tools == python_tools != tools
    assert _OBJECT.validate_python(rust_logged["model_parameters"])["tools"] == python_tools
    assert _OBJECT.validate_python(python_logged["model_parameters"])["tools"] == python_tools


class _Warnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


_SCHEMA: Final = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    },
}
_NATIVE_OUTPUT_REJECTED: Final = ResponseSpec(
    body={"message": "output_config.format: Extra inputs are not permitted"}, status=400
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_params", "rejected"),
    (
        pytest.param({"output_config": {"format": _SCHEMA}}, False, id="native-output"),
        pytest.param({"output_config": {"format": _SCHEMA}}, True, id="native-output-rejected"),
        pytest.param(
            {"model": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0", "output_format": _SCHEMA},
            False,
            id="inlined-schema",
        ),
        pytest.param(
            {"max_tokens": 1000, "context_management": {"edits": [{"type": "clear_thinking_20251015"}]}},
            False,
            id="clear-thinking",
        ),
        pytest.param({"service_tier": "auto"}, False, id="service-tier"),
    ),
)
async def test_bedrock_invoke_warns_what_python_warns(
    server: RecordingServer, request_params: Mapping[str, object], rejected: bool
) -> None:
    server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    warnings: Final = _Warnings()
    logger: Final = logging.getLogger("LiteLLM")
    logger.addHandler(warnings)
    try:
        if rejected:
            server.enqueue(_NATIVE_OUTPUT_REJECTED)
        await _python(server, **request_params)
        python_warnings: Final = list(warnings.messages)
        warnings.messages.clear()
        if rejected:
            server.enqueue(_NATIVE_OUTPUT_REJECTED)
        await _rust(server, **request_params)
    finally:
        logger.removeHandler(warnings)

    assert warnings.messages == python_warnings


@pytest.mark.asyncio
async def test_bedrock_invoke_stream_bills_the_terminal_usage_after_the_caller_detaches(
    server: RecordingServer,
) -> None:
    start, *middle, (_, delta), stop = MESSAGES_EVENTS
    events: Final = (start[1], *(data for _, data in middle), {**delta, "usage": {"output_tokens": 412}}, stop[1])
    server.default_response = ResponseSpec(
        body=b"".join(_aws_frame(event) for event in events), body_chunks=len(events), chunk_delay=0.1
    )
    recorder: Final = _Recorder()

    with rebound(litellm, "callbacks", [recorder]):
        for entry, count in ((_python, 1), (_rust, 2)):
            stream = cast(
                AsyncGenerator[bytes], await entry(server, stream=True)
            )  # cast-ok: a streamed Messages answer is an async generator
            await anext(stream)
            await stream.aclose()
            await recorder.wait_for(count)

    python_logged, rust_logged = recorder.logged
    assert _billed(rust_logged) == _billed(python_logged)
    assert (
        _OBJECT.validate_python(_OBJECT.validate_python(rust_logged[1]["response"])["usage"])["completion_tokens"]
        == 412
    )


_SPLIT_MODEL: Final = "anthropic.claude-fixture-mantle-split"
_CLEAR_THINKING: Final = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bedrock_adaptive", "thinking"),
    [
        pytest.param(True, {"type": "adaptive"}, id="bedrock-adaptive"),
        pytest.param(False, None, id="bedrock-not-adaptive"),
    ],
)
async def test_bedrock_mantle_runs_the_invoke_helpers_on_the_bedrock_capabilities(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, bedrock_adaptive: bool, thinking: object
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        _SPLIT_MODEL,
        {"litellm_provider": "bedrock", "mode": "chat", "supports_adaptive_thinking": bedrock_adaptive},
    )
    monkeypatch.setitem(
        litellm.model_cost,
        f"bedrock_mantle/{_SPLIT_MODEL}",
        {"litellm_provider": "bedrock_mantle", "mode": "chat", "supports_adaptive_thinking": not bedrock_adaptive},
    )
    model_cost_table.changed_in_place()

    await _RUST_ENTRY(
        model=f"bedrock_mantle/{_SPLIT_MODEL}",
        messages=list(MESSAGES),
        max_tokens=32000,
        api_key="test-key",
        api_base=server.base_url,
        context_management=_CLEAR_THINKING,
    )

    assert_served_natively(server)
    body: Final = _OBJECT.validate_python(server.requests[0].body)
    assert body.get("thinking") == thinking


MANTLE_MODEL: Final = "bedrock_mantle/anthropic.claude-sonnet-5"
_MANTLE_HEADERS: Final = (
    "authorization",
    "anthropic-version",
    "anthropic-beta",
    "anthropic-workspace-id",
    "x-api-key",
)


@pytest.fixture
def local_beta_headers(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    reload_beta_headers_config()
    yield
    monkeypatch.delenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS")
    reload_beta_headers_config()


def _mantle_arguments(server: RecordingServer, kwargs: Mapping[str, object]) -> dict[str, object]:
    return {
        "model": MANTLE_MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 8,
        "api_base": server.base_url,
        "api_key": "test-bearer",
        "litellm_call_id": str(uuid.uuid4()),
        **kwargs,
    }


async def _mantle(entry: Callable[..., Awaitable[object]], server: RecordingServer, **kwargs: object) -> object:
    return await entry(**_mantle_arguments(server, kwargs))


def _mantle_wire(request: RecordedRequest) -> dict[str, object]:
    return {
        "path": request.path,
        "body": request.body,
        "headers": {name: request.headers.get(name) for name in _MANTLE_HEADERS},
    }


_PYTHON_WIRE: Final = _OBJECT.validate_json((Path(__file__).parent / "bedrock_mantle_python_wire.json").read_bytes())


def _betas(request: RecordedRequest) -> list[str] | None:
    header: Final = request.headers.get("anthropic-beta")
    return None if header is None else header.split(",")


_SAFEGUARDS: Final = [{"type": "dangerous_tool_use", "classifier_context": {"v": 1, "permission_mode": "auto"}}]
_CLEAR_TOOL_USES: Final = {"edits": [{"type": "clear_tool_uses_20250919"}]}
_WEATHER_TOOL: Final = {
    "name": "get_weather",
    "description": "Look up the weather",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_beta_headers")
@pytest.mark.parametrize(
    ("request_params", "betas"),
    (
        pytest.param({}, None, id="plain"),
        pytest.param(
            {
                "extra_headers": {
                    "anthropic-beta": "claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27"
                }
            },
            ["claude-code-20250219", "context-management-2025-06-27", "interleaved-thinking-2025-05-14"],
            id="accepted-betas",
        ),
        pytest.param(
            {
                "extra_headers": {
                    "anthropic-beta": "code-execution-2025-08-25,context-1m-2025-08-07,files-api-2025-04-14"
                }
            },
            ["context-1m-2025-08-07"],
            id="rejected-betas-dropped",
        ),
        pytest.param(
            {"extra_headers": {"anthropic-beta": "code-execution-2025-08-25"}}, None, id="every-beta-rejected"
        ),
        pytest.param(
            {"extra_headers": {"anthropic-beta": "advanced-tool-use-2025-11-20"}},
            ["tool-search-tool-2025-10-19"],
            id="advanced-tool-use-renamed",
        ),
        pytest.param(
            {"extra_headers": {"anthropic-beta": "context-1m-2025-08-07"}, "context_management": _CLEAR_TOOL_USES},
            ["context-1m-2025-08-07", "context-management-2025-06-27"],
            id="feature-beta-joins",
        ),
        pytest.param({"safeguards": _SAFEGUARDS}, ["dangerous-tool-use-2026-09-03"], id="safeguards-beta"),
        pytest.param(
            {"extra_headers": {"anthropic-beta": "dangerous-tool-use-2026-09-03"}},
            ["dangerous-tool-use-2026-09-03"],
            id="client-dangerous-tool-use",
        ),
        pytest.param(
            {
                "extra_headers": {"anthropic-beta": "context-1m-2025-08-07"},
                "context_management": _CLEAR_TOOL_USES,
                "anthropic_version": "bedrock-2023-05-31",
            },
            ["context-1m-2025-08-07", "context-management-2025-06-27"],
            id="version-stays-out-of-the-body",
        ),
        pytest.param(
            {
                "context_management": {
                    "edits": [{"type": "clear_thinking_20251015", "keep": "all"}, {"type": "clear_tool_uses_20250919"}]
                },
                "thinking": {"type": "adaptive"},
            },
            ["context-management-2025-06-27"],
            id="clear-thinking-with-thinking",
        ),
        pytest.param({"tools": [_WEATHER_TOOL], "tool_choice": {"type": "auto"}}, None, id="tools"),
        pytest.param({"aws_bedrock_project_id": "proj_abc123def456"}, None, id="workspace"),
    ),
)
async def test_bedrock_mantle_sends_the_request_python_sends(
    server: RecordingServer,
    request: pytest.FixtureRequest,
    request_params: Mapping[str, object],
    betas: list[str] | None,
) -> None:
    server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)

    rust_response: Final = await _mantle(_RUST_ENTRY, server, **request_params)

    [rust_sent] = server.requests
    assert _mantle_wire(rust_sent) == _PYTHON_WIRE[f"request/{request.node.callspec.id}"]
    assert rust_sent.path.endswith("/anthropic/v1/messages")
    assert rust_sent.headers["authorization"] == "Bearer test-bearer"
    assert rust_sent.headers["anthropic-version"] == "2023-06-01"
    assert _betas(rust_sent) == betas
    body: Final = _OBJECT.validate_python(rust_sent.body)
    assert body["model"] == "anthropic.claude-sonnet-5"
    assert {"anthropic_version", "anthropic_beta", "aws_bedrock_project_id"}.isdisjoint(body)
    assert _public(rust_response) == MESSAGES_RESPONSE


@pytest.mark.asyncio
async def test_bedrock_mantle_sends_a_proxy_clients_betas_filtered_as_python_does(
    server: RecordingServer, local_beta_headers: None
) -> None:
    from litellm.proxy.litellm_pre_call_utils import add_provider_specific_headers_to_request

    proxy_request: Final[dict[str, object]] = {}  # mutable-ok: the proxy helper fills the request it is handed
    add_provider_specific_headers_to_request(
        data=proxy_request,
        headers={
            "anthropic-beta": "claude-code-20250219,fast-mode-2026-02-01,interleaved-thinking-2025-05-14",
            "anthropic-version": "2023-06-01",
            "user-agent": "claude-cli/2.1.239",
        },
    )

    await _mantle(_RUST_ENTRY, server, **proxy_request)

    [rust_sent] = server.requests
    assert _mantle_wire(rust_sent) == _PYTHON_WIRE["proxy_betas"]
    assert _betas(rust_sent) == ["claude-code-20250219", "interleaved-thinking-2025-05-14"]


@pytest.mark.asyncio
async def test_bedrock_mantle_strips_the_region_prefix_from_the_model_as_python_does(server: RecordingServer) -> None:
    model: Final = "bedrock_mantle/us-east-2/anthropic.claude-haiku-4-5"

    await _mantle(_RUST_ENTRY, server, model=model)

    [rust_sent] = server.requests
    assert _mantle_wire(rust_sent) == _PYTHON_WIRE["region_prefix"]
    assert _OBJECT.validate_python(rust_sent.body)["model"] == "anthropic.claude-haiku-4-5"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    (
        pytest.param(MANTLE_MODEL, id="bedrock-mantle"),
        pytest.param("bedrock/mantle/anthropic.claude-mythos-preview", id="mantle-route"),
    ),
)
async def test_bedrock_mantle_signs_against_the_messages_url_as_python_does(
    server: RecordingServer, model: str
) -> None:
    sigv4: Final = {"model": model, "api_key": None, **AWS}

    await _mantle(_RUST_ENTRY, server, **sigv4)

    [rust_sent] = server.requests
    expected: Final = _OBJECT.validate_python(_PYTHON_WIRE[f"signing/{model}"])
    assert list(_signing(rust_sent)) == expected["signing"]
    assert rust_sent.body == expected["body"]
    assert "/us-east-1/bedrock/aws4_request" in rust_sent.headers["authorization"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    (
        pytest.param(MANTLE_MODEL, id="bedrock-mantle"),
        pytest.param("bedrock/mantle/anthropic.claude-mythos-preview", id="mantle-route"),
    ),
)
async def test_bedrock_mantle_stream_passes_the_sse_through_as_python_does(server: RecordingServer, model: str) -> None:
    server.default_response = ResponseSpec(body=None, events=MESSAGES_EVENTS)
    route: Final = {"model": model, **({} if model == MANTLE_MODEL else {"api_key": None, **AWS})}

    rust_chunks: Final = await _drain(await _mantle(_RUST_ENTRY, server, stream=True, **route))

    [rust_sent] = server.requests
    assert _OBJECT.validate_python(rust_sent.body)["stream"] is True
    assert rust_sent.body == _PYTHON_WIRE[f"stream/{model}"]
    assert _sse_events(rust_chunks) == tuple(event for _, event in MESSAGES_EVENTS)
    assert b"event: message_stop" in b"".join(rust_chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "usage"),
    (
        pytest.param(
            {name: value for name, value in MESSAGES_RESPONSE.items() if name != "usage"},
            {"input_tokens": 0, "output_tokens": 0},
            id="usage-backfilled",
        ),
        pytest.param(
            {**MESSAGES_RESPONSE, "usage": {"input_tokens": 42, "output_tokens": 7, "cache_read_input_tokens": 5}},
            {"input_tokens": 42, "output_tokens": 7, "cache_read_input_tokens": 5},
            id="usage-kept",
        ),
    ),
)
async def test_bedrock_mantle_route_answers_the_usage_python_answers(
    server: RecordingServer, answer: Mapping[str, object], usage: Mapping[str, int]
) -> None:
    server.default_response = ResponseSpec(body=dict(answer))
    route: Final = {"model": "bedrock/mantle/anthropic.claude-opus-4-8", "api_key": None, **AWS}

    rust_response: Final = await _mantle(_RUST_ENTRY, server, **route)

    assert _public(rust_response) == {**answer, "usage": usage}
    rust_usage: Final = _OBJECT.validate_python(_OBJECT.validate_python(rust_response)["usage"])
    assert {name: rust_usage.get(name) for name in usage} == usage


@pytest.mark.asyncio
@pytest.mark.parametrize("model", (MANTLE_MODEL, "bedrock/mantle/anthropic.claude-opus-4-8"))
@pytest.mark.parametrize("usage", ([1], "", 0), ids=("array", "string", "number"))
async def test_bedrock_mantle_refuses_a_usage_that_is_not_an_object(
    server: RecordingServer, model: str, usage: object
) -> None:
    server.default_response = ResponseSpec(body={**MESSAGES_RESPONSE, "usage": usage})
    route: Final = {"model": model, "api_key": None, **AWS}

    with pytest.raises(litellm.APIError) as raised:
        await _mantle(_RUST_ENTRY, server, **route)

    assert type(raised.value) is litellm.APIError
    assert raised.value.status_code == 500
    assert raised.value.llm_provider == model.split("/")[0]  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped
    assert_served_natively(server)


_REFUSED_MODELS: Final = (
    # (model, the model Python main's handler logs, the model its Bedrock invoke transform names)
    pytest.param(
        "bedrock_mantle/us-west-2/anthropic.claude-sonnet-5",
        "anthropic.claude-sonnet-5",
        "anthropic.claude-sonnet-5",
        id="region-prefix",
    ),
    pytest.param(
        "bedrock_mantle/us-west-2/mantle/anthropic.claude-sonnet-5",
        "mantle/anthropic.claude-sonnet-5",
        "anthropic.claude-sonnet-5",
        id="region-prefix-mantle-segment",
    ),
    pytest.param(
        "bedrock/mantle/anthropic.claude-opus-4-8",
        "mantle/anthropic.claude-opus-4-8",
        "anthropic.claude-opus-4-8",
        id="mantle-route",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "logged_model", "raised_model"), _REFUSED_MODELS)
async def test_bedrock_mantle_refusal_logs_and_names_the_model_python_does(
    server: RecordingServer, model: str, logged_model: str, raised_model: str
) -> None:
    recorder: Final = _Recorder()
    refused: Final = {"model": model, "tools": [{"type": "web_search_20250305", "name": "web_search"}]}

    with rebound(litellm, "callbacks", [recorder]), pytest.raises(litellm.BadRequestError) as raised:
        await _mantle(_RUST_ENTRY, server, **refused)
    await recorder.wait_for(1)

    assert type(raised.value) is litellm.BadRequestError
    assert raised.value.model == raised_model  # pyright: ignore[reportUnknownMemberType]  # the public error leaves model untyped
    assert raised.value.llm_provider == "bedrock"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped
    # Python main raises the Bedrock invoke transform's own error, unwrapped by an HTTP handler.
    assert str(raised.value).startswith("litellm.BadRequestError: Bedrock does not support")
    [(kind, logged)] = recorder.logged
    assert (kind, logged["model"]) == ("failure", logged_model)
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "logged_model", "raised_model"),
    (
        # (model, the model Python main's failure log names once the call was sent, the model Rust raises)
        pytest.param(
            "bedrock_mantle/us-west-2/anthropic.claude-sonnet-5",
            "anthropic.claude-sonnet-5",
            "anthropic.claude-sonnet-5",
            id="region-prefix",
        ),
        pytest.param(
            "bedrock_mantle/us-west-2/mantle/anthropic.claude-sonnet-5",
            "anthropic.claude-sonnet-5",
            "anthropic.claude-sonnet-5",
            id="region-prefix-mantle-segment",
        ),
        pytest.param(
            "bedrock/mantle/anthropic.claude-opus-4-8",
            "bedrock/anthropic.claude-opus-4-8",
            "anthropic.claude-opus-4-8",
            id="mantle-route",
        ),
    ),
)
async def test_bedrock_mantle_unbillable_usage_names_the_model_it_sent(
    server: RecordingServer, model: str, logged_model: str, raised_model: str
) -> None:
    server.default_response = ResponseSpec(body={**MESSAGES_RESPONSE, "usage": [1]})
    recorder: Final = _Recorder()

    with rebound(litellm, "callbacks", [recorder]), pytest.raises(litellm.APIError) as raised:
        await _mantle(_RUST_ENTRY, server, model=model, api_key=None, **AWS)
    await recorder.wait_for(1)

    # Python main raises a bare TypeError here; Rust raises an APIError naming the model it sent.
    assert type(raised.value) is litellm.APIError
    assert raised.value.status_code == 500
    assert raised.value.model == raised_model  # pyright: ignore[reportUnknownMemberType]  # the public error leaves model untyped
    assert raised.value.llm_provider == model.split("/")[0]  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped
    [(kind, logged)] = recorder.logged
    assert (kind, logged["model"]) == ("failure", logged_model)
    assert_served_natively(server)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", (MANTLE_MODEL, "bedrock/mantle/anthropic.claude-opus-4-8"))
async def test_bedrock_mantle_logs_the_tools_rust_shaped_and_leaves_the_callers_alone(
    server: RecordingServer, model: str
) -> None:
    server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    recorder: Final = _Recorder()
    tools: Final = [{"type": "custom", "name": "", "description": "d", "input_schema": {"type": "custom"}}]
    sent_tools: Final = json.loads(json.dumps(tools))

    with rebound(litellm, "callbacks", [recorder]):
        await _mantle(_RUST_ENTRY, server, model=model, tools=sent_tools)
        await recorder.wait_for(1)

    [(_, logged)] = recorder.logged
    shaped: Final = [
        {"type": "custom", "name": "litellm_unnamed_tool_0", "description": "d", "input_schema": {"type": "object"}}
    ]
    assert _OBJECT.validate_python(logged["model_parameters"])["tools"] == shaped
    assert _OBJECT.validate_python(server.requests[0].body)["tools"] == shaped
    assert sent_tools == tools


_SIGV4_ONLY: Final = {
    "api_key": None,
    "aws_access_key_id": "test-access-key",
    "aws_secret_access_key": "test-secret-key",
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "raised"),
    (
        pytest.param({"aws_region_name": "US_WEST_2"}, litellm.BadRequestError, id="bedrock-mantle-region"),
        pytest.param(
            {"model": "bedrock/mantle/anthropic.claude-opus-4-8", **_SIGV4_ONLY, "aws_region_name": "US_WEST_2"},
            ValueError,
            id="mantle-route-region",
        ),
    ),
)
async def test_bedrock_mantle_invalid_region_raises_what_python_raises(
    server: RecordingServer, route: Mapping[str, object], raised: type[Exception]
) -> None:
    with pytest.raises(raised) as rust_raised:
        await _mantle(_RUST_ENTRY, server, **route)

    assert type(rust_raised.value) is raised
    assert getattr(rust_raised.value, "status_code", None) == (400 if raised is litellm.BadRequestError else None)
    if raised is litellm.BadRequestError:
        # Python refuses the region while resolving the provider, so its error names no provider and the request's model.
        assert getattr(rust_raised.value, "llm_provider", None) == ""
        assert getattr(rust_raised.value, "model", None) == MANTLE_MODEL
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_aws_credentials")
@pytest.mark.parametrize(
    ("route", "raised"),
    (
        pytest.param({}, ValueError, id="bedrock-mantle"),
        pytest.param(
            {"aws_web_identity_token": "os.environ/AWS_WEB_IDENTITY_TOKEN"},
            AwsAuthError,
            id="bedrock-mantle-web-identity",
        ),
        pytest.param({"model": "bedrock/mantle/anthropic.claude-opus-4-8"}, NoCredentialsError, id="mantle-route"),
    ),
)
async def test_bedrock_mantle_credential_failure_raises_what_python_raises(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, route: Mapping[str, object], raised: type[Exception]
) -> None:
    monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
    identity: Final = (
        {"aws_role_name": "arn:aws:iam::123456789012:role/bedrock", "aws_session_name": "litellm-session"}
        if "aws_web_identity_token" in route
        else {}
    )
    unsigned: Final = {"api_key": None, "aws_region_name": "us-east-1", **route, **identity}

    with pytest.raises(raised) as rust_raised:
        await _mantle(_RUST_ENTRY, server, **unsigned)

    assert type(rust_raised.value) is raised
    assert getattr(rust_raised.value, "status_code", None) == (400 if raised is AwsAuthError else None)
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_aws_credentials")
async def test_bedrock_invoke_missing_profile_raises_what_python_raises(server: RecordingServer) -> None:
    unsigned: Final = {"aws_access_key_id": None, "aws_secret_access_key": None, "aws_profile_name": "absent-profile"}

    with pytest.raises(ProfileNotFound) as python_raised:
        await _python(server, **unsigned)
    with pytest.raises(ProfileNotFound) as rust_raised:
        await _rust(server, **unsigned)

    assert rust_raised.value.kwargs == python_raised.value.kwargs == {"profile": "absent-profile"}
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_aws_credentials")
@pytest.mark.parametrize(
    ("route", "raised"),
    (
        pytest.param({}, ValueError, id="bedrock-mantle"),
        pytest.param({"model": "bedrock/mantle/anthropic.claude-opus-4-8"}, ProfileNotFound, id="mantle-route"),
    ),
)
async def test_bedrock_mantle_missing_profile_raises_what_python_raises(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, route: Mapping[str, object], raised: type[Exception]
) -> None:
    monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
    unsigned: Final = {"api_key": None, "aws_region_name": "us-east-1", "aws_profile_name": "absent-profile", **route}

    with pytest.raises(raised) as rust_raised:
        await _mantle(_RUST_ENTRY, server, **unsigned)

    assert type(rust_raised.value) is raised
    if raised is ProfileNotFound:
        assert rust_raised.value.kwargs == {"profile": "absent-profile"}
    assert server.requests == []


_STS_ACCESS_DENIED: Final = ResponseSpec(
    body=(
        b'<ErrorResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/"><Error><Type>Sender</Type>'
        b"<Code>AccessDenied</Code><Message>denied</Message></Error><RequestId>r</RequestId></ErrorResponse>"
    ),
    status=403,
    headers={"content-type": "text/xml"},
)


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_aws_credentials")
async def test_bedrock_mantle_denied_assume_role_raises_the_botocore_client_error(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from botocore.exceptions import ClientError

    monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
    server.default_response = _STS_ACCESS_DENIED
    route: Final = {
        "api_key": None,
        "aws_region_name": "us-east-1",
        **_SIGV4_ONLY,
        "aws_role_name": "arn:aws:iam::123456789012:role/bedrock",
        "aws_session_name": "litellm-session",
        "aws_sts_endpoint": server.base_url,
    }

    with pytest.raises(ClientError) as raised:
        await _mantle(_RUST_ENTRY, server, **route)

    assert raised.value.response.get("Error", {}).get("Code") == "AccessDenied"
    assert raised.value.operation_name == "AssumeRole"
    assert all(request.method == "POST" and "/v1/messages" not in request.path for request in server.requests)


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_aws_credentials")
async def test_bedrock_mantle_refusal_without_credentials_raises_the_transforms_bad_request(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)
    refused: Final = {"api_key": None, "tools": [{"type": "web_search_20250305", "name": "web_search"}]}

    with pytest.raises(litellm.BadRequestError) as rust_raised:
        await _mantle(_RUST_ENTRY, server, **refused)

    assert type(rust_raised.value) is litellm.BadRequestError
    assert rust_raised.value.status_code == 400
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("model", (MANTLE_MODEL, "bedrock/mantle/anthropic.claude-opus-4-8"))
async def test_bedrock_mantle_reads_its_capabilities_from_the_rust_catalog(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    from litellm.rust_bridge.host import model_capabilities

    def python_probe(*_args: object) -> object:
        raise AssertionError("a Mantle call reads its capabilities from the Rust catalog")

    monkeypatch.setattr(model_capabilities, "anthropic_model_capabilities", python_probe)
    route: Final = {"model": model, **({} if model == MANTLE_MODEL else {"api_key": None, **AWS})}

    await _mantle(_RUST_ENTRY, server, **route)

    assert_served_natively(server)


@pytest.mark.asyncio
async def test_bedrock_mantle_route_sends_the_project_id_as_the_workspace_header(server: RecordingServer) -> None:
    await _mantle(
        _RUST_ENTRY,
        server,
        model="bedrock/mantle/anthropic.claude-opus-4-8",
        api_key=None,
        aws_bedrock_project_id="proj_abc123def456",
        **AWS,
    )

    [sent] = server.requests
    assert sent.headers["anthropic-workspace-id"] == "proj_abc123def456"
    assert "aws_bedrock_project_id" not in _OBJECT.validate_python(sent.body)
