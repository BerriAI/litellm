"""
Regression tests for LIT-4313: the native `sagemaker/` streaming path must
forward each AWS event-stream frame as it arrives instead of buffering to a
fixed 1024-byte threshold and then draining a burst of tokens.

The buffering came from `response.aiter_bytes(chunk_size=1024)`: httpx's
ByteChunker withholds bytes until `chunk_size` accumulates, so the first token
could not be produced until enough later frames had arrived to cross 1024 bytes,
inflating TTFT and turning a steady provider stream into gap-then-burst delivery.
"""

import binascii
import json
import struct
from typing import AsyncIterator, Final, Iterator
from unittest.mock import MagicMock

import httpx
import litellm
import pytest
import respx

from litellm.llms.sagemaker.common_utils import SagemakerError
from litellm.llms.sagemaker.completion.handler import SagemakerLLM


@pytest.fixture
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


def _encode_header(name: str, value: str) -> bytes:
    name_b = name.encode("utf-8")
    value_b = value.encode("utf-8")
    return struct.pack("B", len(name_b)) + name_b + struct.pack("B", 7) + struct.pack(">H", len(value_b)) + value_b


def _encode_event_frame(payload: bytes) -> bytes:
    """Encode one AWS event-stream message that botocore's EventStreamBuffer decodes."""
    headers = {
        ":event-type": "PayloadPart",
        ":content-type": "application/json",
        ":message-type": "event",
    }
    headers_b = b"".join(_encode_header(k, v) for k, v in headers.items())
    total_len = 16 + len(headers_b) + len(payload)
    prelude = struct.pack(">I", total_len) + struct.pack(">I", len(headers_b))
    prelude_crc = struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message = prelude + prelude_crc + headers_b + payload
    message_crc = struct.pack(">I", binascii.crc32(message) & 0xFFFFFFFF)
    return message + message_crc


def _token_frame(text: str) -> bytes:
    # SageMaker HF TGI streaming payloads are `{"token": {"text": ...}}` blobs.
    sse = "data: " + json.dumps({"token": {"text": text}}) + "\n\n"
    return _encode_event_frame(sse.encode("utf-8"))


def _make_frames(n: int) -> list[bytes]:
    frames = [_token_frame(f"token{i} ") for i in range(n)]
    assert all(len(f) < 1024 for f in frames)
    return frames


class _CountingSyncStream(httpx.SyncByteStream):
    """Yields provider frames one at a time and records how many have been pulled."""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = frames
        self.consumed = 0

    def __iter__(self) -> Iterator[bytes]:
        for frame in self._frames:
            self.consumed += 1
            yield frame


class _CountingAsyncStream(httpx.AsyncByteStream):
    """Yields provider frames one at a time and records how many have been pulled."""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = frames
        self.consumed = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            self.consumed += 1
            yield frame


class _FakeSyncClient:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    def post(self, *args, **kwargs) -> httpx.Response:
        return self._response


class _FakeAsyncClient:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    async def post(self, *args, **kwargs) -> httpx.Response:
        return self._response


def test_sync_native_streaming_forwards_each_frame_incrementally():
    """Each token must be emitted after exactly one newly-pulled source frame.

    With the old `chunk_size=1024` the httpx chunker would swallow several small
    frames before yielding, so the first token would arrive only after `consumed`
    had already crossed multiple frames, and tokens would then replay in a burst.
    """
    frames = _make_frames(24)
    stream = _CountingSyncStream(frames)
    response = httpx.Response(200, stream=stream)

    completion_stream = SagemakerLLM().make_sync_call(
        api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/phi-4/invocations-response-stream",
        headers={},
        data="",
        logging_obj=MagicMock(),
        client=_FakeSyncClient(response),
    )

    consumed_at_token = []
    texts = []
    for chunk in completion_stream:
        if chunk is not None and chunk["text"]:
            consumed_at_token.append(stream.consumed)
            texts.append(chunk["text"])

    assert texts == [f"token{i} " for i in range(len(frames))]
    assert consumed_at_token == list(range(1, len(frames) + 1))


def test_sync_native_streaming_raises_sagemaker_error_on_non_200():
    response = httpx.Response(500, text="boom")

    with pytest.raises(SagemakerError) as exc_info:
        SagemakerLLM().make_sync_call(
            api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/phi-4/invocations-response-stream",
            headers={},
            data="",
            logging_obj=MagicMock(),
            client=_FakeSyncClient(response),
        )

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_async_native_streaming_forwards_each_frame_incrementally():
    """Each token must be emitted after exactly one newly-pulled source frame.

    With the old `chunk_size=1024` the httpx chunker would swallow several small
    frames before yielding, so the first token would arrive only after `consumed`
    had already crossed multiple frames, and tokens would then replay in a burst.
    """
    frames = _make_frames(24)
    stream = _CountingAsyncStream(frames)
    response = httpx.Response(200, stream=stream)

    completion_stream = await SagemakerLLM().make_async_call(
        api_base="https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/phi-4/invocations-response-stream",
        headers={},
        data="",
        logging_obj=MagicMock(),
        client=_FakeAsyncClient(response),
    )

    consumed_at_token = []
    texts = []
    async for chunk in completion_stream:
        if chunk is not None and chunk["text"]:
            consumed_at_token.append(stream.consumed)
            texts.append(chunk["text"])

    assert texts == [f"token{i} " for i in range(len(frames))]
    assert consumed_at_token == list(range(1, len(frames) + 1))


def test_load_credentials_assumes_role_with_external_id(monkeypatch):
    """A trust policy requiring sts:ExternalId must be satisfied by the deployment's aws_external_id."""
    import datetime

    import boto3
    from botocore.exceptions import ClientError
    from unittest.mock import patch

    monkeypatch.delenv("AWS_EXTERNAL_ID", raising=False)

    class FakeSTSClient:
        def get_caller_identity(self):
            return {"Arn": "arn:aws:iam::111111111111:user/litellm-proxy-pod"}

        def assume_role(self, **params):
            if params.get("ExternalId") != "external-id-sm-completion":
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "is not authorized to perform: sts:AssumeRole"}},
                    "AssumeRole",
                )
            return {
                "Credentials": {
                    "AccessKeyId": "ASIASMCOMPROLEKEY",
                    "SecretAccessKey": "assumed-secret",
                    "SessionToken": "assumed-session-token",
                    "Expiration": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30),
                }
            }

    optional_params = {
        "aws_access_key_id": "AKIASMCOMPCALLERKEY",
        "aws_secret_access_key": "pod-caller-secret",
        "aws_region_name": "us-east-1",
        "aws_role_name": "arn:aws:iam::999999999999:role/litellm-sm-completion-role",
        "aws_session_name": "litellm-sm-completion-session",
        "aws_external_id": "external-id-sm-completion",
    }

    with patch.object(boto3, "client", return_value=FakeSTSClient()):
        credentials, aws_region_name = SagemakerLLM()._load_credentials(optional_params)

    assert credentials.access_key == "ASIASMCOMPROLEKEY"
    assert credentials.token == "assumed-session-token"
    assert aws_region_name == "us-east-1"
    assert "aws_external_id" not in optional_params


def test_load_credentials_assumes_role_with_session_tags(monkeypatch):
    """A trust policy gated on sts:TagSession only admits the session when the deployment's tags are sent."""
    import datetime

    import boto3
    from botocore.exceptions import ClientError
    from unittest.mock import patch

    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)
    monkeypatch.delenv("AWS_ROLE_ARN", raising=False)
    tags = [{"Key": "team", "Value": "genai"}]

    class FakeSTSClient:
        def get_caller_identity(self):
            return {"Arn": "arn:aws:iam::111111111111:user/litellm-proxy-pod"}

        def assume_role(self, **params):
            if list(params.get("Tags", ())) != tags:
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "is not authorized to perform: sts:TagSession"}},
                    "AssumeRole",
                )
            return {
                "Credentials": {
                    "AccessKeyId": "ASIASMCOMPTAGGED",
                    "SecretAccessKey": "assumed-secret",
                    "SessionToken": "assumed-session-token",
                    "Expiration": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30),
                }
            }

    optional_params = {
        "aws_access_key_id": "AKIASMCOMPCALLERKEY",
        "aws_secret_access_key": "pod-caller-secret",
        "aws_region_name": "us-east-1",
        "aws_role_name": "arn:aws:iam::999999999999:role/litellm-sm-completion-role",
        "aws_session_name": "litellm-sm-completion-session",
        "aws_session_tags": tags,
    }

    with patch.object(boto3, "client", return_value=FakeSTSClient()):
        credentials, aws_region_name = SagemakerLLM()._load_credentials(optional_params)

    assert credentials.access_key == "ASIASMCOMPTAGGED"
    assert aws_region_name == "us-east-1"
    assert "aws_session_tags" not in optional_params


def test_missing_botocore_keeps_dependency_identity():
    from unittest.mock import patch

    import pytest

    from litellm.llms.sagemaker.completion.handler import SagemakerLLM

    with patch.dict("sys.modules", {"botocore": None}):
        with pytest.raises(ModuleNotFoundError, match="pip install boto3") as caught:
            SagemakerLLM()._load_credentials({})
    assert caught.value.name == "botocore"


def test_installed_botocore_signs_the_completion_request():
    from botocore.credentials import Credentials

    from litellm.llms.sagemaker.completion.handler import SagemakerLLM

    request = SagemakerLLM()._prepare_request(
        credentials=Credentials("test-key", "test-secret"), model="test-endpoint", data={"inputs": "ping"},
        messages=[], litellm_params={}, optional_params={}, aws_region_name="us-west-2",
    )
    assert request.body == b'{"inputs": "ping"}'
    assert "/us-west-2/sagemaker/aws4_request" in request.headers["Authorization"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("httpx_transport")
@pytest.mark.usefixtures("fake_provider_credentials")
async def test_acompletion_sagemaker_non_stream(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWS_REGION_NAME", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    url: Final = (
        "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/"
        "jumpstart-dft-hf-textgeneration1-mp-20240815-185614/invocations"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(200, json={"generated_text": "SageMaker reply"})
    )
    response: Final = await litellm.acompletion(
        model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=80,
        cost_per_second=0.000420,
        num_retries=0,
    )

    assert route.called
    assert str(route.calls[0].request.url) == url
    assert json.loads(route.calls[0].request.content) == {
        "inputs": "hi",
        "parameters": {"temperature": 0.2, "max_new_tokens": 80},
    }
    assert response.choices[0].message.content == "SageMaker reply"


@pytest.mark.usefixtures("httpx_transport")
@pytest.mark.usefixtures("fake_provider_credentials")
def test_completion_sagemaker_non_stream(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWS_REGION_NAME", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    url: Final = (
        "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/"
        "jumpstart-dft-hf-textgeneration1-mp-20240815-185614/invocations"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(200, json={"generated_text": "SageMaker reply"})
    )
    response: Final = litellm.completion(
        model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=80,
        cost_per_second=0.000420,
        num_retries=0,
    )

    assert route.called
    assert str(route.calls[0].request.url) == url
    assert json.loads(route.calls[0].request.content) == {
        "inputs": "hi",
        "parameters": {"temperature": 0.2, "max_new_tokens": 80},
    }
    assert response.choices[0].message.content == "SageMaker reply"


@pytest.mark.usefixtures("httpx_transport")
@pytest.mark.usefixtures("fake_provider_credentials")
def test_completion_sagemaker_non_stream_with_aws_params(respx_mock: respx.MockRouter) -> None:
    url: Final = (
        "https://runtime.sagemaker.us-west-5.amazonaws.com/endpoints/"
        "jumpstart-dft-hf-textgeneration1-mp-20240815-185614/invocations"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(200, json={"generated_text": "SageMaker reply"})
    )
    response: Final = litellm.completion(
        model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=80,
        cost_per_second=0.000420,
        aws_access_key_id="unit-test-access-key",
        aws_secret_access_key="unit-test-secret",
        aws_region_name="us-west-5",
        num_retries=0,
    )

    assert route.called
    assert str(route.calls[0].request.url) == url
    assert json.loads(route.calls[0].request.content) == {
        "inputs": "hi",
        "parameters": {"temperature": 0.2, "max_new_tokens": 80},
    }
    assert response.choices[0].message.content == "SageMaker reply"


@pytest.mark.usefixtures("httpx_transport")
@pytest.mark.usefixtures("fake_provider_credentials")
def test_completion_sagemaker_prompt_template_non_stream(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "known_tokenizer_config", {})
    tokenizer_route: Final = respx_mock.get(
        "https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/raw/main/tokenizer_config.json"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "bos_token": {"content": "<｜begin▁of▁sentence｜>"},
                "eos_token": {"content": "<|EOT|>"},
                "chat_template": (
                    "{{ bos_token }}You are a coding assistant\n"
                    "{% for message in messages %}### Instruction:\n{{ message['content'] }}\n{% endfor %}"
                    "### Response:"
                ),
            },
        )
    )
    url: Final = (
        "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/deepseek_coder_6.7_instruct/invocations"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(200, json={"generated_text": "SageMaker reply"})
    )
    response: Final = litellm.completion(
        model="sagemaker/deepseek_coder_6.7_instruct",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=80,
        hf_model_name="deepseek-ai/deepseek-coder-6.7b-instruct",
        aws_region_name="us-east-1",
        num_retries=0,
    )

    assert tokenizer_route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {
        "inputs": "<｜begin▁of▁sentence｜>You are a coding assistant\n### Instruction:\nhi\n### Response:",
        "parameters": {"temperature": 0.2, "max_new_tokens": 80},
    }
    assert response.choices[0].message.content == "SageMaker reply"


@pytest.mark.usefixtures("httpx_transport")
@pytest.mark.usefixtures("fake_provider_credentials")
@pytest.mark.parametrize(
    ("environment", "call_kwargs", "expected_region"),
    [
        ({}, {}, "us-west-2"),
        ({"AWS_REGION_NAME": "us-east-1"}, {}, "us-east-1"),
        ({"AWS_REGION_NAME": "eu-west-1"}, {"aws_region_name": "us-east-1"}, "us-east-1"),
    ],
    ids=["default_region", "environment_region", "config_region_beats_environment"],
)
def test_sagemaker_region_resolution(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    call_kwargs: dict[str, str],
    expected_region: str,
) -> None:
    monkeypatch.delenv("AWS_REGION_NAME", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    url: Final = f"https://runtime.sagemaker.{expected_region}.amazonaws.com/endpoints/mock-endpoint/invocations"
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(200, json={"generated_text": "SageMaker reply"})
    )
    litellm.completion(
        model="sagemaker/mock-endpoint",
        messages=[{"role": "user", "content": "Hello, world!"}],
        num_retries=0,
        **call_kwargs,
    )

    assert route.call_count == 1
    assert str(route.calls[0].request.url) == url


@pytest.mark.asyncio
@pytest.mark.usefixtures("httpx_transport")
@pytest.mark.usefixtures("fake_provider_credentials")
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_completion_sagemaker(
    respx_mock: respx.MockRouter,
    sync_mode: bool,
) -> None:
    url: Final = (
        "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/"
        "jumpstart-dft-hf-textgeneration1-mp-20240815-185614/invocations"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(200, json={"generated_text": "SageMaker reply"})
    )
    rate: Final = 0.000420
    response: Final = (
        litellm.completion(
            model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=80,
            cost_per_second=rate,
            aws_region_name="us-east-1",
            num_retries=0,
        )
        if sync_mode
        else await litellm.acompletion(
            model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=80,
            cost_per_second=rate,
            aws_region_name="us-east-1",
            num_retries=0,
        )
    )
    response._response_ms = 1000.0

    assert route.called
    assert response.choices[0].message.content == "SageMaker reply"
    assert litellm.completion_cost(completion_response=response) == pytest.approx(rate)


@pytest.mark.asyncio
@pytest.mark.usefixtures("httpx_transport")
@pytest.mark.usefixtures("fake_provider_credentials")
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_completion_sagemaker_stream(
    respx_mock: respx.MockRouter,
    sync_mode: bool,
) -> None:
    url: Final = (
        "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/"
        "jumpstart-dft-hf-textgeneration1-mp-20240815-185614/invocations-response-stream"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(200, content=_token_frame("hello") + _token_frame(" world"))
    )
    stream: Final = (
        litellm.completion(
            model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            messages=[{"role": "user", "content": "hi - what is ur name"}],
            stream=True,
            temperature=0.2,
            max_tokens=80,
            cost_per_second=0.000420,
            aws_region_name="us-east-1",
            num_retries=0,
        )
        if sync_mode
        else await litellm.acompletion(
            model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            messages=[{"role": "user", "content": "hi - what is ur name"}],
            stream=True,
            temperature=0.2,
            max_tokens=80,
            cost_per_second=0.000420,
            aws_region_name="us-east-1",
            num_retries=0,
        )
    )
    chunks: Final = tuple(stream) if sync_mode else tuple([chunk async for chunk in stream])

    assert json.loads(route.calls[0].request.content) == {
        "inputs": "hi - what is ur name",
        "parameters": {"temperature": 0.2, "max_new_tokens": 80},
        "stream": True,
    }
    assert chunks[0].choices[0].delta.role == "assistant"
    assert tuple(chunk.choices[0].delta.content for chunk in chunks) == ("hello", " world", None)
    assert tuple(chunk.choices[0].finish_reason for chunk in chunks) == (None, None, "stop")


@pytest.mark.usefixtures("httpx_transport")
@pytest.mark.usefixtures("fake_provider_credentials")
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_completion_sagemaker_streaming_bad_request(
    respx_mock: respx.MockRouter,
    sync_mode: bool,
) -> None:
    url: Final = (
        "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/"
        "jumpstart-dft-hf-textgeneration1-mp-20240815-185614/invocations-response-stream"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(
            400,
            json={"Message": "The request payload failed validation.", "__type": "ValidationException"},
        )
    )

    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.completion(
            model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            aws_region_name="us-east-1",
            num_retries=0,
        ) if sync_mode else await litellm.acompletion(
            model="sagemaker/jumpstart-dft-hf-textgeneration1-mp-20240815-185614",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            aws_region_name="us-east-1",
            num_retries=0,
        )

    assert route.called
    assert exc_info.value.status_code == 400
    assert exc_info.value.llm_provider == "sagemaker"
