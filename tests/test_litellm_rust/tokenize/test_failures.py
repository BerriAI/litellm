from dataclasses import dataclass
from typing import Final

import pytest

import litellm
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.routing import assert_served_natively, call_public

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_invalid_prompt_is_a_public_bad_request_before_provider_io(
    server: RecordingServer, asynchronous: bool
) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError) as caught:
        await call_public(
            litellm.tokenize,
            litellm.atokenize,
            {
                "model": "acme/model",
                "custom_llm_provider": "hosted_vllm",
                "prompt": None,
            },
            server,
            asynchronous=asynchronous,
            num_retries=0,
        )
    assert caught.value.status_code == 400
    assert "prompt" in str(caught.value).lower()
    assert caught.value.model == "acme/model"
    assert caught.value.llm_provider == "tokenize"
    assert_served_natively(server, count=0)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_bare_vllm_is_rejected_before_provider_io(server: RecordingServer, asynchronous: bool) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError) as caught:
        await call_public(
            litellm.tokenize,
            litellm.atokenize,
            {"model": "vllm/acme/model", "prompt": "hello"},
            server,
            asynchronous=asynchronous,
            num_retries=0,
        )
    assert caught.value.status_code == 400
    assert_served_natively(server, count=0)


async def test_azure_tokenization_is_rejected_before_provider_io(server: RecordingServer) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError):
        await litellm.atokenize(
            model="azure/gpt-4o-mini",
            prompt="hello",
            api_base=server.base_url,
            api_key="test-key",
            num_retries=0,
        )
    assert_served_natively(server, count=0)


async def test_watsonx_token_ids_are_a_bad_request_before_provider_io(server: RecordingServer) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError) as caught:
        await litellm.atokenize(
            model="watsonx/ibm/granite-3-8b-instruct",
            prompt="Hello",
            api_base=server.base_url,
            api_key="test-key",
            include=["token_ids"],
            num_retries=0,
        )
    assert caught.value.status_code == 400
    assert "include" in str(caught.value)
    assert_served_natively(server, count=0)


class ProjectionAbort(BaseException):
    pass


class UnprintableProjectionError(ValueError):
    def __str__(self) -> str:
        raise RuntimeError("exception formatting failed")


@dataclass(frozen=True)
class FailingTimeout:
    error: BaseException

    def __float__(self) -> float:
        raise self.error


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "error_type", [ProjectionAbort, UnprintableProjectionError], ids=["base-exception", "mapper-failure"]
)
async def test_projection_preserves_original_exception_when_public_mapping_cannot_apply(
    server: RecordingServer, asynchronous: bool, error_type: type[BaseException]
) -> None:
    server.expected_requests = 0
    original: Final = error_type("timeout conversion failed")
    with pytest.raises(error_type) as caught:
        await call_public(
            litellm.tokenize,
            litellm.atokenize,
            {"model": "hosted_vllm/acme/model", "prompt": "hello"},
            server,
            asynchronous=asynchronous,
            timeout=FailingTimeout(original),
            num_retries=0,
        )
    assert caught.value is original
    assert_served_natively(server, count=0)


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(401, litellm.AuthenticationError), (503, litellm.ServiceUnavailableError)],
    ids=["invalid-credentials", "iam-unavailable"],
)
async def test_iam_failure_keeps_public_status_without_sending_tokenization(
    server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    error_type: type[litellm.AuthenticationError | litellm.ServiceUnavailableError],
) -> None:
    monkeypatch.delenv("WATSONX_TOKEN", raising=False)
    monkeypatch.delenv("WATSONX_ZENAPIKEY", raising=False)
    monkeypatch.setenv("WATSONX_IAM_URL", f"{server.base_url}/identity/token")
    server.default_response = ResponseSpec(status=status, body={"errorMessage": "IAM rejected test credentials"})
    with pytest.raises(error_type) as caught:
        await call_public(
            litellm.tokenize,
            litellm.atokenize,
            {
                "model": "watsonx/ibm/model",
                "custom_llm_provider": "watsonx",
                "prompt": "hello",
                "project_id": "test-project",
            },
            server,
            asynchronous=True,
            num_retries=0,
        )
    assert caught.value.status_code == status
    assert_served_natively(server)
    assert [request.path for request in server.requests] == ["/identity/token"]
