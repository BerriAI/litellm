from collections.abc import Mapping
from typing import Final, Literal

from pydantic import BaseModel, ValidationError


class ProviderFailure(BaseModel):
    status_code: int
    provider: Literal["bedrock"] | None = None
    error_code: str | None = None
    request_id: str | None = None
    call_id: str | None = None
    evidence: Literal["headers", "body", "stream_event"] | None = None

    def junit_properties(self) -> tuple[tuple[str, str], ...]:
        fields: Final = (
            ("status_code", self.status_code),
            ("provider", self.provider),
            ("error_code", self.error_code),
            ("request_id", self.request_id),
            ("call_id", self.call_id),
            ("evidence", self.evidence),
        )
        return tuple((f"upstream_{key}", str(value)) for key, value in fields if value is not None)


class ProviderUnavailableError(AssertionError):
    def __init__(self, failure: ProviderFailure, body: str) -> None:
        self.failure: Final = failure
        super().__init__(f"{failure.model_dump_json(exclude_none=True)}; body={body[:1000]}")


class NetworkFailureError(AssertionError):
    def __init__(self, failure: ProviderFailure, message: str) -> None:
        self.failure: Final = failure
        super().__init__(f"HTTP transfer failed; {failure.model_dump_json(exclude_none=True)}; body={message[:1000]}")


class _ErrorMessage(BaseModel):
    message: str = ""


class _NativeError(BaseModel):
    error: str = ""


class _ErrorEnvelope(BaseModel):
    error: _ErrorMessage | None = None
    detail: _NativeError | None = None


_BEDROCK_UNAVAILABLE: Final = "Bedrock is unable to process your request."


def _bedrock_unavailable_body(body: str, expected_provider: Literal["bedrock"] | None) -> bool:
    try:
        envelope: Final = _ErrorEnvelope.model_validate_json(body)
        if envelope.error is not None:
            message: Final = envelope.error.message
            return message.startswith("litellm.ServiceUnavailableError: BedrockException - ") and (
                _BEDROCK_UNAVAILABLE in message
            )
        if expected_provider == "bedrock" and envelope.detail is not None:
            return _ErrorMessage.model_validate_json(envelope.detail.error).message == _BEDROCK_UNAVAILABLE
    except ValidationError:
        return False
    return False


def provider_failure(
    status_code: int,
    body: str,
    headers: Mapping[str, str],
    *,
    expected_provider: Literal["bedrock"] | None = None,
    stream_error_code: str | None = None,
) -> ProviderFailure:
    normalized: Final = {key.lower(): value for key, value in headers.items()}
    request_id: Final = normalized.get("llm_provider-x-amzn-requestid") or normalized.get("x-amzn-requestid")
    error_code: Final = normalized.get("llm_provider-x-amzn-errortype") or normalized.get("x-amzn-errortype")
    call_id: Final = normalized.get("x-litellm-call-id")
    if (
        200 <= status_code < 300
        and expected_provider == "bedrock"
        and stream_error_code == "serviceUnavailableException"
    ):
        return ProviderFailure(
            status_code=status_code,
            provider="bedrock",
            error_code="ServiceUnavailableException",
            request_id=request_id,
            call_id=call_id,
            evidence="stream_event",
        )
    if status_code == 503:
        bedrock_body: Final = _bedrock_unavailable_body(body, expected_provider)
        if (
            request_id
            and error_code
            and error_code.split(":", 1)[0] == "ServiceUnavailableException"
            and (expected_provider == "bedrock" or bedrock_body)
        ):
            return ProviderFailure(
                status_code=status_code,
                provider="bedrock",
                error_code="ServiceUnavailableException",
                request_id=request_id,
                call_id=call_id,
                evidence="headers",
            )
        if bedrock_body:
            return ProviderFailure(
                status_code=status_code,
                provider="bedrock",
                error_code="ServiceUnavailableException",
                request_id=request_id,
                call_id=call_id,
                evidence="body",
            )
    return ProviderFailure(status_code=status_code, request_id=request_id, call_id=call_id, error_code=error_code)


def raise_if_provider_unavailable(
    status_code: int,
    body: str,
    headers: Mapping[str, str],
    *,
    expected_provider: Literal["bedrock"] | None = None,
    stream_error_code: str | None = None,
) -> None:
    failure: Final = provider_failure(
        status_code, body, headers, expected_provider=expected_provider, stream_error_code=stream_error_code
    )
    if failure.provider is not None:
        raise ProviderUnavailableError(failure, body)
