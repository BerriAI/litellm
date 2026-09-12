import json
from types import MappingProxyType
from typing import Final

import pytest
from e2e_http import StreamingResponse, UnknownApiError, require_successful_call, unwrap, unwrap_status
from provider_diagnostics import ProviderUnavailableError, provider_failure

_BODY: Final = json.dumps(
    {
        "error": {
            "message": (
                'litellm.ServiceUnavailableError: BedrockException - {"message":"Bedrock is unable to process your request."}'
            )
        }
    }
)
_HEADERS: Final = MappingProxyType(
    {
        "LLM_Provider-X-Amzn-RequestId": "aws-request-1",
        "LLM_Provider-X-Amzn-ErrorType": "ServiceUnavailableException:http://internal.amazon.com/",
        "X-Litellm-Call-Id": "proxy-call-1",
    }
)


def _assert_success(helper: str) -> None:
    match helper:
        case "unwrap":
            unwrap(UnknownApiError(status_code=503, body=_BODY, headers=dict(_HEADERS)))
        case "unwrap_status":
            unwrap_status(UnknownApiError(status_code=503, body=_BODY, headers=dict(_HEADERS)), 200)
        case _:
            require_successful_call(StreamingResponse(status_code=503, body=_BODY, headers=dict(_HEADERS)))


class TestProviderFailure:
    def test_aws_headers_preserve_attribution_and_both_request_ids(self) -> None:
        result: Final = provider_failure(503, "unavailable", _HEADERS, expected_provider="bedrock")
        assert result.provider == "bedrock"
        assert result.error_code == "ServiceUnavailableException"
        assert result.request_id == "aws-request-1"
        assert result.call_id == "proxy-call-1"
        assert result.evidence == "headers"

    def test_another_aws_service_is_not_mislabeled_as_bedrock(self) -> None:
        result: Final = provider_failure(503, "service unavailable", _HEADERS)
        assert result.provider is None
        assert result.request_id == "aws-request-1"

    def test_proxy_bedrock_error_body_remains_identifiable_without_headers(self) -> None:
        result: Final = provider_failure(503, _BODY, {})
        assert result.provider == "bedrock" and result.evidence == "body"
        assert result.request_id is None

    def test_native_error_requires_explicit_bedrock_route_context(self) -> None:
        body: Final = json.dumps(
            {"detail": {"error": json.dumps({"message": "Bedrock is unable to process your request."})}}
        )
        assert provider_failure(503, body, {}).provider is None
        result: Final = StreamingResponse(status_code=503, body=body)
        with pytest.raises(ProviderUnavailableError) as caught:
            require_successful_call(result, expected_provider="bedrock")
        assert caught.value.failure.evidence == "body"

    @pytest.mark.parametrize(
        "status,body",
        [
            (503, '{"error":{"message":"proxy overloaded"}}'),
            (503, "Bedrock is unable to process your request."),
            (503, '{"error":{"message":"Bedrock is unable to process your request."}}'),
            (503, '{"detail":{"error":"not JSON"}}'),
            (429, _BODY),
            (403, _BODY),
            (500, _BODY),
        ],
    )
    def test_ambiguous_and_non_availability_errors_are_not_retryable(self, status: int, body: str) -> None:
        assert provider_failure(status, body, {}, expected_provider="bedrock").provider is None
        with pytest.raises(AssertionError) as caught:
            unwrap(UnknownApiError(status_code=status, body=body))
        assert not isinstance(caught.value, ProviderUnavailableError)

    @pytest.mark.parametrize("helper", ["unwrap", "unwrap_status", "raw"])
    def test_all_assertion_paths_emit_the_same_provider_failure(self, helper: str) -> None:
        with pytest.raises(ProviderUnavailableError) as caught:
            _assert_success(helper)
        assert caught.value.failure == provider_failure(503, _BODY, _HEADERS)
