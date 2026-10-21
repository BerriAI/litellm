from typing import Final, Literal, TypeAlias

import httpx
import openai
import pytest
import respx

import litellm
from tests.unit.proxy.conftest import httpx_transport

pytestmark: Final = pytest.mark.usefixtures(httpx_transport.__name__)
Provider: TypeAlias = Literal["anthropic", "gemini"]
Operation: TypeAlias = Literal["delete", "get", "cancel"]
CancelProvider: TypeAlias = Literal["openai", "azure"]


def _invoke_unsupported_response_operation(provider: Provider, operation: Operation) -> object:
    match operation:
        case "delete":
            return litellm.delete_responses(
                response_id="resp_unsupported",
                custom_llm_provider=provider,
                api_key="sk-test",
            )
        case "get":
            return litellm.get_responses(
                response_id="resp_unsupported",
                custom_llm_provider=provider,
                api_key="sk-test",
            )
        case "cancel":
            return litellm.cancel_responses(
                response_id="resp_unsupported",
                custom_llm_provider=provider,
                api_key="sk-test",
            )


async def _invoke_cancel_response(sync_mode: bool, call_kwargs: dict[str, object]) -> object:
    if sync_mode:
        return litellm.cancel_responses(**call_kwargs)
    return await litellm.acancel_responses(**call_kwargs)


@pytest.mark.parametrize(
    ("provider", "operation"),
    (
        ("anthropic", "delete"),
        ("anthropic", "get"),
        ("anthropic", "cancel"),
        ("gemini", "delete"),
        ("gemini", "get"),
        ("gemini", "cancel"),
    ),
)
def test_unsupported_response_lifecycle_operation_fails_before_http(provider: Provider, operation: Operation) -> None:
    with respx.mock() as mock_router:
        with pytest.raises(litellm.APIConnectionError) as exc_info:
            _invoke_unsupported_response_operation(provider, operation)
        calls: Final = tuple(mock_router.calls)

    assert exc_info.value.status_code == 500
    assert f"not supported for {provider}" in str(exc_info.value)
    assert calls == ()


@pytest.mark.parametrize(
    ("provider", "sync_mode"),
    (("openai", True), ("openai", False), ("azure", True), ("azure", False)),
)
@pytest.mark.asyncio
async def test_cancel_responses_404_surfaces_openai_api_error_with_exact_url(
    provider: CancelProvider, sync_mode: bool
) -> None:
    response_id: Final = "resp_missing"
    error_body: Final = {
        "error": {
            "message": "Response was not found",
            "type": "invalid_request_error",
            "code": "not_found",
        }
    }
    api_base: Final = (
        "https://api.openai.com/v1" if provider == "openai" else "https://example-resource.openai.azure.com"
    )
    expected_url: Final = (
        f"{api_base}/responses/{response_id}/cancel"
        if provider == "openai"
        else f"{api_base}/openai/responses/{response_id}/cancel?api-version=2025-03-01-preview"
    )

    with respx.mock() as mock_router:
        route: Final = mock_router.post(expected_url).mock(
            return_value=httpx.Response(status_code=404, json=error_body)
        )
        call_kwargs: Final = {
            "custom_llm_provider": provider,
            "api_key": "sk-test",
            "api_base": api_base,
            "api_version": "2025-03-01-preview",
            "response_id": response_id,
        }
        with pytest.raises(openai.APIError) as exc_info:
            await _invoke_cancel_response(sync_mode, call_kwargs)
        requests: Final = tuple(route.calls)

    assert exc_info.value.status_code == 404
    assert len(requests) == 1
    assert str(requests[0].request.url) == expected_url
