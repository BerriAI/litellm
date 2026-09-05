from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from litellm.rust_bridge.request import (
    NativeRequestContext,
    PreparedNativeCall,
    call_native,
    provider_connection_params,
    provider_request_params,
)


@dataclass(frozen=True, slots=True)
class TokenProvider:
    def __call__(self) -> str:
        raise AssertionError("collecting a request must not invoke its token provider")


def test_credentials_and_callables_cannot_enter_the_provider_payload() -> None:
    provider: Final = TokenProvider()
    params: Final = {
        "max_tokens": 23,
        "azure_client_secret": "azure-secret",
        "vertex_credentials": {"private_key": "google-secret"},
        "azure_ad_token_provider": provider,
    }

    assert provider_request_params(params) == {"max_tokens": 23}
    assert provider_connection_params(params) == {
        "azure_client_secret": "azure-secret",
        "vertex_credentials": {"private_key": "google-secret"},
    }


def test_native_handoff_preserves_the_callable_without_invoking_it() -> None:
    provider: Final = TokenProvider()
    context: Final = NativeRequestContext(litellm_call_id="call-auth")

    def native(
        request: str,
        *,
        context: NativeRequestContext,
        callback_adapter: object | None = None,
        auth_provider: object | None = None,
    ) -> object:
        assert request == "request"
        assert context.litellm_call_id == "call-auth"
        assert callback_adapter is None
        return auth_provider

    result: Final = call_native(native, PreparedNativeCall(request="request", context=context, auth_provider=provider))
    assert result is provider
