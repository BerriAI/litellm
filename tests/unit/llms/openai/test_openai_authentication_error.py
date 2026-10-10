from typing import Final

import httpx
import pytest
import respx

import litellm


def test_openai_401_maps_to_authentication_error(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {
                    "message": "Incorrect API key provided",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )
    )

    with pytest.raises(litellm.AuthenticationError) as error:
        litellm.completion(
            model="openai/migration-auth-error-model",
            messages=[{"role": "user", "content": "hello"}],
            api_key="invalid-test-key",
            max_retries=0,
        )

    assert error.value.status_code == 401
    assert "Incorrect API key provided" in str(error.value)
    assert route.calls.last.request.headers["Authorization"] == "Bearer invalid-test-key"


def test_openai_invalid_role_error_preserves_provider_fields(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "message": "Invalid role",
                    "type": "invalid_request_error",
                    "param": "messages[0].role",
                    "code": "invalid_value",
                }
            },
        )
    )

    with pytest.raises(litellm.BadRequestError) as error:
        litellm.completion(
            model="openai/migration-invalid-role-model",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            max_retries=0,
        )

    assert error.value.code == "invalid_value"
    assert error.value.param == "messages[0].role"
    assert error.value.type == "invalid_request_error"
    assert isinstance(error.value.response, httpx.Response)
