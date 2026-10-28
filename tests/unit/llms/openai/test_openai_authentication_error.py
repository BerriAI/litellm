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
    assert str(error.value).startswith("litellm.AuthenticationError: AuthenticationError: OpenAIException - ")
    assert "Incorrect API key provided" in str(error.value)
    assert route.calls.last.request.headers["Authorization"] == "Bearer invalid-test-key"


async def _complete(sync_mode: bool, request: dict[str, object]) -> object:
    if sync_mode:
        return litellm.completion(**request)
    return await litellm.acompletion(**request)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_openai_invalid_role_error_preserves_provider_fields(
    sync_mode: bool, stream: bool, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "message": "Invalid value: 'usera'. Supported values are: 'system', 'assistant', 'user'.",
                    "type": "invalid_request_error",
                    "param": "messages[0].role",
                    "code": "invalid_value",
                }
            },
        )
    )
    request: Final = {
        "model": "openai/migration-invalid-role-model",
        "messages": [{"role": "usera", "content": "hi"}],
        "api_key": "test-key",
        "max_retries": 0,
        "stream": stream,
    }

    with pytest.raises(litellm.BadRequestError) as error:
        await _complete(sync_mode, request)

    assert str(error.value).startswith("litellm.BadRequestError: OpenAIException - Invalid value")
    assert error.value.code == "invalid_value"
    assert error.value.param == "messages[0].role"
    assert error.value.type == "invalid_request_error"
    assert error.value.response.status_code == 400
