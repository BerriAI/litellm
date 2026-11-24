import json
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.proxy.proxy_server import _should_include_fallback_errors
from litellm.router import Router


def _two_group_fallback_router() -> Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "primary-model",
                "litellm_params": {"model": "openai/gpt-fake", "api_key": "sk-fake"},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {"model": "openai/gpt-fake-2", "api_key": "sk-fake"},
            },
        ],
        fallbacks=[{"primary-model": ["fallback-model"]}],
    )


def _additional_headers(response: object) -> dict:
    return get_hidden_params_dict(response).get("additional_headers", {})


@pytest.mark.asyncio
async def test_fastest_response_marks_the_winning_response_hidden_params() -> None:
    router: Final = _two_group_fallback_router()

    response: Final = await router.abatch_completion_fastest_response(
        model="primary-model, fallback-model",
        messages=[{"role": "user", "content": "Hello"}],
        mock_testing_fallbacks=True,
        mock_response="fastest response",
    )

    assert response.hidden_params["fastest_response_batch_completion"] is True


@pytest.mark.asyncio
async def test_include_fallback_errors_propagates_through_router():
    router = _two_group_fallback_router()

    response = await router.acompletion(
        model="primary-model",
        messages=[{"role": "user", "content": "Hello"}],
        mock_testing_fallbacks=True,
        mock_response="fallback success",
        include_fallback_errors=True,
    )

    headers = _additional_headers(response)
    assert headers["x-litellm-attempted-fallbacks"] == 1
    errors = json.loads(headers["x-litellm-fallback-errors"])
    assert isinstance(errors, list) and len(errors) >= 1
    assert set(errors[0].keys()) == {"message", "type", "param", "code"}


@pytest.mark.asyncio
async def test_router_omits_fallback_errors_without_opt_in():
    router = _two_group_fallback_router()

    response = await router.acompletion(
        model="primary-model",
        messages=[{"role": "user", "content": "Hello"}],
        mock_testing_fallbacks=True,
        mock_response="fallback success",
    )

    headers = _additional_headers(response)
    assert headers["x-litellm-attempted-fallbacks"] == 1
    assert "x-litellm-fallback-errors" not in headers


@pytest.mark.asyncio
async def test_set_response_headers_adds_model_group_to_streaming_wrapper():
    class StreamingWrapper:
        def __init__(self):
            self._hidden_params = {"additional_headers": {"x-existing": "keep"}}

    router = litellm.Router(model_list=[])
    response = StreamingWrapper()

    result = await router.set_response_headers(
        response=response,
        model_group="fallback-model",
    )

    assert result is response
    assert response._hidden_params["additional_headers"] == {
        "x-existing": "keep",
        "x-litellm-model-group": "fallback-model",
    }


def test_should_include_fallback_errors_gated_by_operator_setting():
    request_data: dict = {"include_fallback_errors": True}

    import litellm.proxy.proxy_server as ps

    original = ps.general_settings.copy() if isinstance(ps.general_settings, dict) else {}
    try:
        ps.general_settings = {}
        assert _should_include_fallback_errors(request_data) is False

        ps.general_settings = {"expose_fallback_errors_to_caller": False}
        assert _should_include_fallback_errors(request_data) is False

        ps.general_settings = {"expose_fallback_errors_to_caller": True}
        assert _should_include_fallback_errors(request_data) is True

        ps.general_settings = {"expose_fallback_errors_to_caller": True}
        assert _should_include_fallback_errors({}) is False
    finally:
        ps.general_settings = original
