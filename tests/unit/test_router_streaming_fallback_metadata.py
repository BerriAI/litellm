import json
from typing import Final
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.proxy.proxy_server import _should_include_fallback_errors
from litellm.router import Router
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.types.llms.base import HiddenParams


def test_apply_fallback_hidden_params_copies_from_fallback_response():
    fallback_errors = [
        {
            "message": "litellm.RateLimitError: upstream limited request",
            "type": "RateLimitError",
            "param": None,
            "code": "429",
        }
    ]
    chunk = litellm.ModelResponseStream(
        id="test",
        model="openai/internal-fallback",
        choices=[],
    )
    chunk._hidden_params = {
        "additional_headers": {"x-existing-chunk-header": "keep"},
        "model_id": "chunk-model-id",
    }
    fallback_response = MagicMock()
    fallback_response._hidden_params = {
        "additional_headers": {
            "x-litellm-attempted-fallbacks": 1,
            "x-litellm-model-group": "fallback-model",
            "x-litellm-fallback-errors": json.dumps(fallback_errors),
        },
        "api_base": "https://fallback.example",
    }

    Router._apply_fallback_hidden_params_to_item(
        fallback_item=chunk,
        prepared_fallback_hidden_params=Router._prepare_fallback_hidden_params(
            fallback_response
        ),
    )

    assert chunk._hidden_params["api_base"] == "https://fallback.example"
    assert chunk._hidden_params["model_id"] == "chunk-model-id"
    assert chunk._hidden_params["additional_headers"] == {
        "x-existing-chunk-header": "keep",
        "x-litellm-attempted-fallbacks": 1,
        "x-litellm-model-group": "fallback-model",
        "x-litellm-fallback-errors": json.dumps(fallback_errors),
    }


def test_apply_fallback_hidden_params_updates_plain_duck_chunk():
    class PlainChunk:
        def __init__(self) -> None:
            self._hidden_params = {
                "additional_headers": {"x-existing-chunk-header": "keep"},
                "model_id": "chunk-model-id",
            }

    chunk: Final = PlainChunk()
    fallback_response = {
        "_hidden_params": {
            "additional_headers": {
                "x-litellm-attempted-fallbacks": 1,
            },
            "api_base": "https://fallback.example",
        }
    }

    Router._apply_fallback_hidden_params_to_item(
        fallback_item=chunk,
        prepared_fallback_hidden_params=Router._prepare_fallback_hidden_params(fallback_response),
    )

    assert chunk._hidden_params["api_base"] == "https://fallback.example"
    assert chunk._hidden_params["model_id"] == "chunk-model-id"
    assert chunk._hidden_params["additional_headers"] == {
        "x-existing-chunk-header": "keep",
        "x-litellm-attempted-fallbacks": 1,
    }


def test_apply_fallback_hidden_params_normalizes_hidden_params_on_plain_duck_chunk():
    class PlainChunk:
        def __init__(self) -> None:
            self._hidden_params = HiddenParams(
                api_base="https://original.example",
                model_id="original-model-id",
                additional_headers={"x-existing-chunk-header": "keep"},
            )

    chunk: Final = PlainChunk()
    Router._apply_fallback_hidden_params_to_item(
        fallback_item=chunk,
        prepared_fallback_hidden_params=(
            {"api_base": "https://fallback.example", "model_id": "fallback-model-id"},
            {"x-litellm-attempted-fallbacks": 1},
        ),
    )

    assert isinstance(chunk._hidden_params, dict)
    assert chunk._hidden_params["api_base"] == "https://fallback.example"
    assert chunk._hidden_params["model_id"] == "fallback-model-id"
    assert chunk._hidden_params["additional_headers"] == {
        "x-existing-chunk-header": "keep",
        "x-litellm-attempted-fallbacks": 1,
    }


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


def test_prepare_fallback_hidden_params_no_additional_headers():
    class FakeResponse:
        _hidden_params = {"api_base": "http://example.com"}

    hidden_params, headers = Router._prepare_fallback_hidden_params(FakeResponse())
    assert hidden_params == {"api_base": "http://example.com"}
    assert headers == {}


def test_apply_fallback_hidden_params_to_item_none_item():
    Router._apply_fallback_hidden_params_to_item(
        None, ({"api_base": "http://fallback.example"}, {"x-custom": "value"})
    )


def test_apply_fallback_hidden_params_to_item_no_existing_additional_headers():
    chunk: Final = litellm.ModelResponseStream(
        id="test",
        model="openai/internal-fallback",
        choices=[],
    )
    chunk.hidden_params["model_id"] = "test-id"
    Router._apply_fallback_hidden_params_to_item(
        chunk,
        (
            {"api_base": "http://fallback.example"},
            {"x-litellm-attempted-fallbacks": 1},
        ),
    )

    assert chunk.hidden_params["api_base"] == "http://fallback.example"
    assert chunk.hidden_params["model_id"] == "test-id"
    assert chunk.hidden_params["additional_headers"] == {
        "x-litellm-attempted-fallbacks": 1
    }


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
