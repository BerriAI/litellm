import json

from pydantic import BaseModel

from litellm.router_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    add_retry_headers_to_response,
    get_fallback_errors_from_headers,
    get_hidden_params_dict,
)


class StreamingWrapper:
    def __init__(self):
        self._hidden_params = {"additional_headers": {"x-existing": "keep"}}


def test_add_fallback_headers_to_streaming_wrapper():
    response = StreamingWrapper()

    result = add_fallback_headers_to_response(
        response=response,
        attempted_fallbacks=1,
    )

    assert result is response
    assert response._hidden_params["additional_headers"] == {
        "x-existing": "keep",
        "x-litellm-attempted-fallbacks": 1,
    }


def test_add_fallback_headers_serializes_fallback_errors():
    response = StreamingWrapper()
    fallback_errors = [
        {
            "message": "litellm.RateLimitError: upstream limited request",
            "type": "RateLimitError",
            "param": None,
            "code": "429",
        }
    ]

    result = add_fallback_headers_to_response(
        response=response,
        attempted_fallbacks=1,
        fallback_errors=fallback_errors,
    )

    assert result is response
    assert response._hidden_params["additional_headers"][
        "x-litellm-attempted-fallbacks"
    ] == 1
    assert (
        json.loads(
            response._hidden_params["additional_headers"]["x-litellm-fallback-errors"]
        )
        == fallback_errors
    )


def test_add_retry_headers_to_streaming_wrapper():
    response = StreamingWrapper()

    result = add_retry_headers_to_response(
        response=response,
        attempted_retries=2,
        max_retries=3,
    )

    assert result is response
    assert response._hidden_params["additional_headers"] == {
        "x-existing": "keep",
        "x-litellm-attempted-retries": 2,
        "x-litellm-max-retries": 3,
    }


def test_get_hidden_params_dict_with_pydantic_model_hidden_params():
    class InnerHiddenParams(BaseModel):
        additional_headers: dict = {}

    class Response:
        def __init__(self):
            self._hidden_params = InnerHiddenParams(
                additional_headers={"x-custom": "value"}
            )

    result = get_hidden_params_dict(Response())
    assert result == {"additional_headers": {"x-custom": "value"}}


def test_get_hidden_params_dict_with_no_hidden_params():
    class PlainResponse:
        pass

    assert get_hidden_params_dict(PlainResponse()) == {}


def test_add_fallback_headers_when_no_existing_additional_headers():
    class NoHeadersWrapper:
        def __init__(self):
            self._hidden_params = {}

    response = NoHeadersWrapper()
    result = add_fallback_headers_to_response(response=response, attempted_fallbacks=2)

    assert result is response
    assert response._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"] == 2


def test_add_fallback_headers_returns_none_when_response_is_none():
    result = add_fallback_headers_to_response(response=None, attempted_fallbacks=1)
    assert result is None


def test_add_fallback_headers_returns_unchanged_when_response_has_no_hidden_params():
    class PlainObject:
        pass

    obj = PlainObject()
    result = add_fallback_headers_to_response(response=obj, attempted_fallbacks=1)
    assert result is obj
    assert not hasattr(obj, "_hidden_params")


def test_get_fallback_errors_from_headers_existing_list_passthrough():
    errors = [{"message": "err", "type": "T", "param": None, "code": "400"}]
    result = get_fallback_errors_from_headers({"x-litellm-fallback-errors": errors})
    assert result == errors


def test_get_fallback_errors_from_headers_invalid_json_returns_empty():
    result = get_fallback_errors_from_headers(
        {"x-litellm-fallback-errors": "not-valid-json-{"}
    )
    assert result == []


def test_get_fallback_errors_from_headers_missing_key_returns_empty():
    result = get_fallback_errors_from_headers({})
    assert result == []
