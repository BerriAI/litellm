from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter

from litellm import ModelResponse
from litellm.litellm_core_utils.core_helpers import (
    get_provider_response_headers_from_hidden_params,
    set_provider_response_headers_in_hidden_params,
)
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge.response_metadata import mark_rust_response, with_provider_headers

_HEADERS: Final = TypeAdapter(dict[str, str])
_ADDITIONAL: Final = TypeAdapter(dict[str, object])
HEADERS: Final = (
    ("x-provider-trace", "first"),
    ("x-provider-trace", "second"),
    ("x-ratelimit-remaining-requests", "41"),
)


@pytest.mark.parametrize("as_dict", (False, True), ids=("model", "dict"))
def test_provider_headers_preserve_response_identity_and_existing_metadata(as_dict: bool) -> None:
    response: Final = {"id": "test", "_hidden_params": {"cache_key": "cache"}} if as_dict else ModelResponse()
    original: Final = mark_rust_response(response)
    result: Final = with_provider_headers(original, HEADERS)
    metadata: Final = get_hidden_params_dict(result)
    headers: Final = _HEADERS.validate_python(metadata["headers"])
    additional: Final = _ADDITIONAL.validate_python(metadata["additional_headers"])
    assert result is response
    assert headers["x-provider-trace"] == "first, second"
    assert additional["llm_provider-x-provider-trace"] == "first, second"
    assert additional["x-ratelimit-remaining-requests"] == "41"
    assert additional["x-litellm-rust"] == "true"
    if as_dict:
        assert metadata["cache_key"] == "cache"
        assert additional["x-litellm-cache-key"] == metadata["cache_key"]


def test_native_responses_record_headers_the_way_python_handlers_do() -> None:
    native: Final = with_provider_headers(ModelResponse(), HEADERS)
    python: Final = ModelResponse()
    set_provider_response_headers_in_hidden_params(python, httpx.Headers(list(HEADERS)))
    assert get_provider_response_headers_from_hidden_params(native) == get_provider_response_headers_from_hidden_params(
        python
    )
    assert get_hidden_params_dict(native)["additional_headers"] == get_hidden_params_dict(python)["additional_headers"]


def test_a_response_without_provider_headers_is_left_alone() -> None:
    response: Final = ModelResponse()
    assert with_provider_headers(response, ()) is response
    assert "headers" not in get_hidden_params_dict(response)
