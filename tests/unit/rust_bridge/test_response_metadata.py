from typing import Final

import pytest

from litellm import ModelResponse
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge.response_metadata import mark_rust_response, with_provider_headers


@pytest.mark.parametrize("as_dict", (False, True), ids=("model", "dict"))
def test_provider_headers_preserve_response_identity_and_existing_metadata(as_dict: bool) -> None:
    response: Final = {"id": "test", "_hidden_params": {"cache_key": "cache"}} if as_dict else ModelResponse()
    original: Final = mark_rust_response(response)
    headers: Final = (("x-provider-trace", "first"), ("x-provider-trace", "second"), ("set-cookie", "private"))
    result: Final = with_provider_headers(original, headers, headers[:2])
    metadata: Final = get_hidden_params_dict(result)
    additional: Final = metadata["additional_headers"]
    assert result is response
    assert metadata["provider_response_headers"] == headers
    assert additional["x-provider-trace"] == "first, second"
    assert additional["llm_provider-x-provider-trace"] == additional["x-provider-trace"]
    assert additional["x-litellm-rust"] == "true"
    assert "set-cookie" not in additional
    if as_dict:
        assert metadata["cache_key"] == "cache"
        assert additional["x-litellm-cache-key"] == metadata["cache_key"]
