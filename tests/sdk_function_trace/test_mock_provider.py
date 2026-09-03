from __future__ import annotations

from contextlib import ExitStack
from typing import Final
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from tests.sdk_function_trace.mock_provider import MockProviderResponse, mock_provider


def test_mock_provider_preserves_error_response() -> None:
    response: Final = MockProviderResponse(429, (("retry-after", "2"),), b'{"error":"rate limited"}')
    with mock_provider(response) as api_base:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(api_base, data=b"{}"), timeout=5)
        with error.value as received:
            assert received.code == 429
            assert received.headers["retry-after"] == "2"
            assert received.read() == response.body


@pytest.mark.parametrize("request_count", [0, 2])
def test_mock_provider_rejects_missing_or_duplicate_requests(request_count: int) -> None:
    response: Final = MockProviderResponse(200, (), b"{}")
    with ExitStack() as stack:
        api_base: Final = stack.enter_context(mock_provider(response))
        for _ in range(request_count):
            with urlopen(Request(api_base, data=b"{}"), timeout=5) as received:
                assert received.read() == response.body
        with pytest.raises(AssertionError, match=f"expected one provider request, received {request_count}"):
            stack.close()
