from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import Final
from urllib.request import urlopen

import pytest

from tests._support.provider import SharedProvider, shared_provider
from tests._support.wire import Reply


def _get(url: str) -> bytes:
    with urlopen(url) as response:
        return response.read()


def test_checked_rejects_unused_replies() -> None:
    with pytest.raises(AssertionError, match="queued provider replies were never requested"):
        with shared_provider() as server, server.checked("unused"):
            server.expect(Reply(body=b'{"unused":true}'))


def test_checked_rejects_unread_requests() -> None:
    def send_unread_request() -> None:
        with shared_provider() as server, server.checked("unread"):
            server.expect(Reply(body=b'{"unread":true}'))
            with urlopen(f"{server.wire.url}/unread") as response:
                assert response.read() == b'{"unread":true}'

    with pytest.raises(AssertionError, match="the test never read the provider requests"):
        send_unread_request()


def test_port_zero_providers_serve_concurrently() -> None:
    with ExitStack() as stack:
        first: Final[SharedProvider] = stack.enter_context(shared_provider())
        second: Final[SharedProvider] = stack.enter_context(shared_provider())
        assert first.wire.url != second.wire.url

        with first.checked("first"), second.checked("second"), ThreadPoolExecutor(max_workers=2) as executor:
            first.expect(Reply(body=b'{"source":"first"}'))
            second.expect(Reply(body=b'{"source":"second"}'))
            first_response: Final = executor.submit(_get, f"{first.wire.url}/first")
            second_response: Final = executor.submit(_get, f"{second.wire.url}/second")
            assert first_response.result() == b'{"source":"first"}'
            assert second_response.result() == b'{"source":"second"}'
            first_requests: Final = first.received()
            second_requests: Final = second.received()
            assert tuple(request.target for request in first_requests) == ("/first",)
            assert tuple(request.target for request in second_requests) == ("/second",)
