from typing import Final

import httpx
import openai
import pytest

from litellm.llms.custom_httpx.transport_errors import (
    TransportErrorKind,
    chained_transport_error_kind,
    transport_error_classes,
    transport_error_kind,
)

REQUEST: Final = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _raised_from(error: Exception, cause: Exception) -> Exception:
    try:
        raise error from cause
    except Exception as raised:
        return raised


def _raised_while_handling(error: Exception, handled: Exception) -> Exception:
    try:
        try:
            raise handled
        except Exception:
            raise error
    except Exception as raised:
        return raised


@pytest.mark.parametrize("module_name", ("httpx", "httpx2"))
@pytest.mark.parametrize(
    ("class_name", "expected"),
    (
        ("ReadTimeout", TransportErrorKind.TIMEOUT),
        ("ConnectTimeout", TransportErrorKind.TIMEOUT),
        ("ReadError", TransportErrorKind.DROPPED_CONNECTION),
        ("RemoteProtocolError", TransportErrorKind.DROPPED_CONNECTION),
        ("ConnectError", None),
        ("WriteError", None),
    ),
)
def test_transport_error_kind_matches_httpx_and_httpx2_alike(
    module_name: str, class_name: str, expected: TransportErrorKind | None
):
    module: Final = pytest.importorskip(module_name)
    error: Final = getattr(module, class_name)("transport failure")

    assert transport_error_kind(error) is expected
    assert chained_transport_error_kind(error) is expected
    assert all(isinstance(error, transport_error_classes(kind)) is (kind is expected) for kind in TransportErrorKind)


def test_transport_error_kind_ignores_values_that_are_not_transport_errors():
    assert transport_error_kind(ValueError("decoder failure")) is None
    assert transport_error_kind({"error": "not an exception"}) is None


def test_chained_kind_reads_the_transport_error_an_sdk_error_was_raised_from():
    httpx2: Final = pytest.importorskip("httpx2")
    dropped: Final = _raised_from(openai.APIConnectionError(request=REQUEST), httpx2.ReadError("peer closed"))
    timed_out: Final = _raised_from(openai.APITimeoutError(request=REQUEST), httpx2.ReadTimeout("timed out"))
    twice_wrapped: Final = _raised_from(ValueError("outer"), timed_out)

    assert transport_error_kind(dropped) is None
    assert chained_transport_error_kind(dropped) is TransportErrorKind.DROPPED_CONNECTION
    assert chained_transport_error_kind(timed_out) is TransportErrorKind.TIMEOUT
    assert chained_transport_error_kind(twice_wrapped) is TransportErrorKind.TIMEOUT
    assert chained_transport_error_kind(openai.APIConnectionError(request=REQUEST)) is None


def test_chained_kind_ignores_an_error_raised_while_handling_a_transport_error():
    contextual: Final = _raised_while_handling(ValueError("bug in a handler"), httpx.ReadError("connection reset"))

    assert contextual.__context__ is not None
    assert chained_transport_error_kind(contextual) is None


def test_chained_kind_stops_on_a_cause_cycle():
    cyclic: Final = ValueError("caused by itself")
    cyclic.__cause__ = cyclic

    assert chained_transport_error_kind(cyclic) is None
