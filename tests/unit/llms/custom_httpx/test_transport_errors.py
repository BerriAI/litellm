from typing import Final

import httpx
import openai
import pytest

import litellm
from litellm.llms.custom_httpx.transport_errors import (
    TransportErrorKind,
    as_public_exception,
    classify_transport_error,
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


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ReadTimeout("timed out"), TransportErrorKind.TIMEOUT),
        (httpx.ConnectTimeout("timed out"), TransportErrorKind.TIMEOUT),
        (httpx.ReadError("connection reset"), TransportErrorKind.DROPPED_CONNECTION),
        (httpx.RemoteProtocolError("incomplete body"), TransportErrorKind.DROPPED_CONNECTION),
        (httpx.ConnectError("refused"), None),
        (ValueError("not a transport error"), None),
    ],
)
def test_classifies_legacy_httpx_errors(error: Exception, expected: TransportErrorKind | None):
    assert classify_transport_error(error) is expected


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("ReadTimeout", TransportErrorKind.TIMEOUT),
        ("WriteTimeout", TransportErrorKind.TIMEOUT),
        ("ReadError", TransportErrorKind.DROPPED_CONNECTION),
        ("RemoteProtocolError", TransportErrorKind.DROPPED_CONNECTION),
        ("ConnectError", None),
    ],
)
def test_classifies_httpx2_errors(class_name: str, expected: TransportErrorKind | None):
    httpx2 = pytest.importorskip("httpx2")

    assert classify_transport_error(getattr(httpx2, class_name)("transport failure")) is expected


def test_classifies_the_sdk_wrapper_by_the_error_it_was_raised_from():
    httpx2 = pytest.importorskip("httpx2")

    dropped: Final = _raised_from(openai.APIConnectionError(request=REQUEST), httpx2.ReadError("peer closed"))
    timed_out: Final = _raised_from(openai.APITimeoutError(request=REQUEST), httpx2.ReadTimeout("timed out"))

    assert classify_transport_error(dropped) is TransportErrorKind.DROPPED_CONNECTION
    assert classify_transport_error(timed_out) is TransportErrorKind.TIMEOUT
    assert classify_transport_error(openai.APIConnectionError(request=REQUEST)) is None


def test_implicit_context_is_not_a_cause():
    contextual: Final = _raised_while_handling(
        ValueError("raised while handling the read error"), httpx.ReadError("connection reset")
    )

    assert contextual.__context__ is not None
    assert classify_transport_error(contextual) is None


def test_a_cause_chain_deeper_than_the_walk_is_not_scanned_forever():
    chain: Final = _raised_from(ValueError("top"), httpx.ReadError("bottom"))
    assert classify_transport_error(chain) is TransportErrorKind.DROPPED_CONNECTION

    cyclic: Final = ValueError("self-caused")
    cyclic.__cause__ = cyclic
    assert classify_transport_error(cyclic) is None


def test_as_public_exception_maps_each_transport_kind():
    httpx2 = pytest.importorskip("httpx2")

    timeout: Final = as_public_exception(httpx2.ReadTimeout("timed out"), model="gpt-5.6", llm_provider="openai")
    dropped: Final = as_public_exception(httpx2.ReadError("peer closed"), model="gpt-5.6", llm_provider="openai")

    assert isinstance(timeout, litellm.Timeout)
    assert timeout.llm_provider == "openai"
    assert timeout.model == "gpt-5.6"
    assert "timed out" in str(timeout)
    assert isinstance(dropped, litellm.APIConnectionError)
    assert dropped.llm_provider == "openai"
    assert "peer closed" in str(dropped)
    assert as_public_exception(ValueError("unrelated"), model="gpt-5.6", llm_provider="openai") is None
