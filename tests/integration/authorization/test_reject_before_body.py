"""A request the proxy can already refuse from its headers gets its refusal before the body arrives.

A client retrying 429s with multi-megabyte image payloads used to make every worker buffer each payload
in full before answering, so the refusals themselves consumed the memory. The rejection must not depend
on the body, so these tests send only the headers and the first kilobyte of a declared 8 MB body, then
read the answer through the same socket.
"""

import socket
import uuid
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway

pytestmark: Final = pytest.mark.shares_worker_state

_DECLARED_BODY_BYTES: Final = 8 * 1024 * 1024
_SENT_BODY_BYTES: Final = 1024


def _small_chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"reject before body {uuid.uuid4().hex}"}]},
        key=key,
    )


def _status_before_body_completes(gateway: Gateway, model: str, key: str) -> int:
    """POST a chat completion, send 1 KB of its declared 8 MB body, and return the status the proxy answers with.

    The socket stays open with the body unfinished, so an answer here can only have been made from the
    headers. A proxy that waits for the whole body never answers and the read times out.
    """
    url: Final = gateway.client.base_url
    head: Final = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: {url.host}\r\nAuthorization: Bearer {key}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {_DECLARED_BODY_BYTES}\r\n\r\n"
    ).encode()
    partial_body: Final = (b'{"model": "' + model.encode() + b'", "messages": [{"role": "user", "content": "').ljust(
        _SENT_BODY_BYTES, b"A"
    )
    with socket.create_connection((url.host, url.port or 80), timeout=gateway.client.timeout.read) as connection:
        connection.sendall(head + partial_body)
        status_line: Final = connection.recv(4096).split(b"\r\n", 1)[0]
    return int(status_line.split()[1])


def test_rate_limited_key_is_refused_before_its_body_arrives(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model], rpm_limit=1, max_parallel_requests=1)
        admitted: Final = _small_chat(gateway, model, key)
        assert admitted.status_code == 200, admitted.text
        refused: Final = _small_chat(gateway, model, key)
        assert refused.status_code == 429, refused.text

        assert _status_before_body_completes(gateway, model, key) == 429


def test_blocked_key_is_refused_before_its_body_arrives(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        admitted: Final = _small_chat(gateway, model, key)
        assert admitted.status_code == 200, admitted.text
        blocked: Final = gateway.request("POST", "/key/block", {"key": key})
        assert blocked.status_code == 200, blocked.text
        refused: Final = _small_chat(gateway, model, key)
        assert refused.status_code == 401, refused.text

        assert _status_before_body_completes(gateway, model, key) == 401
