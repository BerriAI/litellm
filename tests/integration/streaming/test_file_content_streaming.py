import threading
import uuid
from collections.abc import Callable
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

STREAM_CHUNK_BYTES: Final = 1024 * 1024
HEAD: Final = b"h" * STREAM_CHUNK_BYTES
TAIL: Final = b'{"custom_id": "tail", "response": {"status_code": 200}}\n'


def _file_content_gated_after_head(gate: threading.Event) -> Callable[[Request], Reply]:
    def respond(_request: Request) -> Reply:
        return Reply(content_type="application/octet-stream", chunks=(HEAD, TAIL), gate_after_first=gate)

    return respond


@pytest.mark.covers("streaming.file_content.body_reaches_client_before_upstream_finishes_sending")
def test_file_content_streams_the_first_megabyte_to_the_client_before_the_upstream_sends_the_rest(
    gateway: Gateway,
) -> None:
    file_id: Final = "file-" + uuid.uuid4().hex
    gate: Final = threading.Event()
    with gateway.scenario() as scenario, wire_server(_file_content_gated_after_head(gate)) as wire:
        model: Final = scenario.model(api_base=wire.url + "/v1")
        with gateway.client.stream(
            "GET",
            f"/v1/files/{file_id}/content",
            params={"model": model},
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            chunks: Final = response.iter_bytes(chunk_size=STREAM_CHUNK_BYTES)
            head: Final = next(chunks)
            assert head == HEAD, f"First {len(head)} bytes differ from the upstream head before the gate was released"
            gate.set()
            rest: Final = b"".join(chunks)
        assert rest == TAIL, rest
        requests: Final = wire.drain()
        assert len(requests) == 1, requests
        assert requests[0].method == "GET", requests[0]
        assert requests[0].target == f"/v1/files/{file_id}/content", requests[0].target
        assert requests[0].headers["authorization"] == "Bearer integration-provider-key", requests[0].headers
        assert requests[0].body == b"", requests[0].body
