from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest
from _s3_v2_support import RecordingS3Sink, s3_config
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.parametrize("stream", [False, True])
def test_streaming_failure_retries_upload_one_s3_failure(
    gateway: Gateway,
    tmp_path: Path,
    stream: bool,
) -> None:
    sink: Final = RecordingS3Sink()

    def provider(request: Request) -> Reply:
        assert request.method == "POST", request.method
        assert request.target == "/v1/chat/completions", request.target
        return Reply(
            status=503,
            body=json.dumps(
                {
                    "error": {
                        "message": "synthetic upstream failure",
                        "type": "server_error",
                        "param": None,
                        "code": "synthetic_failure",
                    }
                }
            ).encode(),
        )

    with wire_server(provider) as upstream, wire_server(sink.respond) as bucket:
        config: Final = s3_config(tmp_path, bucket.url, {}, settings={"num_retries": 2})
        with (
            owned_proxy(
                gateway,
                tmp_path,
                {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "2"},
                config=config,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                api_base=upstream.url + "/v1",
                api_key="synthetic-provider-key",
            )
            key: Final = scenario.key(models=[model])
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "synthetic streaming failure"}],
                    "stream": stream,
                },
                key=key,
            )

            assert not 200 <= response.status_code < 300, f"{response.status_code}: {response.text}"
            requests: Final = upstream.drain()
            assert tuple(request.method for request in requests) == ("POST", "POST", "POST")
            first_payloads: Final = eventually(
                lambda: sink.payloads(),
                lambda payloads: len(payloads) >= 1,
                seconds=30,
            )
            assert first_payloads[0]["status"] == "failure"
            # A full three-interval window is needed to establish that no later duplicate arrives.
            duplicate_attempts: Final = eventually(
                lambda: sink.attempts,
                lambda attempts: attempts > 1,
                seconds=6,
                return_last_on_timeout=True,
            )
            assert duplicate_attempts == 1
            assert sink.attempts == 1
            payloads_after_window: Final = sink.payloads()
            assert len(payloads_after_window) == 1
            assert payloads_after_window[0]["status"] == "failure"
