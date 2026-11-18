import base64
import itertools
import json
import logging
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import boto3
import httpx
import pytest
import yaml
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.eventstream import EventStreamBuffer
from botocore.exceptions import BotoCoreError

from tests.integration._support.client import Gateway, eventually, gateway_from_environment
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

pytestmark: Final = pytest.mark.skipif(
    os.environ.get("LIVE_QA") != "1", reason="requires LIVE_QA=1 and Bedrock credentials"
)
_LOG: Final = logging.getLogger(__name__)

_BEDROCK_MODEL: Final = "anthropic.claude-haiku-4-5-20251001-v1:0"

_CALLBACK: Final = """
from pathlib import Path
from litellm.integrations.custom_logger import CustomLogger

class Capture(CustomLogger):
    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        try:
            async for chunk in response:
                request_data["qa_late_field"] = "set"
                Path(__file__).with_name("stream-seen").touch()
                yield chunk
        finally:
            Path(__file__).with_name("iterator-finally").touch()

    async def async_post_call_failure_hook(
        self, request_data, original_exception, user_api_key_dict, traceback_str=None,
    ):
        import json
        import traceback
        Path(__file__).with_name("callback.json").write_text(json.dumps({
            "field": request_data.get("qa_late_field"),
            "failure": type(original_exception).__name__,
            "hook_stack": [frame.name for frame in traceback.extract_stack() if frame.filename.endswith(("common_request_processing.py", "callbacks_legacy_python.py", "litellm_logging.py"))],
        }))

capture = Capture()
"""

_FORWARD_ERROR: Final = threading.Event()


def _upstream(request: Request) -> Reply:
    if request.target == "/__observations" and request.method == "DELETE":
        return Reply()
    assert request.target.startswith("/model/"), request.target
    url: Final = "https://bedrock-runtime.us-east-1.amazonaws.com" + request.target
    credentials: Final = boto3.Session(profile_name="litellm-dev").get_credentials().get_frozen_credentials()
    signed: Final = AWSRequest(method="POST", url=url, data=request.body, headers={"content-type": "application/json"})
    SigV4Auth(credentials, "bedrock", "us-east-1").add_auth(signed)
    with httpx.stream("POST", url, headers=dict(signed.headers), content=request.body, timeout=30) as response:
        _LOG.info("BEDROCK HTTP: %s", response.status_code)
        assert response.status_code == 200, response.read().decode()[:400]
        buf: Final = EventStreamBuffer()
        opening: Final = bytearray()
        for data in response.iter_bytes():
            opening.extend(data)
            buf.add_data(data)
            if any(b'"content_block_delta"' in base64.b64decode(json.loads(event.payload)["bytes"]) for event in buf):
                break
        else:
            raise AssertionError("Bedrock did not stream a text delta")
    return Reply(
        content_type="application/vnd.amazon.eventstream",
        chunks=(bytes(opening), b"unused"),
        gate_after_first=_FORWARD_ERROR,
        abort_after=1,
    )


@pytest.fixture(scope="module")
def running_qa(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Path]]:
    try:
        credentials: Final = boto3.Session(profile_name="litellm-dev").get_credentials()
        if credentials is None:
            pytest.skip("litellm-dev AWS credentials are required")
        credentials.get_frozen_credentials()
    except (BotoCoreError, OSError) as error:
        pytest.skip(f"litellm-dev AWS credentials unavailable: {type(error).__name__}")
    root: Final = tmp_path_factory.mktemp("callback-qa")
    (root / "qa_callback.py").write_text(_CALLBACK)
    with wire_server(_upstream) as wire:
        config: Final = root / "config.yaml"
        stock: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        stock["model_list"] = [
            {
                "model_name": "callback-bedrock",
                "litellm_params": {
                    "model": f"bedrock/invoke/us.{_BEDROCK_MODEL}",
                    "api_base": wire.url,
                    "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
                    "aws_secret_access_key": "scripted-secret",
                    "aws_region_name": "us-east-1",
                },
            }
        ]
        stock.setdefault("litellm_settings", {}).update(
            callbacks=["qa_callback.capture"], anthropic_sse_ping_interval_seconds=0.2
        )
        config.write_text(yaml.safe_dump(stock))
        with pytest.MonkeyPatch.context() as environment:
            environment.setenv("INTEGRATION_UPSTREAM_URL", wire.url)
            with (
                gateway_from_environment() as shared,
                owned_proxy(shared, root, {}, config=config, workers=2) as gateway,
            ):
                yield gateway, root


def test_callback_after_disconnect(running_qa: tuple[Gateway, Path]) -> None:
    gateway, root = running_qa
    _FORWARD_ERROR.clear()
    for name in ("callback.json", "iterator-finally"):
        (root / name).unlink(missing_ok=True)
    try:
        with gateway.client.stream(
            "POST",
            "/v1/messages",
            headers={"Authorization": f"Bearer {gateway.key}"},
            json={
                "model": "callback-bedrock",
                "max_tokens": 32,
                "stream": True,
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        ) as response:
            assert response.status_code == 200, response.read()
            assert any("event: content_block_delta" in text for text in itertools.accumulate(response.iter_text()))
        _LOG.info("CLIENT: closed after text delta, before relay cut")
        eventually((root / "iterator-finally").exists, bool, seconds=2, return_last_on_timeout=True)
        assert not (root / "callback.json").exists()
    finally:
        _FORWARD_ERROR.set()
    path: Final = root / "callback.json"
    if os.environ.get("EXPECT_CALLBACK") == "0":
        assert not eventually(path.exists, bool, seconds=6, return_last_on_timeout=True)
        _LOG.info("CALLBACK: missing")
        return
    eventually(path.exists, bool, seconds=8)
    seen: Final = json.loads(path.read_text())
    _LOG.info("CALLBACK: %s", seen)
    assert seen["failure"] == "ReadError", seen
    assert seen["field"] == (None if os.environ.get("EXPECT_STALE") == "1" else "set"), seen
