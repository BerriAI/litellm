import itertools
import json
import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml

from tests.integration._support.client import Gateway, eventually, gateway_from_environment
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server
from tests.integration.spend.test_disconnected_bedrock_messages_stream_billing import _message_start, _text_delta

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
    assert json.loads(request.body)["messages"] == [{"role": "user", "content": "Say hello"}], request.body
    opening: Final = _message_start("msg_disconnect_callback") + _text_delta("hello")
    _LOG.info("BEDROCK HTTP: 200 (scripted peer)")
    return Reply(
        content_type="application/vnd.amazon.eventstream",
        chunks=(opening, b"unused"),
        gate_after_first=_FORWARD_ERROR,
        abort_after=1,
    )


@pytest.fixture(scope="module")
def running_qa(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Path]]:
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
    eventually(path.exists, bool, seconds=8)
    seen: Final = json.loads(path.read_text())
    _LOG.info("CALLBACK: %s", seen)
    assert seen["failure"] == "ReadError", seen
    assert seen["field"] == "set", seen
