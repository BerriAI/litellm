import base64
import json
from pathlib import Path
from typing import Final

import pytest
import yaml

from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

_CHAT_SHAPED_HOOK: Final = """
from litellm.integrations.custom_logger import CustomLogger


class ChatShapedHook(CustomLogger):
    async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
        return response.choices[0].message.content


instance = ChatShapedHook()
"""

_VIDEO_JOB: Final = {
    "id": "video_hook_isolation",
    "object": "video",
    "status": "queued",
    "model": "sora-2",
    "seconds": "4",
    "size": "720x1280",
}


@pytest.mark.covers("other.observability.callbacks.raising_success_deployment_hook_keeps_video_response")
def test_video_response_survives_chat_shaped_success_deployment_hook(gateway: Gateway, tmp_path: Path) -> None:
    def upstream(request: Request) -> Reply:
        assert request.target == "/v1/videos", request.target
        assert b"a cat" in request.body, request.body[:300]
        return Reply(body=json.dumps(_VIDEO_JOB).encode())

    (tmp_path / "chat_shaped_hook.py").write_text(_CHAT_SHAPED_HOOK)
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update({"callbacks": ["chat_shaped_hook.instance"]})
    path: Final = tmp_path / "hook.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        wire_server(upstream) as provider,
        owned_proxy(gateway, tmp_path, {}, config=path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/sora-2", api_base=provider.url + "/v1")
        response: Final = candidate.request("POST", "/v1/videos", {"model": model, "prompt": "a cat"})
        assert response.status_code == 200, response.text
        encoded_id: Final = response.json()["id"].removeprefix("video_")
        assert f"video_id:{_VIDEO_JOB['id']}" in base64.b64decode(encoded_id).decode(), response.text
        assert response.json()["status"] == _VIDEO_JOB["status"], response.text
