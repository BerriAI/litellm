import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

_MODEL: Final = "bytedance/seedance-2.5/text-to-video"
_MP4: Final = b"\x00\x00\x00\x18ftypmp42" + uuid.uuid4().bytes * 4


@pytest.mark.covers("other.provider_wire.fal_ai.video_queue_create_status_and_content_download")
def test_fal_video_create_status_and_content_follow_queue_wire_contract(gateway: Gateway) -> None:
    request_id: Final = "fal-req-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.target == f"/files/{request_id}.mp4":
            assert request.method == "GET"
            return Reply(body=_MP4, content_type="video/mp4")
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        if request.method == "POST":
            assert request.target == f"/{_MODEL}"
            assert json.loads(request.body) == {
                "prompt": "a cat playing volleyball on a beach",
                "duration": "4",
                "resolution": "720p",
                "aspect_ratio": "16:9",
            }
            return Reply(
                body=json.dumps({"status": "IN_QUEUE", "request_id": request_id, "queue_position": 0}).encode()
            )
        assert request.method == "GET"
        if request.target == f"/bytedance/seedance-2.5/requests/{request_id}/status":
            return Reply(body=json.dumps({"status": "COMPLETED", "request_id": request_id}).encode())
        assert request.target == f"/bytedance/seedance-2.5/requests/{request_id}"
        return Reply(body=json.dumps({"video": {"url": f"{wire_url}/files/{request_id}.mp4"}}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        wire_url: Final = wire.url
        model: Final = scenario.model(
            model=f"fal_ai/{_MODEL}",
            api_base=wire.url,
            api_key="synthetic-fal-key",
        )
        created: Final = gateway.post(
            "/v1/videos",
            {
                "model": model,
                "prompt": "a cat playing volleyball on a beach",
                "seconds": "4",
                "size": "1280x720",
            },
        )
        assert created["status"] == "queued"
        video_id: Final = created["id"]
        assert isinstance(video_id, str) and video_id
        status: Final = gateway.get(f"/v1/videos/{video_id}")
        assert status["status"] == "completed"
        content: Final = gateway.request("GET", f"/v1/videos/{video_id}/content")
        assert content.status_code == 200, content.text
        assert content.headers["content-type"].startswith("video/mp4")
        assert content.content == _MP4
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", f"/{_MODEL}"),
            ("GET", f"/bytedance/seedance-2.5/requests/{request_id}/status"),
            ("GET", f"/bytedance/seedance-2.5/requests/{request_id}"),
            ("GET", f"/files/{request_id}.mp4"),
        ]
