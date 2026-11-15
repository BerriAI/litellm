"""A provider registered in `custom_provider_map` with a `BaseVideoConfig` serves /v1/videos create, status and content."""

import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Final

from integration._support.client import Gateway
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

_SUPPORT_MODULE: Final = Path(__file__).resolve().parents[1] / "_support" / "custom_video_provider.py"
_PROMPT: Final = "a cat walking across a bridge"
_MP4: Final = b"\x00\x00\x00\x18ftypmp42" + b"queue-video-bytes"


def _queue_backend(request: Request) -> Reply:
    if (request.method, request.target) == ("POST", "/jobs"):
        return Reply(body=json.dumps({"job_id": "job-7", "state": "queued"}).encode())
    if (request.method, request.target) == ("GET", "/jobs/job-7"):
        return Reply(body=json.dumps({"job_id": "job-7", "state": "done", "seconds": 8}).encode())
    if (request.method, request.target) == ("GET", "/jobs/job-7/file"):
        return Reply(body=_MP4, content_type="video/mp4")
    return Reply(status=404, body=json.dumps({"error": f"no route {request.method} {request.target}"}).encode())


def _write_config(path: Path, wire: Wire) -> Path:
    shutil.copy(_SUPPORT_MODULE, path.parent / "custom_video_provider.py")
    path.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": "queue-video",
                        "litellm_params": {
                            "model": "queue-video/seedance-1",
                            "api_base": wire.url,
                            "api_key": "queue-secret",
                        },
                    },
                    {
                        "model_name": "image-only-video",
                        "litellm_params": {
                            "model": "image-only/seedance-1",
                            "api_base": wire.url,
                            "api_key": "unused",
                        },
                    },
                ],
                "litellm_settings": {
                    "custom_provider_map": [
                        {"provider": "queue-video", "custom_handler": "custom_video_provider.VIDEO_CONFIG"},
                        {"provider": "image-only", "custom_handler": "custom_video_provider.IMAGE_ONLY_HANDLER"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _lower_headers(request: Request) -> dict[str, str]:
    return {name.lower(): value for name, value in request.headers.items()}


def test_custom_provider_map_base_video_config_serves_create_status_and_content(
    gateway: Gateway, tmp_path: Path
) -> None:
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_queue_backend))
        config: Final = _write_config(tmp_path / "custom-video.yaml", wire)
        owned: Final = stack.enter_context(owned_proxy_process(gateway, tmp_path, {}, config=config))

        created: Final = owned.gateway.request(
            "POST", "/v1/videos", {"model": "queue-video", "prompt": _PROMPT, "seconds": "8"}
        )
        assert created.status_code == 200, created.text
        created_body: Final = created.json()
        assert created_body["status"] == "queued", created_body
        video_id: Final = created_body["id"]
        assert isinstance(video_id, str) and video_id.startswith("video_"), created_body

        submits: Final = wire.drain()
        assert [(r.method, r.target) for r in submits] == [("POST", "/jobs")], submits
        assert json.loads(submits[0].body) == {"model": "seedance-1", "prompt": _PROMPT, "seconds": "8"}
        assert _lower_headers(submits[0])["x-api-key"] == "queue-secret"

        status: Final = owned.gateway.request("GET", f"/v1/videos/{video_id}")
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "completed", status.text
        assert status.json()["id"] == video_id, status.text

        content: Final = owned.gateway.request("GET", f"/v1/videos/{video_id}/content")
        assert content.status_code == 200, content.text
        assert content.content == _MP4

        polls: Final = wire.drain()
        assert [(r.method, r.target) for r in polls] == [("GET", "/jobs/job-7"), ("GET", "/jobs/job-7/file")], polls
        assert all(_lower_headers(r)["x-api-key"] == "queue-secret" for r in polls)


def test_custom_provider_without_a_video_config_is_refused_with_400(gateway: Gateway, tmp_path: Path) -> None:
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_queue_backend))
        config: Final = _write_config(tmp_path / "custom-video.yaml", wire)
        owned: Final = stack.enter_context(owned_proxy_process(gateway, tmp_path, {}, config=config))

        refused: Final = owned.gateway.request("POST", "/v1/videos", {"model": "image-only-video", "prompt": _PROMPT})
        assert refused.status_code == 400, refused.text
        assert "video generation is not supported for image-only" in refused.text
        assert wire.drain() == ()
