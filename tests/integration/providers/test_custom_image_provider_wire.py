"""A `CustomLLM` in `custom_provider_map` is handed the caller's OpenAI image params on /v1/images/generations and /edits."""

import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Final

from integration._support.client import Gateway
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

_SUPPORT_MODULE: Final = Path(__file__).resolve().parents[1] / "_support" / "custom_image_provider.py"
_PROMPT: Final = "a red cube on a white table"
_PNG: Final = b"\x89PNG\r\n\x1a\n" + b"studio-pixels"


def _studio_backend(request: Request) -> Reply:
    return Reply(body=json.dumps({"data": [{"b64_json": "c3R1ZGlv"}]}).encode())


def _write_config(path: Path, wire: Wire) -> Path:
    shutil.copy(_SUPPORT_MODULE, path.parent / "custom_image_provider.py")
    path.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": "studio-image",
                        "litellm_params": {"model": "studio/flare-1", "api_base": wire.url},
                    },
                    {
                        "model_name": "studio-image-fixed-format",
                        "litellm_params": {
                            "model": "studio/flare-1",
                            "api_base": wire.url,
                            "additional_drop_params": ["response_format", "user"],
                        },
                    },
                ],
                "litellm_settings": {
                    "custom_provider_map": [
                        {"provider": "studio", "custom_handler": "custom_image_provider.IMAGE_HANDLER"}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_custom_image_handler_receives_the_callers_openai_generation_params(gateway: Gateway, tmp_path: Path) -> None:
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_studio_backend))
        config: Final = _write_config(tmp_path / "custom-image.yaml", wire)
        owned: Final = stack.enter_context(owned_proxy_process(gateway, tmp_path, {}, config=config))

        generated: Final = owned.gateway.request(
            "POST",
            "/v1/images/generations",
            {
                "model": "studio-image",
                "prompt": _PROMPT,
                "n": 2,
                "size": "1024x1536",
                "quality": "low",
                "response_format": "b64_json",
                "user": "artist-7",
                "studio_preset": "matte",
            },
        )
        assert generated.status_code == 200, generated.text

        renders: Final = wire.drain()
        assert [(r.method, r.target) for r in renders] == [("POST", "/render")], renders
        assert json.loads(renders[0].body) == {
            "model": "flare-1",
            "prompt": _PROMPT,
            "received": {
                "n": 2,
                "size": "1024x1536",
                "quality": "low",
                "response_format": "b64_json",
                "user": "artist-7",
                "studio_preset": "matte",
            },
            "has_mask": False,
        }


def test_custom_image_handler_receives_the_callers_openai_edit_params(gateway: Gateway, tmp_path: Path) -> None:
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_studio_backend))
        config: Final = _write_config(tmp_path / "custom-image.yaml", wire)
        owned: Final = stack.enter_context(owned_proxy_process(gateway, tmp_path, {}, config=config))

        edited: Final = owned.gateway.request_multipart(
            "/v1/images/edits",
            {
                "model": "studio-image",
                "prompt": _PROMPT,
                "n": "2",
                "size": "1024x1536",
                "quality": "low",
                "response_format": "b64_json",
                "user": "artist-7",
            },
            {"image": ("cube.png", _PNG, "image/png"), "mask": ("mask.png", _PNG, "image/png")},
        )
        assert edited.status_code == 200, edited.text

        retouches: Final = wire.drain()
        assert [(r.method, r.target) for r in retouches] == [("POST", "/retouch")], retouches
        assert json.loads(retouches[0].body) == {
            "model": "flare-1",
            "prompt": _PROMPT,
            "received": {
                "n": 2,
                "size": "1024x1536",
                "quality": "low",
                "response_format": "b64_json",
                "user": "artist-7",
            },
            "has_mask": True,
        }


def test_additional_drop_params_keeps_named_image_params_from_the_custom_handler(
    gateway: Gateway, tmp_path: Path
) -> None:
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_studio_backend))
        config: Final = _write_config(tmp_path / "custom-image.yaml", wire)
        owned: Final = stack.enter_context(owned_proxy_process(gateway, tmp_path, {}, config=config))
        params: Final = {"n": "2", "size": "1024x1536", "response_format": "b64_json", "user": "artist-7"}

        generated: Final = owned.gateway.request(
            "POST",
            "/v1/images/generations",
            {"model": "studio-image-fixed-format", "prompt": _PROMPT, **params, "n": 2},
        )
        assert generated.status_code == 200, generated.text
        edited: Final = owned.gateway.request_multipart(
            "/v1/images/edits",
            {"model": "studio-image-fixed-format", "prompt": _PROMPT, **params},
            {"image": ("cube.png", _PNG, "image/png")},
        )
        assert edited.status_code == 200, edited.text

        calls: Final = wire.drain()
        assert [(r.target, json.loads(r.body)["received"]) for r in calls] == [
            ("/render", {"n": 2, "size": "1024x1536"}),
            ("/retouch", {"n": 2, "size": "1024x1536"}),
        ], calls
