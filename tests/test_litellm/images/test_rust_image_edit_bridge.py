"""Tests for vLLM image-edit routing through the optional Rust bridge."""

from __future__ import annotations

import importlib
from pathlib import Path

import httpx
import pytest

import litellm
from litellm.exceptions import APIConnectionError
from litellm.images import main as image_main

rust_bridge = importlib.import_module("litellm.ocr.rust_bridge")

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-image"


class RecordingImageEditBridge:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        model: str,
        images: list[dict[str, object]],
        prompt: str | None,
        mask: dict[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": model,
                "images": images,
                "prompt": prompt,
                "mask": mask,
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": custom_llm_provider,
                "extra_headers": extra_headers,
                "optional_params": optional_params,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {"created": 1, "data": [{"b64_json": "ZmFrZS1pbWFnZQ=="}]}


class RecordingAsyncImageEditBridge:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def __call__(
        self,
        model: str,
        images: list[dict[str, object]],
        prompt: str | None,
        mask: dict[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": model,
                "images": images,
                "prompt": prompt,
                "mask": mask,
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": custom_llm_provider,
                "extra_headers": extra_headers,
                "optional_params": optional_params,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {"created": 1, "data": [{"b64_json": "YXN5bmMtaW1hZ2U="}]}


@pytest.fixture(autouse=True)
def reset_rust_bridge() -> None:
    rust_bridge.use_litellm_rust(
        False, ocr=None, aocr=None, image_edit=None, aimage_edit=None
    )
    yield
    rust_bridge.use_litellm_rust(
        False, ocr=None, aocr=None, image_edit=None, aimage_edit=None
    )


def test_vllm_image_edit_routes_to_rust_bridge() -> None:
    bridge = RecordingImageEditBridge()
    litellm.use_litellm_rust(True, image_edit=bridge)

    response = litellm.image_edit(
        model="vllm/qwen-image-edit",
        image=PNG_BYTES,
        prompt="make it brighter",
        quality="high",
        size="1024x1024",
        api_base="http://localhost:8000",
        api_key="sk-test",
        extra_headers={"x-trace-id": "trace-1"},
        timeout=12.5,
    )

    assert response.data[0].b64_json == "ZmFrZS1pbWFnZQ=="
    assert len(bridge.calls) == 1
    call = bridge.calls[0]
    assert call["model"] == "qwen-image-edit"
    assert call["prompt"] == "make it brighter"
    assert call["api_key"] == "sk-test"
    assert call["api_base"] == "http://localhost:8000"
    assert call["custom_llm_provider"] == "vllm"
    assert call["extra_headers"] == {"x-trace-id": "trace-1"}
    assert call["optional_params"] == {"quality": "high", "size": "1024x1024"}
    assert call["timeout_seconds"] == 12.5
    images = call["images"]
    assert isinstance(images, list)
    assert images[0]["filename"] == "image-0.png"
    assert images[0]["content_type"] == "image/png"
    assert images[0]["data_base64"] == "iVBORw0KGgpmYWtlLWltYWdl"


@pytest.mark.asyncio
async def test_vllm_aimage_edit_routes_to_async_rust_bridge() -> None:
    bridge = RecordingAsyncImageEditBridge()
    litellm.use_litellm_rust(True, aimage_edit=bridge)

    response = await litellm.aimage_edit(
        model="vllm/qwen-image-edit",
        image=[PNG_BYTES],
        prompt="make it darker",
        api_base="http://localhost:8000",
    )

    assert response.data[0].b64_json == "YXN5bmMtaW1hZ2U="
    assert len(bridge.calls) == 1
    assert bridge.calls[0]["model"] == "qwen-image-edit"
    assert bridge.calls[0]["prompt"] == "make it darker"


def test_vllm_image_edit_respects_disabled_rust_bridge() -> None:
    bridge = RecordingImageEditBridge()
    litellm.use_litellm_rust(False, image_edit=bridge)

    with pytest.raises(
        APIConnectionError, match="image edit is not supported for vllm"
    ):
        litellm.image_edit(
            model="vllm/qwen-image-edit",
            image=PNG_BYTES,
            prompt="make it brighter",
            api_base="http://localhost:8000",
        )

    assert bridge.calls == []


def test_vllm_image_edit_missing_path_raises_before_bridge_call(
    tmp_path: Path,
) -> None:
    bridge = RecordingImageEditBridge()
    missing_image = tmp_path / "missing-image.png"
    litellm.use_litellm_rust(True, image_edit=bridge)

    with pytest.raises(APIConnectionError, match="Image file path does not exist"):
        litellm.image_edit(
            model="vllm/qwen-image-edit",
            image=str(missing_image),
            prompt="make it brighter",
            api_base="http://localhost:8000",
        )

    assert bridge.calls == []


def test_timeout_to_seconds_falls_back_to_non_read_timeout() -> None:
    timeout = httpx.Timeout(connect=7.0, read=None, write=8.0, pool=9.0)

    assert image_main._timeout_to_seconds(timeout) == 7.0
