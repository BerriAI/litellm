"""Tests for vLLM image-edit routing through the optional Rust bridge."""

from __future__ import annotations

import importlib
from io import BytesIO
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


class TextFileLike:
    def __init__(self, data: str) -> None:
        self.data = data
        self.position = 0

    def tell(self) -> int:
        return self.position

    def seek(self, position: int) -> None:
        self.position = position

    def read(self, *_args: object, **_kwargs: object) -> str:
        self.position = len(self.data)
        return self.data


class BytearrayFileLike:
    def __init__(self, data: bytes) -> None:
        self.data = bytearray(data)
        self.position = 0

    def tell(self) -> int:
        return self.position

    def seek(self, position: int) -> None:
        self.position = position

    def read(self, *_args: object, **_kwargs: object) -> bytearray:
        self.position = len(self.data)
        return self.data


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


def test_timeout_to_seconds_returns_none_when_all_timeout_values_unset() -> None:
    timeout = httpx.Timeout(connect=None, read=None, write=None, pool=None)

    assert image_main._timeout_to_seconds(timeout) is None


def test_rust_image_file_part_reads_bytearray() -> None:
    part = image_main._rust_image_file_part(bytearray(PNG_BYTES), "default.png")

    assert part["filename"] == "default.png"
    assert part["content_type"] == "image/png"
    assert part["data_base64"] == "iVBORw0KGgpmYWtlLWltYWdl"


def test_rust_image_file_part_preserves_bytesio_position() -> None:
    image = BytesIO(PNG_BYTES)
    image.seek(4)

    part = image_main._rust_image_file_part(image, "buffer.png")

    assert image.tell() == 4
    assert part["filename"] == "buffer.png"
    assert part["content_type"] == "image/png"
    assert part["data_base64"] == "iVBORw0KGgpmYWtlLWltYWdl"


def test_rust_image_file_part_uses_named_file_object_filename() -> None:
    image = BytesIO(PNG_BYTES)
    image.name = "/tmp/named-source.png"  # type: ignore[attr-defined]

    part = image_main._rust_image_file_part(image, "buffer.png")

    assert part["filename"] == "named-source.png"
    assert part["data_base64"] == "iVBORw0KGgpmYWtlLWltYWdl"


def test_rust_image_file_part_reads_text_file_like() -> None:
    image = TextFileLike("plain-text")
    image.seek(5)

    part = image_main._rust_image_file_part(image, "text.png")

    assert image.tell() == 5
    assert part["filename"] == "text.png"
    assert part["data_base64"] == "cGxhaW4tdGV4dA=="


def test_rust_image_file_part_reads_bytearray_file_like() -> None:
    image = BytearrayFileLike(PNG_BYTES)
    image.seek(3)

    part = image_main._rust_image_file_part(image, "bytes.png")

    assert image.tell() == 3
    assert part["filename"] == "bytes.png"
    assert part["data_base64"] == "iVBORw0KGgpmYWtlLWltYWdl"


def test_rust_image_file_part_reads_filesystem_path(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    image_path.write_bytes(PNG_BYTES)

    part = image_main._rust_image_file_part(str(image_path), "fallback.png")

    assert part["filename"] == "source.png"
    assert part["data_base64"] == "iVBORw0KGgpmYWtlLWltYWdl"


def test_rust_image_file_part_reads_tuple_metadata() -> None:
    part = image_main._rust_image_file_part(
        ("custom.jpeg", BytesIO(PNG_BYTES), "image/jpeg"),
        "fallback.png",
    )

    assert part["filename"] == "custom.jpeg"
    assert part["content_type"] == "image/jpeg"
    assert part["data_base64"] == "iVBORw0KGgpmYWtlLWltYWdl"


def test_rust_image_file_part_rejects_unsupported_input() -> None:
    with pytest.raises(TypeError, match="Unsupported image file type"):
        image_main._rust_image_file_part(object(), "fallback.png")
