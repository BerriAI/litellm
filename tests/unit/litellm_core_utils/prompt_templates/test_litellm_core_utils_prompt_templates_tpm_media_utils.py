import base64
import copy
import json
from collections.abc import Mapping
from typing import Final

import pytest

from litellm.litellm_core_utils.prompt_templates.tpm_media_utils import (
    split_embedding_media_for_tpm,
    split_media_for_tpm,
)


@pytest.mark.parametrize(
    "part,projected,media",
    [
        (
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": "", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE", "detail": "high"}},
        ),
        (
            {"type": "input_image", "image_url": "https://example.com/image.png", "detail": "low"},
            {"type": "input_image", "image_url": "", "detail": "low"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "low"}},
        ),
        (
            {"type": "computer_screenshot", "image_url": "data:image/png;base64,IMAGE"},
            {"type": "computer_screenshot", "image_url": ""},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE", "detail": "auto"}},
        ),
        (
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "IMAGE"}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE", "detail": "auto"}},
        ),
        (
            {"type": "input_audio", "input_audio": {"data": "AUDIO", "format": "wav"}},
            {"type": "input_audio", "input_audio": {"data": "", "format": "wav"}},
            {"type": "input_audio", "input_audio": {"data": "AUDIO", "format": "wav"}},
        ),
        (
            {"type": "file", "file": {"file_data": "IMAGE", "format": "image/png", "filename": "image.png"}},
            {"type": "file", "file": {"file_data": "", "format": "image/png", "filename": "image.png"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE", "detail": "auto"}},
        ),
        (
            {"type": "input_file", "file_data": "data:audio/wav;base64,AUDIO", "filename": "audio.wav"},
            {"type": "input_file", "file_data": "", "filename": "audio.wav"},
            {"type": "input_audio", "input_audio": {"data": "AUDIO", "format": "wav"}},
        ),
    ],
)
def test_tpm_projection_blanks_only_declared_media_transport(
    part: Mapping[str, object], projected: Mapping[str, object], media: Mapping[str, object],
) -> None:
    text: Final = {"type": "text", "text": "data:image/png;base64,IMAGE"}
    request: Final = [{"role": "user", "content": [text, {**part, "cache_control": {"type": "ephemeral"}}]}]
    original: Final = copy.deepcopy(request)
    assert split_media_for_tpm(request) == (
        [{"role": "user", "content": [text, {**projected, "cache_control": {"type": "ephemeral"}}]}],
        (media,),
    )
    assert request == original


def test_tpm_projection_retains_nested_and_native_history_around_media() -> None:
    image: Final = {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE", "detail": "high"}}
    empty_image: Final = {"type": "image_url", "image_url": {"url": "", "detail": "high"}}
    document: Final = {"type": "document", "title": "keep", "source": {"type": "content", "content": [image]}}
    empty_document: Final = {**document, "source": {"type": "content", "content": [empty_image]}}
    tool: Final = {"type": "tool_result", "tool_use_id": "call", "content": [document, {"type": "text", "text": "keep"}]}
    empty_tool: Final = {**tool, "content": [empty_document, {"type": "text", "text": "keep"}]}
    request: Final = [
        {"role": "user", "content": [tool]},
        {"type": "function_call_output", "call_id": "call", "output": [image]},
        {"type": "custom_tool_call_output", "call_id": "custom", "output": [image]},
        {"type": "computer_call_output", "call_id": "computer", "output": {
            "type": "computer_screenshot", "image_url": "data:image/png;base64,IMAGE",
        }},
        {"type": "image_generation_call", "id": "ig", "result": "IMAGE", "status": "completed"},
        {"type": "code_interpreter_call", "id": "ci", "code": "keep()", "outputs": [
            {"type": "logs", "logs": "keep"}, {"type": "image", "url": "https://example.com/image.png"},
        ]},
    ]
    original: Final = copy.deepcopy(request)
    projected, media = split_media_for_tpm(request)
    assert projected == [
        {"role": "user", "content": [empty_tool]},
        {"type": "function_call_output", "call_id": "call", "output": [empty_image]},
        {"type": "custom_tool_call_output", "call_id": "custom", "output": [empty_image]},
        {"type": "computer_call_output", "call_id": "computer", "output": {
            "type": "computer_screenshot", "image_url": "",
        }},
        {"type": "image_generation_call", "id": "ig", "result": "", "status": "completed"},
        {"type": "code_interpreter_call", "id": "ci", "code": "keep()", "outputs": [
            {"type": "logs", "logs": "keep"}, {"type": "image", "url": ""},
        ]},
    ]
    assert media == (
        image, image, image,
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE", "detail": "auto"}},
        {"type": "image_url", "image_url": {"url": "data:application/octet-stream;base64,IMAGE", "detail": "auto"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "auto"}},
    )
    assert request == original


def test_tpm_projection_preserves_non_media_identity_and_opaque_values() -> None:
    image: Final = {"type": "input_image", "image_url": "data:image/png;base64,OPAQUE"}
    request: Final = [
        {"role": "user", "content": json.dumps(image)},
        {"type": "custom_tool_call_output", "call_id": "call", "output": json.dumps(image)},
        {"type": "function_call", "call_id": "call", "name": "keep", "arguments": {"content": [image]}},
        {"type": "file_search_call", "results": [{"text": "keep", "content": [image]}]},
        {"type": "image_generation_call", "id": "ig", "status": "in_progress", "result": None},
        {"role": "user", "content": [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "PDF"}},
            {"type": "document", "source": {"type": "text", "data": json.dumps(image)}},
            {"type": "file", "file": {"format": "application/pdf", "file_data": "PDF"}},
            {"type": "text", "text": "keep", "metadata": {"content": [image]}},
        ]},
    ]
    projected, media = split_media_for_tpm(request)
    assert projected is request
    assert media == ()


@pytest.mark.parametrize("mime", ["application/pdf", "video/mp4", "audio/wav", "image/png"])
def test_embedding_media_split_holds_at_most_one_copy_of_a_large_payload(mime: str) -> None:
    import gc
    import tracemalloc

    uri: Final = f"data:{mime};base64," + base64.b64encode(bytes(3 * 1024 * 1024)).decode("ascii")
    split_embedding_media_for_tpm(f"data:{mime};base64,AAAA")
    gc.collect()
    tracemalloc.start()
    try:
        _, media = split_embedding_media_for_tpm(uri)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert media
    assert peak < 1.5 * len(uri)
