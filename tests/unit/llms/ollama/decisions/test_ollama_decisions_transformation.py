import base64
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.llms.base_llm.decisions.transformation import systemone_request_to_ir
from litellm.llms.ollama.decisions.transformation import OllamaDecisionsConfig
from litellm.llms.openai.decisions.transformation import openai_request_to_ir
from litellm.types.decisions import DecisionsRequestBody, OpenAIDecisionRequestBody

_IMAGE: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a1XkAAAAASUVORK5CYII="
_DATA_URL: Final = f"data:image/png;base64,{_IMAGE}"
_SYSTEMONE: Final = TypeAdapter(DecisionsRequestBody)
_OPENAI: Final = TypeAdapter(OpenAIDecisionRequestBody)
_IMAGES: Final = TypeAdapter(list[str])


@pytest.mark.parametrize("image", (_DATA_URL, {"content_type": "image/png", "base64": _IMAGE}))
def test_systemone_images_decode_as_native_ollama_image_bytes(image: str | dict[str, str]) -> None:
    request: Final = systemone_request_to_ir(
        _SYSTEMONE.validate_python(
            {
                "state": "Read the label.",
                "images": [image],
                "questions": {"label": {"type": "choice", "criteria": {"17": None, "62": None}}},
            }
        )
    )
    body: Final = OllamaDecisionsConfig().transform_decisions_request("clef", request, "ollama")
    images: Final = _IMAGES.validate_python(body["images"])

    assert [base64.b64decode(value, validate=True) for value in images] == [base64.b64decode(_IMAGE)]


def test_openai_image_input_decodes_as_native_ollama_image_bytes() -> None:
    request: Final = openai_request_to_ir(
        _OPENAI.validate_python(
            {
                "input": [{"role": "user", "content": [{"type": "input_image", "image_url": _DATA_URL}]}],
                "questions": [
                    {"type": "choice", "instructions": "Read the label.", "choices": [{"value": "17"}, {"value": "62"}]}
                ],
            }
        )
    )
    body: Final = OllamaDecisionsConfig().transform_decisions_request("clef", request, "ollama")
    images: Final = _IMAGES.validate_python(body["images"])

    assert [base64.b64decode(value, validate=True) for value in images] == [base64.b64decode(_IMAGE)]
