import asyncio
from typing import Final, Literal

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.cloudflare.extraction.guardrail_translation.handler import ExtractionHandler
from litellm.types.extraction import ExtractionResponse
from litellm.types.utils import GenericGuardrailAPIInputs

MANY_FIELDS: Final = 2_000


class _MaskingGuardrail(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(guardrail_name="masking")

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        return GenericGuardrailAPIInputs(texts=[text.replace("secret", "[MASKED]") for text in inputs.get("texts", [])])


@pytest.mark.benchmark
def test_extraction_request_guardrail_over_many_field_instructions() -> None:
    data: Final = {
        "model": "invoice-reader",
        "fields": {
            f"field_{index}": {"type": "text", "instructions": f"secret {index}"} for index in range(MANY_FIELDS)
        },
    }

    result: Final = asyncio.run(ExtractionHandler().process_input_messages(data, _MaskingGuardrail()))

    assert result["fields"] == {
        f"field_{index}": {"type": "text", "instructions": f"[MASKED] {index}"} for index in range(MANY_FIELDS)
    }


@pytest.mark.benchmark
def test_extraction_response_guardrail_over_many_text_answers() -> None:
    response: Final = ExtractionResponse.model_validate(
        {
            "model": "invoice-reader",
            "answers": {f"field_{index}": {"type": "text", "value": f"secret {index}"} for index in range(MANY_FIELDS)},
            "truncated": False,
            "usage": {"input_tokens": 52, "output_tokens": 0},
        }
    )

    result: Final = asyncio.run(ExtractionHandler().process_output_response(response, _MaskingGuardrail()))

    assert isinstance(result, ExtractionResponse)
    assert {name: answer.value for name, answer in result.answers.items()} == {
        f"field_{index}": f"[MASKED] {index}" for index in range(MANY_FIELDS)
    }
