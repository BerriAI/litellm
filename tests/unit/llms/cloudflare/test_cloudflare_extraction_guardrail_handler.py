from typing import Final, Literal

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.cloudflare.extraction.guardrail_translation.handler import ExtractionHandler
from litellm.types.extraction import ExtractionResponse
from litellm.types.utils import CallTypes, GenericGuardrailAPIInputs


class _TaggingGuardrail(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(guardrail_name="tagging")
        self.seen: tuple[str, tuple[str, ...]] | None = None
        self.seen_images: tuple[str, ...] = ()

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = tuple(inputs.get("texts", []))
        self.seen = (input_type, texts)
        self.seen_images = tuple(inputs.get("images", []))
        return GenericGuardrailAPIInputs(texts=[f"{text} [GUARDRAILED]" for text in texts])


class _MaskingGuardrail(CustomGuardrail):
    def __init__(self, secret: str, mask: str) -> None:
        super().__init__(guardrail_name="masking")
        self.secret = secret
        self.mask = mask

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        return GenericGuardrailAPIInputs(
            texts=[text.replace(self.secret, self.mask) for text in inputs.get("texts", [])]
        )


@pytest.mark.parametrize("call_type", (CallTypes.extraction, CallTypes.aextraction))
def test_extraction_call_types_resolve_to_the_extraction_handler(call_type: CallTypes) -> None:
    assert get_guardrail_translation_mapping(call_type) is ExtractionHandler


@pytest.mark.asyncio
async def test_request_state_and_field_instructions_are_guardrailed_on_a_copy() -> None:
    guardrail: Final = _TaggingGuardrail()
    data: Final = {
        "model": "invoice-reader",
        "state": "Invoice from Acme Tools, contact jane@acme.example",
        "fields": {
            "vendor": {"type": "text", "instructions": "Who sent the invoice?"},
            "total": {"type": "number", "unit": "USD"},
        },
        "images": ["data:image/png;base64,AA=="],
    }

    result: Final = await ExtractionHandler().process_input_messages(data, guardrail)

    assert guardrail.seen == (
        "request",
        ("Invoice from Acme Tools, contact jane@acme.example", "Who sent the invoice?", "USD"),
    )
    assert {key: value for key, value in result.items() if key != "metadata"} == {
        **{key: value for key, value in data.items() if key != "metadata"},
        "state": "Invoice from Acme Tools, contact jane@acme.example [GUARDRAILED]",
        "fields": {
            "vendor": {"type": "text", "instructions": "Who sent the invoice? [GUARDRAILED]"},
            "total": {"type": "number", "unit": "USD [GUARDRAILED]"},
        },
    }
    assert data["state"] == "Invoice from Acme Tools, contact jane@acme.example"


@pytest.mark.asyncio
async def test_a_request_with_no_text_or_images_skips_the_guardrail() -> None:
    guardrail: Final = _TaggingGuardrail()
    data: Final = {"model": "invoice-reader", "fields": {"total": {"type": "number"}}}

    assert await ExtractionHandler().process_input_messages(data, guardrail) is data
    assert guardrail.seen is None


@pytest.mark.asyncio
async def test_request_images_reach_the_guardrail_as_data_urls() -> None:
    guardrail: Final = _TaggingGuardrail()
    data: Final = {
        "model": "invoice-reader",
        "fields": {"total": {"type": "number"}},
        "images": [{"content_type": "image/png", "base64": "iVBORw0KGgo="}, "data:image/jpeg;base64,/9j/"],
    }

    result: Final = await ExtractionHandler().process_input_messages(data, guardrail)

    assert guardrail.seen == ("request", ())
    assert guardrail.seen_images == ("data:image/png;base64,iVBORw0KGgo=", "data:image/jpeg;base64,/9j/")
    assert result["images"] == data["images"]


@pytest.mark.asyncio
async def test_response_answers_and_state_text_are_guardrailed_on_a_copy_without_stale_spans() -> None:
    guardrail: Final = _TaggingGuardrail()
    response: Final = ExtractionResponse.model_validate(
        {
            "model": "invoice-reader",
            "answers": {
                "vendor": {"type": "text", "value": "Acme Tools", "guess": "Acme Tools", "span": [19, 29]},
                "items": {"type": "list", "value": ["12", "hammers"], "spans": [[51, 53], [54, 61]]},
                "total": {"type": "number", "value": 174, "guess": 174},
                "contact": {"type": "text", "value": None, "p_null": 0.98},
            },
            "truncated": False,
            "state_text": "Invoice from Acme Tools",
            "usage": {"input_tokens": 52, "output_tokens": 0},
        }
    )
    response.set_hidden_params({"response_cost": 0.0000052})

    result: Final = await ExtractionHandler().process_output_response(response, guardrail)

    assert guardrail.seen == ("response", ("Acme Tools", "Acme Tools", "12", "hammers", "Invoice from Acme Tools"))
    assert isinstance(result, ExtractionResponse)
    assert result.model_dump(mode="json") == {
        **response.model_dump(mode="json"),
        "answers": {
            "vendor": {"type": "text", "value": "Acme Tools [GUARDRAILED]", "guess": "Acme Tools [GUARDRAILED]"},
            "items": {"type": "list", "value": ["12 [GUARDRAILED]", "hammers [GUARDRAILED]"]},
            "total": {"type": "number", "value": 174, "guess": 174},
            "contact": {"type": "text", "value": None, "p_null": 0.98},
        },
        "state_text": "Invoice from Acme Tools [GUARDRAILED]",
    }
    assert response.answers["vendor"].value == "Acme Tools"
    assert result.hidden_params["response_cost"] == 0.0000052


@pytest.mark.asyncio
async def test_spans_into_the_callers_own_state_survive_a_response_guardrail() -> None:
    response: Final = ExtractionResponse.model_validate(
        {
            "model": "invoice-reader",
            "answers": {"vendor": {"type": "text", "value": "Acme Tools", "span": [19, 29]}},
            "truncated": False,
            "usage": {"input_tokens": 52, "output_tokens": 0},
        }
    )

    result: Final = await ExtractionHandler().process_output_response(response, _TaggingGuardrail())

    assert isinstance(result, ExtractionResponse)
    assert result.answers["vendor"].model_dump(mode="json") == {
        "type": "text",
        "value": "Acme Tools [GUARDRAILED]",
        "span": [19, 29],
    }


@pytest.mark.asyncio
async def test_a_response_guardrail_change_to_a_typed_answer_nulls_it_instead_of_breaking_its_type() -> None:
    guardrail: Final = _MaskingGuardrail("2026-03-03", "[DATE]")
    request_data: Final = {
        "fields": {
            "invoice_date": {"type": "date"},
            "due_dates": {"type": "list", "items": "date"},
            "notes": {"type": "list", "items": "text"},
        }
    }
    response: Final = ExtractionResponse.model_validate(
        {
            "model": "invoice-reader",
            "answers": {
                "invoice_date": {"type": "date", "value": "2026-03-03", "guess": "2026-03-03", "span": [30, 40]},
                "due_dates": {"type": "list", "value": ["2026-03-03", "2026-04-02"], "spans": [[30, 40], [50, 60]]},
                "notes": {"type": "list", "value": ["paid 2026-03-03"], "spans": [[70, 85]]},
                "received": {"type": "date", "value": "2026-02-27", "span": [90, 100]},
            },
            "truncated": False,
            "usage": {"input_tokens": 52, "output_tokens": 0},
        }
    )

    result: Final = await ExtractionHandler().process_output_response(response, guardrail, request_data=request_data)

    assert isinstance(result, ExtractionResponse)
    assert {name: answer.model_dump(mode="json") for name, answer in result.answers.items()} == {
        "invoice_date": {"type": "date", "value": None},
        "due_dates": {"type": "list", "value": None},
        "notes": {"type": "list", "value": ["paid [DATE]"], "spans": [[70, 85]]},
        "received": {"type": "date", "value": "2026-02-27", "span": [90, 100]},
    }
