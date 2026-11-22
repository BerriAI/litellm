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
        request_data: dict[str, object],
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
        request_data: dict[str, object],
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

    assert guardrail.seen == (
        "response",
        ("Acme Tools", "12", "hammers", "174", "Invoice from Acme Tools"),
    )
    assert isinstance(result, ExtractionResponse)
    assert result.model_dump(mode="json") == {
        **response.model_dump(mode="json"),
        "answers": {
            "vendor": {"type": "text", "value": "Acme Tools [GUARDRAILED]", "guess": "Acme Tools [GUARDRAILED]"},
            "items": {"type": "list", "value": ["12 [GUARDRAILED]", "hammers [GUARDRAILED]"]},
            "total": {"type": "number", "value": None},
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


@pytest.mark.asyncio
async def test_a_response_guardrail_that_masks_a_number_nulls_every_answer_carrying_it() -> None:
    request_data: Final = {
        "fields": {
            "phone_text": {"type": "text"},
            "phone_number": {"type": "number"},
            "phone_list": {"type": "list", "items": "number"},
            "phone_mean": {"type": "estimate"},
            "phone_median": {"type": "estimate"},
            "phone_p10": {"type": "estimate"},
            "phone_p90": {"type": "estimate"},
            "phone_quantile": {"type": "estimate"},
            "phone_forecast": {"type": "forecast"},
            "total": {"type": "number"},
        }
    }
    response: Final = ExtractionResponse.model_validate(
        {
            "model": "invoice-reader",
            "answers": {
                "phone_text": {"type": "text", "value": "4155550123", "span": [10, 20]},
                "phone_number": {"type": "number", "value": 4155550123, "guess": 4155550123, "span": [10, 20]},
                "phone_list": {"type": "list", "value": [4155550123], "spans": [[10, 20]]},
                "phone_mean": {
                    "type": "estimate",
                    "value": None,
                    "mean": 4155550123.4,
                    "p10": 4155550000,
                    "p_null": 0.6,
                },
                "phone_median": {
                    "type": "estimate",
                    "value": None,
                    "median": 4155550123,
                    "p90": 4155550200,
                    "p_null": 0.6,
                },
                "phone_p10": {"type": "estimate", "value": None, "p10": 4155550123, "p_null": 0.6},
                "phone_p90": {"type": "estimate", "value": None, "p90": 4155550123.2, "p_null": 0.6},
                "phone_quantile": {
                    "type": "estimate",
                    "value": None,
                    "quantiles": [4155550000, 4155550123, 4155550200],
                    "p_null": 0.6,
                },
                "phone_forecast": {
                    "type": "forecast",
                    "value": None,
                    "steps": [{"value": 12, "p10": 10}, {"value": 13, "p90": 4155550123}],
                    "p_null": 0.6,
                },
                "total": {"type": "number", "value": 174, "guess": 174},
            },
            "truncated": False,
            "usage": {"input_tokens": 52, "output_tokens": 0},
        }
    )

    result: Final = await ExtractionHandler().process_output_response(
        response, _MaskingGuardrail("4155550123", "[PHONE]"), request_data=request_data
    )

    assert isinstance(result, ExtractionResponse)
    assert {name: answer.model_dump(mode="json") for name, answer in result.answers.items()} == {
        "phone_text": {"type": "text", "value": "[PHONE]", "span": [10, 20]},
        "phone_number": {"type": "number", "value": None},
        "phone_list": {"type": "list", "value": None},
        "phone_mean": {"type": "estimate", "value": None, "p_null": 0.6},
        "phone_median": {"type": "estimate", "value": None, "p_null": 0.6},
        "phone_p10": {"type": "estimate", "value": None, "p_null": 0.6},
        "phone_p90": {"type": "estimate", "value": None, "p_null": 0.6},
        "phone_quantile": {"type": "estimate", "value": None, "p_null": 0.6},
        "phone_forecast": {"type": "forecast", "value": None, "p_null": 0.6},
        "total": {"type": "number", "value": 174, "guess": 174},
    }


@pytest.mark.asyncio
async def test_a_response_guardrail_does_not_scan_the_fraction_digits_of_an_estimate() -> None:
    days_late: Final = {
        "type": "estimate",
        "value": None,
        "mean": 3.4525830151341586,
        "median": 2.4525830151341586,
        "p_null": 0.2,
    }
    response: Final = ExtractionResponse.model_validate(
        {
            "model": "invoice-reader",
            "answers": {"days_late": days_late},
            "truncated": False,
            "usage": {"input_tokens": 52, "output_tokens": 0},
        }
    )

    result: Final = await ExtractionHandler().process_output_response(
        response,
        _MaskingGuardrail("4525830151341586", "[CARD]"),
        request_data={"fields": {"days_late": {"type": "estimate"}}},
    )

    assert isinstance(result, ExtractionResponse)
    assert result.answers["days_late"].model_dump(mode="json") == days_late


@pytest.mark.asyncio
async def test_a_response_guardrail_sees_the_decimal_text_of_a_fractional_number_answer() -> None:
    response: Final = ExtractionResponse.model_validate(
        {
            "model": "invoice-reader",
            "answers": {
                "unit_price": {"type": "number", "value": 14.5, "guess": 14.5, "span": [30, 36]},
                "prices": {"type": "list", "value": [14.5, 2.25], "spans": [[30, 36], [40, 45]]},
            },
            "truncated": False,
            "usage": {"input_tokens": 52, "output_tokens": 0},
        }
    )

    result: Final = await ExtractionHandler().process_output_response(
        response,
        _MaskingGuardrail("14.5", "[PRICE]"),
        request_data={"fields": {"unit_price": {"type": "number"}, "prices": {"type": "list", "items": "number"}}},
    )

    assert isinstance(result, ExtractionResponse)
    assert {name: answer.model_dump(mode="json") for name, answer in result.answers.items()} == {
        "unit_price": {"type": "number", "value": None},
        "prices": {"type": "list", "value": None},
    }


@pytest.mark.asyncio
async def test_request_enum_options_and_nested_object_property_specs_are_guardrailed_on_a_copy() -> None:
    guardrail: Final = _MaskingGuardrail("acme", "[ORG]")
    data: Final = {
        "model": "invoice-reader",
        "state": "Order from acme",
        "fields": {
            "vendor_tier": {
                "type": "enum",
                "enum": ["acme-gold", "standard", 1],
                "instructions": "Pick tier for acme",
            },
            "item": {
                "type": "object",
                "instructions": "Item from acme",
                "properties": {
                    "sku": {"type": "text", "instructions": "SKU for acme"},
                    "amount": {"type": "number", "unit": "acme-usd"},
                    "channel": {"type": "enum", "enum": ["acme-direct", "partner"]},
                },
            },
            "rows": {
                "type": "list",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "text", "instructions": "Label for acme"},
                        "tag": {"type": "enum", "enum": ["acme-vip", "other"]},
                    },
                },
            },
        },
    }

    result: Final = await ExtractionHandler().process_input_messages(data, guardrail)

    assert result["state"] == "Order from [ORG]"
    assert result["fields"] == {
        "vendor_tier": {
            "type": "enum",
            "enum": ["[ORG]-gold", "standard", 1],
            "instructions": "Pick tier for [ORG]",
        },
        "item": {
            "type": "object",
            "instructions": "Item from [ORG]",
            "properties": {
                "sku": {"type": "text", "instructions": "SKU for [ORG]"},
                "amount": {"type": "number", "unit": "[ORG]-usd"},
                "channel": {"type": "enum", "enum": ["[ORG]-direct", "partner"]},
            },
        },
        "rows": {
            "type": "list",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "text", "instructions": "Label for [ORG]"},
                    "tag": {"type": "enum", "enum": ["[ORG]-vip", "other"]},
                },
            },
        },
    }
    assert data["state"] == "Order from acme"
    assert data["fields"]["vendor_tier"]["enum"] == ["acme-gold", "standard", 1]
    assert data["fields"]["item"]["properties"]["channel"]["enum"] == ["acme-direct", "partner"]


@pytest.mark.asyncio
async def test_response_guardrail_nulls_modified_enum_and_compute_answers_and_keeps_unchanged() -> None:
    request_data: Final = {
        "fields": {
            "payment": {"type": "enum", "enum": ["secret-card", "cash"]},
            "safe_enum": {"type": "enum", "enum": ["cash", "paypal"]},
            "flag": {"type": "boolean"},
            "total": {"type": "compute", "unit": "USD"},
            "safe_compute": {"type": "compute", "unit": "USD"},
        }
    }
    response: Final = ExtractionResponse.model_validate(
        {
            "model": "clairic-27b",
            "answers": {
                "payment": {"type": "enum", "value": "secret-card", "raw": '"secret-card"'},
                "safe_enum": {"type": "enum", "value": "cash", "raw": '"cash"'},
                "flag": {"type": "boolean", "value": True},
                "total": {"type": "compute", "value": 4155550123, "guess": 4155550123},
                "safe_compute": {"type": "compute", "value": 42, "guess": 42},
            },
            "truncated": False,
            "usage": {"input_tokens": 60, "output_tokens": 12},
        }
    )

    masked_enum: Final = await ExtractionHandler().process_output_response(
        response,
        _MaskingGuardrail("secret-card", "[REDACTED]"),
        request_data=request_data,
    )
    assert isinstance(masked_enum, ExtractionResponse)
    assert masked_enum.answers["payment"].model_dump(mode="json") == {"type": "enum", "value": None}
    assert masked_enum.answers["safe_enum"].model_dump(mode="json") == {
        "type": "enum",
        "value": "cash",
        "raw": '"cash"',
    }
    assert masked_enum.answers["flag"].value is True
    assert masked_enum.answers["safe_compute"].value == 42

    masked_compute: Final = await ExtractionHandler().process_output_response(
        response,
        _MaskingGuardrail("4155550123", "[PHONE]"),
        request_data=request_data,
    )
    assert isinstance(masked_compute, ExtractionResponse)
    assert masked_compute.answers["total"].model_dump(mode="json") == {"type": "compute", "value": None}


@pytest.mark.asyncio
async def test_response_guardrail_masks_object_text_properties_and_nulls_non_text_properties() -> None:
    request_data: Final = {
        "fields": {
            "item": {
                "type": "object",
                "properties": {
                    "name": {"type": "text"},
                    "qty": {"type": "number"},
                    "color": {"type": "enum", "enum": ["secret-red", "blue"]},
                },
            },
            "rows": {
                "type": "list",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "text"},
                        "qty": {"type": "number"},
                    },
                },
            },
        }
    }
    response: Final = ExtractionResponse.model_validate(
        {
            "model": "clairic-27b",
            "answers": {
                "item": {
                    "type": "object",
                    "value": {"name": "jane-mug", "qty": 3, "color": "secret-red"},
                    "raw": '{"name":"jane-mug","qty":3,"color":"secret-red"}',
                },
                "rows": {
                    "type": "list",
                    "value": [{"name": "jane-mug", "qty": 3}, {"name": "plain-cup", "qty": 999}],
                    "property_spans": [{"name": [10, 18], "qty": [8, 9]}, {"name": [25, 34], "qty": [21, 24]}],
                    "raw": '[{"name":"jane-mug","qty":3},{"name":"plain-cup","qty":999}]',
                },
            },
            "state_text": "Order for jane-mug and plain-cup",
            "truncated": False,
            "usage": {"input_tokens": 80, "output_tokens": 20},
        }
    )

    masked_text: Final = await ExtractionHandler().process_output_response(
        response,
        _MaskingGuardrail("jane", "[USER]"),
        request_data=request_data,
    )
    assert isinstance(masked_text, ExtractionResponse)
    assert masked_text.state_text == "Order for [USER]-mug and plain-cup"
    assert masked_text.answers["item"].model_dump(mode="json") == {
        "type": "object",
        "value": {"name": "[USER]-mug", "qty": 3, "color": "secret-red"},
    }
    assert masked_text.answers["rows"].model_dump(mode="json") == {
        "type": "list",
        "value": [{"name": "[USER]-mug", "qty": 3}, {"name": "plain-cup", "qty": 999}],
    }
    assert response.answers["item"].value == {"name": "jane-mug", "qty": 3, "color": "secret-red"}

    masked_enum_prop: Final = await ExtractionHandler().process_output_response(
        response,
        _MaskingGuardrail("secret-red", "[COLOR]"),
        request_data=request_data,
    )
    assert isinstance(masked_enum_prop, ExtractionResponse)
    assert masked_enum_prop.answers["item"].model_dump(mode="json") == {"type": "object", "value": None}

    masked_num_prop: Final = await ExtractionHandler().process_output_response(
        response,
        _MaskingGuardrail("999", "[NUM]"),
        request_data=request_data,
    )
    assert isinstance(masked_num_prop, ExtractionResponse)
    assert masked_num_prop.answers["rows"].model_dump(mode="json") == {"type": "list", "value": None}
