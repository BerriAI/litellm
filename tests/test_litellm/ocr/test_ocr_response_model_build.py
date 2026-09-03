"""`OCRResponse` must be constructible.

The model declares a field literally named `object`, which shadows the builtin
inside the class namespace. Pydantic resolves the deferred annotations against
that namespace, so any *other* field annotated with the builtin `object`
resolves to the string "ocr" and pydantic treats it as an unresolved forward
reference — the model then cannot be built at all.
"""

from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse


def test_ocr_response_can_be_constructed():
    response = OCRResponse(
        pages=[OCRPage(index=0, markdown="# hello")],
        model="mistral-ocr-latest",
    )

    assert response.object == "ocr"
    assert response.pages[0].markdown == "# hello"


def test_ocr_response_accepts_tables_and_key_value_pairs():
    response = OCRResponse(
        pages=[OCRPage(index=0, markdown="# hello")],
        model="mistral-ocr-latest",
        tables=[{"rows": [["a", "b"]]}],
        keyValuePairs=[{"key": "invoice_no", "value": "42"}],
    )

    assert response.tables == [{"rows": [["a", "b"]]}]
    assert response.keyValuePairs == [{"key": "invoice_no", "value": "42"}]
