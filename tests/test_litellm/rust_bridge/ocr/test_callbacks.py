from litellm.rust_bridge.ocr.callbacks import response as build_ocr_response


def test_rust_ocr_response_retains_provider_native_response():
    provider_response = {"status": "succeeded", "analyzeResult": {"content": "native"}}
    response = build_ocr_response(
        {
            "pages": [],
            "model": "prebuilt-layout",
            "document_annotation": None,
            "usage_info": {"pages_processed": 0},
            "object": "ocr",
            "provider_native_response": provider_response,
        }
    )

    assert response.get_provider_native_response() == provider_response
    assert response.model_dump().get("provider_native_response") is None
