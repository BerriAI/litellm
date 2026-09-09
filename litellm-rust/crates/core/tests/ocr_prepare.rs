use super::*;

#[test]
fn typed_provider_and_model_select_the_request_contract() {
    let cases = [
        ("mistral", "future-model", OcrIntegrationKind::Mistral),
        (
            "azure_ai",
            "doc-intelligence/prebuilt-read",
            OcrIntegrationKind::AzureDocumentIntelligence,
        ),
        ("azure_ai", "future-model", OcrIntegrationKind::AzureMistral),
        (
            "vertex_ai",
            "deepseek-ocr",
            OcrIntegrationKind::VertexDeepSeek,
        ),
        (
            "vertex_ai",
            "future-model",
            OcrIntegrationKind::VertexMistral,
        ),
        ("reducto", "parse-legacy", OcrIntegrationKind::ReductoLegacy),
        ("reducto", "parse-v3", OcrIntegrationKind::ReductoV3),
        ("reducto", "future-model", OcrIntegrationKind::ReductoV3),
    ];

    for (provider, model, expected) in cases {
        let provider = provider.parse().expect("known provider");
        let model = OcrModel::from(model);
        assert_eq!(resolve_ocr_integration(provider, &model), expected);
    }
}

#[test]
fn integration_kind_exposes_its_routing_provider() {
    let cases = [
        (OcrIntegrationKind::Mistral, OcrProvider::Mistral, "mistral"),
        (
            OcrIntegrationKind::AzureMistral,
            OcrProvider::AzureAi,
            "azure_ai",
        ),
        (
            OcrIntegrationKind::AzureDocumentIntelligence,
            OcrProvider::AzureAi,
            "azure_ai",
        ),
        (
            OcrIntegrationKind::VertexMistral,
            OcrProvider::VertexAi,
            "vertex_ai",
        ),
        (
            OcrIntegrationKind::VertexDeepSeek,
            OcrProvider::VertexAi,
            "vertex_ai",
        ),
        (
            OcrIntegrationKind::ReductoV3,
            OcrProvider::Reducto,
            "reducto",
        ),
        (
            OcrIntegrationKind::ReductoLegacy,
            OcrProvider::Reducto,
            "reducto",
        ),
    ];

    for (integration, expected_provider, expected_name) in cases {
        assert_eq!(integration.provider(), expected_provider);
        assert_eq!(integration.provider().as_str(), expected_name);
    }
}

#[test]
fn unknown_models_are_preserved_for_passthrough() {
    let model = OcrModel::from("future-ocr-v9");
    assert_eq!(model, OcrModel::Passthrough("future-ocr-v9".to_string()));
    assert_eq!(model.as_str(), "future-ocr-v9");
}

#[test]
fn unknown_providers_are_rejected_by_the_typed_boundary() {
    assert!("future_provider".parse::<OcrProvider>().is_err());
}
