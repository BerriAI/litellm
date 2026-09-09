use super::*;

#[test]
fn typed_provider_and_model_select_the_request_contract() {
    let cases = [
        ("mistral", "future-model", OcrProviderKind::Mistral),
        (
            "azure_ai",
            "doc-intelligence/prebuilt-read",
            OcrProviderKind::AzureDocumentIntelligence,
        ),
        ("azure_ai", "future-model", OcrProviderKind::AzureAi),
        ("reducto", "parse-legacy", OcrProviderKind::ReductoLegacy),
        ("reducto", "parse-v3", OcrProviderKind::ReductoV3),
        ("reducto", "future-model", OcrProviderKind::ReductoV3),
    ];

    for (provider, model, expected) in cases {
        let provider = provider.parse().expect("known provider");
        let model = OcrModel::from(model);
        assert_eq!(ocr_provider_config(provider, &model), expected);
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
