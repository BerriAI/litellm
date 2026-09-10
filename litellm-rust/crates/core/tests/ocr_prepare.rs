use super::*;
use rstest::rstest;

#[rstest]
#[case::mistral_default("mistral", "future-model", OcrIntegrationKind::Mistral)]
#[case::azure_document_intelligence(
    "azure_ai",
    "doc-intelligence/prebuilt-read",
    OcrIntegrationKind::AzureDocumentIntelligence
)]
#[case::azure_mistral("azure_ai", "future-model", OcrIntegrationKind::AzureMistral)]
#[case::reducto_legacy("reducto", "parse-legacy", OcrIntegrationKind::ReductoLegacy)]
#[case::reducto_v3("reducto", "parse-v3", OcrIntegrationKind::ReductoV3)]
#[case::reducto_default("reducto", "future-model", OcrIntegrationKind::ReductoV3)]
fn typed_provider_and_model_select_the_request_contract(
    #[case] provider: &str,
    #[case] model: &str,
    #[case] expected: OcrIntegrationKind,
) {
    let provider = provider.parse().expect("known provider");
    let model = OcrModel::from(model);
    assert_eq!(resolve_ocr_integration(provider, &model), expected);
}

#[rstest]
#[case::mistral(OcrIntegrationKind::Mistral, OcrProvider::Mistral, "mistral")]
#[case::azure_mistral(OcrIntegrationKind::AzureMistral, OcrProvider::AzureAi, "azure_ai")]
#[case::azure_document_intelligence(
    OcrIntegrationKind::AzureDocumentIntelligence,
    OcrProvider::AzureAi,
    "azure_ai"
)]
#[case::reducto_v3(OcrIntegrationKind::ReductoV3, OcrProvider::Reducto, "reducto")]
#[case::reducto_legacy(OcrIntegrationKind::ReductoLegacy, OcrProvider::Reducto, "reducto")]
fn integration_kind_exposes_its_routing_provider(
    #[case] integration: OcrIntegrationKind,
    #[case] expected_provider: OcrProvider,
    #[case] expected_name: &str,
) {
    assert_eq!(integration.provider(), expected_provider);
    assert_eq!(integration.provider().as_str(), expected_name);
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
