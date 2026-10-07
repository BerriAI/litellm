use litellm_model_catalog::{
    Catalog, PricingCatalog, Provenance, bundled_pricing_catalog, canonical_provider,
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};

#[fixture]
fn catalog() -> PricingCatalog {
    PricingCatalog::parse(
        &serde_json::to_vec(&json!({
            "openai/model": {"litellm_provider": "openai", "aliases": ["model-short"]},
            "cohere/model": {"litellm_provider": "cohere"},
            "cohere_chat/model": {"litellm_provider": "cohere_chat"},
            "gemini/model": {"litellm_provider": "gemini"},
            "vertex_ai/model": {"litellm_provider": "vertex_ai-language-models"},
            "bedrock/anthropic.model-v1:0": {"litellm_provider": "bedrock"},
            "bedrock_mantle/anthropic.model": {"litellm_provider": "bedrock_mantle"},
            "fireworks_ai/accounts/fireworks/models/model": {"litellm_provider": "fireworks_ai"},
            "azure/model": {"litellm_provider": "azure"},
            "ft:model": {"litellm_provider": "openai"},
            "perplexity/perplexity/model": {"litellm_provider": "perplexity"}
        }))
        .unwrap(),
    )
    .unwrap()
}

#[rstest]
#[case::exact("openai/model", None, None, Some("openai/model"))]
#[case::alias("MODEL-SHORT", None, None, Some("openai/model"))]
#[case::provider("model", Some("openai"), None, Some("openai/model"))]
#[case::canonical("model", Some("gcp.gemini"), None, Some("gemini/model"))]
#[case::legacy("model", None, Some("gemini"), Some("gemini/model"))]
#[case::both("model", Some("gcp.gemini"), Some("gemini"), Some("gemini/model"))]
#[case::ambiguous("model", Some("cohere"), None, None)]
#[case::prefix_narrows("cohere_chat/model", Some("cohere"), None, Some("cohere_chat/model"))]
#[case::legacy_narrows(
    "model",
    Some("cohere"),
    Some("cohere_chat"),
    Some("cohere_chat/model")
)]
#[case::conflict("model", Some("cohere"), Some("openai"), None)]
#[case::dated("model-2026-10-07", Some("openai"), None, Some("openai/model"))]
#[case::version("model-001", Some("gcp.gemini"), None, Some("gemini/model"))]
#[case::vertex_variant("model", Some("gcp.vertex_ai"), None, Some("vertex_ai/model"))]
#[case::bedrock(
    "us.anthropic.model-v1:0",
    Some("aws.bedrock"),
    None,
    Some("bedrock/anthropic.model-v1:0")
)]
#[case::bedrock_route(
    "bedrock/converse/us.anthropic.model-v1:0",
    Some("aws.bedrock"),
    None,
    Some("bedrock/anthropic.model-v1:0")
)]
#[case::mantle_region(
    "bedrock_mantle/us-east-1/anthropic.model",
    None,
    None,
    Some("bedrock_mantle/anthropic.model")
)]
#[case::fireworks(
    "model",
    Some("fireworks_ai"),
    None,
    Some("fireworks_ai/accounts/fireworks/models/model")
)]
#[case::finetune("ft:model:org:suffix:id", Some("openai"), None, Some("ft:model"))]
#[case::repeated_provider(
    "perplexity/model",
    Some("perplexity"),
    None,
    Some("perplexity/perplexity/model")
)]
#[case::unknown("unknown", Some("openai"), None, None)]
#[case::empty_model("", None, None, None)]
#[case::provider_mismatch("openai/model", Some("anthropic"), None, None)]
fn observed_identifiers_resolve_only_one_catalog_row(
    catalog: PricingCatalog,
    #[case] model: &str,
    #[case] provider: Option<&str>,
    #[case] legacy: Option<&str>,
    #[case] expected: Option<&str>,
) {
    assert_eq!(
        catalog
            .resolve(
                model,
                &[provider, legacy].into_iter().flatten().collect::<Vec<_>>()
            )
            .map(|row| row.canonical_key),
        expected
    );
}

#[rstest]
#[case::provider_aliases("model", &["gcp.gemini", "gemini", "gcp.gemini"], Some("gemini/model"))]
#[case::conflicting_providers("model", &["gcp.gemini", "gemini", "openai"], None)]
#[case::empty_provider("openai/model", &["openai", ""], None)]
fn hosting_provider_names_must_agree_through_the_shared_catalog(
    catalog: PricingCatalog,
    #[case] model: &str,
    #[case] providers: &[&str],
    #[case] expected: Option<&str>,
) {
    assert_eq!(
        catalog
            .resolve(model, providers)
            .map(|row| row.canonical_key),
        expected
    );
}

#[rstest]
#[case::number_string(json!("1e-7"))]
#[case::malformed(json!("bad"))]
#[case::boolean(json!(true))]
#[case::null(json!(null))]
#[case::zero(json!(0.0))]
fn pricing_view_retains_uninterpreted_rates(#[case] rate: Value) {
    let source = serde_json::to_vec(&json!({
        "model": {"litellm_provider": "openai", "input_cost_per_token": rate, "aliases": ["alias"]}
    }))
    .unwrap();
    let catalog = PricingCatalog::parse(&source).unwrap();
    let row = catalog.lookup("ALIAS").unwrap();

    assert_eq!(row.canonical_key, "model");
    assert_eq!(row.fields.get("input_cost_per_token"), Some(&rate));
    assert!(!row.fields.contains_key("output_cost_per_token"));
    if rate.is_string() || rate.is_boolean() {
        assert!(Catalog::parse(&source, Provenance::default()).is_err());
    }
}

#[rstest]
fn strict_and_pricing_views_share_alias_resolution() {
    let source = br#"{"Model":{"litellm_provider":"test","aliases":["short"]}}"#;
    let strict = Catalog::parse(source, Provenance::default()).unwrap();
    let pricing = PricingCatalog::parse(source).unwrap();
    let strict_row = strict.lookup("SHORT").unwrap();
    let pricing_row = pricing.lookup("SHORT").unwrap();

    assert_eq!(strict_row.canonical_key, pricing_row.canonical_key);
    assert_eq!(strict_row.matched_key, pricing_row.matched_key);
    assert_eq!(strict_row.entry.fields(), pricing_row.fields);
}

#[rstest]
fn bundled_catalog_uses_the_checked_in_source() {
    let source: Value = serde_json::from_slice(include_bytes!(
        "../../../../model_prices_and_context_window.json"
    ))
    .unwrap();
    let catalog = bundled_pricing_catalog().unwrap();
    let (name, value) = source
        .as_object()
        .unwrap()
        .iter()
        .find(|(name, value)| {
            !matches!(name.as_str(), "sample_spec" | "fallback_generalizations")
                && value.get("input_cost_per_token").is_some()
        })
        .unwrap();

    assert_eq!(
        catalog
            .lookup(name)
            .unwrap()
            .fields
            .get("input_cost_per_token"),
        value.get("input_cost_per_token")
    );
    assert!(std::ptr::eq(catalog, bundled_pricing_catalog().unwrap()));
}

#[rstest]
#[case::as_declared(false)]
#[case::uppercase(true)]
fn provider_mapping_uses_the_same_data_as_the_exporter(#[case] uppercase: bool) {
    let providers: std::collections::BTreeMap<String, String> = serde_json::from_slice(
        include_bytes!("../../../../litellm/integrations/otel/model/providers.json"),
    )
    .unwrap();

    let resolved: std::collections::BTreeMap<_, _> = providers
        .keys()
        .map(|provider| {
            let observed = if uppercase {
                provider.to_uppercase()
            } else {
                provider.clone()
            };
            (provider.clone(), canonical_provider(&observed).to_owned())
        })
        .collect();
    assert_eq!(resolved, providers);
    assert_eq!(canonical_provider("custom-provider"), "custom-provider");
}

#[rstest]
#[case::explicit("azure_ai/model_router/deployment", &[])]
#[case::hinted("my-model-router-deployment", &["azure.ai.inference"])]
#[case::different_serving_provider("azure_ai/model_router/deployment", &["openai"])]
fn route_context_does_not_supply_served_token_rates(
    #[case] model: &str,
    #[case] providers: &[&str],
) {
    let catalog = PricingCatalog::parse(
        br#"{
        "azure_ai/model_router":{"input_cost_per_token":1,"supports_token_only_pricing":false}
    }"#,
    )
    .unwrap();
    let contexts = catalog.request_contexts(model, providers).unwrap();
    assert_eq!(contexts.len(), 1);
    assert_eq!(contexts[0].canonical_key, "azure_ai/model_router");
    assert!(catalog.resolve(model, providers).is_none());
}

#[rstest]
#[case::routed("azure_ai/model-router/deployment", false)]
#[case::opaque("deployment", true)]
#[case::other_provider("openai/model-router/deployment", true)]
#[case::agent_route("azure_ai/agents/model-router", true)]
fn missing_routing_contract_is_distinct_from_an_opaque_alias(
    #[case] model: &str,
    #[case] opaque: bool,
) {
    let catalog = PricingCatalog::parse(br#"{"unrelated":{"litellm_provider":"openai"}}"#).unwrap();
    assert_eq!(
        catalog.request_contexts(model, &[]).map(|rows| rows.len()),
        opaque.then_some(0)
    );
}
