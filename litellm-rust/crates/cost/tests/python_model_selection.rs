use std::collections::HashMap;

use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::model_selection::{
    ModelSelectionRequest, get_hidden_str_for_cost_calc, get_response_model,
    model_contains_known_llm_provider, strip_unregistered_leading_segments,
};
use rstest::rstest;
use serde_json::{Value, json};

const PROVIDERS: &[&str] = &["openai", "bedrock", "vertex_ai", "anthropic", "dashscope"];

fn request<'a>(model: Option<&'a str>, provider: Option<&'a str>) -> ModelSelectionRequest<'a> {
    ModelSelectionRequest {
        model,
        response: None,
        hidden_params: None,
        base_model: None,
        custom_pricing: false,
        provider,
        router_model_id: None,
        region_name: None,
        known_providers: PROVIDERS,
    }
}

#[rstest]
#[case(json!({"input_cost_per_token": 0.003}), "openai/router-id")]
#[case(json!({"input_cost_per_second": 0.003}), "openai/router-id")]
#[case(json!({"input_cost_per_query": 0.003}), "openai/router-id")]
#[case(json!({"tiered_pricing": []}), "openai/router-id")]
#[case(json!({"input_cost_per_token": null}), "openai/requested")]
fn custom_pricing_selects_only_a_priced_router_id(
    #[case] router_entry: Value,
    #[case] expected: &str,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([("router-id".to_owned(), router_entry)]));
    assert_eq!(
        catalog.select_model_name_for_cost_calc(ModelSelectionRequest {
            custom_pricing: true,
            router_model_id: Some("router-id"),
            ..request(Some("requested"), Some("openai"))
        }),
        Some(expected.to_owned())
    );
}

#[rstest]
fn explicit_base_model_ignores_private_response_model_and_region() {
    let catalog = ModelInfoCatalog::new(HashMap::new());
    let response = json!({"model": "response-model"});
    let hidden = json!({"provider_response_model": "private-model", "region_name": "eu"});
    assert_eq!(
        catalog.select_model_name_for_cost_calc(ModelSelectionRequest {
            response: Some(&response),
            hidden_params: Some(&hidden),
            base_model: Some("base-model"),
            ..request(Some("requested"), Some("openai"))
        }),
        Some("openai/base-model".to_owned())
    );
}

#[rstest]
fn private_response_model_uses_its_region_and_resolves_alias() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "bedrock/us-east-1/claude".to_owned(),
        json!({"input_cost_per_token": 1e-6}),
    )]));
    let response = json!({"model": "requested"});
    let hidden = json!({
        "provider_response_model": "us-east-1/claude",
        "region_name": "us-east-1"
    });
    assert_eq!(
        catalog.select_model_name_for_cost_calc(ModelSelectionRequest {
            response: Some(&response),
            hidden_params: Some(&hidden),
            ..request(None, Some("bedrock"))
        }),
        Some("bedrock/us-east-1/claude".to_owned())
    );
}

#[rstest]
fn response_alias_strips_unregistered_segment_only_when_a_price_resolves() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "vertex_ai/claude".to_owned(),
        json!({"input_cost_per_token": 1e-6}),
    )]));
    let response = json!({"model": "vertex/claude"});
    assert_eq!(
        catalog.select_model_name_for_cost_calc(ModelSelectionRequest {
            response: Some(&response),
            ..request(None, Some("vertex_ai"))
        }),
        Some("vertex_ai/claude".to_owned())
    );
    assert_eq!(
        catalog.select_model_name_for_cost_calc(ModelSelectionRequest {
            response: Some(&json!({"model": "team/unknown"})),
            ..request(None, Some("vertex_ai"))
        }),
        Some("vertex_ai/team/unknown".to_owned())
    );
}

#[rstest]
fn custom_priced_slash_router_id_keeps_its_own_catalog_entry() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "vertex/claude".to_owned(),
            json!({"input_cost_per_token": 7e-6}),
        ),
        (
            "vertex_ai/claude".to_owned(),
            json!({"input_cost_per_token": 1e-6}),
        ),
    ]));
    assert_eq!(
        catalog.select_model_name_for_cost_calc(ModelSelectionRequest {
            custom_pricing: true,
            router_model_id: Some("vertex/claude"),
            ..request(Some("claude"), Some("vertex_ai"))
        }),
        Some("vertex_ai/vertex/claude".to_owned())
    );
}

#[rstest]
fn response_model_and_hidden_model_precede_requested_model() {
    let catalog = ModelInfoCatalog::new(HashMap::new());
    assert_eq!(
        catalog.select_model_name_for_cost_calc(ModelSelectionRequest {
            response: Some(&json!({"model": "served"})),
            hidden_params: Some(&json!({"model": "hidden"})),
            ..request(Some("requested"), Some("openai"))
        }),
        Some("openai/served".to_owned())
    );
    assert_eq!(
        catalog.select_model_name_for_cost_calc(ModelSelectionRequest {
            response: Some(&json!({})),
            hidden_params: Some(&json!({"model": "hidden"})),
            ..request(Some("requested"), Some("openai"))
        }),
        Some("openai/hidden".to_owned())
    );
    assert_eq!(
        catalog.select_model_name_for_cost_calc(request(Some("requested"), Some("openai"))),
        Some("openai/requested".to_owned())
    );
}

#[rstest]
fn model_selection_helpers_ignore_invalid_hidden_values_and_stop_at_provider_segments() {
    assert_eq!(
        get_response_model(Some(&json!({"model": "served"}))),
        Some("served")
    );
    assert_eq!(
        get_hidden_str_for_cost_calc(Some(&json!({"model": ""})), "model"),
        None
    );
    assert!(model_contains_known_llm_provider("openai/model", PROVIDERS));
    assert!(!model_contains_known_llm_provider("team/model", PROVIDERS));
    let catalog = HashMap::from([(
        "vertex_ai/claude".to_owned(),
        json!({"input_cost_per_token": 1e-6}),
    )]);
    assert_eq!(
        strip_unregistered_leading_segments("vertex_ai/openai/claude", None, PROVIDERS, &catalog,),
        "vertex_ai/openai/claude"
    );
}
