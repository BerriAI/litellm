#![allow(clippy::disallowed_types)]

// mirrors: test_litellm/test_cost_calculator.py::test_select_model_name_keeps_base_model_free_of_region

use std::collections::HashMap;

use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::model_selection::{
    ModelSelectionRequest, get_hidden_str_for_cost_calc, get_provider_for_cost_calc,
    get_response_model, model_contains_known_llm_provider, strip_unregistered_leading_segments,
};
use rstest::rstest;
use serde_json::{Value, json};

fn pricing<'a>(
    catalog: &'a ModelInfoCatalog,
    request: ModelSelectionRequest<'a>,
    logging_details: Option<&'a Value>,
) -> Option<(String, Value)> {
    catalog
        .pricing_entry_for_cost_calc(request, logging_details)
        .map(|selected| (selected.key.into_owned(), selected.info.into_owned()))
}

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
    assert!(model_contains_known_llm_provider("openai/model"));
    assert!(!model_contains_known_llm_provider("team/model"));
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "vertex_ai/claude".to_owned(),
        json!({"input_cost_per_token": 1e-6}),
    )]));
    assert_eq!(
        strip_unregistered_leading_segments("vertex_ai/openai/claude", None, &catalog),
        "vertex_ai/openai/claude"
    );
}

#[rstest]
#[case::provider_prefix(Some("xai/model"), None, Some("xai"))]
#[case::explicit_provider_wins(Some("xai/model"), Some("anthropic"), Some("anthropic"))]
#[case::no_model(None, None, None)]
#[case::unlisted_bare_provider_is_not_inferred(Some("custom-xai"), None, None)]
#[case::unknown_bare_model(Some("unknown"), None, None)]
#[case::bedrock_converse_routes_to_bedrock(Some("converse-model"), None, Some("bedrock"))]
#[case::vertex_family_routes_to_vertex_ai(Some("vertex-model"), None, Some("vertex_ai"))]
#[case::ai21_chat(Some("ai21-model"), None, Some("ai21_chat"))]
#[case::ai21_prefix(Some("ai21/any"), None, Some("ai21_chat"))]
#[case::cohere_chat_under_cohere_prefix(Some("cohere/chat-model"), None, Some("cohere_chat"))]
#[case::azure_hosted_mistral(Some("azure/mistral-model"), None, Some("openai"))]
#[case::anthropic_text(Some("anthropic/claude-2"), None, Some("anthropic_text"))]
#[case::openai_finetune(Some("ft:gpt-4o:org::id"), None, Some("openai"))]
#[case::replicate_version_id(Some(&*format!("owner/model:{}", "a".repeat(64))), None, Some("replicate"))]
#[case::wildcard(Some("*"), None, Some("openai"))]
fn provider_inference_follows_get_llm_provider(
    #[case] model: Option<&str>,
    #[case] explicit: Option<&str>,
    #[case] expected: Option<&str>,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        ("custom-xai".to_owned(), json!({"litellm_provider": "xai"})),
        (
            "converse-model".to_owned(),
            json!({"litellm_provider": "bedrock_converse"}),
        ),
        (
            "vertex-model".to_owned(),
            json!({"litellm_provider": "vertex_ai-language-models"}),
        ),
        (
            "ai21-model".to_owned(),
            json!({"litellm_provider": "ai21", "mode": "chat"}),
        ),
        (
            "chat-model".to_owned(),
            json!({"litellm_provider": "cohere_chat"}),
        ),
        (
            "mistral/mistral-model".to_owned(),
            json!({"litellm_provider": "mistral"}),
        ),
    ]));
    assert_eq!(
        get_provider_for_cost_calc(model, explicit, &catalog).as_deref(),
        expected
    );
}

#[rstest]
fn pricing_entry_prefers_registered_deployment_to_served_model() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "router-id".to_owned(),
            json!({"input_cost_per_token": 0.03}),
        ),
        (
            "openai/served".to_owned(),
            json!({"input_cost_per_token": 0.01}),
        ),
    ]));
    let response = json!({"model": "served"});
    let request = ModelSelectionRequest {
        response: Some(&response),
        custom_pricing: true,
        router_model_id: Some("router-id"),
        ..request(Some("requested"), Some("openai"))
    };
    assert_eq!(
        pricing(&catalog, request, None),
        Some((
            "router-id".to_owned(),
            json!({"input_cost_per_token": 0.03})
        ))
    );
}

#[rstest]
fn pricing_entry_reads_deployment_metadata_before_published_price() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/served".to_owned(),
        json!({"input_cost_per_token": 0.01}),
    )]));
    let response = json!({"model": "served"});
    let logging = json!({"litellm_params": {
        "metadata": {"model_info": {"input_cost_per_token": 0.02}},
        "litellm_metadata": {"model_info": {"input_cost_per_token": 0.03}}
    }});
    let selected = ModelSelectionRequest {
        response: Some(&response),
        custom_pricing: true,
        ..request(Some("requested"), Some("openai"))
    };
    assert_eq!(
        pricing(&catalog, selected, Some(&logging)),
        Some((
            "requested".to_owned(),
            json!({"input_cost_per_token": 0.02})
        ))
    );
    assert_eq!(
        pricing(
            &catalog,
            ModelSelectionRequest {
                custom_pricing: false,
                ..selected
            },
            Some(&logging)
        ),
        Some((
            "openai/served".to_owned(),
            json!({"input_cost_per_token": 0.01})
        ))
    );
}

#[rstest]
fn pricing_entry_falls_back_from_unpriced_base_to_served_model() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/served".to_owned(),
        json!({"input_cost_per_token": 0.01}),
    )]));
    let response = json!({"model": "served"});
    assert_eq!(
        pricing(
            &catalog,
            ModelSelectionRequest {
                response: Some(&response),
                base_model: Some("unpriced"),
                ..request(Some("requested"), Some("openai"))
            },
            None,
        ),
        Some((
            "openai/served".to_owned(),
            json!({"input_cost_per_token": 0.01})
        ))
    );
}

#[rstest]
#[case::no_requested_model(None, Some("openai/gpt-4o"))]
#[case::requested_model_wins(Some("requested"), Some("openai/requested"))]
fn custom_pricing_without_a_priced_router_id_falls_back_to_the_response_model(
    #[case] model: Option<&str>,
    #[case] expected: Option<&str>,
) {
    let catalog = ModelInfoCatalog::new(HashMap::new());
    let response = json!({"model": "gpt-4o"});
    assert_eq!(
        catalog.select_model_name_for_cost_calc(ModelSelectionRequest {
            custom_pricing: true,
            response: Some(&response),
            ..request(model, Some("openai"))
        }),
        expected.map(str::to_owned)
    );
}

// expected value recorded from cost_calculator.py::_select_model_name_for_cost_calc
#[rstest]
fn served_response_model_alone_prices_in_the_request_region() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "bedrock/us-east-1/served".to_owned(),
        json!({"litellm_provider": "bedrock"}),
    )]));
    let response = json!({"model": "served"});
    assert_eq!(
        catalog.select_model_name_for_cost_calc(ModelSelectionRequest {
            response: Some(&response),
            region_name: Some("us-east-1"),
            ..request(Some("requested"), Some("bedrock"))
        }),
        Some("bedrock/us-east-1/served".to_owned())
    );
}

// expected values recorded from cost_calculator.py::_strip_unregistered_leading_segments
#[rstest]
#[case::region_segment_stays_in_the_head(
    "bedrock/us-east-1/x",
    Some("us-east-1"),
    "bedrock/us-east-1/x"
)]
#[case::unmatched_region_segment_is_strippable("bedrock/us-east-1/x", None, "bedrock/x")]
#[case::two_segments_never_take_a_region_head(
    "bedrock/us-east-1",
    Some("us-east-1"),
    "bedrock/us-east-1"
)]
#[case::single_segment_is_unchanged("solo", None, "solo")]
#[case::alias_after_the_region_is_stripped(
    "bedrock/us-west-2/team/y",
    Some("us-west-2"),
    "bedrock/us-west-2/y"
)]
#[case::other_region_keeps_the_model(
    "bedrock/us-west-2/team/y",
    Some("eu-west-1"),
    "bedrock/us-west-2/team/y"
)]
fn strip_unregistered_leading_segments_matches_python(
    #[case] model: &str,
    #[case] region: Option<&str>,
    #[case] expected: &str,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        ("bedrock/x".to_owned(), json!({})),
        ("solo/".to_owned(), json!({})),
        ("bedrock/us-west-2/y".to_owned(), json!({})),
        ("bedrock/us-east-1/".to_owned(), json!({})),
    ]));
    assert_eq!(
        strip_unregistered_leading_segments(model, region, &catalog),
        expected
    );
}
