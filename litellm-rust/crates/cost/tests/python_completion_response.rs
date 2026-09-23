use std::collections::HashMap;

use litellm_cost::catalog::ModelInfoCatalog;
use litellm_cost::completion_input::{CompletionInputRequest, ResponseKind};
use litellm_cost::completion_response::{
    CompletionResponseCostError, CompletionResponseCostRequest, completion_cost_from_response,
};
use litellm_cost::model_selection::ModelSelectionRequest;
use rstest::rstest;
use serde_json::{Value, json};

const PROVIDERS: &[&str] = &["anthropic", "exa_ai", "openai", "recraft"];

fn request<'a>(
    response: Option<&'a Value>,
    model: Option<&'a str>,
    provider: Option<&'a str>,
    discount: &'a Value,
    margin: &'a Value,
) -> CompletionResponseCostRequest<'a> {
    CompletionResponseCostRequest {
        input: CompletionInputRequest {
            model_selection: ModelSelectionRequest {
                model,
                response,
                hidden_params: None,
                base_model: None,
                custom_pricing: false,
                provider,
                router_model_id: None,
                region_name: None,
                known_providers: PROVIDERS,
            },
            call_type: None,
            response_kind: Some(ResponseKind::Completion),
            service_tier: None,
            optional_params: None,
        },
        fallback_usage: None,
        provider,
        region: None,
        data_residency: None,
        vertex_location: None,
        at: "2026-01-01T12:00Z".parse().unwrap(),
        response_time_ms: None,
        prompt_characters: None,
        completion_characters: None,
        transcription_duration_seconds: None,
        request_model: None,
        deployment_info: None,
        image_quality: None,
        image_size: None,
        image_count: None,
        built_in_tool_cost: 0.0,
        additional_costs: &[],
        discount_config: discount,
        margin_config: margin,
    }
}

#[rstest]
fn response_cost_falls_back_to_served_model_and_bills_served_tier() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "anthropic/served".to_owned(),
        json!({
            "input_cost_per_token": 0.01,
            "output_cost_per_token": 0.02,
            "input_cost_per_token_priority": 0.03,
            "output_cost_per_token_priority": 0.04
        }),
    )]));
    let response = json!({
        "model": "served",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "service_tier": "priority"}
    });
    let discount = json!({"anthropic": 0.1});
    let margin = json!({"global": 0.2});
    let auto = json!("auto");
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                model_selection: ModelSelectionRequest {
                    base_model: Some("unpriced"),
                    ..request(
                        Some(&response),
                        Some("requested"),
                        Some("anthropic"),
                        &discount,
                        &margin,
                    )
                    .input
                    .model_selection
                },
                service_tier: Some(&auto),
                ..request(
                    Some(&response),
                    Some("requested"),
                    Some("anthropic"),
                    &discount,
                    &margin,
                )
                .input
            },
            built_in_tool_cost: 0.1,
            additional_costs: &[0.05],
            ..request(
                Some(&response),
                Some("requested"),
                Some("anthropic"),
                &discount,
                &margin,
            )
        },
    )
    .unwrap();
    assert_eq!(result.model, "served");
    assert_eq!(
        result.prepared.model_candidates[0].as_deref(),
        Some("anthropic/unpriced")
    );
    assert_eq!(result.prepared.service_tier.as_deref(), Some("priority"));
    assert!((result.cost.original - 0.65).abs() < 1e-12);
    assert!((result.cost.total - 0.702).abs() < 1e-12);
}

#[rstest]
fn search_response_uses_query_list_without_token_usage() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "exa_ai/search".to_owned(),
        json!({"input_cost_per_query": 0.002}),
    )]));
    let response = json!({"model": "search"});
    let optional_params = json!({"query": ["a", "b", "c"]});
    let empty = json!({});
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("search"),
                optional_params: Some(&optional_params),
                ..request(
                    Some(&response),
                    Some("search"),
                    Some("exa_ai"),
                    &empty,
                    &empty,
                )
                .input
            },
            ..request(
                Some(&response),
                Some("search"),
                Some("exa_ai"),
                &empty,
                &empty,
            )
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.006).abs() < 1e-12);
}

#[rstest]
fn transcription_response_uses_hidden_duration_when_tokens_are_absent() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/transcribe".to_owned(),
        json!({"input_cost_per_second": 0.02}),
    )]));
    let response =
        json!({"model": "transcribe", "usage": {"prompt_tokens": 0, "completion_tokens": 0}});
    let hidden = json!({"audio_transcription_duration": 3.0});
    let empty = json!({});
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("transcription"),
                model_selection: ModelSelectionRequest {
                    hidden_params: Some(&hidden),
                    ..request(
                        Some(&response),
                        Some("transcribe"),
                        Some("openai"),
                        &empty,
                        &empty,
                    )
                    .input
                    .model_selection
                },
                ..request(
                    Some(&response),
                    Some("transcribe"),
                    Some("openai"),
                    &empty,
                    &empty,
                )
                .input
            },
            ..request(
                Some(&response),
                Some("transcribe"),
                Some("openai"),
                &empty,
                &empty,
            )
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.06).abs() < 1e-12);
}

#[rstest]
fn retrieve_batch_response_uses_batch_rates() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/batch".to_owned(),
        json!({
            "input_cost_per_token_batches": 0.01,
            "output_cost_per_token_batches": 0.02
        }),
    )]));
    let response = json!({
        "model": "batch",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    });
    let empty = json!({});
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("retrieve_batch"),
                ..request(
                    Some(&response),
                    Some("batch"),
                    Some("openai"),
                    &empty,
                    &empty,
                )
                .input
            },
            ..request(
                Some(&response),
                Some("batch"),
                Some("openai"),
                &empty,
                &empty,
            )
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.2).abs() < 1e-12);
}

#[rstest]
fn hidden_provider_selects_pricing_and_margin_after_model_fallback() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "anthropic/served".to_owned(),
        json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.0}),
    )]));
    let response =
        json!({"model": "served", "usage": {"prompt_tokens": 10, "completion_tokens": 0}});
    let hidden = json!({"custom_llm_provider": "anthropic"});
    let discount = json!({"anthropic": 0.1});
    let empty = json!({});
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                model_selection: ModelSelectionRequest {
                    hidden_params: Some(&hidden),
                    ..request(
                        Some(&response),
                        Some("requested"),
                        Some("openai"),
                        &discount,
                        &empty,
                    )
                    .input
                    .model_selection
                },
                ..request(
                    Some(&response),
                    Some("requested"),
                    Some("openai"),
                    &discount,
                    &empty,
                )
                .input
            },
            ..request(
                Some(&response),
                Some("requested"),
                Some("openai"),
                &discount,
                &empty,
            )
        },
    )
    .unwrap();
    assert_eq!(result.model, "served");
    assert!((result.cost.total - 0.09).abs() < 1e-12);
}

#[rstest]
fn explicit_base_model_suppresses_regional_catalog_lookup() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "openai/base".to_owned(),
            json!({"input_cost_per_token": 0.01, "output_cost_per_token": 0.0}),
        ),
        (
            "openai/eu/base".to_owned(),
            json!({"input_cost_per_token": 0.02, "output_cost_per_token": 0.0}),
        ),
    ]));
    let response = json!({"usage": {"prompt_tokens": 10, "completion_tokens": 0}});
    let hidden = json!({"region_name": "eu"});
    let empty = json!({});
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                model_selection: ModelSelectionRequest {
                    base_model: Some("base"),
                    hidden_params: Some(&hidden),
                    ..request(
                        Some(&response),
                        Some("requested"),
                        Some("openai"),
                        &empty,
                        &empty,
                    )
                    .input
                    .model_selection
                },
                ..request(
                    Some(&response),
                    Some("requested"),
                    Some("openai"),
                    &empty,
                    &empty,
                )
                .input
            },
            region: Some("eu"),
            ..request(
                Some(&response),
                Some("requested"),
                Some("openai"),
                &empty,
                &empty,
            )
        },
    )
    .unwrap();
    assert_eq!(result.model, "openai/base");
    assert!((result.cost.total - 0.1).abs() < 1e-12);
}

#[rstest]
fn image_response_uses_image_route_before_discount_and_margin() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "recraft/image".to_owned(),
        json!({"output_cost_per_image": 0.2}),
    )]));
    let response = json!({"data": [{}, {}]});
    let discount = json!({"recraft": 0.5});
    let margin = json!({"global": 0.5});
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                response_kind: Some(ResponseKind::ImageGeneration),
                call_type: Some("image_generation"),
                ..request(
                    Some(&response),
                    Some("image"),
                    Some("recraft"),
                    &discount,
                    &margin,
                )
                .input
            },
            ..request(
                Some(&response),
                Some("image"),
                Some("recraft"),
                &discount,
                &margin,
            )
        },
    )
    .unwrap();
    assert!((result.cost.total - 0.4).abs() < 1e-12);
    assert_eq!(result.cost.discount_percent, 0.0);
    assert_eq!(result.cost.margin_percent, 0.0);
}

#[rstest]
fn video_response_prefers_provider_total_without_deployment_and_multiplies_custom_rate() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/video".to_owned(),
        json!({"output_cost_per_second": 0.02, "output_cost_per_second_1080p": 0.08}),
    )]));
    let response = json!({"usage": {
        "duration_seconds": 10.0,
        "video_resolution": " 1080P ",
        "video_count": 2,
        "provider_reported_cost_usd": 0.31
    }});
    let discount = json!({"openai": 0.5});
    let empty = json!({});
    let base = request(
        Some(&response),
        Some("video"),
        Some("openai"),
        &discount,
        &empty,
    );
    let reported = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("create_video"),
                ..base.input
            },
            ..base
        },
    )
    .unwrap();
    assert!((reported.cost.total - 0.31).abs() < 1e-12);
    let deployment = json!({"output_cost_per_video_per_second": 0.05});
    let custom = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("video_edit"),
                model_selection: ModelSelectionRequest {
                    custom_pricing: true,
                    ..base.input.model_selection
                },
                ..base.input
            },
            deployment_info: Some(&deployment),
            ..base
        },
    )
    .unwrap();
    assert!((custom.cost.total - 1.0).abs() < 1e-12);
    assert_eq!(custom.cost.discount_percent, 0.0);
}

#[rstest]
fn video_status_poll_does_not_bill_response_time_as_generation_duration() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/video".to_owned(),
        json!({"mode": "video_generation", "output_cost_per_second": 0.2}),
    )]));
    let response = json!({"status": "completed"});
    let empty = json!({});
    let result = completion_cost_from_response(
        &catalog,
        CompletionResponseCostRequest {
            input: CompletionInputRequest {
                call_type: Some("video_retrieve"),
                ..request(
                    Some(&response),
                    Some("video"),
                    Some("openai"),
                    &empty,
                    &empty,
                )
                .input
            },
            response_time_ms: Some(2000.0),
            ..request(
                Some(&response),
                Some("video"),
                Some("openai"),
                &empty,
                &empty,
            )
        },
    )
    .unwrap();
    assert_eq!(result.cost.total, 0.0);
}

#[rstest]
fn unsupported_call_does_not_silently_bill_as_tokens() {
    let catalog = ModelInfoCatalog::new(HashMap::new());
    let response = json!({"model": "image"});
    let empty = json!({});
    assert_eq!(
        completion_cost_from_response(
            &catalog,
            CompletionResponseCostRequest {
                input: CompletionInputRequest {
                    call_type: Some("image_generation"),
                    ..request(
                        Some(&response),
                        Some("image"),
                        Some("openai"),
                        &empty,
                        &empty
                    )
                    .input
                },
                ..request(
                    Some(&response),
                    Some("image"),
                    Some("openai"),
                    &empty,
                    &empty
                )
            }
        ),
        Err(CompletionResponseCostError::UnsupportedCallType)
    );
}
