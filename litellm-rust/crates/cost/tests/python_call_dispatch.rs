use std::collections::HashMap;

use litellm_cost::catalog::{CatalogCallError, CostCall, ModelCostRequest, ModelInfoCatalog};
use litellm_cost::responses_usage::ChatUsage;
use litellm_cost::usage_dispatch::get_usage_object;
use rstest::rstest;
use serde_json::json;

fn request<'a>(
    model: &'a str,
    provider: Option<&'a str>,
    usage: &'a ChatUsage,
) -> ModelCostRequest<'a> {
    ModelCostRequest {
        model,
        provider,
        region: None,
        usage,
        service_tier: None,
        data_residency: None,
        vertex_location: None,
        at: "2026-01-01T12:00Z".parse().unwrap(),
        response_time_ms: None,
    }
}

fn usage() -> ChatUsage {
    get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120
    }}))
    .unwrap()
    .unwrap()
}

fn token<'a>(call_type: &'a str, request_model: Option<&'a str>) -> CostCall<'a> {
    CostCall::Token {
        call_type,
        prompt_characters: None,
        completion_characters: None,
        request_model,
    }
}

#[rstest]
fn token_call_routes_vertex_character_and_preserves_wall_clock_priority() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "vertex_ai/character".to_owned(),
            json!({"input_cost_per_character": 0.002, "output_cost_per_character": 0.003}),
        ),
        (
            "vertex_ai/duration".to_owned(),
            json!({"mode": "chat", "input_cost_per_second": 0.5, "input_cost_per_character": 0.002}),
        ),
    ]));
    let usage = usage();
    assert_eq!(
        catalog
            .cost_per_token_for_call(
                request("character", Some("vertex_ai"), &usage),
                CostCall::Token {
                    call_type: "completion",
                    prompt_characters: Some(10.0),
                    completion_characters: Some(20.0),
                    request_model: None,
                },
            )
            .unwrap(),
        (0.02, 0.06)
    );
    assert_eq!(
        catalog
            .cost_per_token_for_call(
                ModelCostRequest {
                    response_time_ms: Some(2000.0),
                    ..request("duration", Some("vertex_ai"), &usage)
                },
                CostCall::Token {
                    call_type: "completion",
                    prompt_characters: Some(10.0),
                    completion_characters: Some(20.0),
                    request_model: None,
                },
            )
            .unwrap(),
        (1.0, 0.0)
    );
}

#[rstest]
fn token_call_routes_together_embedding_and_azure_router() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "together-ai-embedding-151m-to-350m".to_owned(),
            json!({"input_cost_per_token": 4e-8, "output_cost_per_token": 0.0}),
        ),
        (
            "azure_ai/routed".to_owned(),
            json!({"input_cost_per_token": 0.002, "output_cost_per_token": 0.003}),
        ),
        (
            "azure_ai/model_router".to_owned(),
            json!({"input_cost_per_token": 0.001}),
        ),
    ]));
    let usage = usage();
    assert_eq!(
        catalog
            .cost_per_token_for_call(
                request("model-200m", Some("together_ai"), &usage),
                token("aembedding", None),
            )
            .unwrap(),
        (4e-6, 0.0)
    );
    let (prompt, completion) = catalog
        .cost_per_token_for_call(
            request("routed", Some("azure_ai"), &usage),
            token("completion", Some("azure_ai/model_router")),
        )
        .unwrap();
    assert!((prompt - 0.3).abs() < 1e-12);
    assert!((completion - 0.06).abs() < 1e-12);
}

#[rstest]
fn speech_and_transcription_calls_select_character_token_or_duration_rates() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "openai/speech".to_owned(),
            json!({"input_cost_per_character": 0.002, "input_cost_per_token": 0.003}),
        ),
        (
            "openai/transcribe".to_owned(),
            json!({"input_cost_per_token": 0.003, "output_cost_per_token": 0.004, "input_cost_per_second": 0.01}),
        ),
    ]));
    let usage = usage();
    assert_eq!(
        catalog
            .cost_per_token_for_call(
                request("speech", Some("openai"), &usage),
                CostCall::Speech {
                    prompt_characters: Some(5.0),
                },
            )
            .unwrap(),
        (0.01, 0.0)
    );
    assert_eq!(
        catalog
            .cost_per_token_for_call(
                request("transcribe", Some("openai"), &usage),
                CostCall::Transcription {
                    duration_seconds: 2.0,
                },
            )
            .unwrap(),
        (0.3, 0.08)
    );
    assert_eq!(
        catalog
            .cost_per_token_for_call(
                request("transcribe", Some("openai"), &ChatUsage::default()),
                CostCall::Transcription {
                    duration_seconds: 2.0,
                },
            )
            .unwrap(),
        (0.02, 0.0)
    );
}

#[rstest]
fn retrieval_and_search_calls_use_their_reported_units() {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "cohere/rerank".to_owned(),
            json!({"input_cost_per_query": 0.25}),
        ),
        (
            "vertex_ai/search_api".to_owned(),
            json!({"input_cost_per_query": 0.4}),
        ),
        (
            "exa_ai/search".to_owned(),
            json!({"input_cost_per_query": 0.003}),
        ),
    ]));
    let usage = usage();
    assert_eq!(
        catalog
            .cost_per_token_for_call(
                request("rerank", Some("cohere"), &usage),
                CostCall::Rerank {
                    billed_units: Some(&json!({"search_units": 3})),
                },
            )
            .unwrap(),
        (0.75, 0.0)
    );
    assert_eq!(
        catalog
            .cost_per_token_for_call(
                request("search_api", Some("vertex_ai"), &usage),
                CostCall::VectorStoreSearch {
                    api_type: Some("search_api"),
                },
            )
            .unwrap(),
        (0.4, 0.0)
    );
    assert_eq!(
        catalog
            .cost_per_token_for_call(
                request("search", Some("exa_ai"), &usage),
                CostCall::Search {
                    number_of_queries: Some(0),
                    optional_params: &json!({}),
                },
            )
            .unwrap(),
        (0.003, 0.0)
    );
    assert_eq!(
        catalog.cost_per_token_for_call(
            request("rerank", None, &usage),
            CostCall::Rerank { billed_units: None },
        ),
        Err(CatalogCallError::MissingProvider)
    );
}
