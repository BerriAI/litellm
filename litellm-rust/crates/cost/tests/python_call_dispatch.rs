#![allow(clippy::disallowed_types)]
// mirrors: test_litellm/test_cost_calculator.py::test_per_query_priced_rerank_deployment_completion_cost_is_nonzero
use litellm_cost::cost_calculator::cost_per_token_for_call;
use litellm_cost::error::CostError;

use std::collections::HashMap;

use litellm_cost::catalog::{CostCall, ModelCostRequest, ModelInfoCatalog};
use litellm_cost::responses_usage::ChatUsage;
use litellm_cost::together_cost::together_pricing_model;
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
        cost_per_token_for_call(
            &catalog,
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
        cost_per_token_for_call(
            &catalog,
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
    let together_model =
        together_pricing_model(&catalog, "model-200m", Some("together_ai"), "aembedding");
    assert_eq!(
        cost_per_token_for_call(
            &catalog,
            request(&together_model, Some("together_ai"), &usage),
            token("aembedding", None),
        )
        .unwrap(),
        (4e-6, 0.0)
    );
    let (prompt, completion) = cost_per_token_for_call(
        &catalog,
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
        cost_per_token_for_call(
            &catalog,
            request("speech", Some("openai"), &usage),
            CostCall::Speech {
                prompt_characters: Some(5.0),
            },
        )
        .unwrap(),
        (0.01, 0.0)
    );
    assert_eq!(
        cost_per_token_for_call(
            &catalog,
            request("transcribe", Some("openai"), &usage),
            CostCall::Transcription {
                duration_seconds: 2.0,
            },
        )
        .unwrap(),
        (0.3, 0.08)
    );
    assert_eq!(
        cost_per_token_for_call(
            &catalog,
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
        cost_per_token_for_call(
            &catalog,
            request("rerank", Some("cohere"), &usage),
            CostCall::Rerank {
                billed_units: Some(&json!({"search_units": 3})),
            },
        )
        .unwrap(),
        (0.75, 0.0)
    );
    assert_eq!(
        cost_per_token_for_call(
            &catalog,
            request("search_api", Some("vertex_ai"), &usage),
            CostCall::VectorStoreSearch {
                api_type: Some("search_api"),
            },
        )
        .unwrap(),
        (0.4, 0.0)
    );
    assert_eq!(
        cost_per_token_for_call(
            &catalog,
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
        cost_per_token_for_call(
            &catalog,
            request("rerank", None, &usage),
            CostCall::Rerank { billed_units: None },
        ),
        Err(CostError::MissingProvider)
    );
}

#[rstest]
fn ocr_call_layers_deployment_and_published_rates_with_credit_priority() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "mistral/ocr".to_owned(),
        json!({
            "ocr_cost_per_credit": 0.25,
            "ocr_cost_per_page": 0.004,
            "annotation_cost_per_page": 0.01
        }),
    )]));
    let usage = usage();
    let deployment = json!({"ocr_cost_per_page": 0.05});
    let credit_response = json!({"usage_info": {
        "credits": 4.0,
        "pages_processed": 3,
        "pages_processed_annotation": 2
    }});
    assert_eq!(
        cost_per_token_for_call(
            &catalog,
            request("ocr", Some("mistral"), &usage),
            CostCall::Ocr {
                response: &credit_response,
                deployment_info: Some(&deployment),
            },
        )
        .unwrap(),
        (1.0, 0.0)
    );
    let page_response = json!({"usage_info": {
        "pages_processed": 3,
        "pages_processed_annotation": 2
    }});
    let (prompt, completion) = cost_per_token_for_call(
        &catalog,
        request("ocr", Some("mistral"), &usage),
        CostCall::Ocr {
            response: &page_response,
            deployment_info: Some(&deployment),
        },
    )
    .unwrap();
    assert!((prompt - 0.17).abs() < 1e-12);
    assert_eq!(completion, 0.0);
}

#[rstest]
fn ocr_call_distinguishes_missing_pages_from_unpriced_usage() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "mistral/priced".to_owned(),
        json!({"ocr_cost_per_page": 0.004}),
    )]));
    let usage = usage();
    let missing_pages = json!({"usage_info": {}});
    assert_eq!(
        cost_per_token_for_call(
            &catalog,
            request("priced", Some("mistral"), &usage),
            CostCall::Ocr {
                response: &missing_pages,
                deployment_info: None,
            },
        ),
        Err(CostError::MissingPages)
    );
    assert_eq!(
        cost_per_token_for_call(
            &catalog,
            request("unpriced", Some("mistral"), &usage),
            CostCall::Ocr {
                response: &missing_pages,
                deployment_info: None,
            },
        )
        .unwrap(),
        (0.0, 0.0)
    );
    assert_eq!(
        cost_per_token_for_call(
            &catalog,
            request("priced", Some("mistral"), &usage),
            CostCall::Ocr {
                response: &json!({}),
                deployment_info: None,
            },
        ),
        Err(CostError::MissingUsage)
    );
}

#[rstest]
fn batch_call_selects_independent_thresholds_modalities_and_region() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/batch".to_owned(),
        json!({
            "input_cost_per_token_batches": 1e-6,
            "input_cost_per_token_above_100_tokens_batches": 2e-6,
            "output_cost_per_token_batches": 3e-6,
            "cache_read_input_token_cost_batches": 0.2e-6,
            "cache_read_input_token_cost_above_100_tokens_batches": 0.4e-6,
            "input_cost_per_audio_token_batches": 5e-6,
            "regional_processing_uplift_multiplier_eu": 1.1
        }),
    )]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "prompt_tokens_details": {"cached_tokens": 20, "audio_tokens": 10}
    }}))
    .unwrap()
    .unwrap();
    let (prompt, completion) = cost_per_token_for_call(
        &catalog,
        ModelCostRequest {
            data_residency: Some("eu"),
            ..request("batch", Some("openai"), &usage)
        },
        CostCall::Batch {
            deployment_info: None,
        },
    )
    .unwrap();
    assert!((prompt - (90.0 * 2e-6 + 10.0 * 5e-6 + 20.0 * 0.4e-6) * 1.1).abs() < 1e-12);
    assert!((completion - 30.0 * 3e-6 * 1.1).abs() < 1e-12);
}

#[rstest]
fn batch_call_falls_back_from_unpriced_deployment_and_uses_inclusive_xai_tier() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "xai/batch".to_owned(),
        json!({
            "input_cost_per_token_batches": 1e-6,
            "input_cost_per_token_above_100_tokens_batches": 2e-6,
            "output_cost_per_token_batches": 3e-6
        }),
    )]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 20
    }}))
    .unwrap()
    .unwrap();
    assert_eq!(
        cost_per_token_for_call(
            &catalog,
            request("batch", Some("xai"), &usage),
            CostCall::Batch {
                deployment_info: Some(&json!({"id": "deployment"})),
            },
        )
        .unwrap(),
        (100.0 * 2e-6, 20.0 * 3e-6)
    );
    assert_eq!(
        cost_per_token_for_call(
            &catalog,
            request("unknown", Some("xai"), &usage),
            CostCall::Batch {
                deployment_info: None,
            },
        )
        .unwrap(),
        (0.0, 0.0)
    );
}

#[rstest]
fn batch_call_parses_k_threshold_and_numeric_string_rate() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/batch".to_owned(),
        json!({
            "input_cost_per_token_batches": 1e-6,
            "input_cost_per_token_above_100k_tokens_batches": "2e-6",
            "output_cost_per_token_batches": 3e-6
        }),
    )]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100_001,
        "completion_tokens": 2
    }}))
    .unwrap()
    .unwrap();
    let (prompt, completion) = cost_per_token_for_call(
        &catalog,
        request("batch", Some("openai"), &usage),
        CostCall::Batch {
            deployment_info: None,
        },
    )
    .unwrap();
    assert!((prompt - 100_001.0 * 2e-6).abs() < 1e-12);
    assert!((completion - 2.0 * 3e-6).abs() < 1e-12);
}

#[rstest]
fn batch_call_invalid_highest_tier_falls_back_to_flat_rate() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/batch".to_owned(),
        json!({
            "input_cost_per_token_batches": 1e-6,
            "input_cost_per_token_above_100_tokens_batches": 2e-6,
            "input_cost_per_token_above_200_tokens_batches": "invalid"
        }),
    )]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 201,
        "completion_tokens": 0
    }}))
    .unwrap()
    .unwrap();
    assert_eq!(
        cost_per_token_for_call(
            &catalog,
            request("batch", Some("openai"), &usage),
            CostCall::Batch {
                deployment_info: None,
            },
        )
        .unwrap(),
        (201.0 * 1e-6, 0.0)
    );
}

#[rstest]
fn generic_token_cost_is_gated_on_token_or_tiered_pricing() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "audio-only/model".to_owned(),
        json!({"input_cost_per_audio_token": 1e-5, "output_cost_per_audio_token": 2e-5}),
    )]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "prompt_tokens_details": {"audio_tokens": 100},
        "completion_tokens_details": {"audio_tokens": 20}
    }}))
    .unwrap()
    .unwrap();
    assert_eq!(
        cost_per_token_for_call(
            &catalog,
            request("audio-only/model", Some("mistral"), &usage),
            token("completion", None),
        )
        .unwrap(),
        (0.0, 0.0),
        "providers without a dedicated calculator bill zero when no token or tiered pricing exists"
    );
    let (input, output) = cost_per_token_for_call(
        &catalog,
        request("audio-only/model", Some("anthropic"), &usage),
        token("completion", None),
    )
    .unwrap();
    assert!(
        (input - 100.0 * 1e-5).abs() < 1e-12 && (output - 20.0 * 2e-5).abs() < 1e-12,
        "anthropic dispatches before the gate in Python and still bills modality rates"
    );
}

#[rstest]
fn anthropic_totals_apply_the_geo_multiplier_other_providers_do_not() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "anthropic/model".to_owned(),
        json!({
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 4e-6,
            "provider_specific_entry": {"us": 1.4}
        }),
    )]));
    let usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "inference_geo": "us"
    }}))
    .unwrap()
    .unwrap();
    let (input, output) = cost_per_token_for_call(
        &catalog,
        request("model", Some("anthropic"), &usage),
        token("completion", None),
    )
    .unwrap();
    assert!((input - 100.0 * 2e-6 * 1.4).abs() < 1e-12);
    assert!((output - 50.0 * 4e-6 * 1.4).abs() < 1e-12);
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/model".to_owned(),
        json!({
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 4e-6,
            "provider_specific_entry": {"us": 1.4}
        }),
    )]));
    let (input, output) = cost_per_token_for_call(
        &catalog,
        request("model", Some("openai"), &usage),
        token("completion", None),
    )
    .unwrap();
    assert!((input - 100.0 * 2e-6).abs() < 1e-12);
    assert!((output - 50.0 * 4e-6).abs() < 1e-12);
}
