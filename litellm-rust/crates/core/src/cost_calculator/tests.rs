use super::calculate_cost;
use super::pricing::init_pricing_db;
use super::types::*;

fn setup_test_pricing() {
    let mut db = PricingDatabase::new();

    db.insert(
        "gpt-4o".to_string(),
        ModelPricing {
            litellm_provider: Some("openai".to_string()),
            input_cost_per_token: Some(2.5e-6),
            output_cost_per_token: Some(1e-5),
            cache_read_input_tokens_cost: Some(1.25e-6),
            cache_creation_input_token_cost: None,
            cache_creation_input_token_cost_above_1hr: None,
            input_cost_per_token_priority: Some(4.25e-6),
            output_cost_per_token_priority: Some(1.7e-5),
            input_cost_per_token_flex: None,
            output_cost_per_token_flex: None,
            input_cost_per_token_above_128k_tokens: None,
            output_cost_per_token_above_128k_tokens: None,
            input_cost_per_token_above_200k_tokens: None,
            output_cost_per_token_above_200k_tokens: None,
            cache_creation_input_token_cost_above_200k_tokens: None,
            cache_read_input_tokens_cost_above_200k_tokens: None,
            output_cost_per_reasoning_token: None,
            input_cost_per_audio_token: None,
            output_cost_per_audio_token: None,
            input_cost_per_character: None,
            output_cost_per_character: None,
        },
    );

    db.insert(
        "claude-3-opus".to_string(),
        ModelPricing {
            litellm_provider: Some("anthropic".to_string()),
            input_cost_per_token: Some(1.5e-5),
            output_cost_per_token: Some(7.5e-5),
            cache_read_input_tokens_cost: Some(1.5e-6),
            cache_creation_input_token_cost: Some(1.875e-5),
            cache_creation_input_token_cost_above_1hr: Some(3e-5),
            input_cost_per_token_priority: None,
            output_cost_per_token_priority: None,
            input_cost_per_token_flex: None,
            output_cost_per_token_flex: None,
            input_cost_per_token_above_128k_tokens: None,
            output_cost_per_token_above_128k_tokens: None,
            input_cost_per_token_above_200k_tokens: Some(6e-6),
            output_cost_per_token_above_200k_tokens: Some(9e-5),
            cache_creation_input_token_cost_above_200k_tokens: Some(7.5e-6),
            cache_read_input_tokens_cost_above_200k_tokens: Some(6e-7),
            output_cost_per_reasoning_token: None,
            input_cost_per_audio_token: None,
            output_cost_per_audio_token: None,
            input_cost_per_character: None,
            output_cost_per_character: None,
        },
    );

    db.insert(
        "gpt-4o-audio".to_string(),
        ModelPricing {
            litellm_provider: Some("openai".to_string()),
            input_cost_per_token: Some(2.5e-6),
            output_cost_per_token: Some(1e-5),
            cache_read_input_tokens_cost: None,
            cache_creation_input_token_cost: None,
            cache_creation_input_token_cost_above_1hr: None,
            input_cost_per_token_priority: None,
            output_cost_per_token_priority: None,
            input_cost_per_token_flex: None,
            output_cost_per_token_flex: None,
            input_cost_per_token_above_128k_tokens: None,
            output_cost_per_token_above_128k_tokens: None,
            input_cost_per_token_above_200k_tokens: None,
            output_cost_per_token_above_200k_tokens: None,
            cache_creation_input_token_cost_above_200k_tokens: None,
            cache_read_input_tokens_cost_above_200k_tokens: None,
            output_cost_per_reasoning_token: None,
            input_cost_per_audio_token: Some(1e-5),
            output_cost_per_audio_token: Some(2e-5),
            input_cost_per_character: None,
            output_cost_per_character: None,
        },
    );

    db.insert(
        "o1-preview".to_string(),
        ModelPricing {
            litellm_provider: Some("openai".to_string()),
            input_cost_per_token: Some(1.5e-5),
            output_cost_per_token: Some(6e-5),
            cache_read_input_tokens_cost: None,
            cache_creation_input_token_cost: None,
            cache_creation_input_token_cost_above_1hr: None,
            input_cost_per_token_priority: None,
            output_cost_per_token_priority: None,
            input_cost_per_token_flex: None,
            output_cost_per_token_flex: None,
            input_cost_per_token_above_128k_tokens: None,
            output_cost_per_token_above_128k_tokens: None,
            input_cost_per_token_above_200k_tokens: None,
            output_cost_per_token_above_200k_tokens: None,
            cache_creation_input_token_cost_above_200k_tokens: None,
            cache_read_input_tokens_cost_above_200k_tokens: None,
            output_cost_per_reasoning_token: Some(6e-5),
            input_cost_per_audio_token: None,
            output_cost_per_audio_token: None,
            input_cost_per_character: None,
            output_cost_per_character: None,
        },
    );

    init_pricing_db(db);
}

#[test]
fn test_basic_cost_calculation() {
    setup_test_pricing();

    let request = CostRequest {
        model: "gpt-4o",
        usage: Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: None,
    };

    let response = calculate_cost(&request).unwrap();

    let expected_prompt = 100.0 * 2.5e-6;
    let expected_completion = 50.0 * 1e-5;

    assert!((response.prompt_cost_usd - expected_prompt).abs() < 1e-12);
    assert!((response.completion_cost_usd - expected_completion).abs() < 1e-12);
}

#[test]
fn test_cost_with_cache_hit() {
    setup_test_pricing();

    let request = CostRequest {
        model: "gpt-4o",
        usage: Usage {
            prompt_tokens: 1000,
            completion_tokens: 100,
            total_tokens: 1100,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: 0,
                cache_hit_tokens: 800,
                cache_creation_tokens: 0,
                text_tokens: 200,
                audio_tokens: 0,
                image_tokens: 0,
            }),
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: None,
    };

    let response = calculate_cost(&request).unwrap();

    let expected_prompt = (200.0 * 2.5e-6) + (800.0 * 1.25e-6);
    let expected_completion = 100.0 * 1e-5;

    assert!((response.prompt_cost_usd - expected_prompt).abs() < 1e-12);
    assert!((response.completion_cost_usd - expected_completion).abs() < 1e-12);
}

#[test]
fn test_cost_with_cache_creation() {
    setup_test_pricing();

    let request = CostRequest {
        model: "claude-3-opus",
        usage: Usage {
            prompt_tokens: 1000,
            completion_tokens: 100,
            total_tokens: 1100,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: 0,
                cache_hit_tokens: 0,
                cache_creation_tokens: 500,
                text_tokens: 500,
                audio_tokens: 0,
                image_tokens: 0,
            }),
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: None,
    };

    let response = calculate_cost(&request).unwrap();

    let expected_prompt = (500.0 * 1.5e-5) + (500.0 * 1.875e-5);
    let expected_completion = 100.0 * 7.5e-5;

    assert!((response.prompt_cost_usd - expected_prompt).abs() < 1e-12);
    assert!((response.completion_cost_usd - expected_completion).abs() < 1e-12);
}

#[test]
fn test_cost_with_service_tier_priority() {
    setup_test_pricing();

    let request = CostRequest {
        model: "gpt-4o",
        usage: Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: Some("priority"),
    };

    let response = calculate_cost(&request).unwrap();

    let expected_prompt = 100.0 * 4.25e-6;
    let expected_completion = 50.0 * 1.7e-5;

    assert!((response.prompt_cost_usd - expected_prompt).abs() < 1e-12);
    assert!((response.completion_cost_usd - expected_completion).abs() < 1e-12);
}

#[test]
fn test_cost_with_threshold_pricing() {
    setup_test_pricing();

    let request = CostRequest {
        model: "claude-3-opus",
        usage: Usage {
            prompt_tokens: 250_000,
            completion_tokens: 1000,
            total_tokens: 251_000,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: None,
    };

    let response = calculate_cost(&request).unwrap();

    let expected_prompt = 250_000.0 * 6e-6;
    let expected_completion = 1000.0 * 9e-5;

    assert!((response.prompt_cost_usd - expected_prompt).abs() < 1e-9);
    assert!((response.completion_cost_usd - expected_completion).abs() < 1e-9);
}

#[test]
fn test_cost_with_audio_tokens() {
    setup_test_pricing();

    let request = CostRequest {
        model: "gpt-4o-audio",
        usage: Usage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            total_tokens: 1500,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: 0,
                cache_hit_tokens: 0,
                cache_creation_tokens: 0,
                text_tokens: 800,
                audio_tokens: 200,
                image_tokens: 0,
            }),
            completion_tokens_details: Some(CompletionTokensDetails {
                text_tokens: 400,
                audio_tokens: 100,
                reasoning_tokens: 0,
            }),
        },
        custom_llm_provider: None,
        service_tier: None,
    };

    let response = calculate_cost(&request).unwrap();

    let expected_prompt = (800.0 * 2.5e-6) + (200.0 * 1e-5);
    let expected_completion = (400.0 * 1e-5) + (100.0 * 2e-5);

    assert!((response.prompt_cost_usd - expected_prompt).abs() < 1e-12);
    assert!((response.completion_cost_usd - expected_completion).abs() < 1e-12);
}

#[test]
fn test_cost_with_reasoning_tokens() {
    setup_test_pricing();

    let request = CostRequest {
        model: "o1-preview",
        usage: Usage {
            prompt_tokens: 100,
            completion_tokens: 200,
            total_tokens: 300,
            prompt_tokens_details: None,
            completion_tokens_details: Some(CompletionTokensDetails {
                text_tokens: 50,
                audio_tokens: 0,
                reasoning_tokens: 150,
            }),
        },
        custom_llm_provider: None,
        service_tier: None,
    };

    let response = calculate_cost(&request).unwrap();

    let expected_prompt = 100.0 * 1.5e-5;
    let expected_completion = (50.0 * 6e-5) + (150.0 * 6e-5);

    assert!((response.prompt_cost_usd - expected_prompt).abs() < 1e-12);
    assert!((response.completion_cost_usd - expected_completion).abs() < 1e-12);
}

#[test]
fn test_total_cost_calculation() {
    setup_test_pricing();

    let request = CostRequest {
        model: "gpt-4o",
        usage: Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: None,
    };

    let response = calculate_cost(&request).unwrap();
    let total = response.total_cost_usd();

    assert!((total - (response.prompt_cost_usd + response.completion_cost_usd)).abs() < 1e-12);
}

#[test]
fn test_unknown_model_returns_error() {
    setup_test_pricing();

    let request = CostRequest {
        model: "unknown-model",
        usage: Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: None,
    };

    let result = calculate_cost(&request);
    assert!(result.is_err());
}

#[test]
fn test_zero_tokens_zero_cost() {
    setup_test_pricing();

    let request = CostRequest {
        model: "gpt-4o",
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: None,
    };

    let response = calculate_cost(&request).unwrap();

    assert_eq!(response.prompt_cost_usd, 0.0);
    assert_eq!(response.completion_cost_usd, 0.0);
}

#[test]
fn test_provider_prefix_stripping() {
    setup_test_pricing();

    let prefixed_models = [
        "openai/gpt-4o",
        "anthropic/claude-3-opus",
        "bedrock/claude-3-opus",
        "azure/gpt-4o",
    ];

    for model in &prefixed_models {
        let request = CostRequest {
            model: model,
            usage: Usage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            },
            custom_llm_provider: None,
            service_tier: None,
        };

        let result = calculate_cost(&request);
        assert!(
            result.is_ok(),
            "Should find pricing for prefixed model: {model}"
        );
    }
}

#[test]
fn test_provider_prefix_stripping_matches_direct_lookup() {
    setup_test_pricing();

    let direct = calculate_cost(&CostRequest {
        model: "gpt-4o",
        usage: Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: None,
    })
    .unwrap();

    let prefixed = calculate_cost(&CostRequest {
        model: "openai/gpt-4o",
        usage: Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: None,
    })
    .unwrap();

    assert!((direct.total_cost_usd() - prefixed.total_cost_usd()).abs() < 1e-12);
}
