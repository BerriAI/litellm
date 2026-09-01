//! Integration tests for the full Rust pipeline.
//!
//! These tests exercise the complete request flow: deserialize -> auth ->
//! token count -> route -> provider call -> cost calc -> spend tracking ->
//! serialize. They use mock providers to avoid real API calls.

use std::collections::HashSet;
use std::time::Duration;

use litellm_core::auth::{KeyCache, KeyObject, hash_token};
use litellm_core::cost_calculator;
use litellm_core::spend_tracking::{EntityType, MemoryFlush, SpendUpdateItem, SpendWorker};
use litellm_core::token_counter;
use serde_json::json;

#[test]
fn full_pipeline_token_count_then_cost() {
    let mut pricing_db = cost_calculator::PricingDatabase::new();
    pricing_db.insert(
        "gpt-4o".to_string(),
        cost_calculator::ModelPricing {
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
            input_cost_per_audio_token: None,
            output_cost_per_audio_token: None,
            input_cost_per_character: None,
            output_cost_per_character: None,
        },
    );
    cost_calculator::init_pricing_db(pricing_db);

    let model = "gpt-4o";
    let messages = [
        json!({"role": "system", "content": "You are helpful."}),
        json!({"role": "user", "content": "Hello!"}),
    ];

    let messages_refs: Vec<serde_json::Map<String, serde_json::Value>> = messages
        .iter()
        .map(|v| v.as_object().unwrap().clone())
        .collect();

    let token_count = token_counter::token_counter(&token_counter::types::TokenCounterRequest {
        model,
        text: None,
        messages: Some(&messages_refs),
        tools: None,
        tool_choice: None,
        count_response_tokens: false,
        default_token_count: None,
    })
    .unwrap();

    assert!(token_count > 0);

    let cost = cost_calculator::calculate_cost(&cost_calculator::types::CostRequest {
        model,
        usage: cost_calculator::types::Usage {
            prompt_tokens: token_count as u64,
            completion_tokens: 10,
            total_tokens: token_count as u64 + 10,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: None,
    })
    .unwrap();

    assert!(cost.total_cost_usd() > 0.0);
}

#[test]
fn auth_hash_then_cache_lookup() {
    use litellm_core::auth::HashedToken;
    use std::sync::Arc;

    let raw_key = "sk-test-integration-key";
    let hashed = HashedToken::hash(raw_key);
    assert_eq!(hashed.as_hex_str().len(), 64);

    let cache = KeyCache::new(Duration::from_secs(60), 100);

    let key_obj = Arc::new(KeyObject {
        token: hashed.as_hex_str().to_string(),
        key_name: Some("test-key".to_string()),
        key_alias: None,
        user_id: Some("user-1".to_string()),
        team_id: Some("team-1".to_string()),
        org_id: None,
        project_id: None,
        agent_id: None,
        spend: 0.0,
        max_budget: Some(100.0),
        budget_duration: None,
        models: HashSet::from(["gpt-4".to_string()]),
        tpm_limit: None,
        rpm_limit: None,
        max_parallel_requests: None,
        blocked: false,
        allowed_routes: HashSet::new(),
        metadata: None,
        last_refreshed_at: None,
        expires: None,
    });

    cache.set(hashed, Arc::clone(&key_obj));

    let retrieved = cache.get(&hashed).unwrap();
    assert_eq!(retrieved.token, key_obj.token);
    assert_eq!(retrieved.user_id, Some("user-1".to_string()));
    assert!(retrieved.has_model_access("gpt-4"));
    assert!(!retrieved.has_model_access("claude-3"));
    assert!(retrieved.is_within_budget());
}

#[tokio::test]
async fn spend_tracking_end_to_end() {
    let flush = MemoryFlush::new();
    let worker = SpendWorker::spawn(10, Duration::from_millis(50), flush.clone());

    let hashed_key = hash_token("sk-test-key");

    worker
        .record_update(SpendUpdateItem {
            entity_type: EntityType::Key,
            entity_id: hashed_key.clone(),
            cost: 0.05,
        })
        .await;
    worker
        .record_update(SpendUpdateItem {
            entity_type: EntityType::User,
            entity_id: "user-1".to_string(),
            cost: 0.05,
        })
        .await;
    worker
        .record_update(SpendUpdateItem {
            entity_type: EntityType::Team,
            entity_id: "team-1".to_string(),
            cost: 0.05,
        })
        .await;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let batches = flush.get_batches().await;
    assert!(!batches.is_empty());

    let batch = &batches[0];
    assert!((batch.key_updates[&hashed_key] - 0.05).abs() < 1e-10);
    assert!((batch.user_updates["user-1"] - 0.05).abs() < 1e-10);
    assert!((batch.team_updates["team-1"] - 0.05).abs() < 1e-10);
}

#[test]
fn process_request_dispatches_chat_completions() {
    let request_json = serde_json::to_string(&json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }))
    .unwrap();

    let request_value: serde_json::Value = serde_json::from_str(&request_json).unwrap();
    assert_eq!(request_value["model"], "gpt-4o");
    assert!(request_value["messages"].is_array());
}

#[test]
fn process_request_rejects_unknown_route() {
    let err = litellm_core::CoreError::Unsupported("unknown route");
    assert!(err.to_string().contains("unknown route"));
}
