use std::sync::Arc;

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{CacheEntry, CacheKey, CacheKeyInput, CacheScope, ResponseCache};
use rstest::rstest;
use serde_json::{Value, json};

fn key(input: CacheKeyInput) -> CacheKey {
    ResponseCache::new(Arc::new(InMemoryCache::<CacheEntry>::default()))
        .key(&input, &CacheScope::default())
}

fn input(parameters: Value) -> CacheKeyInput {
    CacheKeyInput::new("responses", parameters)
}

#[rstest]
#[case::boolean(json!(false), json!("false"))]
#[case::number(json!(1), json!("1"))]
#[case::array_order(json!([1, 2]), json!([2, 1]))]
#[case::nested_null(json!({"a": null}), json!({}))]
fn parameter_keys_preserve_value_identity(#[case] first: Value, #[case] second: Value) {
    assert_ne!(
        key(input(json!({"input": first}))),
        key(input(json!({"input": second}))),
    );
}

#[rstest]
#[case::api_parameter("temperature")]
#[case::messages_system("system")]
#[case::messages_top_k("top_k")]
#[case::responses_instructions("instructions")]
#[case::unknown_parameter("x-anything")]
fn every_parameter_changes_the_key(#[case] name: &str) {
    let base = json!({"input": "hello"});
    let varied = json!({"input": "hello", name: 1});
    assert_ne!(key(input(base)), key(input(varied)));
}

#[rstest]
fn surfaces_never_share_a_key() {
    assert_ne!(
        key(CacheKeyInput::new("responses", json!({"input": "hello"}))),
        key(CacheKeyInput::new("messages", json!({"input": "hello"}))),
    );
}
