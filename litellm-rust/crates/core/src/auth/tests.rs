use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use super::cache::KeyCache;
use super::hash::{HashedToken, hash_token, hash_token_if_needed};
use super::types::KeyObject;

fn make_key(token: &str) -> KeyObject {
    KeyObject {
        token: hash_token(token),
        key_name: None,
        key_alias: None,
        user_id: Some("user-1".to_string()),
        team_id: Some("team-1".to_string()),
        org_id: None,
        project_id: None,
        agent_id: None,
        spend: 0.0,
        max_budget: Some(100.0),
        budget_duration: None,
        models: HashSet::from(["gpt-4".to_string(), "gpt-4o".to_string()]),
        tpm_limit: Some(10_000),
        rpm_limit: Some(100),
        max_parallel_requests: None,
        blocked: false,
        allowed_routes: HashSet::new(),
        metadata: None,
        last_refreshed_at: None,
        expires: None,
    }
}

#[test]
fn hash_token_produces_64_char_hex() {
    let hash = hash_token("sk-test-key-123");
    assert_eq!(hash.len(), 64);
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn hash_token_is_deterministic() {
    let hash1 = hash_token("sk-test-key-123");
    let hash2 = hash_token("sk-test-key-123");
    assert_eq!(hash1, hash2);
}

#[test]
fn hash_token_different_inputs_different_hashes() {
    let hash1 = hash_token("sk-key-1");
    let hash2 = hash_token("sk-key-2");
    assert_ne!(hash1, hash2);
}

#[test]
fn hash_token_produces_valid_sha256() {
    use sha2::{Digest, Sha256};

    let input = "sk-1234567890abcdef";
    let hash = hash_token(input);

    let expected_bytes = Sha256::digest(input.as_bytes());
    let mut expected_hex = String::with_capacity(64);
    for b in expected_bytes {
        expected_hex.push_str(&format!("{:02x}", b));
    }

    assert_eq!(hash, expected_hex);
}

#[test]
fn hash_token_if_needed_hashes_sk_prefix() {
    let result = hash_token_if_needed("sk-my-key");
    assert_eq!(result.len(), 64);
    assert_eq!(result, hash_token("sk-my-key"));
}

#[test]
fn hash_token_if_needed_passes_through_already_hashed() {
    let already_hashed = "a".repeat(64);
    let result = hash_token_if_needed(&already_hashed);
    assert_eq!(result, already_hashed);
}

#[test]
fn hashed_token_is_stack_allocated() {
    let token = HashedToken::hash("sk-test");
    assert_eq!(token.as_hex_str().len(), 64);
    assert_eq!(token.as_hex_str(), hash_token("sk-test"));
}

#[test]
fn cache_set_and_get() {
    let cache = KeyCache::new(Duration::from_secs(60), 100);
    let key = Arc::new(make_key("sk-test"));
    let hashed = HashedToken::hash("sk-test");

    cache.set(hashed, Arc::clone(&key));
    let retrieved = cache.get(&hashed).unwrap();

    assert_eq!(retrieved.token, key.token);
    assert_eq!(retrieved.user_id, key.user_id);
    assert_eq!(retrieved.team_id, key.team_id);
}

#[test]
fn cache_miss_returns_none() {
    let cache = KeyCache::new(Duration::from_secs(60), 100);
    let hashed = HashedToken::hash("nonexistent");
    assert!(cache.get(&hashed).is_none());
}

#[test]
fn cache_remove() {
    let cache = KeyCache::new(Duration::from_secs(60), 100);
    let key = Arc::new(make_key("sk-test"));
    let hashed = HashedToken::hash("sk-test");

    cache.set(hashed, Arc::clone(&key));
    assert!(cache.get(&hashed).is_some());

    cache.remove(&hashed);
    assert!(cache.get(&hashed).is_none());
}

#[test]
fn cache_len_and_is_empty() {
    let cache = KeyCache::new(Duration::from_secs(60), 100);
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    let key = Arc::new(make_key("sk-test"));
    let hashed = HashedToken::hash("sk-test");
    cache.set(hashed, key);
    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 1);
}

#[test]
fn cache_evicts_oldest_at_capacity() {
    let cache = KeyCache::new(Duration::from_secs(60), 2);

    let key1 = Arc::new(make_key("sk-key-1"));
    let key2 = Arc::new(make_key("sk-key-2"));
    let key3 = Arc::new(make_key("sk-key-3"));
    let h1 = HashedToken::hash("sk-key-1");
    let h2 = HashedToken::hash("sk-key-2");
    let h3 = HashedToken::hash("sk-key-3");

    cache.set(h1, Arc::clone(&key1));
    cache.set(h2, Arc::clone(&key2));
    assert_eq!(cache.len(), 2);

    cache.set(h3, Arc::clone(&key3));
    assert_eq!(cache.len(), 2);

    assert!(cache.get(&h1).is_none());
    assert!(cache.get(&h2).is_some());
    assert!(cache.get(&h3).is_some());
}

#[test]
fn key_object_model_access_with_allowed_models() {
    let key = make_key("sk-test");
    assert!(key.has_model_access("gpt-4"));
    assert!(key.has_model_access("gpt-4o"));
    assert!(!key.has_model_access("claude-3"));
}

#[test]
fn key_object_model_access_empty_means_all() {
    let mut key = make_key("sk-test");
    key.models = HashSet::new();
    assert!(key.has_model_access("gpt-4"));
    assert!(key.has_model_access("claude-3"));
    assert!(key.has_model_access("any-model"));
}

#[test]
fn key_object_route_access_empty_means_all() {
    let key = make_key("sk-test");
    assert!(key.allowed_routes.is_empty());
    assert!(key.has_route_access("/v1/chat/completions"));
    assert!(key.has_route_access("/v1/models"));
}

#[test]
fn key_object_route_access_with_allowed_routes() {
    let mut key = make_key("sk-test");
    key.allowed_routes = HashSet::from(["/v1/chat/completions".to_string()]);
    assert!(key.has_route_access("/v1/chat/completions"));
    assert!(!key.has_route_access("/v1/models"));
}

#[test]
fn key_object_budget_check_within_budget() {
    let key = make_key("sk-test");
    assert!(key.is_within_budget());
}

#[test]
fn key_object_budget_check_over_budget() {
    let mut key = make_key("sk-test");
    key.spend = 150.0;
    key.max_budget = Some(100.0);
    assert!(!key.is_within_budget());
}

#[test]
fn key_object_budget_check_no_limit() {
    let mut key = make_key("sk-test");
    key.max_budget = None;
    key.spend = 999_999.0;
    assert!(key.is_within_budget());
}

#[test]
fn key_object_not_expired_without_expiry() {
    let key = make_key("sk-test");
    assert!(!key.is_expired());
}

#[test]
fn key_object_expired_with_past_expiry() {
    let mut key = make_key("sk-test");
    key.expires = Some(chrono::Utc::now() - chrono::TimeDelta::seconds(1));
    assert!(key.is_expired());
}

#[test]
fn key_object_not_expired_with_future_expiry() {
    let mut key = make_key("sk-test");
    key.expires = Some(chrono::Utc::now() + chrono::TimeDelta::hours(1));
    assert!(!key.is_expired());
}

#[test]
fn key_object_blocked_check() {
    let mut key = make_key("sk-test");
    assert!(!key.blocked);
    key.blocked = true;
    assert!(key.blocked);
}
