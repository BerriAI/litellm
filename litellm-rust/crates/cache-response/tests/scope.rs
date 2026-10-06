use std::sync::Arc;

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheEntry, CacheKey, CacheKeyInput, CacheScope, CacheTarget, ResponseCache,
};
use rstest::rstest;
use serde_json::json;

fn key(scope: &CacheScope) -> CacheKey {
    ResponseCache::new(Arc::new(InMemoryCache::<CacheEntry>::default())).key(
        &CacheKeyInput::new(
            "messages",
            CacheTarget::ModelGroup("group".into()),
            json!({"prompt": "hello"}),
        ),
        scope,
    )
}

#[rstest]
#[case::same_caller(
    CacheScope::caller("a", "b", "c"),
    CacheScope::caller("a", "b", "c"),
    true
)]
#[case::different_credential(
    CacheScope::caller("a", "b", "c"),
    CacheScope::caller("a", "b", "d"),
    false
)]
#[case::field_boundaries(
    CacheScope::caller("a", "bc", "d"),
    CacheScope::caller("ab", "c", "d"),
    false
)]
#[case::caller_and_shared(CacheScope::caller("a", "b", "c"), CacheScope::Shared, false)]
#[case::empty_caller_and_shared(CacheScope::Caller(String::new()), CacheScope::Shared, false)]
fn only_the_same_scope_shares_a_key(
    #[case] first: CacheScope,
    #[case] second: CacheScope,
    #[case] shared: bool,
) {
    assert_eq!(key(&first) == key(&second), shared);
}
