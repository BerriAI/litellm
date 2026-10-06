use std::sync::Arc;

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheCredential, CacheEntry, CacheKey, CacheKeyInput, CacheScope, Deployment, ResponseCache,
};
use rstest::rstest;
use serde_json::json;

fn key(credential: Option<CacheCredential>) -> CacheKey {
    ResponseCache::new(Arc::new(InMemoryCache::<CacheEntry>::default())).key(
        &CacheKeyInput::new(
            "messages",
            Deployment::new("claude", None, None),
            json!({"prompt": "hello"}),
        ),
        &CacheScope {
            credential,
            model_group: None,
        },
    )
}

#[rstest]
#[case::same_credential(
    Some(CacheCredential::new("a", "b", "c")),
    Some(CacheCredential::new("a", "b", "c")),
    true
)]
#[case::without_credentials(None, None, true)]
#[case::different_credential_ids(
    Some(CacheCredential::new("a", "b", "c")),
    Some(CacheCredential::new("a", "b", "d")),
    false
)]
#[case::field_boundaries(
    Some(CacheCredential::new("a", "bc", "d")),
    Some(CacheCredential::new("ab", "c", "d")),
    false
)]
#[case::credential_and_none(Some(CacheCredential::new("a", "b", "c")), None, false)]
#[case::empty_credential_and_none(Some(CacheCredential::new("", "", "")), None, false)]
fn only_the_same_credential_shares_a_key(
    #[case] first: Option<CacheCredential>,
    #[case] second: Option<CacheCredential>,
    #[case] shared: bool,
) {
    assert_eq!(key(first) == key(second), shared);
}
