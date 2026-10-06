use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_cache::ExactCacheContext;
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheEntry, CacheKeyInput, CacheKeyRequest, CacheOptions, CacheScope, RequestRewrite,
    ResponseCache, ResponseCacheConfig, ResponseCacheRequest, ResponseCacheService, get_cache_key,
};
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case::valid(1, "messages", Some(7))]
#[case::unknown_version(2, "messages", None)]
#[case::another_surface(1, "responses", None)]
fn envelopes_require_a_matching_surface_and_version(
    #[case] version: u32,
    #[case] surface: &str,
    #[case] expected: Option<u32>,
) {
    let envelope: litellm_cache_response::ResponseEnvelope<u32> =
        serde_json::from_value(json!({"version":version,"surface":surface,"output":7})).unwrap();
    assert_eq!(envelope.decode("messages"), expected);
}
