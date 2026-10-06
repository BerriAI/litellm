use std::{sync::Arc, time::Duration};

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheAccess, CacheEntry, CacheKey, CacheKeyInput, CacheTarget, ResponseCache,
    ResponseCacheRequest,
};
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case::same_scope("tenant-a", "tenant-a", true)]
#[case::different_scope("tenant-a", "tenant-b", false)]
#[case::empty_isolated_scope("", "", true)]
#[tokio::test]
async fn isolated_policy_controls_actual_entry_reuse(
    #[case] first: &str,
    #[case] second: &str,
    #[case] hit: bool,
    #[values(false, true)] override_policy: bool,
) {
    use litellm_cache_response::{CachePolicy, CacheScope, ScopedCache};
    let service = Arc::new(ResponseCache::new(Arc::new(
        InMemoryCache::<CacheEntry>::default(),
    )));
    let request = |scope| {
        ScopedCache::new(service.clone(), scope)
            .options(override_policy.then_some(CachePolicy {
                ttl: Some(Duration::from_secs(30)),
                ..CachePolicy::default()
            }))
            .request("messages", input(json!({"prompt":"hello"})))
    };
    service
        .async_store(
            &request(CacheScope::Isolated(first.into())),
            json!({"answer":7}),
            Duration::ZERO,
        )
        .await
        .unwrap();
    assert_eq!(
        service
            .async_lookup(
                &request(CacheScope::Isolated(second.into())),
                Duration::ZERO
            )
            .await
            .unwrap(),
        hit.then(|| json!({"answer":7}))
    );
    assert_eq!(
        service
            .async_lookup(&request(CacheScope::Shared), Duration::ZERO)
            .await
            .unwrap(),
        None
    );
}

#[rstest]
fn policy_does_not_change_logical_identity() {
    use litellm_cache_response::{CacheOptions, CachePolicy, CacheScope};
    let input = input(json!({"input":"hello"}));
    let baseline = CacheOptions::new(CacheScope::Shared);
    let controlled = CacheOptions {
        policy: CachePolicy {
            ttl: Some(Duration::from_secs(9)),
            max_age: Some(Duration::from_secs(3)),
            no_cache: true,
            no_store: true,
            caching: Some(false),
        },
        ..baseline.clone()
    }
    .request("responses", input.clone());
    let original = baseline.request("responses", input);
    assert_eq!(key(&original), key(&controlled));
    assert_eq!(controlled.context.ttl, Some(Duration::from_secs(9)));
    assert_eq!(controlled.max_age, Some(Duration::from_secs(3)));
    assert_eq!(controlled.access, CacheAccess::NONE);
}

#[rstest]
fn preset_keys_keep_isolated_callers_separate() {
    use litellm_cache_response::{CacheOptions, CacheScope};
    let input = CacheKeyInput::Preset("explicit".into());
    let first =
        CacheOptions::new(CacheScope::Isolated("first".into())).request("responses", input.clone());
    let second = CacheOptions::new(CacheScope::Isolated("second".into()))
        .request("responses", input.clone());
    let shared = CacheOptions::new(CacheScope::Shared).request("responses", input);
    assert_ne!(key(&first), key(&second));
    assert_eq!(key(&shared).as_str(), "explicit");
}

#[rstest]
fn surfaces_never_share_a_key() {
    use litellm_cache_response::{CacheOptions, CacheScope};
    let options = CacheOptions::new(CacheScope::Shared);
    assert_ne!(
        key(&options
            .clone()
            .request("responses", input(json!({"input":"hello"})))),
        key(&options.request("messages", input(json!({"input":"hello"})))),
    );
}

fn input(parameters: serde_json::Value) -> CacheKeyInput {
    CacheKeyInput::request(CacheTarget::ModelGroup("group".into()), parameters)
}

fn key(request: &ResponseCacheRequest) -> CacheKey {
    ResponseCache::new(Arc::new(InMemoryCache::<CacheEntry>::default())).key(request)
}
