use std::{sync::Arc, time::Duration};

use litellm_cache::ExactCacheContext;
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheAccess, CacheEntry, CacheKey, CacheKeyInput, CacheKeyRequest, CacheOptions, CacheScope,
    RequestRewrite, ResponseCache, ResponseCacheRequest,
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
            .request("test", "messages", json!({"prompt":"hello"}))
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
    let input = CacheKeyInput::from_parameters(json!({"model":"deployment", "input":"hello"}));
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
    .request_with_key("test", "responses", input.clone());
    let original = baseline.request_with_key("test", "responses", input);
    assert_eq!(key(&original), key(&controlled));
    assert_eq!(controlled.context.ttl, Some(Duration::from_secs(9)));
    assert_eq!(controlled.max_age, Some(Duration::from_secs(3)));
    assert_eq!(controlled.access, CacheAccess::NONE);
}

#[rstest]
fn preset_keys_keep_isolated_callers_separate() {
    use litellm_cache_response::{CacheOptions, CacheScope};
    let input = CacheKeyInput {
        preset: Some("explicit".into()),
        ..Default::default()
    };
    let first = CacheOptions::new(CacheScope::Isolated("first".into())).request_with_key(
        "test",
        "responses",
        input.clone(),
    );
    let second = CacheOptions::new(CacheScope::Isolated("second".into())).request_with_key(
        "test",
        "responses",
        input.clone(),
    );
    let shared = CacheOptions::new(CacheScope::Shared).request_with_key("test", "responses", input);
    assert_ne!(key(&first), key(&second));
    assert_eq!(key(&shared).as_str(), "explicit");
}

#[rstest]
#[case::logical(false, false)]
#[case::logical_with_rewrite(true, false)]
#[case::preset(false, true)]
#[case::preset_with_rewrite(true, true)]
fn isolated_presets_ignore_rewrites(#[case] rewritten: bool, #[case] preset: bool) {
    let input = |rewritten: bool| CacheKeyInput {
        preset: preset.then(|| "explicit".into()),
        rewritten_request: rewritten.then(|| CacheKeyRequest {
            url: "https://provider.test/infer".into(),
            headers: vec![],
            body: json!({"input":"changed"}),
        }),
        ..CacheKeyInput::from_parameters(json!({"model":"provider"}))
    };
    let options = CacheOptions::new(CacheScope::Isolated("caller".into()));
    let baseline = options
        .clone()
        .request_with_key("test", "responses", input(false));
    let request = options.request_with_key("test", "responses", input(rewritten));
    assert_eq!(
        request.rewrite,
        match rewritten {
            true => RequestRewrite::Rewritten,
            false => RequestRewrite::Unchanged,
        }
    );
    assert_eq!(request.scope, CacheScope::Isolated("caller".into()));
    assert_eq!(key(&request) == key(&baseline), !rewritten || preset);
    assert_eq!(
        request
            .clone()
            .with_context(ExactCacheContext::default())
            .rewrite,
        request.rewrite
    );
}

fn key(request: &ResponseCacheRequest) -> CacheKey {
    ResponseCache::new(Arc::new(InMemoryCache::<CacheEntry>::default())).key(request)
}
