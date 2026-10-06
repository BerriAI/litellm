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
    CacheAccess, CacheEntry, CacheKeyInput, CacheKeyRequest, CacheOptions, CacheScope, RequestRewrite,
    ResponseCache, ResponseCacheConfig, ResponseCacheRequest, ResponseCacheService, get_cache_key,
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
    use litellm_cache_response::{CacheKeyContext, CacheOptions, CachePolicy, CacheScope};
    let input = CacheKeyInput::from_parameters(json!({"model":"deployment", "input":"hello"}));
    let baseline = CacheOptions {
        key_context: CacheKeyContext {
            model_group: Some("logical".into()),
            ..Default::default()
        },
        ..CacheOptions::new(CacheScope::Shared)
    };
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
    assert_eq!(get_cache_key(&original.key), get_cache_key(&controlled.key));
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
    assert_ne!(get_cache_key(&first.key), get_cache_key(&second.key));
    assert_eq!(get_cache_key(&shared.key), "explicit");
}

#[rstest]
fn retained_logical_parameters_override_provider_transformation() {
    use litellm_cache_response::{CacheKeyContext, CacheOptions, CacheScope};
    let logical = CacheKeyInput::from_parameters(json!({"model":"requested", "max_tokens":32}));
    let context = CacheKeyContext {
        model_group: Some("logical-group".into()),
        ..Default::default()
    };
    let expected = CacheOptions {
        key_context: context.clone(),
        ..CacheOptions::new(CacheScope::Shared)
    }
    .request_with_key("test", "chat_completions", logical.clone());
    let options = CacheOptions {
        key_input: Some(logical),
        key_context: context,
        ..CacheOptions::new(CacheScope::Shared)
    };
    let first = options.clone().request_with_key(
        "test",
        "chat_completions",
        CacheKeyInput::from_parameters(json!({"model":"deployment-a", "max_new_tokens":32})),
    );
    let second = options.request_with_key(
        "test",
        "chat_completions",
        CacheKeyInput::from_parameters(json!({"model":"deployment-b", "max_new_tokens":64})),
    );
    assert_eq!(get_cache_key(&first.key), get_cache_key(&expected.key));
    assert_eq!(get_cache_key(&second.key), get_cache_key(&expected.key));
}

#[rstest]
#[case::logical(false, false)]
#[case::logical_with_rewrite(true, false)]
#[case::preset(false, true)]
#[case::preset_with_rewrite(true, true)]
fn selected_key_input_retains_request_state(#[case] rewritten: bool, #[case] preset: bool) {
    let logical = CacheKeyInput {
        preset: preset.then(|| "explicit".into()),
        ..CacheKeyInput::from_parameters(json!({"model":"logical", "input":"original"}))
    };
    let options = CacheOptions {
        key_input: Some(logical),
        ..CacheOptions::new(CacheScope::Isolated("caller".into()))
    };
    let baseline = options
        .clone()
        .request("test", "responses", json!({"model":"provider"}));
    let request = options.request_with_key(
        "test",
        "responses",
        CacheKeyInput {
            rewritten_request: rewritten.then(|| CacheKeyRequest {
                url: "https://provider.test/infer".into(),
                headers: vec![],
                body: json!({"input":"changed"}),
            }),
            ..CacheKeyInput::from_parameters(json!({"model":"provider"}))
        },
    );
    assert_eq!(
        request.rewrite,
        match rewritten {
            true => RequestRewrite::Rewritten,
            false => RequestRewrite::Unchanged,
        }
    );
    assert_eq!(
        get_cache_key(&request.key) == get_cache_key(&baseline.key),
        !rewritten || preset
    );
    assert_eq!(
        request
            .clone()
            .with_context(ExactCacheContext::default())
            .rewrite,
        request.rewrite
    );
}
