use std::sync::Arc;

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheEntry, CacheKey, CacheKeyInput, CacheOptions, CacheScope, CacheTarget, ResponseCache,
    ResponseCacheConfig,
};
use rstest::rstest;
use serde_json::{Value, json};

fn key_in(namespace: &str, input: CacheKeyInput) -> CacheKey {
    ResponseCache::new(Arc::new(InMemoryCache::<CacheEntry>::default()))
        .with_config(ResponseCacheConfig {
            namespace: namespace.into(),
            ..ResponseCacheConfig::default()
        })
        .key(&CacheOptions::new(CacheScope::Shared).request("responses", input))
}

fn key(input: CacheKeyInput) -> CacheKey {
    key_in("test", input)
}

fn group(name: &str, parameters: Value) -> CacheKeyInput {
    CacheKeyInput::request(CacheTarget::ModelGroup(name.into()), parameters)
}

#[rstest]
#[case::without_namespace("")]
#[case::with_namespace("team")]
fn preset_keys_are_used_verbatim(#[case] namespace: &str) {
    assert_eq!(
        key_in(namespace, CacheKeyInput::Preset("preset".into())).as_str(),
        "preset"
    );
}

#[rstest]
#[case::without_namespace("", "inference-v4:")]
#[case::with_namespace("litellm", "litellm:inference-v4:")]
fn the_namespace_prefixes_generated_keys_exactly_once(
    #[case] namespace: &str,
    #[case] prefix: &str,
) {
    let key = key_in(
        namespace,
        group("litellm", json!({"input": "litellm:inference-v4:x"})),
    );
    let hash = key.as_str().strip_prefix(prefix).unwrap();
    assert_eq!(hash.len(), 64);
    assert!(hash.chars().all(|char| char.is_ascii_hexdigit()));
}

#[rstest]
fn deployments_in_one_model_group_share_a_key() {
    let request = json!({"input": "hello"});
    assert_eq!(
        key(CacheKeyInput::request(
            CacheTarget::resolve(
                Some("gpt-5"),
                "gpt-5",
                Some("azure"),
                Some("https://azure.example")
            ),
            request.clone()
        )),
        key(CacheKeyInput::request(
            CacheTarget::resolve(Some("gpt-5"), "gpt-5", Some("openai"), None),
            request
        )),
    );
}

#[rstest]
#[case::different_groups(
    CacheTarget::resolve(Some("fast"), "gpt-5", Some("openai"), None),
    CacheTarget::resolve(Some("smart"), "gpt-5", Some("openai"), None)
)]
#[case::different_models_without_a_group(
    CacheTarget::resolve(None, "gpt-5", Some("openai"), None),
    CacheTarget::resolve(None, "gpt-5-mini", Some("openai"), None)
)]
#[case::different_providers_without_a_group(
    CacheTarget::resolve(None, "gpt-5", Some("openai"), None),
    CacheTarget::resolve(None, "gpt-5", Some("azure"), None)
)]
#[case::different_api_bases_without_a_group(
    CacheTarget::resolve(None, "gpt-5", Some("openai"), Some("https://a.example")),
    CacheTarget::resolve(None, "gpt-5", Some("openai"), Some("https://b.example"))
)]
#[case::group_and_model_with_the_same_name(
    CacheTarget::resolve(Some("gpt-5"), "gpt-5", None, None),
    CacheTarget::resolve(None, "gpt-5", None, None)
)]
fn distinct_targets_never_share_a_key(#[case] first: CacheTarget, #[case] second: CacheTarget) {
    let request = json!({"input": "hello"});
    assert_ne!(
        key(CacheKeyInput::request(first, request.clone())),
        key(CacheKeyInput::request(second, request)),
    );
}

#[rstest]
fn parameter_keys_ignore_nested_object_order() {
    let first: Value = serde_json::from_str(
        r#"{"messages":[{"role":"user","content":{"text":"hello","detail":1}}]}"#,
    )
    .unwrap();
    let second: Value = serde_json::from_str(
        r#"{"messages":[{"content":{"detail":1,"text":"hello"},"role":"user"}]}"#,
    )
    .unwrap();
    assert_eq!(key(group("g", first)), key(group("g", second)));
}

#[rstest]
#[case::boolean(json!(false), json!("false"))]
#[case::number(json!(1), json!("1"))]
#[case::array_order(json!([1, 2]), json!([2, 1]))]
#[case::nested_null(json!({"a": null}), json!({}))]
fn parameter_keys_preserve_value_identity(#[case] first: Value, #[case] second: Value) {
    assert_ne!(
        key(group("g", json!({"input": first}))),
        key(group("g", json!({"input": second}))),
    );
}

#[rstest]
fn top_level_null_parameters_match_absent_ones() {
    assert_eq!(
        key(group("g", json!({"input": "hello", "user": null}))),
        key(group("g", json!({"input": "hello"}))),
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
    assert_ne!(key(group("g", base)), key(group("g", varied)));
}
