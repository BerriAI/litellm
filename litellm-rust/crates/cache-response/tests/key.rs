use std::sync::Arc;

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheEntry, CacheKey, CacheKeyInput, CacheScope, Deployment, ResponseCache,
};
use rstest::rstest;
use serde_json::{Value, json};

fn key_in(input: CacheKeyInput, model_group: Option<&str>) -> CacheKey {
    ResponseCache::new(Arc::new(InMemoryCache::<CacheEntry>::default())).key(
        &input,
        &CacheScope {
            credential: None,
            model_group: model_group.map(str::to_owned),
        },
    )
}

fn key(input: CacheKeyInput) -> CacheKey {
    key_in(input, None)
}

fn routed(model_group: Option<&str>, deployment: Deployment) -> CacheKey {
    key_in(
        CacheKeyInput::new("responses", deployment, json!({"input": "hello"})),
        model_group,
    )
}

fn input(parameters: Value) -> CacheKeyInput {
    CacheKeyInput::new("responses", Deployment::new("gpt-5", None, None), parameters)
}

#[rstest]
fn deployments_in_one_model_group_share_a_key() {
    assert_eq!(
        routed(
            Some("gpt-5"),
            Deployment::new("gpt-5", Some("azure"), Some("https://azure.example"))
        ),
        routed(Some("gpt-5"), Deployment::new("gpt-5", Some("openai"), None)),
    );
}

#[rstest]
#[case::different_groups(
    (Some("fast"), Deployment::new("gpt-5", Some("openai"), None)),
    (Some("smart"), Deployment::new("gpt-5", Some("openai"), None))
)]
#[case::different_models_without_a_group(
    (None, Deployment::new("gpt-5", Some("openai"), None)),
    (None, Deployment::new("gpt-5-mini", Some("openai"), None))
)]
#[case::different_providers_without_a_group(
    (None, Deployment::new("gpt-5", Some("openai"), None)),
    (None, Deployment::new("gpt-5", Some("azure"), None))
)]
#[case::different_api_bases_without_a_group(
    (None, Deployment::new("gpt-5", Some("openai"), Some("https://a.example"))),
    (None, Deployment::new("gpt-5", Some("openai"), Some("https://b.example")))
)]
#[case::group_and_model_with_the_same_name(
    (Some("gpt-5"), Deployment::new("gpt-5", None, None)),
    (None, Deployment::new("gpt-5", None, None))
)]
fn distinct_targets_never_share_a_key(
    #[case] first: (Option<&str>, Deployment),
    #[case] second: (Option<&str>, Deployment),
) {
    assert_ne!(routed(first.0, first.1), routed(second.0, second.1));
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
    assert_eq!(key(input(first)), key(input(second)));
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
fn top_level_null_parameters_match_absent_ones() {
    assert_eq!(
        key(input(json!({"input": "hello", "user": null}))),
        key(input(json!({"input": "hello"}))),
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
    let deployment = || Deployment::new("gpt-5", None, None);
    assert_ne!(
        key(CacheKeyInput::new(
            "responses",
            deployment(),
            json!({"input": "hello"})
        )),
        key(CacheKeyInput::new(
            "messages",
            deployment(),
            json!({"input": "hello"})
        )),
    );
}
