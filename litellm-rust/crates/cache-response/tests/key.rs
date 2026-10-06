use std::sync::Arc;

use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheEntry, CacheKey, CacheKeyField, CacheKeyInput, CacheKeyRequest, CacheKeyTransport,
    ResponseCache, ResponseCacheRequest,
};
use rstest::rstest;
use sha2::{Digest, Sha256};

fn field(name: &str, value: Option<&str>) -> CacheKeyField {
    CacheKeyField {
        name: name.into(),
        value: value.map(str::to_owned),
    }
}

fn key(input: &CacheKeyInput) -> CacheKey {
    ResponseCache::new(Arc::new(InMemoryCache::<CacheEntry>::default()))
        .key(&ResponseCacheRequest::new(input.clone()))
}

fn hash(preimage: &[u8]) -> String {
    format!("{:x}", Sha256::digest(preimage))
}

#[rstest]
#[case::without_namespace(None)]
#[case::with_namespace(Some("team"))]
fn preset_keys_are_used_verbatim(#[case] namespace: Option<&str>) {
    let input = CacheKeyInput {
        fields: vec![field("model", Some("a"))],
        preset: Some("preset".into()),
        namespace: namespace.map(str::to_owned),
        ..Default::default()
    };
    assert_eq!(key(&input).as_str(), "preset");
}

#[rstest]
#[case::unchanged(false)]
#[case::rewritten(true)]
fn typed_transport_preserves_existing_key_bytes(#[case] rewritten: bool) {
    let transport = serde_json::json!({
        "provider": "test", "url": "https://provider.test/infer", "headers": [["x-route", "a"]]
    });
    let request = serde_json::json!({
        "url": "https://provider.test/infer", "headers": [["x-route", "a"]], "body": {"input": "changed"}
    });
    let input = CacheKeyInput {
        fields: vec![field("model", Some("model"))],
        transport: Some(serde_json::from_value::<CacheKeyTransport>(transport.clone()).unwrap()),
        rewritten_request: rewritten
            .then(|| serde_json::from_value::<CacheKeyRequest>(request.clone()).unwrap()),
        ..Default::default()
    };
    let preimage = match rewritten {
        false => format!("model: modeltransport: {transport}"),
        true => format!("model: modeltransport: {transport}wire_changes: {request}"),
    };
    assert_eq!(key(&input).as_str(), hash(preimage.as_bytes()));
}

#[rstest]
fn selected_parameters_use_json_value_encoding() {
    let input = CacheKeyInput::from_parameters(serde_json::json!({
        "model": "logical",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.5,
        "stream": false,
        "max_tokens": null,
    }));
    assert_eq!(key(&input).as_str(), hash(
        br#"messages: [{"content":"hello","role":"user"}]model: "logical"stream: falsetemperature: 0.5"#
    ));
}

#[rstest]
fn parameter_keys_ignore_nested_object_order() {
    let first: serde_json::Value = serde_json::from_str(
        r#"{"model":"logical","messages":[{"role":"user","content":{"text":"hello","detail":1}}]}"#,
    )
    .unwrap();
    let second: serde_json::Value = serde_json::from_str(
        r#"{"messages":[{"content":{"detail":1,"text":"hello"},"role":"user"}],"model":"logical"}"#,
    )
    .unwrap();
    assert_eq!(
        key(&CacheKeyInput::from_parameters(first)),
        key(&CacheKeyInput::from_parameters(second)),
    );
}

#[rstest]
#[case::boolean(serde_json::json!(false), serde_json::json!("false"))]
#[case::number(serde_json::json!(1), serde_json::json!("1"))]
#[case::array_order(serde_json::json!([1, 2]), serde_json::json!([2, 1]))]
fn parameter_keys_preserve_value_identity(
    #[case] first: serde_json::Value,
    #[case] second: serde_json::Value,
) {
    assert_ne!(
        key(&CacheKeyInput::from_parameters(
            serde_json::json!({"input": first})
        )),
        key(&CacheKeyInput::from_parameters(
            serde_json::json!({"input": second})
        )),
    );
}

#[rstest]
#[case::api_parameter("temperature")]
#[case::provider_parameter("top_k")]
#[case::unknown_parameter("x-anything")]
fn every_parameter_changes_the_key(#[case] name: &str) {
    let base = serde_json::json!({"model": "logical", "messages": "hello"});
    let varied = serde_json::json!({"model": "logical", "messages": "hello", name: 1});
    assert_ne!(
        key(&CacheKeyInput::from_parameters(base)),
        key(&CacheKeyInput::from_parameters(varied)),
    );
}
