use litellm_cache_response::{
    CacheControls, CacheKeyContext, CacheKeyField, CacheKeyInput, CacheKeyRequest,
    CacheKeyTransport, cache_key, get_cache_key, should_use_cache,
};
use rstest::rstest;
use sha2::{Digest, Sha256};

fn field(name: &str, value: Option<&str>) -> CacheKeyField {
    CacheKeyField {
        name: name.into(),
        value: value.map(str::to_owned),
        api_parameter: true,
        internal_parameter: false,
    }
}

fn hash(preimage: &[u8]) -> String {
    format!("{:x}", Sha256::digest(preimage))
}

#[rstest]
#[case::caching_group_and_checksum(
    CacheKeyContext {
        model_group: Some("group".into()),
        caching_groups: vec![(vec!["group".into()], "['group']".into())],
        file_checksum: Some("checksum".into()),
        ..Default::default()
    },
    Some("team"),
    "team:",
    b"model: ['group']file: checksum".as_slice(),
)]
#[case::model_group_outside_caching_groups(
    CacheKeyContext {
        model_group: Some("group".into()),
        caching_groups: vec![(vec!["other".into()], "['other']".into())],
        file_object_name: Some("object".into()),
        ..Default::default()
    },
    None,
    "",
    b"model: groupfile: object".as_slice(),
)]
#[case::metadata_file_name_before_parameters(
    CacheKeyContext {
        metadata_file_name: Some("metadata".into()),
        parameters_file_name: Some("parameters".into()),
        ..Default::default()
    },
    Some(""),
    "",
    b"model: deploymentfile: metadata".as_slice(),
)]
#[case::parameters_file_name_last(
    CacheKeyContext {
        parameters_file_name: Some("parameters".into()),
        ..Default::default()
    },
    None,
    "",
    b"model: deploymentfile: parameters".as_slice(),
)]
#[case::no_context_keeps_the_request_model(
    CacheKeyContext::default(),
    Some("team"),
    "team:",
    b"model: deployment".as_slice(),
)]
fn keys_match_python_order_groups_files_and_namespaces(
    #[case] context: CacheKeyContext,
    #[case] namespace: Option<&str>,
    #[case] prefix: &str,
    #[case] preimage: &[u8],
) {
    let mut input = CacheKeyInput {
        fields: vec![field("model", Some("deployment")), field("file", None)],
        namespace: namespace.map(str::to_owned),
        ..Default::default()
    };
    context.apply(&mut input);
    let expected = format!("{prefix}{}", hash(preimage));
    assert_eq!(cache_key(&input), expected);
    assert_eq!(get_cache_key(&input), expected);
}

#[rstest]
#[case::api_parameter(true, false, false, true)]
#[case::provider_parameter_when_included(false, false, true, true)]
#[case::provider_parameter_when_excluded(false, false, false, false)]
#[case::internal_parameter_never(false, true, true, false)]
fn keys_hash_api_and_opted_in_provider_parameters(
    #[case] api_parameter: bool,
    #[case] internal_parameter: bool,
    #[case] include_provider_parameters: bool,
    #[case] hashed: bool,
) {
    let input = CacheKeyInput {
        fields: vec![
            field("model", Some("a")),
            CacheKeyField {
                name: "extra".into(),
                value: Some("x".into()),
                api_parameter,
                internal_parameter,
            },
        ],
        include_provider_parameters,
        ..Default::default()
    };
    let preimage: &[u8] = if hashed {
        b"model: aextra: x"
    } else {
        b"model: a"
    };
    assert_eq!(cache_key(&input), hash(preimage));
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
    assert_eq!(cache_key(&input), "preset");
    assert_eq!(get_cache_key(&input), "preset");
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
    assert_eq!(cache_key(&input), hash(preimage.as_bytes()));
}

const ENABLED: CacheControls = CacheControls {
    supported_call_type: true,
    configured: true,
    native_backend: false,
    default_on: true,
    caching: None,
    no_cache: false,
    no_store: false,
    use_cache: false,
};

#[rstest]
#[case::enabled(ENABLED, true, true)]
#[case::default_off(CacheControls { default_on: false, ..ENABLED }, false, false)]
#[case::default_off_with_use_cache(
    CacheControls { default_on: false, use_cache: true, ..ENABLED },
    true,
    true
)]
#[case::no_cache(CacheControls { no_cache: true, ..ENABLED }, false, true)]
#[case::no_store(CacheControls { no_store: true, ..ENABLED }, true, false)]
#[case::no_cache_and_no_store(
    CacheControls { no_cache: true, no_store: true, ..ENABLED },
    false,
    false
)]
#[case::caching_disabled(CacheControls { caching: Some(false), ..ENABLED }, false, false)]
#[case::caching_enabled(CacheControls { caching: Some(true), ..ENABLED }, true, true)]
#[case::unsupported_call_type(
    CacheControls { supported_call_type: false, ..ENABLED },
    false,
    false
)]
#[case::unconfigured(CacheControls { configured: false, ..ENABLED }, false, false)]
fn cache_controls_honor_default_modes_and_directives(
    #[case] controls: CacheControls,
    #[case] reads: bool,
    #[case] writes: bool,
) {
    assert_eq!(controls.reads(), reads);
    assert_eq!(controls.writes(), writes);
    assert_eq!(should_use_cache(controls), reads || writes);
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
    assert_eq!(cache_key(&input), hash(
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
        cache_key(&CacheKeyInput::from_parameters(first)),
        cache_key(&CacheKeyInput::from_parameters(second)),
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
        cache_key(&CacheKeyInput::from_parameters(
            serde_json::json!({"input": first})
        )),
        cache_key(&CacheKeyInput::from_parameters(
            serde_json::json!({"input": second})
        )),
    );
}
