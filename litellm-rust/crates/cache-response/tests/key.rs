use litellm_cache_response::{
    CacheControls, CacheKeyContext, CacheKeyField, CacheKeyInput, CacheKeyParticipation,
    CacheKeyRequest, CacheKeyTransport, get_cache_key, should_use_cache,
};
use rstest::rstest;
use sha2::{Digest, Sha256};

fn field(name: &str, value: Option<&str>) -> CacheKeyField {
    CacheKeyField {
        name: name.into(),
        value: value.map(str::to_owned),
        participation: CacheKeyParticipation::Always,
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
    let input = context.project(CacheKeyInput {
        fields: vec![field("model", Some("deployment")), field("file", None)],
        namespace: namespace.map(str::to_owned),
        ..Default::default()
    });
    let expected = format!("{prefix}{}", hash(preimage));
    assert_eq!(get_cache_key(&input), expected);
}

#[rstest]
#[case::always_without_opt_in(CacheKeyParticipation::Always, false, true)]
#[case::always_with_opt_in(CacheKeyParticipation::Always, true, true)]
#[case::provider_parameter_when_included(CacheKeyParticipation::ProviderOptIn, true, true)]
#[case::provider_parameter_when_excluded(CacheKeyParticipation::ProviderOptIn, false, false)]
#[case::never_without_opt_in(CacheKeyParticipation::Never, false, false)]
#[case::never_with_opt_in(CacheKeyParticipation::Never, true, false)]
fn keys_hash_api_and_opted_in_provider_parameters(
    #[case] participation: CacheKeyParticipation,
    #[case] include_provider_parameters: bool,
    #[case] hashed: bool,
    #[values(None, Some("x"))] value: Option<&str>,
) {
    let input = CacheKeyInput {
        fields: vec![
            field("model", Some("a")),
            CacheKeyField {
                name: "extra".into(),
                value: value.map(str::to_owned),
                participation,
            },
        ],
        include_provider_parameters,
        ..Default::default()
    };
    let preimage: &[u8] = if hashed && value.is_some() {
        b"model: aextra: x"
    } else {
        b"model: a"
    };
    assert_eq!(get_cache_key(&input), hash(preimage));
}

#[rstest]
#[case::api(true, false, CacheKeyParticipation::Always, true, true)]
#[case::api_and_internal(true, true, CacheKeyParticipation::Always, true, true)]
#[case::provider(false, false, CacheKeyParticipation::ProviderOptIn, false, true)]
#[case::internal(false, true, CacheKeyParticipation::Never, false, false)]
fn legacy_flags_preserve_key_participation(
    #[case] api_parameter: bool,
    #[case] internal_parameter: bool,
    #[case] participation: CacheKeyParticipation,
    #[case] included_without_opt_in: bool,
    #[case] included_with_opt_in: bool,
    #[values(false, true)] include_provider_parameters: bool,
) {
    let decoded: CacheKeyField = serde_json::from_value(serde_json::json!({
        "name": "extra", "value": "x", "api_parameter": api_parameter,
        "internal_parameter": internal_parameter,
    }))
    .unwrap();
    assert_eq!(decoded.participation, participation);
    let input = CacheKeyInput {
        fields: vec![field("model", Some("a")), decoded],
        include_provider_parameters,
        ..Default::default()
    };
    let included = match include_provider_parameters {
        true => included_with_opt_in,
        false => included_without_opt_in,
    };
    let preimage: &[u8] = match included {
        true => b"model: aextra: x",
        false => b"model: a",
    };
    let expected = hash(preimage);
    assert_eq!(get_cache_key(&input), expected);
    let round_trip: CacheKeyInput =
        serde_json::from_value(serde_json::to_value(&input).unwrap()).unwrap();
    assert_eq!(get_cache_key(&round_trip), expected);
}

#[rstest]
#[case::always(CacheKeyParticipation::Always, true, false)]
#[case::provider(CacheKeyParticipation::ProviderOptIn, false, false)]
#[case::never(CacheKeyParticipation::Never, false, true)]
fn participation_serializes_to_compatible_python_fields(
    #[case] participation: CacheKeyParticipation,
    #[case] api_parameter: bool,
    #[case] internal_parameter: bool,
) {
    let field = CacheKeyField {
        name: "extra".into(),
        value: Some("x".into()),
        participation,
    };
    assert_eq!(
        serde_json::to_value(field).unwrap(),
        serde_json::json!({
            "name": "extra", "value": "x", "api_parameter": api_parameter,
            "internal_parameter": internal_parameter,
        })
    );
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
    assert_eq!(get_cache_key(&input), "preset");
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
    assert_eq!(get_cache_key(&input), hash(preimage.as_bytes()));
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
    assert_eq!(get_cache_key(&input), hash(
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
        get_cache_key(&CacheKeyInput::from_parameters(first)),
        get_cache_key(&CacheKeyInput::from_parameters(second)),
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
        get_cache_key(&CacheKeyInput::from_parameters(
            serde_json::json!({"input": first})
        )),
        get_cache_key(&CacheKeyInput::from_parameters(
            serde_json::json!({"input": second})
        )),
    );
}
