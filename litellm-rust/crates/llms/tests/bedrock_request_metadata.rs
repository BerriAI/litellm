use litellm_auth_aws::{SigV4Signer, constants::BEDROCK_SERVICE};
use litellm_http::outbound::OutboundRequest;
use litellm_llms::bedrock::request_metadata::{
    BEDROCK_REQUEST_METADATA_HEADER, BEDROCK_REQUEST_METADATA_MAX_PAIRS, BedrockMetadataSource,
    BedrockRequestMetadataInput, bedrock_request_metadata_headers,
    resolve_bedrock_request_metadata,
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[fixture]
fn input() -> BedrockRequestMetadataInput {
    BedrockRequestMetadataInput {
        allowed_fields: [
            "user_api_key_alias",
            "user_api_key_team_alias",
            "user_api_key_user_email",
            "spend_logs_metadata",
        ]
        .map(str::to_string)
        .to_vec(),
        sources: vec![BedrockMetadataSource {
            identity: vec![
                ("user_api_key_alias".into(), "prod-key".into()),
                ("user_api_key_team_alias".into(), "platform".into()),
            ],
            spend_logs: vec![("cost_center".into(), "cc-1".into())],
        }],
    }
}

#[rstest]
#[case::declared_order(false, false)]
#[case::reversed_clients(false, true)]
#[case::reversed_fields(true, false)]
#[case::both_reversed(true, true)]
fn identity_survives_a_full_client_budget(
    mut input: BedrockRequestMetadataInput,
    #[case] reverse_fields: bool,
    #[case] reverse_clients: bool,
) {
    if reverse_fields {
        input.allowed_fields.reverse();
    }
    input.sources[0].spend_logs = (0..BEDROCK_REQUEST_METADATA_MAX_PAIRS)
        .map(|index| (format!("client_{index:02}"), "v".into()))
        .collect();
    if reverse_clients {
        input.sources[0].spend_logs.reverse();
    }
    let resolved = resolve_bedrock_request_metadata(&input);
    assert_eq!(resolved.len(), BEDROCK_REQUEST_METADATA_MAX_PAIRS);
    assert!(resolved.contains(&("user_api_key_alias".into(), "prod-key".into())));
    assert!(resolved.contains(&("user_api_key_team_alias".into(), "platform".into())));
    assert_eq!(
        resolved
            .iter()
            .filter(|(key, _)| key.starts_with("client_"))
            .count(),
        BEDROCK_REQUEST_METADATA_MAX_PAIRS - 2
    );
}

#[rstest]
fn identity_pairs_follow_the_allowed_fields_order() {
    let input = BedrockRequestMetadataInput {
        allowed_fields: vec![
            "user_api_key_team_alias".into(),
            "user_api_key_alias".into(),
        ],
        sources: vec![BedrockMetadataSource {
            identity: vec![
                ("user_api_key_alias".into(), "prod-key".into()),
                ("user_api_key_team_alias".into(), "platform".into()),
            ],
            spend_logs: vec![],
        }],
    };
    assert_eq!(
        resolve_bedrock_request_metadata(&input),
        vec![
            ("user_api_key_team_alias".into(), "platform".into()),
            ("user_api_key_alias".into(), "prod-key".into()),
        ]
    );
}

#[rstest]
#[case::alias_twice(0, "user_api_key_alias")]
#[case::alias_after_team(2, "user_api_key_alias")]
#[case::team_after_clients(4, "user_api_key_team_alias")]
fn repeated_fields_do_not_consume_slots(
    mut input: BedrockRequestMetadataInput,
    #[case] index: usize,
    #[case] repeated: &str,
) {
    input.allowed_fields.insert(index, repeated.to_string());
    input.sources[0].spend_logs = (0..BEDROCK_REQUEST_METADATA_MAX_PAIRS - 1)
        .map(|index| (format!("client_{index:02}"), "v".into()))
        .collect();
    let resolved = resolve_bedrock_request_metadata(&input);
    let expected: Vec<_> = [
        ("user_api_key_alias".into(), "prod-key".into()),
        ("user_api_key_team_alias".into(), "platform".into()),
    ]
    .into_iter()
    .chain(
        (0..BEDROCK_REQUEST_METADATA_MAX_PAIRS - 2)
            .map(|index| (format!("client_{index:02}"), "v".into())),
    )
    .collect();
    assert_eq!(resolved, expected);
}

#[rstest]
fn client_pairs_merge_sources_in_order_and_dedupe_by_key() {
    let input = BedrockRequestMetadataInput {
        allowed_fields: vec!["spend_logs_metadata".into()],
        sources: vec![
            BedrockMetadataSource {
                identity: vec![],
                spend_logs: vec![
                    ("cost_center".into(), "first".into()),
                    ("region_tag".into(), "r1".into()),
                ],
            },
            BedrockMetadataSource {
                identity: vec![],
                spend_logs: vec![
                    ("cost_center".into(), "second".into()),
                    ("team_tag".into(), "t1".into()),
                ],
            },
        ],
    };
    assert_eq!(
        resolve_bedrock_request_metadata(&input),
        vec![
            ("cost_center".into(), "first".into()),
            ("region_tag".into(), "r1".into()),
            ("team_tag".into(), "t1".into()),
        ]
    );
}

#[rstest]
fn client_pairs_are_dropped_when_spend_logs_metadata_is_not_allowed() {
    let input = BedrockRequestMetadataInput {
        allowed_fields: vec!["user_api_key_alias".into()],
        sources: vec![BedrockMetadataSource {
            identity: vec![("user_api_key_alias".into(), "prod-key".into())],
            spend_logs: vec![("cost_center".into(), "cc-1".into())],
        }],
    };
    assert_eq!(
        resolve_bedrock_request_metadata(&input),
        vec![("user_api_key_alias".into(), "prod-key".into())]
    );
}

#[rstest]
#[case::team("user_api_key_team_alias")]
#[case::organization("user_api_key_org_alias")]
#[case::hash("user_api_key_hash")]
fn client_cannot_forge_reserved_identity(
    mut input: BedrockRequestMetadataInput,
    #[case] key: &str,
) {
    input.sources[0].spend_logs = vec![(key.into(), "forged".into())];
    assert_eq!(
        resolve_bedrock_request_metadata(&input),
        input.sources[0].identity
    );
}

#[rstest]
fn invalid_identity_is_dropped_without_rewriting(mut input: BedrockRequestMetadataInput) {
    input.sources[0].identity = vec![
        ("user_api_key_alias".into(), "prod-key".into()),
        ("user_api_key_team_alias".into(), "O'Brien's team".into()),
        ("user_api_key_user_email".into(), "x".repeat(300)),
    ];
    input.sources[0].spend_logs.clear();
    assert_eq!(
        resolve_bedrock_request_metadata(&input),
        [("user_api_key_alias".into(), "prod-key".into())]
    );
}

#[rstest]
#[case::empty_key(String::new(), "v".to_string())]
#[case::key_too_long("k".repeat(257), "v".to_string())]
#[case::key_with_quote("O'Brien".to_string(), "v".to_string())]
#[case::value_too_long("k".to_string(), "v".repeat(257))]
#[case::value_with_quote("k".to_string(), "O'Brien".to_string())]
fn unforwardable_pairs_are_dropped(#[case] key: String, #[case] value: String) {
    let input = BedrockRequestMetadataInput {
        allowed_fields: vec!["spend_logs_metadata".into()],
        sources: vec![BedrockMetadataSource {
            identity: vec![],
            spend_logs: vec![(key, value), ("kept".into(), "ok".into())],
        }],
    };
    assert_eq!(
        resolve_bedrock_request_metadata(&input),
        vec![("kept".into(), "ok".into())]
    );
}

#[rstest]
#[case::empty_value("")]
#[case::file_separator("\u{1c}")]
fn boundary_values_are_kept(#[case] value: &str) {
    let input = BedrockRequestMetadataInput {
        allowed_fields: vec!["spend_logs_metadata".into()],
        sources: vec![BedrockMetadataSource {
            identity: vec![],
            spend_logs: vec![("k".into(), value.into())],
        }],
    };
    assert_eq!(
        resolve_bedrock_request_metadata(&input),
        vec![("k".into(), value.into())]
    );
}

#[rstest]
fn pairs_are_capped_at_sixteen() {
    let input = BedrockRequestMetadataInput {
        allowed_fields: vec!["spend_logs_metadata".into()],
        sources: vec![BedrockMetadataSource {
            identity: vec![],
            spend_logs: (0..BEDROCK_REQUEST_METADATA_MAX_PAIRS + 4)
                .map(|index| (format!("client_{index:02}"), "v".into()))
                .collect(),
        }],
    };
    assert_eq!(
        resolve_bedrock_request_metadata(&input).len(),
        BEDROCK_REQUEST_METADATA_MAX_PAIRS
    );
}

#[rstest]
#[case::not_allowed(false)]
#[case::explicitly_allowed(true)]
fn email_requires_its_own_allowlist_entry(
    mut input: BedrockRequestMetadataInput,
    #[case] allowed: bool,
) {
    input
        .allowed_fields
        .retain(|field| allowed || field != "user_api_key_user_email");
    input.sources[0]
        .identity
        .push(("user_api_key_user_email".into(), "owner@example.com".into()));
    let resolved = resolve_bedrock_request_metadata(&input);
    assert_eq!(
        resolved
            .iter()
            .find(|(key, _)| key == "user_api_key_user_email")
            .map(|(_, value)| value.as_str()),
        allowed.then_some("owner@example.com")
    );
}

#[rstest]
#[case::canonical(BEDROCK_REQUEST_METADATA_HEADER)]
#[case::lowercase("x-amzn-bedrock-request-metadata")]
#[case::mixed("x-AMZN-bedrock-Request-METADATA")]
fn caller_header_is_replaced_in_every_casing(
    input: BedrockRequestMetadataInput,
    #[case] name: &str,
) {
    let headers = bedrock_request_metadata_headers(
        vec![
            (name.into(), "FORGED".into()),
            ("x-other".into(), "keep".into()),
        ],
        &input,
    )
    .unwrap();
    assert_eq!(headers[0], ("x-other".into(), "keep".into()));
    assert_eq!(headers.len(), 2);
    assert_eq!(headers[1].0, BEDROCK_REQUEST_METADATA_HEADER);
    assert_eq!(
        serde_json::from_str::<Value>(&headers[1].1).unwrap(),
        json!({"user_api_key_alias":"prod-key", "user_api_key_team_alias":"platform", "cost_center":"cc-1"})
    );
}

#[rstest]
#[case::empty_sources(vec![])]
#[case::invalid_sources(vec![BedrockMetadataSource { identity: vec![("user_api_key_alias".into(), "O'Brien".into())], spend_logs: vec![] }])]
fn owned_empty_metadata_evicts_the_caller(
    mut input: BedrockRequestMetadataInput,
    #[case] sources: Vec<BedrockMetadataSource>,
) {
    input.sources = sources;
    assert!(resolve_bedrock_request_metadata(&input).is_empty());
    assert!(
        bedrock_request_metadata_headers(
            vec![(BEDROCK_REQUEST_METADATA_HEADER.into(), "FORGED".into())],
            &input
        )
        .unwrap()
        .is_empty()
    );
}

#[rstest]
fn disabled_forwarding_keeps_the_caller_header(input: BedrockRequestMetadataInput) {
    let off = BedrockRequestMetadataInput {
        allowed_fields: vec![],
        ..input
    };
    let headers = vec![(BEDROCK_REQUEST_METADATA_HEADER.into(), "caller-set".into())];
    assert_eq!(
        bedrock_request_metadata_headers(headers.clone(), &off).unwrap(),
        headers
    );
    assert!(resolve_bedrock_request_metadata(&off).is_empty());
}

#[rstest]
fn header_value_matches_pythons_escaped_json_bytes() {
    let input = BedrockRequestMetadataInput {
        allowed_fields: vec!["user_api_key_alias".into(), "spend_logs_metadata".into()],
        sources: vec![BedrockMetadataSource {
            identity: vec![("user_api_key_alias".into(), "prod-key".into())],
            spend_logs: vec![("cost_center".into(), "a\u{a0}b".into())],
        }],
    };
    let headers = bedrock_request_metadata_headers(vec![], &input).unwrap();
    assert_eq!(
        headers[0].1,
        "{\"user_api_key_alias\":\"prod-key\",\"cost_center\":\"a\\u00a0b\"}"
    );
}

fn fixed_clock() -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(1_700_000_000)
}

#[rstest]
fn resolved_metadata_header_is_signed(input: BedrockRequestMetadataInput) {
    let headers = bedrock_request_metadata_headers(vec![], &input).unwrap();
    let signer = SigV4Signer::new(
        "us-east-1".into(),
        BEDROCK_SERVICE,
        litellm_auth_aws::Credentials::new("test", "secret", None, None, "test"),
    )
    .with_clock(fixed_clock);
    let sent = OutboundRequest::signed_json(
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke".into(),
        headers,
        &json!({"messages":[]}),
        None,
        &signer,
    )
    .unwrap();
    assert!(
        sent.header("Authorization")
            .unwrap()
            .contains("x-amzn-bedrock-request-metadata")
    );
    assert_eq!(
        serde_json::from_str::<Value>(sent.header(BEDROCK_REQUEST_METADATA_HEADER).unwrap())
            .unwrap(),
        json!({"user_api_key_alias":"prod-key", "user_api_key_team_alias":"platform", "cost_center":"cc-1"})
    );
}
