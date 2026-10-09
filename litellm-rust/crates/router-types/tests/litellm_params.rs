use litellm_auth_types::{AwsParams, SecretValue, VertexParams};
use litellm_router_types::{LitellmParams, Spelled};
use rstest::rstest;
use serde_json::{Map, Value, json};

fn params(value: Value) -> LitellmParams {
    serde_json::from_value(value).unwrap()
}

#[rstest]
#[case::aws_group(
    json!({"model": "m", "aws_region_name": "eu-west-1", "aws_access_key_id": "AKIA"}),
    LitellmParams {
        model: "m".into(),
        aws: AwsParams {
            aws_region_name: Some("eu-west-1".into()),
            aws_access_key_id: Some("AKIA".into()),
            ..AwsParams::default()
        },
        ..LitellmParams::default()
    },
)]
#[case::vertex_group(
    json!({"model": "m", "vertex_project": "p", "vertex_ai_location": "eu", "vertex_credentials": {"type": "service_account"}}),
    LitellmParams {
        model: "m".into(),
        vertex: VertexParams {
            vertex_project: Some("p".into()),
            vertex_ai_location: Some("eu".into()),
            vertex_credentials: Some(r#"{"type":"service_account"}"#.into()),
            ..VertexParams::default()
        },
        ..LitellmParams::default()
    },
)]
#[case::both_groups_and_the_credentials(
    json!({"model": "m", "api_key": "k", "api_base": "b", "aws_region_name": "eu-west-1", "vertex_location": "us-east5"}),
    LitellmParams {
        model: "m".into(),
        api_key: Some(SecretValue::new("k")),
        api_base: Some("b".into()),
        aws: AwsParams {
            aws_region_name: Some("eu-west-1".into()),
            ..AwsParams::default()
        },
        vertex: VertexParams {
            vertex_location: Some("us-east5".into()),
            ..VertexParams::default()
        },
        ..LitellmParams::default()
    },
)]
#[case::deployment_settings(
    json!({"model": "m", "timeout": "os.environ/T", "rpm": 5, "drop_params": "true", "tags": ["a"], "max_budget": 1.5}),
    LitellmParams {
        model: "m".into(),
        timeout: Some(Spelled::Text("os.environ/T".into())),
        rpm: Some(Spelled::Value(5.0)),
        drop_params: Some(Spelled::Text("true".into())),
        tags: Some(Box::from(["a".to_string()])),
        max_budget: Some(1.5),
        ..LitellmParams::default()
    },
)]
#[case::explicit_null_is_absent(
    json!({"model": "m", "aws_region_name": null, "vertex_project": null, "api_key": null, "tpm": null}),
    LitellmParams { model: "m".into(), ..LitellmParams::default() },
)]
#[case::one_organization(
    json!({"model": "m", "organization": "org-a"}),
    LitellmParams {
        model: "m".into(),
        organization: Some(vec!["org-a".into()]),
        ..LitellmParams::default()
    },
)]
#[case::organizations_the_router_expands(
    json!({"model": "m", "organization": ["org-a", "org-b"]}),
    LitellmParams {
        model: "m".into(),
        organization: Some(vec!["org-a".into(), "org-b".into()]),
        ..LitellmParams::default()
    },
)]
fn deserializes_each_group_from_a_config_or_a_call(
    #[case] value: Value,
    #[case] expected: LitellmParams,
) {
    assert_eq!(params(value), expected);
}

#[rstest]
fn keys_no_field_names_land_in_extra_and_nothing_else_does() {
    let typed = params(json!({
        "model": "m",
        "api_key": "k",
        "aws_region_name": "eu-west-1",
        "vertex_project": "p",
        "rpm": 5,
        "azure_ad_token": "t",
        "messages": [],
    }));

    assert_eq!(
        typed.extra.keys().collect::<Vec<_>>(),
        ["azure_ad_token", "messages"]
    );
    assert_eq!(typed.extra["azure_ad_token"], "t");
    assert_eq!(typed.aws.aws_region_name.as_deref(), Some("eu-west-1"));
    assert_eq!(typed.vertex.vertex_project.as_deref(), Some("p"));
}

#[rstest]
#[case::model_missing(json!({"api_key": "k"}))]
#[case::aws(json!({"model": "m", "aws_region_name": 7}))]
#[case::vertex(json!({"model": "m", "vertex_project": ["p"]}))]
#[case::vertex_credentials(json!({"model": "m", "vertex_credentials": 7}))]
#[case::rate_limit(json!({"model": "m", "max_parallel_requests": "many"}))]
#[case::tags(json!({"model": "m", "tags": "not-a-list"}))]
fn a_param_of_the_wrong_type_is_rejected(#[case] value: Value) {
    assert!(serde_json::from_value::<LitellmParams>(value).is_err());
}

#[rstest]
fn fields_fill_the_model_the_credentials_and_every_group() {
    let filled: Value = LitellmParams::fields()
        .map(|name| (name.to_string(), Value::from(name)))
        .collect::<Map<_, _>>()
        .into();
    let typed = params(filled);
    let nulls = |group: Value| {
        group
            .as_object()
            .unwrap()
            .values()
            .filter(|value| value.is_null())
            .count()
    };

    assert_eq!(typed.model, "model");
    assert_eq!(typed.api_key, Some(SecretValue::new("api_key")));
    assert_eq!(typed.api_base.as_deref(), Some("api_base"));
    assert_eq!(typed.api_version.as_deref(), Some("api_version"));
    assert_eq!(nulls(serde_json::to_value(&typed.aws).unwrap()), 0);
    assert_eq!(nulls(serde_json::to_value(&typed.vertex).unwrap()), 0);
    assert!(typed.extra.is_empty());
}

#[rstest]
fn debug_shows_neither_the_api_key_nor_the_extra_values() {
    let typed =
        params(json!({"model": "m", "api_key": "sk-secret", "azure_ad_token": "token-secret"}));
    let debug = format!("{typed:?}");

    assert!(!debug.contains("sk-secret"));
    assert!(!debug.contains("token-secret"));
    assert!(debug.contains("azure_ad_token"));
}
