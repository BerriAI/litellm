use std::collections::BTreeSet;

use litellm_auth_types::VertexParams;
use rstest::rstest;
use serde_json::{Map, Value, json};

#[rstest]
#[case::current_spelling(
    VertexParams { vertex_project: Some("p".into()), vertex_ai_project: Some("old".into()), ..VertexParams::default() },
    Some("p")
)]
#[case::legacy_spelling(
    VertexParams { vertex_ai_project: Some("old".into()), ..VertexParams::default() },
    Some("old")
)]
#[case::blank_falls_through(
    VertexParams { vertex_project: Some("  ".into()), vertex_ai_project: Some("old".into()), ..VertexParams::default() },
    Some("old")
)]
#[case::unset(VertexParams::default(), None)]
fn params_prefer_the_current_spelling_over_the_legacy_one(
    #[case] params: VertexParams,
    #[case] project: Option<&str>,
) {
    assert_eq!(params.project().as_deref(), project);
    let mirrored = VertexParams {
        vertex_credentials: params.vertex_project.clone(),
        vertex_ai_credentials: params.vertex_ai_project.clone(),
        vertex_location: params.vertex_project.clone(),
        vertex_ai_location: params.vertex_ai_project.clone(),
        ..VertexParams::default()
    };
    assert_eq!(mirrored.credentials().as_deref(), project);
    assert_eq!(mirrored.location().as_deref(), project);
}

#[rstest]
#[case::text(json!({"vertex_credentials": "/path/to/key.json"}), Some("/path/to/key.json"))]
#[case::object(json!({"vertex_credentials": {"type": "service_account"}}), Some(r#"{"type":"service_account"}"#))]
#[case::null(json!({"vertex_credentials": null}), None)]
#[case::empty_object_falls_through_to_the_legacy_spelling(
    json!({"vertex_credentials": {}, "vertex_ai_credentials": "/legacy/key.json"}),
    Some("/legacy/key.json")
)]
#[case::absent(json!({}), None)]
fn credentials_deserialize_from_text_or_an_object(
    #[case] params: Value,
    #[case] credentials: Option<&str>,
) {
    let params: VertexParams = serde_json::from_value(params).unwrap();
    assert_eq!(params.credentials().as_deref(), credentials);
}

#[rstest]
#[case::number(json!({"vertex_credentials": 7}))]
#[case::list(json!({"vertex_ai_credentials": ["a"]}))]
#[case::project(json!({"vertex_project": ["p"]}))]
fn a_param_of_the_wrong_type_is_rejected(#[case] params: Value) {
    assert!(serde_json::from_value::<VertexParams>(params).is_err());
}

#[rstest]
fn fields_name_every_param_once() {
    let filled: Value = VertexParams::fields()
        .map(|name| (name.to_string(), Value::from(name)))
        .collect::<Map<_, _>>()
        .into();
    let params: VertexParams = serde_json::from_value(filled).unwrap();
    let expected = VertexParams {
        vertex_credentials: Some("vertex_credentials".into()),
        vertex_project: Some("vertex_project".into()),
        vertex_location: Some("vertex_location".into()),
        vertex_ai_credentials: Some("vertex_ai_credentials".into()),
        vertex_ai_project: Some("vertex_ai_project".into()),
        vertex_ai_location: Some("vertex_ai_location".into()),
    };
    assert_eq!(params, expected);
    assert_eq!(
        VertexParams::fields().collect::<BTreeSet<_>>().len(),
        VertexParams::fields().count()
    );
}

#[rstest]
fn debug_output_does_not_expose_credentials() {
    let params = VertexParams {
        vertex_credentials: Some("current-secret".into()),
        vertex_ai_credentials: Some("legacy-secret".into()),
        vertex_project: Some("project-1".into()),
        ..VertexParams::default()
    };
    let debug = format!("{params:?}");

    assert!(!debug.contains("current-secret"));
    assert!(!debug.contains("legacy-secret"));
    assert!(debug.contains("project-1"));
}

#[rstest]
#[case::current_spelling(Some("p"), Some("legacy"), &[("VERTEXAI_PROJECT", "env")], Some("p"))]
#[case::legacy_spelling(None, Some("legacy"), &[("VERTEXAI_PROJECT", "env")], Some("legacy"))]
#[case::blank_params_fall_to_the_environment(Some(" "), None, &[("VERTEXAI_PROJECT", "env")], Some("env"))]
#[case::nothing(None, None, &[], None)]
fn a_spec_resolves_from_the_params_then_the_environment(
    #[case] vertex_project: Option<&str>,
    #[case] vertex_ai_project: Option<&str>,
    #[case] environment: &[(&str, &str)],
    #[case] expected: Option<&str>,
) {
    let params = VertexParams {
        vertex_project: vertex_project.map(str::to_string),
        vertex_ai_project: vertex_ai_project.map(str::to_string),
        ..VertexParams::default()
    };
    let env = |name: &str| {
        environment
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value.to_string())
    };

    assert_eq!(
        params.resolve(&VertexParams::PROJECT, &env).as_deref(),
        expected
    );
}
