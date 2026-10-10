use std::collections::BTreeMap;

use litellm_auth_gcp::{
    VertexConfig,
    constants::{GOOGLE_APPLICATION_CREDENTIALS_ENV, VERTEX_LOCATION_ENV},
    get_vertex_ai_location, get_vertex_ai_project, get_vertex_ai_project_from_credentials,
};
use litellm_auth_types::VertexParams;
use rstest::rstest;
use serde_json::{Value, json};

fn config(value: Value) -> VertexConfig {
    VertexConfig::from_sourced_optional_params(value.as_object().unwrap(), &BTreeMap::new())
        .unwrap()
}

#[rstest]
fn config_is_typed_and_secrets_are_redacted() {
    let config = config(json!({
        "vertex_credentials":{"private_key":"secret-key"},
        "vertex_project":"project-1",
        "vertex_location":"europe-west4"
    }));
    assert_eq!(config.project_id(), Some("project-1"));
    assert_eq!(config.location(), Some("europe-west4"));
    assert!(!format!("{config:?}").contains("secret-key"));
    assert!(
        VertexConfig::from_sourced_optional_params(
            json!({"vertex_credentials":true}).as_object().unwrap(),
            &BTreeMap::new()
        )
        .is_err()
    );
}

#[rstest]
fn project_and_location_prefer_input_then_environment() {
    let configured =
        config(json!({"vertex_project":"input-project","vertex_location":"input-location"}));
    let env = |name: &str| Some(format!("env-{name}"));
    assert_eq!(
        get_vertex_ai_project(&configured, &env).as_deref(),
        Some("input-project")
    );
    assert_eq!(
        get_vertex_ai_location(&configured, &env).as_deref(),
        Some("input-location")
    );
    let empty = VertexConfig::default();
    assert_eq!(
        get_vertex_ai_project(&empty, &|_| Some("env-project".into())).as_deref(),
        Some("env-project")
    );
    assert_eq!(
        get_vertex_ai_location(&empty, &|name| (name == VERTEX_LOCATION_ENV)
            .then(|| "fallback-location".into()))
        .as_deref(),
        Some("fallback-location")
    );
}

#[rstest]
fn project_is_read_from_the_credentials_that_would_authenticate() {
    let key = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(key.path(), r#"{"project_id":"file-project"}"#).unwrap();
    let path = key.path().to_str().unwrap().to_string();
    let inline = VertexConfig::from_params(&VertexParams {
        vertex_credentials: Some(r#"{"project_id":"inline-project"}"#.into()),
        ..VertexParams::default()
    });
    assert_eq!(
        get_vertex_ai_project_from_credentials(&inline, &|_| None).as_deref(),
        Some("inline-project")
    );
    let from_file = VertexConfig::from_params(&VertexParams {
        vertex_credentials: Some(path.clone()),
        ..VertexParams::default()
    });
    assert_eq!(
        get_vertex_ai_project_from_credentials(&from_file, &|_| None).as_deref(),
        Some("file-project")
    );
    let empty = VertexConfig::default();
    assert_eq!(
        get_vertex_ai_project_from_credentials(&empty, &|name| (name
            == GOOGLE_APPLICATION_CREDENTIALS_ENV)
            .then(|| path.clone()))
        .as_deref(),
        Some("file-project")
    );
    assert_eq!(
        get_vertex_ai_project_from_credentials(&empty, &|_| None),
        None
    );
}
