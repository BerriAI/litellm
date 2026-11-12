use std::collections::BTreeMap;

use litellm_auth_gcp::{
    VertexConfig,
    constants::{
        CLOUDSDK_CONFIG_ENV, CLOUDSDK_CORE_PROJECT_ENV, GCLOUD_PROJECT_ENV,
        GOOGLE_APPLICATION_CREDENTIALS_ENV, GOOGLE_CLOUD_PROJECT_ENV, VERTEX_LOCATION_ENV,
    },
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
    let empty_gcloud = tempfile::tempdir().unwrap();
    let empty_gcloud = empty_gcloud.path().to_str().unwrap().to_string();
    assert_eq!(
        get_vertex_ai_project_from_credentials(&empty, &|name| (name == CLOUDSDK_CONFIG_ENV)
            .then(|| empty_gcloud.clone())),
        None
    );
}

const USER: &str = r#"{"type":"authorized_user","client_id":"c","client_secret":"s","refresh_token":"r","quota_project_id":"quota-project"}"#;
const SERVICE_ACCOUNT: &str = r#"{"type":"service_account","project_id":"key-project"}"#;

#[rstest]
#[case::authorized_user_bills_its_quota_project(Some(USER), None, &[], &[], Some("quota-project"))]
#[case::explicit_credentials_ignore_the_adc_project_variable(
    Some(SERVICE_ACCOUNT), None, &[(GOOGLE_CLOUD_PROJECT_ENV, "env-project")], &[], Some("key-project")
)]
#[case::adc_project_variable(None, None, &[(GOOGLE_CLOUD_PROJECT_ENV, "env-project")], &[], Some("env-project"))]
#[case::legacy_adc_project_variable(None, None, &[(GCLOUD_PROJECT_ENV, "legacy-project")], &[], Some("legacy-project"))]
#[case::adc_project_variable_outranks_the_key_file(
    None, Some(SERVICE_ACCOUNT), &[(GOOGLE_CLOUD_PROJECT_ENV, "env-project")], &[], Some("env-project")
)]
#[case::gcloud_service_account_key(None, Some(SERVICE_ACCOUNT), &[], &[], Some("key-project"))]
#[case::gcloud_user_reads_the_active_configuration(
    None, Some(USER), &[], &[("configurations/config_default", "[core]\naccount = a@b.c\nproject = gcloud-project\n")], Some("gcloud-project")
)]
#[case::gcloud_user_reads_a_named_configuration(
    None, Some(USER), &[],
    &[("active_config", "work\n"), ("configurations/config_work", "[compute]\nproject = wrong\n[core]\nproject = work-project\n")],
    Some("work-project")
)]
#[case::gcloud_core_project_variable(
    None, Some(USER), &[(CLOUDSDK_CORE_PROJECT_ENV, "override-project")],
    &[("configurations/config_default", "[core]\nproject = gcloud-project\n")], Some("override-project")
)]
#[case::gcloud_user_without_a_configured_project(None, Some(USER), &[], &[], None)]
fn project_is_resolved_like_python_load_auth(
    #[case] credentials: Option<&str>,
    #[case] gcloud_credentials: Option<&str>,
    #[case] environment: &[(&str, &str)],
    #[case] gcloud_files: &[(&str, &str)],
    #[case] expected: Option<&str>,
) {
    let gcloud = tempfile::tempdir().unwrap();
    std::fs::create_dir(gcloud.path().join("configurations")).unwrap();
    if let Some(contents) = gcloud_credentials {
        std::fs::write(
            gcloud.path().join("application_default_credentials.json"),
            contents,
        )
        .unwrap();
    }
    for (name, contents) in gcloud_files {
        std::fs::write(gcloud.path().join(name), contents).unwrap();
    }
    let directory = gcloud.path().to_str().unwrap().to_string();
    let lookup = |name: &str| {
        if name == CLOUDSDK_CONFIG_ENV {
            return Some(directory.clone());
        }
        environment
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value.to_string())
    };
    let config = VertexConfig::from_params(&VertexParams {
        vertex_credentials: credentials.map(str::to_string),
        ..VertexParams::default()
    });
    assert_eq!(
        get_vertex_ai_project_from_credentials(&config, &lookup).as_deref(),
        expected
    );
}

#[rstest]
fn project_is_read_from_a_relative_key_file_named_like_json() {
    let key = tempfile::Builder::new()
        .prefix("{vertex")
        .suffix(".json")
        .tempfile_in(".")
        .unwrap();
    std::fs::write(key.path(), r#"{"project_id":"file-project"}"#).unwrap();
    let relative_name = key.path().file_name().unwrap().to_str().unwrap();
    let from_file = VertexConfig::from_params(&VertexParams {
        vertex_credentials: Some(relative_name.into()),
        ..VertexParams::default()
    });
    assert_eq!(
        get_vertex_ai_project_from_credentials(&from_file, &|_| None).as_deref(),
        Some("file-project")
    );
}
