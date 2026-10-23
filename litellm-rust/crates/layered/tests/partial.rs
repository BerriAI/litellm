use litellm_layered::{Layered, Sources, merge, read};
use rstest::rstest;
use serde::{Deserialize, Deserializer};
use serde_json::{Value, json};

#[derive(Layered)]
#[expect(dead_code, reason = "only the derived partial is built")]
struct Credentials {
    #[layer(
        litellm_param = ["project", "legacy_project"],
        litellm_setting = "credential_project",
        env = ["PROJECT", "PROJECT_FALLBACK"],
        read = read::text,
    )]
    project: String,
    #[layer(litellm_param = "token", env = "TOKEN", read = read::text, secret, setting = "API token")]
    token: String,
    #[layer(litellm_param = "document", deserialize_with = document_text)]
    document: String,
    #[layer(env = "LIMIT")]
    limit: u32,
}

fn document_text<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<String>, D::Error> {
    Ok(match Value::deserialize(deserializer)? {
        Value::String(text) => Some(text),
        Value::Object(object) => Some(Value::Object(object).to_string()),
        _ => None,
    })
}

fn params(value: Value) -> serde_json::Map<String, Value> {
    serde_json::from_value(value).unwrap()
}

#[rstest]
#[case::first_alias(json!({"project": "first", "legacy_project": "second"}), Some("first"))]
#[case::second_alias(json!({"legacy_project": "second"}), Some("second"))]
#[case::blank_falls_to_alias(json!({"project": " ", "legacy_project": "second"}), Some("second"))]
#[case::null_falls_to_alias(json!({"project": null, "legacy_project": "second"}), Some("second"))]
#[case::non_string_is_ignored(json!({"project": 5}), None)]
#[case::absent(json!({}), None)]
fn litellm_params_read_aliases_in_order(#[case] value: Value, #[case] expected: Option<&str>) {
    assert_eq!(
        CredentialsPartial::from_litellm_params(&params(value))
            .project
            .as_deref(),
        expected
    );
}

#[rstest]
fn litellm_params_use_the_field_decoder() {
    let partial = CredentialsPartial::from_litellm_params(&params(json!({
        "document": {"type": "service_account"},
        "limit": 5,
    })));
    assert_eq!(
        partial.document.as_deref(),
        Some(r#"{"type":"service_account"}"#)
    );
    assert_eq!(partial.limit, None);
}

#[rstest]
fn merge_takes_each_field_from_the_highest_partial_that_sets_it() {
    let merged = merge([
        CredentialsPartial {
            project: Some("params".into()),
            ..CredentialsPartial::default()
        },
        CredentialsPartial {
            project: Some("env".into()),
            token: Some("env-token".into()),
            ..CredentialsPartial::default()
        },
    ]);
    assert_eq!(merged.project.as_deref(), Some("params"));
    assert_eq!(merged.token.as_deref(), Some("env-token"));
    assert_eq!(
        merge(Vec::<CredentialsPartial>::new()),
        CredentialsPartial::default()
    );
}

#[rstest]
fn debug_redacts_secret_fields() {
    let debug = format!(
        "{:?}",
        CredentialsPartial {
            project: Some("visible-project".into()),
            token: Some("hidden-token".into()),
            ..CredentialsPartial::default()
        }
    );
    assert!(debug.contains("visible-project"));
    assert!(!debug.contains("hidden-token"));
    assert!(debug.contains("REDACTED"));
}

#[rstest]
fn name_lists_follow_the_declaration() {
    assert_eq!(
        Credentials::LITELLM_PARAMS,
        ["project", "legacy_project", "token", "document"]
    );
    assert_eq!(
        Credentials::LITELLM_SETTINGS,
        [("credential_project", "project")]
    );
    assert_eq!(
        Credentials::ENV_NAMES,
        ["PROJECT", "PROJECT_FALLBACK", "TOKEN", "LIMIT"]
    );
    assert_eq!(
        Credentials::PROJECT,
        &Sources {
            setting: "project",
            litellm_params: &["project", "legacy_project"],
            litellm_setting: Some("credential_project"),
            env: &["PROJECT", "PROJECT_FALLBACK"],
        }
    );
    assert_eq!(Credentials::SOURCES[1].setting, "API token");
}

#[rstest]
#[case::every_source(
    Credentials::PROJECT,
    "pass project or legacy_project, set litellm.credential_project, or set PROJECT or PROJECT_FALLBACK"
)]
#[case::params_and_env(Credentials::TOKEN, "pass token or set TOKEN")]
#[case::params_only(Credentials::DOCUMENT, "pass document")]
#[case::env_only(Credentials::LIMIT, "set LIMIT")]
fn guidance_names_exactly_the_declared_sources(#[case] sources: &Sources, #[case] expected: &str) {
    assert_eq!(sources.guidance(), expected);
}
