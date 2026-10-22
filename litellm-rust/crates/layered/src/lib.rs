extern crate self as litellm_layered;

mod env;
mod error;
pub mod read;
mod sources;

pub use env::Env;
pub use error::{Error, Invalid};
pub use litellm_core_utils::settings::Lookup;
pub use litellm_layered_derive::Layered;
pub use sources::Sources;

pub trait Partial: Default {
    fn or(self, lower: Self) -> Self;
}

pub fn merge<P: Partial>(highest_precedence_first: impl IntoIterator<Item = P>) -> P {
    highest_precedence_first
        .into_iter()
        .reduce(P::or)
        .unwrap_or_default()
}

#[doc(hidden)]
pub mod __private {
    use std::fmt;

    use serde_json::{Map, Value};

    pub use serde;
    pub use serde_json;

    use crate::{Error, Invalid, Lookup};

    pub fn from_env<T>(
        env: &(impl Lookup + ?Sized),
        names: &'static [&'static str],
        blank_is_unset: bool,
        read: impl Fn(&str) -> Result<Option<T>, Invalid>,
    ) -> Result<Option<T>, Error> {
        for name in names {
            let Some(value) = env.get(name) else {
                continue;
            };
            if blank_is_unset && value.trim().is_empty() {
                continue;
            }
            if let Some(value) = read(&value).map_err(|invalid| Error { name, invalid })? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    pub fn from_litellm_params<T>(
        params: &Map<String, Value>,
        names: &[&str],
        blank_is_unset: bool,
        decode: impl Fn(&Value) -> Option<T>,
    ) -> Option<T> {
        names
            .iter()
            .filter_map(|name| params.get(*name))
            .filter(|value| !(blank_is_unset && is_blank(value)))
            .find_map(decode)
    }

    fn is_blank(value: &Value) -> bool {
        value.as_str().is_some_and(|text| text.trim().is_empty())
    }

    pub struct Redacted<'a, T>(pub &'a Option<T>);

    impl<T> fmt::Debug for Redacted<'_, T> {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self.0 {
                Some(_) => formatter.write_str("Some(\"[REDACTED]\")"),
                None => formatter.write_str("None"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rstest::rstest;
    use serde::{Deserialize, Deserializer};
    use serde_json::{Value, json};

    use super::*;

    #[derive(Layered)]
    #[allow(dead_code)]
    struct Settings {
        #[layer(env = ["FIRST_NAME", "SECOND_NAME"], read = read::text)]
        name: String,
        #[layer(env = "SWITCH", read = read::enabled)]
        switch: bool,
        #[layer(env = "COUNT", read = read::parsed::<u32>)]
        count: u32,
        #[layer(env = "PORT", read = read::required::<u16>)]
        port: u16,
        #[layer(env = "IDLE", read = read::seconds)]
        idle: Duration,
        #[layer(from_env = window_from_env)]
        window: (u32, u32),
        untouched_by_env: bool,
    }

    fn window_from_env(env: &(impl Lookup + ?Sized)) -> Result<Option<(u32, u32)>, Error> {
        let start = __private::from_env(env, &["WINDOW_START"], false, read::required::<u32>)?;
        let end = __private::from_env(env, &["WINDOW_END"], false, read::required::<u32>)?;
        Ok(start.zip(end))
    }

    #[derive(Layered)]
    #[layer(blank_is_unset)]
    #[allow(dead_code)]
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

    fn document_text<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<String>, D::Error> {
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
    fn each_field_reads_its_env_names_in_order() {
        let partial = SettingsPartial::from_env(&Env::of(&[
            ("SECOND_NAME", "second"),
            ("SWITCH", "True"),
            ("COUNT", " 7 "),
            ("PORT", "8080"),
            ("IDLE", "45"),
            ("WINDOW_START", "1"),
            ("WINDOW_END", "2"),
        ]))
        .unwrap();
        assert_eq!(
            partial,
            SettingsPartial {
                name: Some("second".into()),
                switch: Some(true),
                count: Some(7),
                port: Some(8080),
                idle: Some(Duration::from_secs(45)),
                window: Some((1, 2)),
                untouched_by_env: None,
            }
        );
    }

    #[rstest]
    fn the_first_set_env_name_wins() {
        let partial =
            SettingsPartial::from_env(&Env::of(&[("FIRST_NAME", ""), ("SECOND_NAME", "second")]))
                .unwrap();
        assert_eq!(partial.name.as_deref(), Some(""));
    }

    #[rstest]
    fn a_field_without_env_never_comes_from_env() {
        let partial = SettingsPartial::from_env(&Env::of(&[("UNTOUCHED_BY_ENV", "true")])).unwrap();
        assert_eq!(partial, SettingsPartial::default());
    }

    #[rstest]
    #[case::switch_off("SWITCH", "false")]
    #[case::switch_one("SWITCH", "1")]
    #[case::lax_count("COUNT", "many")]
    fn lax_values_that_do_not_read_are_unset(#[case] name: &str, #[case] value: &str) {
        assert_eq!(
            SettingsPartial::from_env(&Env::of(&[(name, value)])).unwrap(),
            SettingsPartial::default()
        );
    }

    #[rstest]
    #[case::port("PORT", "abc")]
    #[case::port_overflow("PORT", "65536")]
    #[case::from_env_field("WINDOW_START", "soon")]
    fn strict_values_that_do_not_read_fail_naming_the_variable(
        #[case] name: &'static str,
        #[case] value: &str,
    ) {
        let error = SettingsPartial::from_env(&Env::of(&[(name, value)])).unwrap_err();
        assert_eq!(error.name, name);
        assert!(error.to_string().contains(name));
    }

    #[rstest]
    fn blank_values_fall_through_to_the_next_name_when_blank_is_unset() {
        let partial = CredentialsPartial::from_env(&Env::of(&[
            ("PROJECT", "  "),
            ("PROJECT_FALLBACK", "fallback"),
        ]))
        .unwrap();
        assert_eq!(partial.project.as_deref(), Some("fallback"));
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
    fn guidance_names_exactly_the_declared_sources(
        #[case] sources: &Sources,
        #[case] expected: &str,
    ) {
        assert_eq!(sources.guidance(), expected);
    }
}
