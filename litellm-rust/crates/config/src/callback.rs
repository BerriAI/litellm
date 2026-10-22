use litellm_auth_types::SecretValue;

use crate::{Object, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct CallbackSettings {
    pub callback_type: Option<String>,
    pub endpoint: Option<SecretValue>,
    pub headers: Option<Object>,
    pub event_types: Option<Box<[String]>>,
    pub log_format: Option<String>,
    pub max_retries: Option<Spelled<i64>>,
    pub retry_delay: Option<Spelled<f64>>,
    pub timeout: Option<Spelled<f64>>,
    #[serde(flatten)]
    pub additional_fields: Object,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{Config, Spelled};

    fn parse(yaml: &str) -> Config {
        Config::from_yaml(yaml).unwrap()
    }

    #[rstest]
    fn callback_settings_keep_typed_fields_and_unknown_keys() {
        let config = parse(
            "callback_settings:\n  hook:\n    callback_type: generic_api\n    retry_delay: 0.5\n    batch_size: 10\n",
        );
        let hook = &config.callback_settings["hook"];

        assert_eq!(hook.callback_type.as_deref(), Some("generic_api"));
        assert_eq!(hook.retry_delay, Some(Spelled::Value(0.5)));
        assert!(hook.additional_fields.contains_key("batch_size"));
    }

    #[rstest]
    #[case::env_reference("os.environ/RETRIES", Spelled::Text("os.environ/RETRIES".to_owned()))]
    #[case::number("4", Spelled::Value(4))]
    fn spelled_int_fields_keep_references_as_text(
        #[case] spelled: &str,
        #[case] expected: Spelled<i64>,
    ) {
        let config = parse(&format!(
            "callback_settings:\n  hook:\n    max_retries: {spelled}\n"
        ));

        assert_eq!(config.callback_settings["hook"].max_retries, Some(expected));
    }

    #[rstest]
    #[case::callback_endpoint(
        r#"
callback_settings:
  hook: {endpoint: SENTINEL}
"#
    )]
    #[case::callback_headers(
        r#"
callback_settings:
  hook: {headers: {Authorization: SENTINEL}}
"#
    )]
    #[case::callback_extra(
        r#"
callback_settings:
  hook: {vendor_secret: SENTINEL}
"#
    )]
    fn debug_never_prints_callback_values(#[case] yaml: &str) {
        assert!(!format!("{:?}", parse(yaml)).contains("SENTINEL"));
    }
}
