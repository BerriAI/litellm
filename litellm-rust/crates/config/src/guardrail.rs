use litellm_auth_types::SecretValue;

use crate::{Object, Raw, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct Guardrail {
    pub guardrail_id: Option<String>,
    pub guardrail_name: Option<String>,
    pub litellm_params: Option<GuardrailParams>,
    pub guardrail_info: Option<Object>,
    pub policy_template: Option<String>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct GuardrailParams {
    pub guardrail: Option<String>,
    pub mode: Option<Raw>,
    pub default_on: Option<Spelled<bool>>,
    pub api_key: Option<SecretValue>,
    pub api_base: Option<String>,
    #[serde(flatten)]
    pub additional_fields: Object,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{Config, Spelled, Value};

    fn parse(yaml: &str) -> Config {
        Config::from_yaml(yaml).unwrap()
    }

    #[rstest]
    fn guardrail_entries_keep_typed_params_and_unknown_keys() {
        let config = parse(
            r#"
guardrails:
  - guardrail_name: pii
    litellm_params:
      guardrail: presidio
      mode: [pre_call, post_call]
      default_on: "true"
      api_key: os.environ/PRESIDIO_KEY
      presidio_language: en
    guardrail_info:
      owner: security
"#,
        );
        let guardrail = &config.guardrails[0];
        let params = guardrail.litellm_params.as_ref().unwrap();

        assert_eq!(guardrail.guardrail_name.as_deref(), Some("pii"));
        assert_eq!(params.guardrail.as_deref(), Some("presidio"));
        assert!(
            matches!(params.mode.as_ref().unwrap().get(), Value::Sequence(modes) if modes.len() == 2)
        );
        assert_eq!(params.default_on, Some(Spelled::Text("true".to_owned())));
        assert_eq!(
            params.api_key.as_ref().unwrap().expose(),
            "os.environ/PRESIDIO_KEY"
        );
        assert!(params.additional_fields.contains_key("presidio_language"));
        assert!(
            guardrail
                .guardrail_info
                .as_ref()
                .unwrap()
                .contains_key("owner")
        );
    }

    #[rstest]
    #[case::guardrail_key(
        r#"
guardrails:
  - litellm_params: {api_key: SENTINEL}
"#
    )]
    #[case::guardrail_extra(
        r#"
guardrails:
  - litellm_params: {vendor_secret: SENTINEL}
"#
    )]
    #[case::guardrail_info(
        r#"
guardrails:
  - guardrail_info: {owner: SENTINEL}
"#
    )]
    #[case::guardrail_mode(
        r#"
guardrails:
  - litellm_params: {mode: SENTINEL}
"#
    )]
    fn debug_never_prints_guardrail_values(#[case] yaml: &str) {
        assert!(!format!("{:?}", parse(yaml)).contains("SENTINEL"));
    }
}
