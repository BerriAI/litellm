use litellm_auth_types::SecretValue;

use crate::{Object, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct Prompt {
    pub prompt_id: Option<String>,
    pub litellm_params: Option<PromptParams>,
    pub prompt_info: Option<PromptInfo>,
    pub version: Option<Spelled<i64>>,
    pub environment: Option<String>,
    pub created_by: Option<String>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct PromptParams {
    pub prompt_id: Option<String>,
    pub prompt_integration: Option<String>,
    pub api_base: Option<String>,
    pub api_key: Option<SecretValue>,
    pub provider_specific_query_params: Option<Object>,
    pub ignore_prompt_manager_model: Option<Spelled<bool>>,
    pub ignore_prompt_manager_optional_params: Option<Spelled<bool>>,
    pub dotprompt_content: Option<String>,
    #[serde(flatten)]
    pub additional_fields: Object,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct PromptInfo {
    pub prompt_type: Option<String>,
    pub environment: Option<String>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{Config, Spelled};

    fn parse(yaml: &str) -> Config {
        Config::from_yaml(yaml).unwrap()
    }

    #[rstest]
    fn prompt_entries_keep_typed_params_and_info() {
        let config = parse(
            r#"
prompts:
  - prompt_id: greeting
    version: 2
    litellm_params:
      prompt_integration: dotprompt
      ignore_prompt_manager_model: true
      prompt_extension: x
    prompt_info:
      prompt_type: config
"#,
        );
        let prompt = &config.prompts[0];
        let params = prompt.litellm_params.as_ref().unwrap();

        assert_eq!(prompt.prompt_id.as_deref(), Some("greeting"));
        assert_eq!(prompt.version, Some(Spelled::Value(2)));
        assert_eq!(params.prompt_integration.as_deref(), Some("dotprompt"));
        assert_eq!(
            params.ignore_prompt_manager_model,
            Some(Spelled::Value(true))
        );
        assert!(params.additional_fields.contains_key("prompt_extension"));
        assert_eq!(
            prompt.prompt_info.as_ref().unwrap().prompt_type.as_deref(),
            Some("config")
        );
    }

    #[rstest]
    #[case::prompt_key(
        r#"
prompts:
  - litellm_params: {api_key: SENTINEL}
"#
    )]
    #[case::prompt_query_params(
        r#"
prompts:
  - litellm_params: {provider_specific_query_params: {token: SENTINEL}}
"#
    )]
    #[case::prompt_extra(
        r#"
prompts:
  - litellm_params: {vendor_secret: SENTINEL}
"#
    )]
    fn debug_never_prints_prompt_values(#[case] yaml: &str) {
        assert!(!format!("{:?}", parse(yaml)).contains("SENTINEL"));
    }
}
