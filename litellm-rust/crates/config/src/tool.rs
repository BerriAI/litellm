use litellm_auth_types::SecretValue;

use crate::{Object, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct SearchTool {
    pub search_tool_name: Option<String>,
    pub litellm_params: Option<SearchToolParams>,
    pub search_tool_info: Option<SearchToolInfo>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct SearchToolParams {
    pub search_provider: Option<String>,
    pub api_key: Option<SecretValue>,
    pub api_base: Option<String>,
    pub timeout: Option<Spelled<f64>>,
    pub max_retries: Option<Spelled<i64>>,
    pub input_cost_per_query: Option<Spelled<f64>>,
    #[serde(flatten)]
    pub additional_fields: Object,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct SearchToolInfo {
    pub description: Option<String>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct SandboxTool {
    pub sandbox_tool_name: Option<String>,
    pub litellm_params: Option<SandboxToolParams>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct SandboxToolParams {
    pub sandbox_provider: Option<String>,
    pub api_key: Option<SecretValue>,
    pub api_base: Option<String>,
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
    fn search_tools_keep_typed_params_and_info() {
        let config = parse(
            r#"
search_tools:
  - search_tool_name: web
    litellm_params:
      search_provider: perplexity
      max_retries: 3
      timeout: 2.5
      region: eu
    search_tool_info:
      description: web search
"#,
        );
        let tool = &config.search_tools[0];
        let params = tool.litellm_params.as_ref().unwrap();

        assert_eq!(tool.search_tool_name.as_deref(), Some("web"));
        assert_eq!(params.search_provider.as_deref(), Some("perplexity"));
        assert_eq!(params.max_retries, Some(Spelled::Value(3)));
        assert_eq!(params.timeout, Some(Spelled::Value(2.5)));
        assert!(params.additional_fields.contains_key("region"));
        assert_eq!(
            tool.search_tool_info
                .as_ref()
                .unwrap()
                .description
                .as_deref(),
            Some("web search")
        );
    }

    #[rstest]
    fn sandbox_tools_keep_typed_params() {
        let config = parse(
            "sandbox_tools:\n  - sandbox_tool_name: e2b\n    litellm_params: {sandbox_provider: e2b, template: base}\n",
        );
        let tool = &config.sandbox_tools[0];
        let params = tool.litellm_params.as_ref().unwrap();

        assert_eq!(tool.sandbox_tool_name.as_deref(), Some("e2b"));
        assert_eq!(params.sandbox_provider.as_deref(), Some("e2b"));
        assert!(params.additional_fields.contains_key("template"));
    }

    #[rstest]
    #[case::search_key(
        r#"
search_tools:
  - litellm_params: {api_key: SENTINEL}
"#
    )]
    #[case::search_extra(
        r#"
search_tools:
  - litellm_params: {vendor_secret: SENTINEL}
"#
    )]
    #[case::sandbox_key(
        r#"
sandbox_tools:
  - litellm_params: {api_key: SENTINEL}
"#
    )]
    #[case::sandbox_extra(
        r#"
sandbox_tools:
  - litellm_params: {vendor_secret: SENTINEL}
"#
    )]
    fn debug_never_prints_tool_values(#[case] yaml: &str) {
        assert!(!format!("{:?}", parse(yaml)).contains("SENTINEL"));
    }
}
