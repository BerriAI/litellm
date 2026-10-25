use crate::{Object, Raw};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct VectorStore {
    pub vector_store_name: Option<String>,
    pub litellm_params: Option<VectorStoreParams>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct VectorStoreParams {
    pub vector_store_id: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub vector_store_description: Option<String>,
    pub vector_store_metadata: Option<Raw>,
    #[serde(flatten)]
    pub additional_fields: Object,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::Config;

    fn parse(yaml: &str) -> Config {
        Config::from_yaml(yaml).unwrap()
    }

    #[rstest]
    fn vector_store_entries_keep_typed_params_and_unknown_keys() {
        let config = parse(
            r#"
vector_store_registry:
  - vector_store_name: kb
    litellm_params: {vector_store_id: vs_123, custom_llm_provider: openai, region: eu}
"#,
        );
        let store = &config.vector_store_registry[0];
        let params = store.litellm_params.as_ref().unwrap();

        assert_eq!(store.vector_store_name.as_deref(), Some("kb"));
        assert_eq!(params.vector_store_id.as_deref(), Some("vs_123"));
        assert_eq!(params.custom_llm_provider.as_deref(), Some("openai"));
        assert!(params.additional_fields.contains_key("region"));
    }

    #[rstest]
    #[case::vector_store_metadata(
        r#"
vector_store_registry:
  - litellm_params: {vector_store_metadata: SENTINEL}
"#
    )]
    #[case::vector_store_extra(
        r#"
vector_store_registry:
  - litellm_params: {api_key: SENTINEL}
"#
    )]
    fn debug_never_prints_vector_store_values(#[case] yaml: &str) {
        assert!(!format!("{:?}", parse(yaml)).contains("SENTINEL"));
    }
}
