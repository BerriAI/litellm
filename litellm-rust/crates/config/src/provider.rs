use litellm_auth_types::SecretValue;

use crate::{Object, Raw};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct AssistantSettings {
    pub custom_llm_provider: Option<String>,
    pub litellm_params: Option<Object>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct ProviderSettings {
    pub custom_llm_provider: Option<String>,
    pub api_key: Option<SecretValue>,
    pub api_base: Option<String>,
    #[serde(flatten)]
    pub additional_fields: Object,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct VertexConfig {
    pub vertex_project: Option<String>,
    pub vertex_location: Option<String>,
    pub vertex_credentials: Option<Raw>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct Worker {
    pub worker_id: Option<String>,
    pub name: Option<String>,
    pub url: Option<String>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::Config;

    fn parse(yaml: &str) -> Config {
        Config::from_yaml(yaml).unwrap()
    }

    #[rstest]
    #[case::files("files_settings")]
    #[case::finetune("finetune_settings")]
    fn provider_settings_keep_the_provider_and_unknown_keys(#[case] section: &str) {
        let config = parse(&format!(
            "{section}:\n  - custom_llm_provider: azure\n    api_version: '2024-01-01'\n"
        ));
        let settings = config
            .files_settings
            .first()
            .or(config.finetune_settings.first())
            .unwrap();

        assert_eq!(settings.custom_llm_provider.as_deref(), Some("azure"));
        assert!(settings.additional_fields.contains_key("api_version"));
    }

    #[rstest]
    fn single_provider_sections_land_in_typed_fields() {
        let config = parse(
            r#"
assistant_settings:
  custom_llm_provider: openai
  litellm_params:
    api_key: os.environ/OPENAI_API_KEY
default_vertex_config:
  vertex_project: proj
  vertex_location: us-central1
worker_registry:
  - worker_id: eu
    name: Europe
    url: https://eu.example.test
"#,
        );
        let assistant = config.assistant_settings.as_ref().unwrap();
        let vertex = config.default_vertex_config.as_ref().unwrap();
        let worker = &config.worker_registry[0];

        assert_eq!(assistant.custom_llm_provider.as_deref(), Some("openai"));
        assert!(
            assistant
                .litellm_params
                .as_ref()
                .unwrap()
                .contains_key("api_key")
        );
        assert_eq!(vertex.vertex_project.as_deref(), Some("proj"));
        assert_eq!(vertex.vertex_location.as_deref(), Some("us-central1"));
        assert_eq!(worker.worker_id.as_deref(), Some("eu"));
        assert_eq!(worker.name.as_deref(), Some("Europe"));
        assert_eq!(worker.url.as_deref(), Some("https://eu.example.test"));
    }

    #[rstest]
    #[case::vertex_credentials("default_vertex_config: {vertex_credentials: SENTINEL}")]
    #[case::vertex_credentials_map(
        "default_vertex_config: {vertex_credentials: {private_key: SENTINEL}}"
    )]
    #[case::assistant_params("assistant_settings: {litellm_params: {api_key: SENTINEL}}")]
    #[case::files_key(
        r#"
files_settings:
  - api_key: SENTINEL
"#
    )]
    #[case::finetune_extra(
        r#"
finetune_settings:
  - vendor_secret: SENTINEL
"#
    )]
    fn debug_never_prints_provider_values(#[case] yaml: &str) {
        assert!(!format!("{:?}", parse(yaml)).contains("SENTINEL"));
    }
}
